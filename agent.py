"""Core agent loop.

Pipeline each step: auto-dismiss consent → screenshot + page text → inject
plan / memory / tab context → LLM (with confidence) → optional human approval
→ annotate + record → execute → summarize. The reasoning backend is pluggable
"""

import asyncio
import json
import os
from typing import Any, Callable

from actions import Action, execute_action
from browser import BrowserController
from config import config
from llm import LLMProvider, Message
from memory import MemoryManager
from planner import Plan, make_plan, replan
from session import SessionRecorder
from utils import (
    annotate_click,
    auto_dismiss_consent,
    capture_screenshot,
    detect_captcha,
    ensure_dir,
    extract_page_text,
    parse_json_object,
)

_SYSTEM_TEMPLATE = """You are a browser automation agent controlling a Chromium browser.

Each step you receive: a screenshot, the current URL, the extracted page text,
the open tabs, your overall plan, and context from previous steps.

Respond ONLY with a single valid JSON object — no markdown fences, no prose:
{
  "action": "<type>",
  "params": { <fields> },
  "reason": "<one short sentence>",
  "confidence": <integer 0-100, how sure you are this action is correct>
}

━━━ AVAILABLE ACTIONS ━━━

__CLICK_ACTIONS__
  navigate        → {"url": "<full https:// URL>"}
  fill            → {"selector": "<CSS selector>", "text": "<value>"}
  type            → {"text": "<string>"}
  key             → {"key": "Enter"|"Tab"|"Escape"|"ArrowDown"|...}
  scroll          → {"direction": "up"|"down", "amount": <pixels, default 300>}
  switch_tab      → {"tab_id": <int>}     switch the active tab
  close_tab       → {"tab_id": <int>}     close a tab
  done            → {"result": "<the full answer / report>"}

━━━ CRITICAL RULES ━━━

__CLICK_RULE__

CONFIDENCE
  Always include "confidence" (0-100). Be honest: low when guessing a target
  or unsure the action will work, high when the element is clearly visible.

TABS
  New tabs you open become active automatically. The open-tabs list shows ids
  and which is active (*). Use switch_tab / close_tab to manage them.

STAY ON THE PLAN
  Follow the overall plan. Use the context-from-previous-steps memory instead
  of redoing earlier work. Once a sub-goal is done, move to the next one.

USE THE PAGE TEXT — it already contains the content (articles, comments).
  Scrolling does NOT reveal new page text. Don't scroll to "read more".

NO REPEATED / CYCLING ACTIONS
  If an action produced no change, do something DIFFERENT. Never repeat a
  multi-step sequence that already failed. Being stuck does NOT mean leave the
  site — change tactics or finish with "done".

FINISHING & REPORTS
  If the task wants a summary/report, accumulate findings and output "done"
  with the COMPLETE report. A partial sourced answer beats looping to the limit.

SEARCH URLS (never type into search boxes):
  DuckDuckGo: https://duckduckgo.com/?q=...   ← PREFERRED (use this first)
  Bing:       https://www.bing.com/search?q=...
  Scholar:    https://scholar.google.com/scholar?q=...

CAPTCHA / BOT CHALLENGES
  If the page is a CAPTCHA, "unusual traffic", Cloudflare / hCaptcha /
  reCAPTCHA, or "verify you are human" wall: do NOT try to solve it and do
  NOT keep clicking. Immediately `navigate` to a DIFFERENT source for the
  same goal — another site, or a search engine that is not blocking you. If
  it persists you will be auto-redirected. Never interact with
  consent.google.com or reCAPTCHA frames.
"""

_CLICK_ACTIONS = {
    "vision": (
        '  click           → {"x": <int 0-1280>, "y": <int 0-720>, "text": "<optional:\n'
        "                     visible label at that spot>\"}  ← PRIMARY: read the\n"
        "                     screenshot and give the target's centre pixel coordinates.\n"
        '                     Include "text" whenever you can read the element\'s visible\n'
        "                     label — the browser will try clicking it by that text\n"
        "                     first, which survives layout shifts better than raw pixels.\n"
        '  click           → {"text": "<visible text>"}  ← fallback ONLY when you cannot\n'
        "                     determine coordinates."
    ),
    "text": (
        '  find_and_click  → {"text": "<button/link text visible on page>"}  ← PREFERRED\n'
        "                     for all buttons and links.\n"
        '  click           → {"x": <int 0-1280>, "y": <int 0-720>}  ← only when the\n'
        "                     target has no usable visible text."
    ),
}

_CLICK_RULE = {
    "vision": (
        "VISION-BASED CLICKING\n"
        "  Default to coordinate clicks. Read the screenshot, locate the element,\n"
        "  and return its centre x,y (0-1280 wide, 0-720 tall). Use the text\n"
        "  fallback only when coordinates are genuinely unknowable."
    ),
    "text": (
        "TEXT-BASED CLICKING\n"
        "  Prefer find_and_click with the exact visible text of the button or\n"
        "  link. Use a coordinate click only when the element has no usable text."
    ),
}


def build_system_prompt(click_mode: str) -> str:
    mode = click_mode if click_mode in _CLICK_ACTIONS else "vision"
    return _SYSTEM_TEMPLATE.replace(
        "__CLICK_ACTIONS__", _CLICK_ACTIONS[mode]
    ).replace("__CLICK_RULE__", _CLICK_RULE[mode])


LogFn = Callable[[str], None]
StopFn = Callable[[], bool]
ApproveFn = Callable[[Action, int, str], str]
FrameEmitFn = Callable[[str], None] | None


def _detect_cycle(actions: list[str], max_len: int) -> list[str] | None:
    for n in range(2, max_len + 1):
        if len(actions) >= 2 * n and actions[-n:] == actions[-2 * n : -n]:
            return actions[-n:]
    return None








_LOOP_TRACKED_ACTIONS = {"click", "find_and_click", "navigate"}
_LOOP_WINDOW = 5
_LOOP_TRIGGER = 3


def _loop_signature(url: str, action_key: str) -> str:
    """Identity of an action for loop detection: its URL + type + params."""
    return f"{url}|{action_key}"


def _loop_warning(action: Action) -> str:
    """Strict system warning injected at the top of the next prompt."""
    if action.action_type == "click" and {"x", "y"} <= set(action.params):
        return (
            "[SYSTEM WARNING] You have clicked these exact coordinates 3 "
            "times without changing the page state. Do NOT retry this "
            "coordinate click. Try scrolling, refreshing the page, or using "
            "an alternative text-based strategy to find your target link."
        )
    if action.action_type in ("click", "find_and_click"):
        text = action.params.get("text", "")
        return (
            f"[SYSTEM WARNING] You have tried clicking '{text}' 3 times "
            "without changing the page state. Do NOT retry this exact "
            "click. Try scrolling, a different element, or navigating to "
            "an alternative source."
        )
    url = action.params.get("url", "")
    return (
        f"[SYSTEM WARNING] You have navigated to '{url}' 3 times in a row "
        "without changing the page state. Do NOT navigate there again. "
        "Pick a different URL or strategy."
    )


def _save_png(path: str, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)


class BrowserAgent:
    """Drives a browser to complete a task using a pluggable LLM provider."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self._history: list[Message] = []
        self._system_prompt = build_system_prompt(config.click_mode)

    def _trim_history(self) -> None:
        keep_imgs = config.max_history_screenshots
        limit = (keep_imgs + 3) * 2
        if len(self._history) > limit:
            self._history = self._history[-limit:]
        seen = 0
        for msg in reversed(self._history):
            if msg.image is not None:
                seen += 1
                if seen > keep_imgs:
                    msg.image = None

    async def _get_next_action(
        self,
        *,
        task: str,
        screenshot: bytes,
        url: str,
        page_text: str,
        step: int,
        warnings: list[str],
        context_block: str,
        annotated_prev: bytes | None,
        log: LogFn,
    ) -> dict[str, Any]:
        if annotated_prev is not None:
            self._history.append(
                Message(
                    role="user",
                    text=(
                        "Verification of your previous click: the red dot marks "
                        "where you clicked. Confirm it worked; if not, adjust."
                    ),
                    image=annotated_prev,
                )
            )

        warning_block = ""
        if warnings:
            warning_block = "\n\n⚠️ WARNINGS:\n" + "\n".join(
                f"- {w}" for w in warnings
            )
        context = f"{context_block}\n\n" if context_block else ""

        self._history.append(
            Message(
                role="user",
                text=(
                    f"{context}"
                    f"Task: {task}\n"
                    f"URL: {url}\n"
                    f"Step: {step}/{config.max_steps}\n\n"
                    f"Page text:\n{page_text}"
                    f"{warning_block}\n\n"
                    "Reply with ONE JSON object including an integer "
                    '"confidence" (0-100). What is the next action?'
                ),
                image=screenshot,
            )
        )

        last_error: Exception | None = None
        for attempt in range(1, config.max_json_retries + 1):
            raw = await self._provider.generate(self._system_prompt, self._history)
            try:
                data = parse_json_object(raw)
                self._history.append(Message(role="model", text=raw))
                return data
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                log(f"  [WARN] Attempt {attempt}/{config.max_json_retries} — bad JSON: {exc}")
                self._history.append(Message(role="model", text=raw))
                self._history.append(
                    Message(
                        role="user",
                        text=(
                            f"Your response was not valid JSON ({exc}). Reply "
                            "with ONLY a valid JSON object — no markdown, no prose."
                        ),
                    )
                )
        raise last_error

    @staticmethod
    def _tabs_block(browser: BrowserController) -> str:
        tabs = browser.tabs()
        if not tabs:
            return ""
        rows = "\n".join(
            f"  [{t['id']}]{'*' if t['active'] else ' '} {t['url']}" for t in tabs
        )
        return f"OPEN TABS (* = active):\n{rows}"

    async def run(
        self,
        task: str,
        start_url: str,
        log: LogFn = print,
        should_stop: StopFn = lambda: False,
        approve: ApproveFn | None = None,
        supervised: bool | None = None,
        click_mode: str | None = None,
        system_instructions: str | None = None,
        emit_frame: FrameEmitFn = None,
    ) -> str:
        supervised = config.supervised if supervised is None else supervised
        approve = approve or (lambda _a, _s, _p: "y")
        click_mode = (click_mode or config.click_mode).strip().lower()
        click_mode = click_mode if click_mode in ("vision", "text") else "vision"
        self._system_prompt = build_system_prompt(click_mode)
        if system_instructions:
            self._system_prompt += f"\n\n━━━ CUSTOM INSTRUCTIONS ━━━\n{system_instructions}\n"

        ensure_dir(config.debug_dir)
        ensure_dir(config.sessions_dir)

        browser = BrowserController()
        session = SessionRecorder(task, start_url)
        memory = MemoryManager(
            self._provider,
            config.max_history_screenshots,
            config.enable_memory_summary,
        )

        success = False
        result_text = "Reached the maximum number of steps without completing the task."

        try:
            await browser.start()
            await browser.navigate(start_url)

            self._live_browser = browser

            if emit_frame is not None:
                await browser.start_screencast(emit_frame)

            log(f"  Clicking : {click_mode} mode")

            log("\nPlanning…")
            plan: Plan = await make_plan(self._provider, task, start_url)
            log(plan.pretty() if plan.steps else "  (no plan — proceeding anyway)")

            last_url = ""
            last_page_text = ""
            stuck_count = 0
            action_log: list[str] = []
            fail_count = 0
            fail_key: str | None = None
            last_replan_step = -(10 ** 6)
            pending_annotated: bytes | None = None
            captcha_streak = 0
            loop_history: list[str] = []
            pending_loop_warning: str | None = None

            
            for step in range(1, config.max_steps + 1):
                
                if step > config.max_steps:
                    log(f"  [LIMIT] Breaking loop: step {step} has exceeded max_steps of {config.max_steps}")
                    break

                if should_stop():
                    result_text = "Stopped by user."
                    return result_text

                dismissed = await auto_dismiss_consent(browser.page)
                if dismissed:
                    log("  [AUTO] Dismissed consent dialog")
                    await asyncio.sleep(0.5)

                url = browser.current_url()
                screenshot = await capture_screenshot(browser.page)
                page_text = await extract_page_text(browser.page)

                captcha = await detect_captcha(browser.page)
                if captcha:
                    captcha_streak += 1
                    log(f"  [CAPTCHA] {captcha} detected (streak {captcha_streak})")
                    if captcha_streak >= config.captcha_max_attempts:
                        fb = config.captcha_fallback_url
                        log(f"  [CAPTCHA] Persisted — auto-redirecting to {fb}")
                        try:
                            await browser.navigate(fb)
                        except Exception as exc:
                            log(f"  [WARN] captcha redirect failed: {exc}")
                        self._history.append(
                            Message(
                                role="user",
                                text=(
                                    f"The site at {url} was blocked by a "
                                    f"{captcha}. It has been abandoned and the "
                                    f"browser redirected to {fb}. Do NOT go back "
                                    "there — pursue the task from a different "
                                    "source or search engine."
                                ),
                            )
                        )
                        session.add(
                            step=step, url=url, screenshot=screenshot,
                            action={
                                "action": "navigate",
                                "params": {"url": fb},
                                "reason": "auto-escape captcha",
                            },
                            confidence=100,
                            result=f"captcha ({captcha}) — auto-redirected",
                        )
                        memory.record(
                            step=step, url=url, action="navigate",
                            params={"url": fb},
                            reason="auto-escape captcha", confidence=100,
                            outcome=f"captcha ({captcha}) auto-redirected",
                        )
                        await memory.compress()
                        self._trim_history()
                        captcha_streak = 0
                        pending_annotated = None
                        continue
                else:
                    captcha_streak = 0

                if url != last_url or page_text != last_page_text:
                    stuck_count = 0
                    last_url = url
                    last_page_text = page_text
                else:
                    stuck_count += 1

                log(f"\n{'=' * 64}")
                log(f"  Step {step}/{config.max_steps}  |  {url}")

                if (
                    config.replan_after_stuck > 0
                    and stuck_count >= config.replan_after_stuck
                    and step - last_replan_step >= config.replan_after_stuck
                ):
                    log("  [REPLAN] No progress — regenerating the plan")
                    plan = await replan(
                        self._provider,
                        task,
                        url,
                        memory.context_text(),
                        plan.revision + 1,
                    )
                    log(plan.pretty() if plan.steps else "  (replan empty)")
                    last_replan_step = step
                    stuck_count = 0

                warnings: list[str] = []
                non_scroll = [a for a in action_log if not a.startswith("scroll:")]

                if stuck_count >= 3:
                    tail = action_log[-stuck_count:]
                    only_scrolls = bool(tail) and all(
                        a.startswith("scroll:") for a in tail
                    )
                    log(f"  [STUCK] No progress for {stuck_count} steps")
                    warnings.append(
                        f"You scrolled {stuck_count} times with no text change — "
                        "the full content is already in the page text. STOP "
                        "scrolling; act on the next sub-goal or output 'done'."
                        if only_scrolls
                        else f"No progress for {stuck_count} steps on '{url}'. Do "
                        "NOT leave the site or restart. Take a DIFFERENT action "
                        "or output 'done' with what you have."
                    )

                if (
                    len(non_scroll) >= config.max_repeated_action
                    and len(set(non_scroll[-config.max_repeated_action :])) == 1
                ):
                    log(f"  [LOOP] Repeated action: {non_scroll[-1]}")
                    warnings.append(
                        f"You repeated '{non_scroll[-1]}'. It is not working — try "
                        "a different target/action, or finish with 'done'."
                    )

                cycle = _detect_cycle(action_log, config.cycle_max_len)
                if cycle:
                    log(f"  [CYCLE] Repeating {len(cycle)}-step sequence detected")
                    warnings.append(
                        f"You are repeating a {len(cycle)}-step sequence that "
                        "already failed. Break the loop: do it differently or "
                        "output 'done' with what you have."
                    )

                if fail_count == 2:
                    warnings.append(
                        "Your last 2 attempts at this step failed. Do NOT repeat "
                        "the same action. Try a completely different approach to "
                        "accomplish the same goal."
                    )
                elif fail_count >= config.max_step_failures:
                    warnings.append(
                        "You have failed 3 times. Skip this sub-task and move on "
                        "to the next part of the overall task."
                    )
                    fail_count = 0

                if step >= config.max_steps - 3:
                    warnings.append(
                        f"Only {config.max_steps - step + 1} steps left. Output "
                        "'done' now with the best report you can assemble."
                    )

                if captcha:
                    warnings.insert(
                        0,
                        f"This page is a {captcha} bot-check. Do NOT try to "
                        "solve it or click its controls. Use `navigate` NOW to "
                        f"a DIFFERENT source/search engine — '{url}' is blocked.",
                    )

                context_block = "\n\n".join(
                    b
                    for b in (
                        pending_loop_warning,
                        plan.as_prompt_block(),
                        self._tabs_block(browser),
                        memory.context_text(),
                    )
                    if b
                )
                pending_loop_warning = None  

                try:
                    data = await self._get_next_action(
                        task=task,
                        screenshot=screenshot,
                        url=url,
                        page_text=page_text,
                        step=step,
                        warnings=warnings,
                        context_block=context_block,
                        annotated_prev=pending_annotated,
                        log=log,
                    )
                except Exception as exc:
                    msg = str(exc)
                    log(f"  [ERROR] Could not get action: {msg}")
                    if any(c in msg for c in ("404", "403", "401", "API_KEY")):
                        result_text = f"Fatal API error: {msg}"
                        return result_text
                    continue
                pending_annotated = None

                try:
                    confidence = int(data.get("confidence", 40))
                except (TypeError, ValueError):
                    confidence = 40
                confidence = max(0, min(100, confidence))

                action = Action(
                    action_type=data.get("action", "done"),
                    params=data.get("params", {}) or {},
                    reason=data.get("reason", ""),
                    confidence=confidence,
                )
                log(f"  Action : {action}")
                log(f"  Reason : {action.reason}  (confidence: {confidence}%)")

                action_key = f"{action.action_type}:" + json.dumps(
                    action.params, sort_keys=True
                )
                action_log.append(action_key)
                cap = 2 * config.cycle_max_len + 4
                if len(action_log) > cap:
                    action_log.pop(0)

                debug_path = os.path.join(
                    config.debug_dir, f"step_{step:02d}.png"
                )
                _save_png(debug_path, screenshot)

                if action.action_type in _LOOP_TRACKED_ACTIONS:
                    loop_history.append(_loop_signature(url, action_key))
                    loop_history = loop_history[-_LOOP_WINDOW:]
                    if (
                        len(loop_history) >= _LOOP_TRIGGER
                        and len(set(loop_history[-_LOOP_TRIGGER:])) == 1
                    ):
                        pending_loop_warning = _loop_warning(action)
                        log(f"  [ANTI-LOOP] Intercepted repeated action: {action}")
                        self._history.append(
                            Message(role="user", text=pending_loop_warning)
                        )
                        session.add(
                            step=step, url=url, screenshot=screenshot,
                            action=data, confidence=confidence,
                            result="blocked: anti-loop detector intercepted repeat",
                        )
                        memory.record(
                            step=step, url=url, action=action.action_type,
                            params=action.params, reason=action.reason,
                            confidence=confidence,
                            outcome="blocked by anti-loop detector (3x repeat)",
                        )
                        await memory.compress()
                        self._trim_history()
                        continue

                if confidence < config.confidence_pause_below:
                    log(f"  [LOW CONFIDENCE] {confidence}% < {config.confidence_pause_below}")

                if supervised:
                    decision = (
                        approve(action, step, debug_path) or "y"
                    ).strip().lower()[:1]
                    if decision == "n":
                        result_text = "Aborted by user."
                        session.add(
                            step=step, url=url, screenshot=screenshot,
                            action=data, confidence=confidence,
                            result="aborted by user",
                        )
                        return result_text
                    if decision == "s":
                        log("  [SKIP] User skipped this action")
                        self._history.append(
                            Message(
                                role="user",
                                text="User skipped that action. Propose a "
                                "different approach for the same goal.",
                            )
                        )
                        session.add(
                            step=step, url=url, screenshot=screenshot,
                            action=data, confidence=confidence,
                            result="skipped by user",
                        )
                        memory.record(
                            step=step, url=url, action=action.action_type,
                            params=action.params, reason=action.reason,
                            confidence=confidence, outcome="skipped by user",
                        )
                        await memory.compress()
                        self._trim_history()
                        continue
                    if decision == "r":
                        log("  [REPLAN] Requested by user")
                        plan = await replan(
                            self._provider, task, url,
                            memory.context_text(), plan.revision + 1,
                        )
                        log(plan.pretty() if plan.steps else "  (replan empty)")
                        last_replan_step = step
                        continue
                elif confidence < config.confidence_autoskip_below:
                    log(
                        f"  [AUTO-SKIP] confidence {confidence}% < "
                        f"{config.confidence_autoskip_below} — asking for another action"
                    )
                    self._history.append(
                        Message(
                            role="user",
                            text=(
                                f"Your last proposal had confidence {confidence} "
                                "(too low) and was skipped. Choose a DIFFERENT, "
                                "more reliable action."
                            ),
                        )
                    )
                    session.add(
                        step=step, url=url, screenshot=screenshot,
                        action=data, confidence=confidence,
                        result="auto-skipped (low confidence)",
                    )
                    memory.record(
                        step=step, url=url, action=action.action_type,
                        params=action.params, reason=action.reason,
                        confidence=confidence,
                        outcome="auto-skipped (low confidence)",
                    )
                    await memory.compress()
                    self._trim_history()
                    continue

                if action.action_type == "click" and {"x", "y"} <= set(
                    action.params
                ):
                    try:
                        x, y = int(action.params["x"]), int(action.params["y"])
                        annotated = annotate_click(screenshot, x, y, "CLICK")
                        _save_png(
                            os.path.join(
                                config.debug_dir,
                                f"step_{step:02d}_annotated.png",
                            ),
                            annotated,
                        )
                        pending_annotated = annotated
                    except Exception as exc:
                        log(f"  [WARN] Could not annotate screenshot: {exc}")

                outcome: str
                try:
                    result = await execute_action(browser, action)
                    if action_key != fail_key:
                        fail_count = 0
                        fail_key = None
                    outcome = "ok" if result is None else f"done: {result}"
                except Exception as exc:
                    fail_count += 1
                    fail_key = action_key
                    outcome = f"failed: {exc}"
                    log(f"  [ERROR] Action failed ({fail_count}x): {exc}")
                    result = None
                    if action.action_type == "close_tab":
                        self._history.append(
                            Message(
                                role="user",
                                text=(
                                    f"close_tab failed: {exc}. Choose a "
                                    "different action — e.g. switch_tab, "
                                    "navigate, or continue the task in the "
                                    "current tab."
                                ),
                            )
                        )

                session.add(
                    step=step, url=url, screenshot=screenshot,
                    action=data, confidence=confidence, result=outcome,
                )
                memory.record(
                    step=step, url=url, action=action.action_type,
                    params=action.params, reason=action.reason,
                    confidence=confidence, outcome=outcome,
                )
                await memory.compress()
                self._trim_history()

                if result is not None:
                    log(f"\n{'=' * 64}")
                    log(f"  DONE — {result}")
                    success = True
                    result_text = result
                    return result_text

            return result_text

        finally:
            self._live_browser = None
            await browser.stop_screencast()
            try:
                jp, tp = session.save(
                    success=success,
                    final_result=result_text,
                    outdir=config.sessions_dir,
                )
                log(f"\n  Session saved:\n    {jp}\n    {tp}")
            except Exception as exc:
                log(f"  [WARN] Could not save session: {exc}")
            await browser.stop()
