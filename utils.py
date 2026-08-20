"""Screenshot capture, page text extraction, consent dismissal, and helpers."""

import asyncio
import io
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import Page

from config import config


def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if missing and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from a model reply.

    Tolerates ```json fences and leading/trailing prose by falling back to the
    first ``{`` … last ``}`` slice. Raises ``json.JSONDecodeError`` /
    ``ValueError`` if nothing parseable is found.
    """
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def annotate_click(png: bytes, x: int, y: int, label: str = "CLICK") -> bytes:
    """Return a PNG with a red dot (r=12) at *(x, y)* and a yellow text label.

    Used to give the model visual feedback about where it clicked so it can
    verify the action on the following step. Pillow only — no extra deps.
    """
    img = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(img)
    r = 12
    draw.ellipse(
        [x - r, y - r, x + r, y + r],
        fill=(220, 30, 30),
        outline=(255, 255, 255),
        width=2,
    )

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    text = label.upper()
    tx, ty = x + r + 6, y - r
    try:
        box = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(
            [box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(0, 0, 0)
        )
    except Exception:
        pass
    draw.text((tx, ty), text, fill=(255, 215, 0), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_CONSENT_TEXTS = [
    "Accept all", "Accept All", "Accept All Cookies", "Accept cookies",
    "Accept Cookies", "Accept and continue", "Accept & Continue",
    "I Accept", "I accept", "I agree", "I Agree", "AGREE", "Agree",
    "Allow all", "Allow All", "Allow cookies", "Allow All Cookies",
    "Got it", "Okay", "OK", "Consent", "Understood",
    "Accepter tout",    
    "Alle akzeptieren", 
]


_CONSENT_SELECTORS = [
    "#accept-all", "#acceptAll", "#accept_all",
    "#onetrust-accept-btn-handler",
    "#CybotCookiebotDialogBodyButtonAccept",
    "#cookie-accept", "#cookieAccept",
    ".accept-all", ".acceptAll", ".cookie-accept",
    "[data-testid='accept-cookies']",
    "[data-action='accept']",
    "[data-cookieconsent='allow']",
    "[aria-label='Accept all cookies']",
    "[aria-label='Accept all']",
]


async def auto_dismiss_consent(page: Page) -> bool:
    """
    Try to auto-dismiss cookie/GDPR consent dialogs.

    Attempts three strategies in order:
      1. JavaScript click on any interactive element whose text matches known accept phrases
      2. Playwright get_by_role locator for the most common button labels
      3. Direct CSS selector hit on well-known consent widget IDs/classes

    Returns True if something was clicked (dialog likely dismissed).
    """
    
    try:
        clicked = await page.evaluate(
            """(texts) => {
                const lower = texts.map(t => t.toLowerCase());
                const tags = 'button, a[role="button"], [role="button"], '
                           + 'input[type="button"], input[type="submit"]';
                for (const el of document.querySelectorAll(tags)) {
                    const text = (el.textContent || el.innerText || el.value || '').trim();
                    if (lower.some(t => text.toLowerCase().includes(t))) {
                        el.click();
                        return true;
                    }
                }
                return false;
            }""",
            _CONSENT_TEXTS,
        )
        if clicked:
            await asyncio.sleep(0.8)
            return True
    except Exception:
        pass

    
    for text in _CONSENT_TEXTS[:12]:
        try:
            loc = page.get_by_role("button", name=text, exact=False).first
            if await loc.is_visible(timeout=200):
                await loc.click()
                await asyncio.sleep(0.8)
                return True
        except Exception:
            pass

    
    for selector in _CONSENT_SELECTORS:
        try:
            loc = page.locator(selector).first
            if await loc.is_visible(timeout=200):
                await loc.click()
                await asyncio.sleep(0.8)
                return True
        except Exception:
            pass

    return False


_CAPTCHA_URL_SIGNS = {
    "/sorry/": "Google 'unusual traffic'",
    "consent.google": "Google consent wall",
    "ipv4check": "bot check",
    "recaptcha": "reCAPTCHA",
    "hcaptcha": "hCaptcha",
    "challenges.cloudflare.com": "Cloudflare Turnstile",
    "/cdn-cgi/challenge": "Cloudflare challenge",
    "datadome": "DataDome",
    "perimeterx": "PerimeterX",
    "px-captcha": "PerimeterX",
    "captcha": "CAPTCHA",
}


async def detect_captcha(page: Page) -> str | None:
    """Return a short label if the current page is a CAPTCHA / bot wall.

    Combines URL signatures with DOM (known challenge iframes/widgets) and
    body-text phrases ("verify you are human", "unusual traffic", …). Returns
    None on a normal page. Best-effort: never raises.
    """
    try:
        url = (page.url or "").lower()
    except Exception:
        url = ""
    for key, label in _CAPTCHA_URL_SIGNS.items():
        if key in url:
            return label

    try:
        label: str = await page.evaluate(
            """() => {
                const has = (s) => !!document.querySelector(s);
                if (has('iframe[src*="recaptcha"]') || has('.g-recaptcha'))
                    return 'reCAPTCHA';
                if (has('iframe[src*="hcaptcha"]') || has('.h-captcha'))
                    return 'hCaptcha';
                if (has('iframe[src*="challenges.cloudflare.com"]')
                    || has('iframe[src*="turnstile"]')
                    || has('#cf-challenge-running')
                    || has('#challenge-form')
                    || has('#challenge-running'))
                    return 'Cloudflare challenge';
                const t = (document.body ? document.body.innerText : '')
                            .toLowerCase();
                const phrases = [
                    "i'm not a robot", "verify you are human",
                    "are you a human", "unusual traffic",
                    "complete the security check",
                    "checking your browser before", "press and hold",
                    "verify you are not a robot",
                    "enter the characters you see",
                    "to continue, please type the characters"
                ];
                for (const p of phrases) if (t.includes(p)) return 'CAPTCHA';
                return '';
            }"""
        )
        return label or None
    except Exception:
        return None


async def extract_page_text(page: Page, max_chars: int | None = None) -> str:
    """
    Extract visible text from the page.

    Prefers a ``main`` / ``[role=main]`` container (which on comment-heavy
    sites like Reddit contains BOTH the post and the full comment tree) and
    otherwise falls back to the whole body. Deliberately does *not* prefer a
    bare ``article`` element — on Reddit that is the post only and hides every
    comment, which is exactly the content the agent usually needs.

    Scripts, styles, nav/header/footer and hidden nodes are stripped so the
    result is readable content rather than chrome. Returns up to
    ``config.max_page_text_chars`` characters (override via *max_chars*).
    """
    if max_chars is None:
        max_chars = config.max_page_text_chars
    try:
        text: str = await page.evaluate(
            """() => {
                const main = document.querySelector('main, [role="main"]');
                const root = (main || document.body).cloneNode(true);
                root.querySelectorAll(
                    'script, style, noscript, svg, iframe, nav, header, '
                    + 'footer, [aria-hidden="true"]'
                ).forEach(el => el.remove());
                const raw = root.innerText || root.textContent || '';
                // Collapse runs of blank lines to a single blank line
                return raw.replace(/\\n{3,}/g, '\\n\\n').trim();
            }"""
        )
        return text[:max_chars]
    except Exception:
        return ""


async def capture_screenshot(page: Page) -> bytes:
    """
    Capture a PNG screenshot normalised to the configured viewport dimensions.
    Returns raw PNG bytes ready for the Gemini API.
    """
    raw = await page.screenshot(type="png")

    img = Image.open(io.BytesIO(raw))
    if img.size != (config.viewport_width, config.viewport_height):
        img = img.resize(
            (config.viewport_width, config.viewport_height),
            Image.Resampling.LANCZOS,
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
