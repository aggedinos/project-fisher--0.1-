"""Validate, authorize, execute, and verify browser tools."""

from __future__ import annotations

import asyncio
import re

from playwright.async_api import Error as PlaywrightError
from pydantic import ValidationError

from fisher.browser.controller import BrowserController
from fisher.browser.models import StaleElementError
from fisher.models import Observation, PermissionMode, ToolCall, ToolResult, ToolSpec
from fisher.security.permissions import ApprovalCallback, permitted
from fisher.security.policy import SecurityError, SecurityPolicy, safe_url
from fisher.tools.browser_tools import DEFINITIONS, invoke_browser_tool


def _safe_call(call: ToolCall) -> ToolCall:
    return ToolCall(
        name=call.name,
        arguments={
            key: safe_url(value)
            if key == "url" and isinstance(value, str)
            else (
                value
                if key in {"element_id", "tab_id", "key", "delta_y", "x", "y"}
                else "[redacted]"
            )
            for key, value in call.arguments.items()
        },
    )


_URL_IN_ERROR = re.compile(r"(?i)(?:https?|file)://[^\s'\"<>]+")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|token|password|secret|authorization)\s*[:=]\s*[^\s,;]+"
)


def _safe_error(exc: Exception, call: ToolCall) -> str:
    message = _URL_IN_ERROR.sub("[URL redacted]", str(exc))
    message = _SECRET_ASSIGNMENT.sub("[secret redacted]", message)
    for key in ("text", "password", "token", "api_key"):
        value = call.arguments.get(key)
        if isinstance(value, str) and len(value) >= 3:
            message = message.replace(value, "[redacted]")
    return f"{type(exc).__name__}: {message[:240]}"


class ToolRegistry:
    def __init__(self, browser: BrowserController, policy: SecurityPolicy | None = None) -> None:
        self.browser = browser
        self.policy = policy or browser.policy

    def specs(self) -> list[ToolSpec]:
        return [definition.spec() for definition in DEFINITIONS.values()]

    async def execute(
        self,
        call: ToolCall,
        mode: PermissionMode = PermissionMode.SAFE,
        approve: ApprovalCallback | None = None,
    ) -> ToolResult:
        safe_call = _safe_call(call)
        definition = DEFINITIONS.get(call.name)
        if definition is None:
            return ToolResult(
                call=safe_call,
                success=False,
                message="Unknown tool",
                error_code="unknown_tool",
                retryable=False,
            )
        try:
            arguments = definition.arguments.model_validate(call.arguments).model_dump()
        except ValidationError as exc:
            fields = ", ".join(".".join(map(str, error["loc"])) for error in exc.errors())
            return ToolResult(
                call=safe_call,
                success=False,
                message=f"Invalid arguments for {call.name}: {fields}",
                error_code="invalid_arguments",
                retryable=False,
            )
        validated_call = ToolCall(name=call.name, arguments=arguments)
        before: Observation | None = self.browser.last_observation
        try:
            if before is None:
                before = await self.browser.observe()
            assessment = self.policy.classify(validated_call, before)
            if not await permitted(
                mode=PermissionMode(mode),
                call=validated_call,
                risk=assessment.level,
                reason=assessment.reason,
                approve=approve,
            ):
                return ToolResult(
                    call=safe_call,
                    success=False,
                    before_state=before,
                    message=f"Permission required: {assessment.reason}",
                    error_code="permission_required",
                    retryable=False,
                )
            data = await asyncio.wait_for(
                invoke_browser_tool(self.browser, call.name, arguments),
                timeout=definition.timeout_seconds,
            )
            after = await self.browser.observe()
            if call.name == "read_page":
                data = {"observation": after.summary()}
                changed, evidence = False, ["Page observed"]
                success = True
            else:
                from fisher.agent.verifier import verify_change

                changed, evidence = verify_change(validated_call, before, after)
                success = changed
            return ToolResult(
                call=safe_call,
                success=success,
                changed=changed,
                message="Verified change"
                if success and changed
                else ("Page observed" if success else "Action produced no verified change"),
                evidence=evidence,
                error_code=None if success else "verification_failed",
                retryable=not success and definition.retryable,
                before_state=before,
                after_state=after,
                data=data,
            )
        except (SecurityError, StaleElementError) as exc:
            code = "blocked_url" if isinstance(exc, SecurityError) else "stale_element"
            return ToolResult(
                call=safe_call,
                success=False,
                message=str(exc),
                error_code=code,
                retryable=code == "stale_element",
                before_state=before,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                call=safe_call,
                success=False,
                message="Tool timed out",
                error_code="timeout",
                retryable=definition.retryable,
                before_state=before,
            )
        except (ValueError, PlaywrightError, RuntimeError) as exc:
            return ToolResult(
                call=safe_call,
                success=False,
                message=_safe_error(exc, call),
                error_code="browser_error",
                retryable=definition.retryable,
                before_state=before,
            )
