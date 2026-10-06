"""Runtime permission decisions for the three user-facing modes."""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Awaitable, Callable

from fisher.models import PermissionMode, PermissionRequest, RiskLevel, ToolCall
from fisher.security.policy import safe_url

ApprovalCallback = Callable[[PermissionRequest], bool | Awaitable[bool]]


def requires_approval(mode: PermissionMode, risk: RiskLevel, tool_name: str) -> bool:
    """AUTO allows routine and medium actions; HIGH always needs approval."""
    if tool_name == "read_page":
        return False
    if mode == PermissionMode.SUPERVISED:
        return True
    if mode == PermissionMode.SAFE:
        return risk in {RiskLevel.MEDIUM, RiskLevel.HIGH}
    return risk == RiskLevel.HIGH


async def permitted(
    *,
    mode: PermissionMode,
    call: ToolCall,
    risk: RiskLevel,
    reason: str,
    approve: ApprovalCallback | None,
) -> bool:
    if not requires_approval(mode, risk, call.name):
        return True
    if approve is None:
        return False
    safe_arguments = {
        key: (
            safe_url(value)
            if key == "url" and isinstance(value, str)
            else "[redacted]"
            if key in {"text", "password", "token", "api_key"}
            else value
        )
        for key, value in call.arguments.items()
    }
    request = PermissionRequest(
        id=uuid.uuid4().hex,
        call=ToolCall(name=call.name, arguments=safe_arguments),
        risk=risk,
        reason=reason,
    )
    decision = approve(request)
    if inspect.isawaitable(decision):
        decision = await decision
    return decision is True
