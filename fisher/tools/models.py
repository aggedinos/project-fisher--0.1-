"""Strict argument schemas for every tool visible to a model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NavigateArgs(ToolArgs):
    url: str = Field(min_length=8, max_length=4096)


class ReadPageArgs(ToolArgs):
    pass


class ElementArgs(ToolArgs):
    element_id: str = Field(pattern=r"^o\d+-e\d+$")


class TextArgs(ElementArgs):
    text: str = Field(max_length=10000)


class KeyArgs(ToolArgs):
    key: str

    @field_validator("key")
    @classmethod
    def safe_key(cls, value: str) -> str:
        allowed = {
            "Enter",
            "NumpadEnter",
            "Tab",
            "Escape",
            "Backspace",
            "Delete",
            "ArrowUp",
            "ArrowDown",
            "ArrowLeft",
            "ArrowRight",
            "Home",
            "End",
            "PageUp",
            "PageDown",
            "Space",
            "ControlOrMeta+A",
            "Control+A",
            "Meta+A",
        }
        if value not in allowed:
            raise ValueError("Unsupported key; use type_text for ordinary text")
        return value


class ScrollArgs(ToolArgs):
    delta_y: int = Field(ge=-2000, le=2000)

    @field_validator("delta_y")
    @classmethod
    def nonzero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("Scroll amount must be nonzero")
        return value


class TabArgs(ToolArgs):
    tab_id: str = Field(pattern=r"^t\d+$")


class CoordinatesArgs(ToolArgs):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
