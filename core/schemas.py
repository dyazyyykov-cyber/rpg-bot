from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class GeneralStory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1, max_length=4000)
    highlights: Optional[list[str]] = Field(default=None, min_items=0, max_items=8)


class TurnConfig(BaseModel):
    """Runtime configuration for a single turn loop."""

    model_config = ConfigDict(extra="allow")
    group_chat_id: int
    timeout: int = Field(60, ge=5, le=600)
    temperature_general: float = Field(0.7, ge=0.0, le=1.5)
    top_p_general: float = Field(0.9, ge=0.0, le=1.0)
    llm_timeout: int = Field(480, ge=5, le=1200)


class TurnOutputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    general: GeneralStory
    turn: int
    telemetry: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "GeneralStory",
    "TurnConfig",
    "TurnOutputs",
]
