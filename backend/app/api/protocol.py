"""Validated client -> server control messages (see docs/protocol.md). Binary frames are PCM audio."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Msg(BaseModel):
    model_config = ConfigDict(extra="ignore")


class StartMsg(_Msg):
    type: Literal["start"]


class StopMsg(_Msg):
    type: Literal["stop"]


class InterruptMsg(_Msg):
    type: Literal["interrupt"]


class ResetMsg(_Msg):
    type: Literal["reset"]


class PingMsg(_Msg):
    type: Literal["ping"]
    t: float | None = None


class SetVoiceMsg(_Msg):
    type: Literal["set_voice"]
    voice: str = Field(min_length=1, max_length=64)


class PlaybackMsg(_Msg):
    type: Literal["playback"]
    event: Literal["started", "idle"]
    generation_id: int = Field(ge=0)
    seq: int = Field(default=0, ge=0)


ControlMessage = Annotated[
    StartMsg | StopMsg | InterruptMsg | ResetMsg | PingMsg | SetVoiceMsg | PlaybackMsg,
    Field(discriminator="type"),
]
control_adapter: TypeAdapter[ControlMessage] = TypeAdapter(ControlMessage)

MAX_CONTROL_BYTES = 4096


def parse_control(raw: str) -> dict:
    """Parse and validate a JSON control message; raises ValueError on anything malformed."""
    if len(raw) > MAX_CONTROL_BYTES:
        raise ValueError("control message too large")
    return control_adapter.validate_json(raw).model_dump()
