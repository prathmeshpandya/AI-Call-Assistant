"""Session-level conversation memory.

Invariants:
* Only finalized user transcripts enter history (never provisional partials).
* Assistant turns contain only sentences whose playback actually started on the client.
  Sentences that were generated but never heard (because of an interruption) are dropped.
* Consecutive user turns are merged (the user kept talking before hearing any reply).
* Context sent to the LLM is bounded by turn count and character budget.

Extension point: a persistent memory store can implement `MemoryStore` and be consulted in
`build_messages` (e.g. retrieved facts appended to the system prompt). Not used in v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from app.llm.llm_base import ChatMessage

# Appended to an interrupted assistant turn. Tested on Qwen3-4B: an explicit bracketed note made the
# model restart its old answer or refuse; a bare dash plus one system-prompt line made it follow the
# user's new request in 6/6 trials.
INTERRUPTED_NOTE = " —"


class MemoryStore(Protocol):  # pragma: no cover - future extension
    def recall(self, query: str) -> list[str]: ...
    def remember(self, turns: list["Turn"]) -> None: ...


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str
    interrupted: bool = False


@dataclass
class _PendingResponse:
    generation_id: int
    sentences: dict[int, str] = field(default_factory=dict)  # seq -> text (sent to the client)
    played: set[int] = field(default_factory=set)  # seqs whose playback started

    def heard_text(self) -> str:
        return " ".join(self.sentences[s] for s in sorted(self.played) if s in self.sentences)

    def full_text(self) -> str:
        return " ".join(self.sentences[s] for s in sorted(self.sentences))


class ConversationManager:
    def __init__(self, max_turns: int = 12, max_chars: int = 6000) -> None:
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.turns: list[Turn] = []
        self._pending: _PendingResponse | None = None

    # --- user ---------------------------------------------------------------------------------

    def add_user(self, text: str) -> Turn:
        text = " ".join(text.split())
        if not text:
            raise ValueError("empty transcript")
        if self.turns and self.turns[-1].role == "user":
            self.turns[-1].content = f"{self.turns[-1].content} {text}"
            return self.turns[-1]
        turn = Turn("user", text)
        self.turns.append(turn)
        return turn

    # --- assistant ----------------------------------------------------------------------------

    def start_assistant(self, generation_id: int) -> None:
        self._pending = _PendingResponse(generation_id)

    @property
    def pending_generation(self) -> int | None:
        return self._pending.generation_id if self._pending else None

    def record_sentence(self, generation_id: int, seq: int, text: str) -> None:
        if self._pending and self._pending.generation_id == generation_id:
            self._pending.sentences[seq] = text

    def record_played(self, generation_id: int, seq: int) -> None:
        if self._pending and self._pending.generation_id == generation_id:
            self._pending.played.add(seq)

    def current_assistant_text(self) -> str:
        """Everything sent for the in-flight response (used for echo detection)."""
        return self._pending.full_text() if self._pending else ""

    def finish_assistant(self, generation_id: int, interrupted: bool) -> Turn | None:
        """Commit the in-flight response. Returns the stored turn, or None if nothing was heard."""
        pending = self._pending
        if pending is None or pending.generation_id != generation_id:
            return None
        self._pending = None
        text = pending.heard_text() if interrupted else pending.full_text()
        if not text:
            return None
        # Even if every sent sentence started playing, an interruption cut the last one short.
        turn = Turn("assistant", text, interrupted=interrupted)
        self.turns.append(turn)
        return turn

    # --- context ------------------------------------------------------------------------------

    def build_messages(self, system_prompt: str) -> list[ChatMessage]:
        selected: list[Turn] = []
        budget = self.max_chars
        for turn in reversed(self.turns[-self.max_turns :]):
            if len(turn.content) > budget and selected:
                break
            selected.append(turn)
            budget -= len(turn.content)
        selected.reverse()
        while selected and selected[0].role != "user":  # chat templates expect user first
            selected.pop(0)
        messages = [ChatMessage("system", system_prompt)]
        for t in selected:
            content = t.content + (INTERRUPTED_NOTE if t.interrupted else "")
            messages.append(ChatMessage(t.role, content))
        return messages

    def reset(self) -> None:
        self.turns.clear()
        self._pending = None
