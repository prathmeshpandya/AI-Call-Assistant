"""Decides whether user speech during an assistant response is a genuine interruption.

Signals combined to reduce false interruptions:
* sustained voiced time (a cough or click is short),
* the streaming recognizer produced real words (noise produces none),
* an explicit interrupt phrase ("wait", "stop") lowers the bar,
* the words are not just the assistant's own voice leaking back into the mic (echo).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BargeInConfig:
    enabled: bool = True
    min_speech_ms: int = 400
    min_words: int = 2
    keyword_min_speech_ms: int = 200
    keywords: list[str] = field(default_factory=lambda: ["stop", "wait", "hold on", "no"])
    echo_overlap_ratio: float = 0.7


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def looks_like_echo(partial: str, assistant_text: str, ratio: float = 0.7) -> bool:
    """True if most recognised words also appear in what the assistant is currently saying."""
    heard = _words(partial)
    if not heard:
        return False
    spoken = set(_words(assistant_text))
    overlap = sum(1 for w in heard if w in spoken) / len(heard)
    return overlap >= ratio


def has_keyword(partial: str, keywords: list[str]) -> bool:
    text = " " + " ".join(_words(partial)) + " "
    return any(f" {k} " in text for k in keywords)


def should_barge_in(partial: str, speech_ms: float, assistant_text: str, cfg: BargeInConfig) -> bool:
    if not cfg.enabled:
        return False
    words = _words(partial)
    if not words:
        return False
    if looks_like_echo(partial, assistant_text, cfg.echo_overlap_ratio):
        return False
    if has_keyword(partial, cfg.keywords) and speech_ms >= cfg.keyword_min_speech_ms:
        return True
    return speech_ms >= cfg.min_speech_ms and len(words) >= cfg.min_words
