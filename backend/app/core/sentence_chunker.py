"""Incremental sentence segmentation for streaming LLM output -> TTS.

Text arrives token by token. We emit a chunk as soon as a complete sentence is available, so TTS
can start while the LLM keeps generating. Rules:

* Sentence ends at . ! ? or an ellipsis followed by whitespace (lookahead avoids splitting "3.14" or "e.g.").
* Common abbreviations (Dr., e.g., U.S.) and single-letter initials do not end a sentence.
* A newline ends a chunk (lists / paragraphs).
* Chunks shorter than `min_chars` are merged with the following sentence, except the very first chunk
  of a response, which is emitted immediately to minimise time-to-first-audio.
* The first chunk may break at a clause boundary (, ; : -) once it reaches `first_soft_chars`;
  later chunks once they reach `soft_chars`. Shorter chunks start TTS sooner and keep synthesis ahead
  of playback on slow CPUs.
* A run-on longer than `max_chars` is split at the last clause boundary, else the last space.
* Never splits inside a word.
"""

from __future__ import annotations

import re
import unicodedata

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "approx", "dept",
    "est", "inc", "ltd", "co", "corp", "fig", "ca", "cf", "u.s", "u.k", "a.m", "p.m", "mt", "ave",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
# Abbreviations that often END a sentence too ("... until 6 p.m. Then ..."): a sentence break
# is assumed when the next word is capitalised.
_MAY_END_SENTENCE = {"etc", "a.m", "p.m"}
_TERMINATORS = ".!?…"
_CLOSERS = "\"')]”’"
# , ; : need a following space ("3,000" is not a clause break); dashes do not ("session—like").
_CLAUSE_RE = re.compile(r"[,;:](?=\s)|[—–]")


class SentenceChunker:
    def __init__(
        self, min_chars: int = 12, max_chars: int = 220, first_soft_chars: int = 30, soft_chars: int = 40
    ) -> None:
        if min_chars < 1 or max_chars <= min_chars:
            raise ValueError("require 1 <= min_chars < max_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.first_soft_chars = first_soft_chars
        self.soft_chars = soft_chars
        self._buf = ""
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        """Add generated text; return zero or more complete chunks, in order."""
        self._buf += delta
        chunks: list[str] = []
        while True:
            end = self._next_end()
            if end is None:
                break
            chunk = self._buf[:end].strip()
            self._buf = self._buf[end:]
            if chunk:
                chunks.append(chunk)
                self._emitted += 1
        return chunks

    def flush(self) -> str | None:
        """Return whatever is left (an incomplete final sentence) at the end of generation."""
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._emitted += 1
            return rest
        return None

    # --- internals -------------------------------------------------------------------------

    def _next_end(self) -> int | None:
        min_len = 1 if self._emitted == 0 else self.min_chars
        end = self._sentence_end(min_len)
        if end is not None:
            return end
        soft = self.first_soft_chars if self._emitted == 0 else self.soft_chars
        if len(self._buf) >= soft:
            end = self._clause_end(soft, len(self._buf))
            if end is not None:
                return end
        if len(self._buf) >= self.max_chars:
            return self._clause_end(self.min_chars, self.max_chars) or self._space_end(self.max_chars)
        return None

    def _sentence_end(self, min_len: int) -> int | None:
        buf, n = self._buf, len(self._buf)
        for i, ch in enumerate(buf):
            if ch == "\n":
                if len(buf[:i].strip()) >= min_len:
                    return i + 1
                continue
            if ch not in _TERMINATORS:
                continue
            j = i + 1
            while j < n and (buf[j] in _TERMINATORS or buf[j] in _CLOSERS):
                j += 1
            if j >= n:
                return None  # need lookahead (could be "3.5", "e.g." or "...")
            if not buf[j].isspace():
                continue
            if ch == "." and _is_abbreviation(buf[:i]):
                if _last_word(buf[:i]).lower() not in _MAY_END_SENTENCE:
                    continue
                k = j
                while k < n and buf[k].isspace():
                    k += 1
                if k >= n:
                    return None  # "6 p.m." - wait to see whether a new sentence starts
                if not buf[k].isupper():
                    continue
            if len(buf[:j].strip()) < min_len:
                continue
            return j
        return None

    def _clause_end(self, min_len: int, limit: int) -> int | None:
        """Earliest clause boundary at or after min_len, before limit."""
        buf = self._buf[:limit]
        for m in _CLAUSE_RE.finditer(buf):
            if m.end() >= min_len:
                return m.end()
        return None

    def _space_end(self, limit: int) -> int | None:
        idx = self._buf.rfind(" ", 0, limit)
        return idx + 1 if idx > 0 else None


def _last_word(prefix: str) -> str:
    word = prefix.rsplit(None, 1)[-1] if prefix.strip() else ""
    return word.lstrip("(\"'“‘")


def _is_abbreviation(prefix: str) -> bool:
    word = _last_word(prefix)
    if not word:
        return False
    if word.lower() in _ABBREVIATIONS:
        return True
    # Single-letter initials ("J. R. R. Tolkien"), but not the pronoun "I".
    return len(word) == 1 and word.isupper() and word != "I"


_MARKDOWN = re.compile(r"(^\s*[-*•]\s+|\*+|__|`+|^#+\s*)", re.MULTILINE)


def clean_for_speech(text: str) -> str:
    """Strip markdown and emoji the TTS should not read aloud. Returns '' if nothing speakable remains."""
    text = _MARKDOWN.sub("", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Cs", "Co"))
    text = re.sub(r"\s+", " ", text).strip()
    return text if any(ch.isalnum() for ch in text) else ""
