"""Unit tests for config, audio validation, chunking, turn-taking, barge-in and memory."""

from __future__ import annotations

import numpy as np
import pytest

from app.api.protocol import parse_control
from app.audio.audio_utils import (
    AudioFormatError,
    float32_to_pcm16,
    pack_audio_frame,
    pcm16_to_float32,
    unpack_audio_frame,
    validate_pcm16_frame,
)
from app.config import PROJECT_ROOT, Settings
from app.core.barge_in import BargeInConfig, looks_like_echo, should_barge_in
from app.core.conversation_manager import INTERRUPTED_NOTE, ConversationManager
from app.core.sentence_chunker import SentenceChunker, clean_for_speech
from app.core.turn_detector import TurnConfig, TurnDetector, TurnEvent

# --- configuration ------------------------------------------------------------------------


def test_config_defaults_and_path_resolution():
    s = Settings(_env_file=None)
    assert s.backend_port == 8000
    assert s.llm_base_url.endswith("/v1")
    assert s.resolve(s.stt_model_dir) == (PROJECT_ROOT / "models/sherpa-onnx-streaming-zipformer-en-2023-06-26").resolve()
    assert "Never claim to be human" in s.load_system_prompt()


def test_config_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://example.local:9000/v1/")
    monkeypatch.setenv("END_OF_TURN_SILENCE_MS", "900")
    monkeypatch.setenv("FRONTEND_ORIGINS", "http://a, http://b")
    monkeypatch.setenv("BARGE_IN_KEYWORDS", "Stop, Wait")
    s = Settings(_env_file=None)
    assert s.llm_base_url == "http://example.local:9000/v1"  # trailing slash stripped
    assert s.end_of_turn_silence_ms == 900
    assert s.allowed_origins == ["http://a", "http://b"]
    assert s.barge_in_keyword_list == ["stop", "wait"]


def test_config_rejects_unsupported_audio(monkeypatch):
    monkeypatch.setenv("AUDIO_INPUT_SAMPLE_RATE", "44100")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


# --- audio messages -----------------------------------------------------------------------


@pytest.mark.parametrize("data", [b"", b"\x00", b"\x00" * 40000], ids=["empty", "odd_length", "too_large"])
def test_invalid_audio_frames_rejected(data):
    with pytest.raises(AudioFormatError):
        validate_pcm16_frame(data, max_bytes=32000)


def test_pcm_roundtrip_and_frame_header():
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    back = pcm16_to_float32(float32_to_pcm16(samples))
    assert np.allclose(back, samples, atol=1e-3)
    frame = pack_audio_frame(7, 3, 24000, b"\x01\x02")
    assert unpack_audio_frame(frame) == (7, 3, 24000, b"\x01\x02")


def test_control_message_validation():
    assert parse_control('{"type":"playback","event":"started","generation_id":2,"seq":1}')["seq"] == 1
    for bad in ['{"type":"nope"}', "not json", '{"type":"playback","event":"x","generation_id":1}',
                '{"type":"set_voice","voice":""}', '{"type":"start","pad":"' + "x" * 5000 + '"}']:
        with pytest.raises(ValueError):
            parse_control(bad)


# --- sentence chunking --------------------------------------------------------------------


def stream(chunker: SentenceChunker, text: str, step: int = 3) -> list[str]:
    out: list[str] = []
    for i in range(0, len(text), step):  # simulate token-sized deltas
        out += chunker.feed(text[i : i + step])
    tail = chunker.flush()
    return out + ([tail] if tail else [])


def test_chunker_splits_sentences_in_order():
    text = "Sure. The capital of France is Paris. It has about two million people! Anything else?"
    assert stream(SentenceChunker(min_chars=12), text) == [
        "Sure.",  # first chunk emitted immediately, even though short
        "The capital of France is Paris.",
        "It has about two million people!",
        "Anything else?",
    ]


def test_chunker_handles_abbreviations_decimals_and_initials():
    text = "Dr. Smith measured 3.14 meters, e.g. in the U.S. lab. J. R. R. Tolkien wrote it. Done now."
    chunks = stream(SentenceChunker(min_chars=5, first_soft_chars=200), text)
    assert chunks == [
        "Dr. Smith measured 3.14 meters, e.g. in the U.S. lab.",
        "J. R. R. Tolkien wrote it.",
        "Done now.",
    ]


def test_chunker_merges_short_sentences_after_first():
    chunks = stream(SentenceChunker(min_chars=12), "Okay. Yes. No. That is a longer sentence.")
    assert chunks == ["Okay.", "Yes. No. That is a longer sentence."]


def test_chunker_waits_for_lookahead_before_splitting():
    c = SentenceChunker()
    assert c.feed("The value is 3.") == []  # could be 3.5
    assert c.feed("5 today. Next") == ["The value is 3.5 today."]
    assert c.flush() == "Next"


def test_chunker_first_chunk_breaks_at_clause_and_long_runs_split_on_words():
    c = SentenceChunker(min_chars=10, max_chars=60, first_soft_chars=30)
    chunks = stream(c, "Well, if you think about the long term implications, it really depends on the budget "
                       "and timeline and team size and many other factors that matter")
    assert chunks[0] == "Well, if you think about the long term implications,"
    assert all(len(ch) <= 60 for ch in chunks[1:])
    assert " ".join(chunks).split() == ("Well, if you think about the long term implications, it really depends on "
                                        "the budget and timeline and team size and many other factors that matter").split()


def test_chunker_splits_long_later_sentences_at_clauses():
    text = ("The capital of France is Paris. It's famous for the Eiffel Tower, the Louvre Museum, "
            "and its rich culture, fashion, and cuisine.")
    chunks = stream(SentenceChunker(min_chars=12, soft_chars=60), text)
    assert chunks == ["The capital of France is Paris.",
                      "It's famous for the Eiffel Tower, the Louvre Museum, and its rich culture,",
                      "fashion, and cuisine."]


def test_chunker_real_model_output_cases():
    # Observed from Qwen3-4B: unspaced em dash, and "p.m." ending a sentence.
    text = ('Sure! Start by setting a specific goal for each study session—like "I\'ll review Chapter 3 '
            'by 6 p.m." That gives you direction. Meet at 9 a.m. tomorrow, okay?')
    chunks = stream(SentenceChunker(min_chars=12, first_soft_chars=30, soft_chars=40), text)
    assert chunks == ["Sure!",
                      "Start by setting a specific goal for each study session—",
                      'like "I\'ll review Chapter 3 by 6 p.m."',
                      "That gives you direction.",
                      "Meet at 9 a.m. tomorrow, okay?"]


def test_chunker_incomplete_final_sentence_is_flushed():
    assert stream(SentenceChunker(), "I was going to say that") == ["I was going to say that"]


def test_clean_for_speech_strips_markdown_and_emoji():
    assert clean_for_speech("**Great** question! 😀") == "Great question!"
    assert clean_for_speech("- item one") == "item one"
    assert clean_for_speech("who called it *Lutetia*.") == "who called it Lutetia."
    assert clean_for_speech("🙂 ***") == ""


# --- turn detection -----------------------------------------------------------------------


def run_frames(det: TurnDetector, pattern: list[tuple[bool, int]], transcript: str = "hello") -> list[TurnEvent]:
    events = []
    for speech, n in pattern:
        for _ in range(n):
            ev = det.process(speech, 32.0, transcript)
            if ev:
                events.append(ev)
    return events


def test_turn_ends_after_silence_but_not_on_short_pause():
    det = TurnDetector(TurnConfig(end_of_turn_silence_ms=700, min_utterance_ms=200))
    # 640 ms speech, 320 ms pause (thinking), more speech, then 800 ms silence
    events = run_frames(det, [(True, 20), (False, 10), (True, 10), (False, 25)])
    assert events == [TurnEvent.SPEECH_START, TurnEvent.END_OF_TURN]


def test_short_noise_is_discarded():
    det = TurnDetector(TurnConfig(end_of_turn_silence_ms=300, min_utterance_ms=250))
    assert run_frames(det, [(True, 3), (False, 20)]) == [TurnEvent.SPEECH_START, TurnEvent.DISCARD]


def test_hesitation_word_extends_end_of_turn_wait():
    cfg = TurnConfig(end_of_turn_silence_ms=500, hesitation_silence_ms=1200)
    det = TurnDetector(cfg)
    assert run_frames(det, [(True, 20), (False, 20)], transcript="i want to go and") == [TurnEvent.SPEECH_START]
    assert run_frames(det, [(False, 20)], transcript="i want to go and") == [TurnEvent.END_OF_TURN]
    # Questions ending in a preposition are complete; they must not get the long wait.
    assert det.required_silence_ms("what is it famous for") == 500


# --- barge-in -----------------------------------------------------------------------------


def test_barge_in_rules():
    cfg = BargeInConfig(min_speech_ms=400, min_words=2, keywords=["wait", "stop"])
    said = "I think the best approach would be to start by"
    assert should_barge_in("no explain the other approach", 600, said, cfg)
    assert not should_barge_in("", 900, said, cfg)  # noise: no words
    assert not should_barge_in("yeah", 600, said, cfg)  # single-word backchannel
    assert not should_barge_in("hello there", 200, said, cfg)  # too brief
    assert should_barge_in("wait", 250, said, cfg)  # explicit interrupt word
    assert not should_barge_in("the best approach", 800, said, cfg)  # echo of our own voice
    assert looks_like_echo("start by", said)
    assert not should_barge_in("hello there friend", 900, said, BargeInConfig(enabled=False))


# --- conversation memory ------------------------------------------------------------------


def test_history_only_keeps_heard_sentences_after_interruption():
    conv = ConversationManager()
    conv.add_user("Tell me about approaches")
    conv.start_assistant(1)
    conv.record_sentence(1, 0, "There are two approaches.")
    conv.record_sentence(1, 1, "The first is to start by planning.")
    conv.record_sentence(1, 2, "The second is unheard.")
    conv.record_played(1, 0)
    conv.record_played(1, 1)
    turn = conv.finish_assistant(1, interrupted=True)
    assert turn is not None and turn.interrupted
    assert turn.content == "There are two approaches. The first is to start by planning."
    conv.add_user("No wait, explain the other approach")
    msgs = conv.build_messages("sys")
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2].content.endswith(INTERRUPTED_NOTE)
    assert "unheard" not in " ".join(m.content for m in msgs)


def test_unanswered_user_turns_merge_and_empty_rejected():
    conv = ConversationManager()
    conv.add_user("What is")
    conv.start_assistant(1)
    assert conv.finish_assistant(1, interrupted=True) is None  # nothing heard -> nothing stored
    conv.add_user("the capital of France")
    assert [t.content for t in conv.turns] == ["What is the capital of France"]
    with pytest.raises(ValueError):
        conv.add_user("   ")


def test_stale_generation_updates_ignored_and_history_bounded():
    conv = ConversationManager(max_turns=4, max_chars=10_000)
    for i in range(10):
        conv.add_user(f"question {i}")
        conv.start_assistant(i)
        conv.record_sentence(i, 0, f"answer {i}")
        conv.record_sentence(i - 1, 1, "stale")  # old generation: ignored
        conv.finish_assistant(i, interrupted=False)
    msgs = conv.build_messages("sys")
    assert len(msgs) == 5 and msgs[1].content == "question 8"
    assert all("stale" not in m.content for m in msgs)
    conv.reset()
    assert conv.build_messages("sys")[1:] == []
