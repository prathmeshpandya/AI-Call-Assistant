"""PCM conversion and validation helpers. All audio format logic lives here, nowhere else."""

from __future__ import annotations

import struct

import numpy as np

# Server -> client audio frame header: generation_id, seq, sample_rate (all uint32 little-endian).
AUDIO_HEADER = struct.Struct("<III")


class AudioFormatError(ValueError):
    """Raised when an incoming audio message is malformed."""


def validate_pcm16_frame(data: bytes, max_bytes: int) -> None:
    if not data:
        raise AudioFormatError("empty audio frame")
    if len(data) > max_bytes:
        raise AudioFormatError(f"audio frame too large ({len(data)} > {max_bytes} bytes)")
    if len(data) % 2 != 0:
        raise AudioFormatError("PCM16 frame must contain an even number of bytes")


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit PCM -> float32 in [-1, 1]."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Cheap linear resampler. Adequate for test tooling; the browser does real-time mic resampling."""
    if src_rate == dst_rate or len(samples) == 0:
        return np.asarray(samples, dtype=np.float32)
    duration = len(samples) / src_rate
    n_out = int(round(duration * dst_rate))
    x_old = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def pack_audio_frame(generation_id: int, seq: int, sample_rate: int, pcm16: bytes) -> bytes:
    return AUDIO_HEADER.pack(generation_id, seq, sample_rate) + pcm16


def unpack_audio_frame(frame: bytes) -> tuple[int, int, int, bytes]:
    if len(frame) < AUDIO_HEADER.size:
        raise AudioFormatError("audio frame shorter than header")
    gen, seq, rate = AUDIO_HEADER.unpack_from(frame)
    return gen, seq, rate, frame[AUDIO_HEADER.size :]


def duration_ms(num_samples: int, sample_rate: int) -> float:
    return 1000.0 * num_samples / sample_rate
