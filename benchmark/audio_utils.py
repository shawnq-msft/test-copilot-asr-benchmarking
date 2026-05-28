import io
import struct
import wave

import numpy as np
import soundfile as sf

from .config import TARGET_CHANNELS, TARGET_SAMPWIDTH, TARGET_SR


def load_audio_as_pcm16_mono_16k(source) -> tuple[bytes, float]:
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(bytes(source))
    data, sr = sf.read(source, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    data = data.squeeze(-1)

    if sr != TARGET_SR:
        n_out = int(round(len(data) * TARGET_SR / sr))
        if n_out <= 0:
            return b"", 0.0
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)

    pcm = np.clip(data, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()
    duration = len(pcm16) / (TARGET_SR * TARGET_CHANNELS * TARGET_SAMPWIDTH)
    return pcm16, duration


def pcm_to_wav_bytes(pcm16: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(TARGET_CHANNELS)
        wf.setsampwidth(TARGET_SAMPWIDTH)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm16)
    return buf.getvalue()


def wav_header_placeholder() -> bytes:
    byte_rate = TARGET_SR * TARGET_CHANNELS * TARGET_SAMPWIDTH
    block_align = TARGET_CHANNELS * TARGET_SAMPWIDTH
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, TARGET_CHANNELS, TARGET_SR,
        byte_rate, block_align, TARGET_SAMPWIDTH * 8,
        b"data", 0xFFFFFFFF,
    )


def truncate_pcm(pcm16: bytes, truncate_s: float) -> bytes:
    max_bytes = int(truncate_s * TARGET_SR * TARGET_SAMPWIDTH * TARGET_CHANNELS)
    max_bytes -= max_bytes % (TARGET_CHANNELS * TARGET_SAMPWIDTH)
    return pcm16[:max_bytes]


def chunk_pcm(pcm16: bytes, chunk_ms: int) -> list[bytes]:
    bytes_per_chunk = int(TARGET_SR * TARGET_CHANNELS * TARGET_SAMPWIDTH * chunk_ms / 1000)
    bytes_per_chunk -= bytes_per_chunk % (TARGET_CHANNELS * TARGET_SAMPWIDTH)
    return [pcm16[i:i + bytes_per_chunk] for i in range(0, len(pcm16), bytes_per_chunk)]


def write_pcm16_wav(pcm16: bytes, path) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pcm_to_wav_bytes(pcm16))
