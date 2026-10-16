"""Audio/video file helpers built on the ffmpeg binary that ships with imageio-ffmpeg."""

import subprocess
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def run_ffmpeg(args: list[str]) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"] + args
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + proc.stderr.decode("utf-8", "replace")[-2000:])


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio file to mono float32 at `sr` Hz."""
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Could not read audio {path}: " + proc.stderr.decode("utf-8", "replace")[-500:])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def save_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save mono or stereo (n, 2) float audio as 16-bit WAV."""
    audio = np.asarray(audio, dtype=np.float32)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())


def audio_duration(path: str | Path) -> float:
    return len(load_audio(path, 16000)) / 16000


def save_frames_as_video(frames, path: str | Path, fps: int) -> None:
    import imageio.v2 as imageio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=9, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame))
    finally:
        writer.close()


def read_video_frames(path: str | Path) -> tuple[list[np.ndarray], float]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    try:
        fps = float(reader.get_meta_data().get("fps", 24))
        frames = [np.asarray(f) for f in reader]
    finally:
        reader.close()
    return frames, fps
