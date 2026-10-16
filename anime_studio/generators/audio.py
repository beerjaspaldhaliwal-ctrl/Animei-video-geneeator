"""Background music (MusicGen) and sound effects (AudioLDM)."""

import logging
import zlib
from pathlib import Path

import numpy as np

from ..device import MODELS, pick_device
from ..media import SAMPLE_RATE, load_audio, save_wav

log = logging.getLogger(__name__)

MAX_MUSIC_SECONDS = 30  # longer scenes loop the track with a crossfade


# ---------------------------------------------------------------- draft synths
def _note(freq, seconds, sr=SAMPLE_RATE):
    t = np.arange(int(seconds * sr)) / sr
    tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(6 * np.pi * freq * t)
    attack = np.minimum(1, t / 0.4)
    release = np.minimum(1, (seconds - t) / 0.6).clip(0)
    return tone * attack * release


def synth_music(prompt: str, seconds: float) -> np.ndarray:
    """Simple chord-pad loop so draft episodes are not silent."""
    rng = np.random.default_rng(zlib.crc32(prompt.encode()))
    sad = any(w in prompt.lower() for w in ("sad", "melanchol", "dark", "mysterious", "tense", "calm", "night"))
    root = 196.0 * 2 ** (rng.integers(0, 7) / 12)
    # I-V-vi-IV in major, i-VI-III-VII in minor (semitone offsets of each chord)
    chords = (
        [[0, 3, 7], [-4, 0, 3], [3, 7, 10], [-2, 2, 5]] if sad else [[0, 4, 7], [7, 11, 14], [9, 12, 16], [5, 9, 12]]
    )
    bar = 2.4 if sad else 1.8
    loop = np.concatenate([sum(_note(root * 2 ** (s / 12), bar) for s in chord) / 3 for chord in chords])
    reps = int(np.ceil(seconds / (len(loop) / SAMPLE_RATE)))
    audio = np.tile(loop, reps)[: int(seconds * SAMPLE_RATE)]
    return (audio * 0.5).astype(np.float32)


def synth_sfx(prompt: str, seconds: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(prompt.encode()))
    n = int(seconds * SAMPLE_RATE)
    noise = rng.standard_normal(n)
    smooth = np.convolve(noise, np.ones(int(5 + rng.integers(0, 40))) / 20, mode="same")
    env = np.exp(-np.linspace(0, 5, n))
    return (smooth * env * 0.6).astype(np.float32)


# ---------------------------------------------------------------- AI models
def _load_musicgen(settings):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = pick_device(settings)
    processor = AutoProcessor.from_pretrained(settings["music_model"])
    model = MusicgenForConditionalGeneration.from_pretrained(settings["music_model"]).to(device)
    return processor, model, device


def _load_audioldm(settings):
    import torch
    from diffusers import AudioLDMPipeline

    device = pick_device(settings)
    dtype = torch.float32 if device == "cpu" else torch.float16
    pipe = AudioLDMPipeline.from_pretrained(settings["sfx_model"], torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


class MusicEngine:
    def __init__(self, settings: dict):
        self.settings = settings

    @property
    def draft(self) -> bool:
        return bool(self.settings.get("draft_mode"))

    def cache_key(self, prompt: str, seconds: float) -> list:
        model = "draft-synth" if self.draft else self.settings["music_model"]
        return ["music-v1", model, prompt, min(MAX_MUSIC_SECONDS, round(seconds))]

    def generate(self, prompt: str, seconds: float, out_path: Path, seed: int = 0) -> None:
        seconds = float(min(MAX_MUSIC_SECONDS, max(5, seconds)))
        if self.draft:
            save_wav(out_path, synth_music(prompt, seconds))
            return
        import torch

        processor, model, device = MODELS.get(
            "musicgen:" + self.settings["music_model"], lambda: _load_musicgen(self.settings)
        )
        torch.manual_seed(int(seed))
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        frame_rate = getattr(model.config.audio_encoder, "frame_rate", 50)
        with torch.no_grad():
            audio = model.generate(
                **inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=int(seconds * frame_rate)
            )
        sr = model.config.audio_encoder.sampling_rate
        _save_resampled(out_path, audio[0, 0].float().cpu().numpy(), sr)


class SfxEngine:
    def __init__(self, settings: dict):
        self.settings = settings

    @property
    def draft(self) -> bool:
        return bool(self.settings.get("draft_mode"))

    def cache_key(self, prompt: str) -> list:
        return ["sfx-v1", "draft-synth" if self.draft else self.settings["sfx_model"], prompt]

    def generate(self, prompt: str, out_path: Path, seconds: float = 3.0, seed: int = 0) -> None:
        if self.draft:
            save_wav(out_path, synth_sfx(prompt))
            return
        import torch

        pipe = MODELS.get("audioldm:" + self.settings["sfx_model"], lambda: _load_audioldm(self.settings))
        generator = torch.Generator("cpu").manual_seed(int(seed))
        result = pipe(
            prompt,
            negative_prompt="low quality, music, speech",
            num_inference_steps=50,
            audio_length_in_s=float(seconds),
            generator=generator,
        )
        audio = np.asarray(result.audios[0], dtype=np.float32)
        _save_resampled(out_path, _trim_silence(audio, 16000), 16000)


def _trim_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    loud = np.where(np.abs(audio) > 0.02 * (np.abs(audio).max() or 1))[0]
    if loud.size == 0:
        return audio
    end = min(len(audio), loud[-1] + int(0.2 * sr))
    return audio[max(0, loud[0] - int(0.02 * sr)) : end]


def _save_resampled(out_path: Path, audio: np.ndarray, sr: int) -> None:
    """Save at the project sample rate (resampled by ffmpeg)."""
    tmp = Path(str(out_path) + ".tmp.wav")
    peak = float(np.max(np.abs(audio))) or 1.0
    save_wav(tmp, audio / peak * 0.9, sr)
    try:
        save_wav(out_path, load_audio(tmp, SAMPLE_RATE))
    finally:
        tmp.unlink(missing_ok=True)
