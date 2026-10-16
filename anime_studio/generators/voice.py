"""Character voices.

The voice name picks the engine automatically:
  * Kokoro (offline, runs on your PC):  af_heart, am_michael, jf_alpha, ...
  * Edge (free Microsoft voices, needs internet): en-US-AnaNeural, ja-JP-NanamiNeural, ...
  * beep: cute blips, used in draft mode
"""

import asyncio
import logging
import re
import tempfile
import zlib
from pathlib import Path

import numpy as np

from ..media import SAMPLE_RATE, load_audio, save_wav

log = logging.getLogger(__name__)

KOKORO_VOICES = {
    "American English": [
        "af_heart",
        "af_bella",
        "af_nicole",
        "af_sarah",
        "af_sky",
        "af_nova",
        "af_aoede",
        "af_kore",
        "af_jessica",
        "af_river",
        "af_alloy",
        "am_michael",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_onyx",
        "am_puck",
        "am_santa",
    ],
    "British English": [
        "bf_emma",
        "bf_alice",
        "bf_isabella",
        "bf_lily",
        "bm_george",
        "bm_daniel",
        "bm_fable",
        "bm_lewis",
    ],
    "Japanese (needs misaki[ja])": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "Chinese (needs misaki[zh])": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi"],
    "Spanish": ["ef_dora", "em_alex"],
    "French": ["ff_siwis"],
    "Hindi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "Italian": ["if_sara", "im_nicola"],
    "Portuguese": ["pf_dora", "pm_alex"],
}
EDGE_VOICES = [
    "en-US-AnaNeural",
    "en-US-AriaNeural",
    "en-US-JennyNeural",
    "en-US-MichelleNeural",
    "en-US-GuyNeural",
    "en-US-ChristopherNeural",
    "en-US-EricNeural",
    "en-US-RogerNeural",
    "en-GB-SoniaNeural",
    "en-GB-MaisieNeural",
    "en-GB-RyanNeural",
    "ja-JP-NanamiNeural",
    "ja-JP-KeitaNeural",
    "hi-IN-SwaraNeural",
    "hi-IN-MadhurNeural",
    "es-ES-ElviraNeural",
    "fr-FR-DeniseNeural",
    "ko-KR-SunHiNeural",
    "zh-CN-XiaoxiaoNeural",
]
ALL_VOICES = [v for group in KOKORO_VOICES.values() for v in group] + EDGE_VOICES + ["beep"]

_FAST = ("excited", "angry", "shout", "yell", "scream", "panic", "scared", "hurry", "fast")
_SLOW = ("sad", "tired", "whisper", "slow", "sleepy", "quiet", "soft", "cry", "sob")


def engine_for(voice: str) -> str:
    if voice == "beep":
        return "beep"
    if re.fullmatch(r"[a-z]{2,3}-[A-Z]{2}(-\w+)?-\w+Neural", voice or ""):
        return "edge"
    return "kokoro"


def emotion_adjust(emotion: str) -> tuple[float, float]:
    """(speed multiplier, volume multiplier) for an emotion like '(angry)'."""
    e = (emotion or "").lower()
    speed = 1.12 if any(w in e for w in _FAST) else 0.88 if any(w in e for w in _SLOW) else 1.0
    volume = 0.6 if "whisper" in e else 1.15 if any(w in e for w in ("shout", "yell", "scream")) else 1.0
    return speed, volume


# ---------------------------------------------------------------- engines
def _beep_voice(text: str, voice_seed: str, speed: float) -> np.ndarray:
    """Animal-Crossing style blips, one per syllable."""
    sr = SAMPLE_RATE
    rng = np.random.default_rng(zlib.crc32(voice_seed.encode()))
    base = 180 + rng.random() * 260
    out = []
    syllable = 0.075 / speed
    for word in re.findall(r"[\w']+|[.,!?;:…]", text):
        if word in ".!?…":
            out.append(np.zeros(int(sr * 0.25 / speed)))
            continue
        if word in ",;:":
            out.append(np.zeros(int(sr * 0.12 / speed)))
            continue
        for _ in range(max(1, len(re.findall(r"[aeiouyAEIOUY]+", word)) or len(word) // 2)):
            n = int(sr * syllable)
            t = np.arange(n) / sr
            freq = base * (0.9 + rng.random() * 0.25)
            env = np.sin(np.pi * np.arange(n) / n) ** 2
            out.append(
                0.25 * env * np.sign(np.sin(2 * np.pi * freq * t)) * 0.6 + 0.15 * env * np.sin(4 * np.pi * freq * t)
            )
            out.append(np.zeros(int(sr * 0.02)))
        out.append(np.zeros(int(sr * 0.05)))
    return np.concatenate(out) if out else np.zeros(sr // 2)


_kokoro_pipelines = {}


def _shorten_espeak_path() -> None:
    """espeak-ng (used by Kokoro for unusual words such as names) only accepts
    data paths shorter than ~160 characters and exits the whole program
    otherwise. When the install path is long, use a copy in the temp folder."""
    try:
        import shutil

        import espeakng_loader
        import misaki.espeak  # noqa: F401  (sets the default data path on import)
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        data = espeakng_loader.get_data_path()
        if len(data) > 100:
            short = Path(tempfile.gettempdir()) / "as-espeak-data"
            if not (short / "phontab").exists():
                shutil.copytree(data, short, dirs_exist_ok=True)
            EspeakWrapper.set_data_path(str(short))
    except Exception as exc:  # optional dependency layout changed; Kokoro may still work
        log.debug("espeak path fix skipped: %s", exc)


def _kokoro(text: str, voice: str, speed: float) -> np.ndarray:
    from kokoro import KPipeline

    lang = voice[0] if voice and voice[0] in "abjzefhip" else "a"
    if lang not in _kokoro_pipelines:
        _shorten_espeak_path()
        # Kokoro is tiny (82M); the CPU is fast enough and avoids GPU memory fights.
        _kokoro_pipelines[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device="cpu")
    pipeline = _kokoro_pipelines[lang]
    chunks = []
    for result in pipeline(text, voice=voice, speed=speed):
        audio = result.audio if hasattr(result, "audio") else result[2]
        if audio is not None:
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    if not chunks:
        raise RuntimeError(f"Kokoro produced no audio for: {text!r}")
    audio = np.concatenate(chunks)
    # resample 24 kHz -> project rate with ffmpeg for good quality
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "k.wav"
        save_wav(src, audio, 24000)
        return load_audio(src, SAMPLE_RATE)


def _edge(text: str, voice: str, speed: float) -> np.ndarray:
    import edge_tts

    rate = f"{round((speed - 1) * 100):+d}%"
    with tempfile.TemporaryDirectory() as tmp:
        mp3 = Path(tmp) / "e.mp3"

        async def run():
            await edge_tts.Communicate(text, voice, rate=rate).save(str(mp3))

        asyncio.run(run())
        return load_audio(mp3, SAMPLE_RATE)


class VoiceEngine:
    def __init__(self, settings: dict):
        self.settings = settings

    def resolve(self, voice: str) -> str:
        return "beep" if self.settings.get("draft_mode") else (voice or "af_heart")

    def cache_key(self, text: str, voice: str, speed: float, emotion: str) -> list:
        return ["voice-v1", self.resolve(voice), text, round(float(speed), 3), emotion]

    def synthesize(
        self, text: str, voice: str, out_path: Path, speed: float = 1.0, emotion: str = "", speaker: str = ""
    ) -> None:
        voice = self.resolve(voice)
        emo_speed, volume = emotion_adjust(emotion)
        speed = float(speed) * emo_speed
        # Strip stage directions like *sigh* or [laughs] that TTS would read aloud.
        clean = re.sub(r"\*[^*]*\*|\[[^\]]*\]", "", text).strip() or text
        engine = engine_for(voice)
        if engine == "beep":
            audio = _beep_voice(clean, speaker or voice, speed)
        elif engine == "edge":
            audio = _edge(clean, voice, speed)
        else:
            audio = _kokoro(clean, voice, speed)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.85
        save_wav(out_path, np.clip(audio * volume, -1, 1), SAMPLE_RATE)
