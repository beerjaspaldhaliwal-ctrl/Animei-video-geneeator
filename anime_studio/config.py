"""Global app settings, stored in settings.json next to app.py."""

import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = APP_DIR / "settings.json"

DEFAULTS = {
    # Where series projects and downloaded AI models are stored.
    "projects_dir": str(APP_DIR / "projects"),
    "models_dir": str(APP_DIR / "models"),
    # Hardware
    "device": "auto",  # auto | cuda | cpu   (AMD ROCm GPUs show up as "cuda")
    "low_vram": False,  # move model parts to RAM when idle (slower, needs less VRAM)
    # Draft mode: no AI models at all. Placeholder pictures, "beep" voices and
    # simple synth music, so you can check script timing in seconds.
    "draft_mode": False,
    # Keyframe images (any SDXL anime model: Hugging Face id or a .safetensors file)
    "image_model": "cagliostrolab/animagine-xl-4.0",
    "image_vae": "madebyollin/sdxl-vae-fp16-fix",
    "image_steps": 28,
    "image_guidance": 5.0,
    "image_width": 1344,
    "image_height": 768,
    "ip_adapter": True,  # use character reference pictures for consistent faces
    "ip_adapter_scale": 0.5,
    # Animation. "motion" = camera moves over the keyframe (fast, always works).
    # "ai" = AI image-to-video for every shot (slow). Shots can override with
    # "| animate: ai" or "| animate: motion" in the script.
    "default_animation": "motion",
    "video_engine": "ltx",  # ltx | wan
    "ltx_model": "Lightricks/LTX-Video",
    "wan_model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "video_steps": 30,
    "video_seconds": 4.0,
    # Music and sound effects
    "music_model": "facebook/musicgen-small",
    "sfx_model": "cvssp/audioldm-s-full-v2",
    # Output video
    "fps": 24,
    "output_width": 1280,
    "output_height": 720,
    "burn_subtitles": True,
    "subtitle_speaker_names": False,
    "title_card": True,
    "end_card": True,
    "voice_volume": 1.0,
    "music_volume": 0.30,
    "sfx_volume": 0.8,
    # Story assistant (optional, needs Ollama: https://ollama.com)
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.1:8b",
}


def load_settings() -> dict:
    settings = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            settings.update(json.loads(SETTINGS_PATH.read_text("utf-8")))
        except (OSError, ValueError):
            pass
    return settings


def save_settings(settings: dict) -> None:
    data = {k: settings[k] for k in DEFAULTS if k in settings}
    SETTINGS_PATH.write_text(json.dumps(data, indent=2), "utf-8")
