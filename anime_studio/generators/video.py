"""AI image-to-video: bring a keyframe to life with LTX-Video or Wan 2.2."""

import logging

from PIL import Image

from ..device import MODELS, bf16_dtype, pick_device, place_pipeline

log = logging.getLogger(__name__)

ENGINES = {
    "ltx": "LTX-Video 2B (faster, ~28 GB download)",
    "wan": "Wan 2.2 TI2V 5B (better quality, slower, ~34 GB download, needs 32 GB RAM)",
}
VIDEO_NEGATIVE = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, deformed face, extra limbs, "
    "static, still image, watermark, text"
)


def _cover_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _load_ltx(settings):
    from diffusers import LTXImageToVideoPipeline

    device = pick_device(settings)
    pipe = LTXImageToVideoPipeline.from_pretrained(settings["ltx_model"], torch_dtype=bf16_dtype(device))
    pipe = place_pipeline(pipe, device, settings, force_offload=True)
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _load_wan(settings):
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    device = pick_device(settings)
    vae = AutoencoderKLWan.from_pretrained(settings["wan_model"], subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(settings["wan_model"], vae=vae, torch_dtype=bf16_dtype(device))
    pipe = place_pipeline(pipe, device, settings, force_offload=True)
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


class VideoEngine:
    # Both models work in multiples of 32 pixels; this is a 16:9-ish size that
    # keeps generation time reasonable on a 16 GB card.
    width = 832
    height = 480

    def __init__(self, settings: dict):
        self.settings = settings

    def cache_key(self) -> list:
        s = self.settings
        engine = s["video_engine"]
        return [engine, s[f"{engine}_model"], s["video_steps"], s["video_seconds"], self.width, self.height]

    def generate(self, image_path: str, prompt: str, seed: int, seconds: float | None = None) -> tuple[list, int]:
        """Return (list of PIL frames, fps)."""
        import torch

        engine = self.settings["video_engine"]
        if engine not in ENGINES:
            raise ValueError(f"Unknown video engine '{engine}'. Use one of: {', '.join(ENGINES)}")
        seconds = float(seconds or self.settings["video_seconds"])
        fps = 24
        image = _cover_resize(Image.open(image_path).convert("RGB"), self.width, self.height)
        generator = torch.Generator("cpu").manual_seed(int(seed))
        steps = int(self.settings["video_steps"])

        if engine == "ltx":
            pipe = MODELS.get("ltx:" + self.settings["ltx_model"], lambda: _load_ltx(self.settings))
            frames = int(seconds * fps) // 8 * 8 + 1  # LTX needs 8k+1 frames
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=VIDEO_NEGATIVE,
                width=self.width,
                height=self.height,
                num_frames=max(9, frames),
                frame_rate=fps,
                num_inference_steps=steps,
                guidance_scale=3.0,
                decode_timestep=0.03,
                decode_noise_scale=0.025,
                generator=generator,
            )
        else:
            pipe = MODELS.get("wan:" + self.settings["wan_model"], lambda: _load_wan(self.settings))
            frames = int(seconds * fps) // 4 * 4 + 1  # Wan needs 4k+1 frames
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=VIDEO_NEGATIVE,
                width=self.width,
                height=self.height,
                num_frames=max(5, frames),
                num_inference_steps=steps,
                guidance_scale=5.0,
                generator=generator,
            )
        return list(result.frames[0]), fps
