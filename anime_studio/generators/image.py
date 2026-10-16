"""Anime keyframe pictures with Stable Diffusion XL (Animagine by default)."""

import colorsys
import hashlib
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from ..device import MODELS, half_dtype, pick_device, place_pipeline
from ..fonts import font_for, wrap_text

log = logging.getLogger(__name__)

IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHTS = "ip-adapter-plus_sdxl_vit-h.safetensors"


# ---------------------------------------------------------------- draft mode
def placeholder_image(prompt: str, width: int, height: int, seed: int = 0, label: str = "") -> Image.Image:
    """A colourful card showing the prompt. Used in draft mode (no AI)."""
    digest = hashlib.md5(f"{prompt}{seed}".encode()).digest()
    hue = digest[0] / 255
    top = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.45, 0.95))
    bottom = tuple(int(c * 255) for c in colorsys.hsv_to_rgb((hue + 0.15) % 1, 0.7, 0.35))
    gradient = Image.linear_gradient("L").resize((width, height))
    img = Image.composite(Image.new("RGB", (width, height), bottom), Image.new("RGB", (width, height), top), gradient)
    draw = ImageDraw.Draw(img)
    # a few soft "bokeh" circles so camera moves are visible
    for i in range(12):
        x = digest[i % 16] / 255 * width
        y = digest[(i + 5) % 16] / 255 * height
        r = 20 + digest[(i + 9) % 16] / 255 * height * 0.18
        draw.ellipse([x - r, y - r, x + r, y + r], fill=tuple(min(255, c + 40) for c in top))
    img = img.filter(ImageFilter.GaussianBlur(6))
    draw = ImageDraw.Draw(img)
    big = font_for(label or "DRAFT", max(24, height // 10))
    draw.text(
        (width * 0.05, height * 0.06), label or "DRAFT", font=big, fill="white", stroke_width=3, stroke_fill="black"
    )
    small = font_for(prompt, max(16, height // 28))
    lines = wrap_text(draw, prompt[:400], small, int(width * 0.9))[:8]
    y = height * 0.25
    for line in lines:
        draw.text((width * 0.05, y), line, font=small, fill="white", stroke_width=2, stroke_fill="black")
        y += small.size * 1.3
    return img


# ---------------------------------------------------------------- SDXL
def _load_sdxl(settings: dict):
    import torch
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

    device = pick_device(settings)
    dtype = half_dtype(device)
    model = settings["image_model"].strip()
    kwargs = {"torch_dtype": dtype}
    if settings.get("image_vae") and dtype == torch.float16:
        kwargs["vae"] = AutoencoderKL.from_pretrained(settings["image_vae"], torch_dtype=dtype)
    if model.lower().endswith((".safetensors", ".ckpt")):
        pipe = StableDiffusionXLPipeline.from_single_file(model, **kwargs)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(model, **kwargs)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe._anime_ip_adapter = False
    if settings.get("ip_adapter"):
        try:
            pipe.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder="sdxl_models",
                weight_name=IP_ADAPTER_WEIGHTS,
                image_encoder_folder="models/image_encoder",
            )
            pipe._anime_ip_adapter = True
        except Exception as exc:
            log.warning("Could not load IP-Adapter, character references disabled: %s", exc)
    pipe._anime_loras = set()
    pipe = place_pipeline(pipe, device, settings)
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _encode_long_prompt(pipe, prompt: str, negative: str):
    """SDXL prompt embeddings without the 77-token limit.

    The text is split into 75-token chunks; each chunk is encoded separately
    and the results are concatenated, so long character descriptions are kept.
    """
    import torch

    device = pipe._execution_device
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    encoders = [pipe.text_encoder, pipe.text_encoder_2]

    def token_chunks(tokenizer, text):
        ids = tokenizer(text, add_special_tokens=False, verbose=False).input_ids
        return [ids[i : i + 75] for i in range(0, len(ids), 75)] or [[]]

    def encode(text, n_chunks):
        per_encoder, pooled = [], None
        for tokenizer, encoder in zip(tokenizers, encoders):
            chunks = token_chunks(tokenizer, text)
            chunks += [[]] * (n_chunks - len(chunks))
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            embeds = []
            for chunk in chunks:
                ids = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]
                ids += [pad] * (77 - len(ids))
                out = encoder(torch.tensor([ids], device=device), output_hidden_states=True)
                if pooled is None and out[0].ndim == 2:
                    pooled = out[0]  # pooled output of the second (projection) encoder
                embeds.append(out.hidden_states[-2])
            per_encoder.append(torch.cat(embeds, dim=1))
        return torch.cat(per_encoder, dim=-1), pooled

    n = max(len(token_chunks(t, x)) for t in tokenizers for x in (prompt, negative))
    with torch.no_grad():
        prompt_embeds, pooled = encode(prompt, n)
        negative_embeds, negative_pooled = encode(negative, n)
    dtype = pipe.unet.dtype
    return (
        prompt_embeds.to(dtype),
        negative_embeds.to(dtype),
        pooled.to(dtype),
        negative_pooled.to(dtype),
    )


def _apply_loras(pipe, loras: list[tuple[str, float]]):
    wanted = {}
    for path, weight in loras:
        if not path or not Path(path).exists():
            continue
        name = "lora_" + hashlib.md5(str(path).encode()).hexdigest()[:10]
        if name not in pipe._anime_loras:
            pipe.load_lora_weights(path, adapter_name=name)
            pipe._anime_loras.add(name)
        wanted[name] = float(weight)
    if wanted:
        pipe.enable_lora()
        pipe.set_adapters(list(wanted), adapter_weights=list(wanted.values()))
    elif pipe._anime_loras:
        pipe.disable_lora()


class ImageEngine:
    def __init__(self, settings: dict):
        self.settings = settings

    @property
    def draft(self) -> bool:
        return bool(self.settings.get("draft_mode"))

    def cache_key(self) -> list:
        s = self.settings
        if self.draft:
            return ["draft-image"]
        return [s["image_model"], s.get("image_vae"), s["image_steps"], s["image_guidance"], s.get("ip_adapter")]

    def generate(
        self,
        prompt: str,
        negative: str,
        seed: int,
        width: int | None = None,
        height: int | None = None,
        reference: str | None = None,
        reference_scale: float | None = None,
        loras: list[tuple[str, float]] = (),
        label: str = "",
    ) -> Image.Image:
        width = int(width or self.settings["image_width"])
        height = int(height or self.settings["image_height"])
        if self.draft:
            return placeholder_image(prompt, width, height, seed, label)

        import torch

        pipe = MODELS.get("sdxl:" + str(self.cache_key()), lambda: _load_sdxl(self.settings))
        _apply_loras(pipe, list(loras))
        embeds = _encode_long_prompt(pipe, prompt, negative)
        kwargs = {}
        if pipe._anime_ip_adapter:
            if reference:
                ref_img = Image.open(reference).convert("RGB")
                pipe.set_ip_adapter_scale(
                    self.settings["ip_adapter_scale"] if reference_scale is None else reference_scale
                )
            else:
                ref_img = Image.new("RGB", (224, 224))
                pipe.set_ip_adapter_scale(0.0)
            kwargs["ip_adapter_image"] = ref_img
        generator = torch.Generator("cpu").manual_seed(int(seed))
        result = pipe(
            prompt_embeds=embeds[0],
            negative_prompt_embeds=embeds[1],
            pooled_prompt_embeds=embeds[2],
            negative_pooled_prompt_embeds=embeds[3],
            width=width,
            height=height,
            num_inference_steps=int(self.settings["image_steps"]),
            guidance_scale=float(self.settings["image_guidance"]),
            generator=generator,
            **kwargs,
        )
        return result.images[0]
