"""GPU detection and a one-model-at-a-time cache to fit in 16 GB of VRAM."""

import gc
import hashlib
import json
import logging
import os
import subprocess
import sys

log = logging.getLogger(__name__)

_GPU_PROBE = """
import json, torch
gpus = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append([i, p.name, int(getattr(p, "is_integrated", 0)), p.multi_processor_count, p.total_memory])
print(json.dumps(gpus))
"""


def isolate_discrete_gpu() -> str | None:
    """Make PyTorch use the graphics card, not the CPU's built-in graphics.

    Ryzen 7000/9000 CPUs have a small integrated GPU that can show up as
    device 0, and PyTorch would pick it. We list the GPUs in a separate
    process and expose only the strongest one. Must run before torch is
    imported. Returns the chosen GPU name, or None if nothing was changed.
    """
    if any(k in os.environ for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        return None
    try:
        result = subprocess.run([sys.executable, "-c", _GPU_PROBE], capture_output=True, text=True, timeout=180)
        gpus = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:  # no torch, probe crashed, ...
        log.debug("GPU probe failed: %s", exc)
        return None
    if len(gpus) < 2:
        return None
    # Prefer non-integrated, then most compute units, then most memory.
    best = max(gpus, key=lambda g: (not g[2], g[3], g[4]))
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = str(best[0])
    log.info("Several GPUs found %s; using %s", [g[1] for g in gpus], best[1])
    return best[1]


def gpu_info() -> dict:
    """Describe the compute device PyTorch can see."""
    try:
        import torch
    except ImportError:
        return {"backend": "none", "name": "PyTorch is not installed", "vram_gb": 0.0}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        backend = "ROCm (AMD)" if getattr(torch.version, "hip", None) else "CUDA (NVIDIA)"
        return {"backend": backend, "name": props.name, "vram_gb": round(props.total_memory / 1024**3, 1)}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"backend": "MPS (Apple)", "name": "Apple GPU", "vram_gb": 0.0}
    return {"backend": "CPU", "name": "No GPU found, running on CPU (very slow)", "vram_gb": 0.0}


def pick_device(settings: dict) -> str:
    import torch

    wanted = settings.get("device", "auto")
    if wanted != "auto":
        return wanted
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def half_dtype(device: str):
    """fp16 on GPUs, fp32 on CPU (CPU fp16 is slow or unsupported)."""
    import torch

    return torch.float32 if device == "cpu" else torch.float16


def bf16_dtype(device: str):
    import torch

    return torch.float32 if device == "cpu" else torch.bfloat16


def content_hash(*parts) -> str:
    data = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:16]


class ModelCache:
    """Keeps only one big model in memory. Loading another frees the previous one."""

    def __init__(self):
        self.key = None
        self.model = None

    def get(self, key: str, loader):
        if self.key == key and self.model is not None:
            return self.model
        self.unload()
        log.info("Loading model: %s", key)
        self.model = loader()
        self.key = key
        return self.model

    def unload(self):
        if self.model is None:
            return
        log.info("Unloading model: %s", self.key)
        self.model = None
        self.key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


MODELS = ModelCache()


def place_pipeline(pipe, device: str, settings: dict, force_offload: bool = False):
    """Move a diffusers pipeline to the GPU, or use CPU offload for big models."""
    if device != "cpu" and (force_offload or settings.get("low_vram")):
        try:
            pipe.enable_model_cpu_offload(device=device)
            return pipe
        except Exception as exc:  # accelerate missing or unsupported device
            log.warning("CPU offload not available (%s); loading fully on %s", exc, device)
    return pipe.to(device)
