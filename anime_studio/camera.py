"""Camera moves over still keyframes (the classic "limited animation" look)."""

import zlib

import numpy as np
from PIL import Image

from .script_parser import CAMERA_MOVES, Shot


def auto_camera(shot: Shot) -> str:
    """Pick a sensible camera move when the script doesn't say."""
    if shot.camera:
        return shot.camera
    d = shot.description.lower()
    if any(w in d for w in ("close-up", "close up", "closeup", "face", "eyes")):
        return "zoom_in"
    if any(w in d for w in ("explosion", "earthquake", "impact", "crash", "punch", "shock")):
        return "shake"
    if any(w in d for w in ("wide shot", "establishing", "landscape", "panorama", "city", "sky")):
        return "pan_right"
    if any(w in d for w in ("reveal", "pull back", "zoom out")):
        return "zoom_out"
    cycle = ["static", "zoom_in", "pan_left", "zoom_out", "pan_right"]
    return cycle[zlib.crc32(shot.description.encode()) % len(cycle)]


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def crop_box(move: str, t: float, src_w: int, src_h: int, aspect: float, frame_index: int = 0, seed: int = 0):
    """Crop rectangle (left, top, right, bottom) for progress t in [0, 1]."""
    if move not in CAMERA_MOVES and move != "still":
        move = "static"
    # largest box of the output aspect ratio that fits in the picture
    if src_w / src_h > aspect:
        base_h, base_w = src_h, src_h * aspect
    else:
        base_w, base_h = src_w, src_w / aspect
    e = _ease(t)
    zoom = {
        "zoom_in": 1.0 + 0.18 * e,
        "zoom_out": 1.18 - 0.18 * e,
        "static": 1.0 + 0.04 * e,
        "still": 1.0,
        "shake": 1.08,
    }.get(move, 1.12)
    w, h = base_w / zoom, base_h / zoom
    cx, cy = src_w / 2, src_h / 2
    free_x, free_y = (src_w - w) / 2, (src_h - h) / 2
    if move == "pan_right":
        cx = src_w / 2 - free_x + 2 * free_x * e
    elif move == "pan_left":
        cx = src_w / 2 + free_x - 2 * free_x * e
    elif move == "pan_down":
        cy = src_h / 2 - free_y + 2 * free_y * e
    elif move == "pan_up":
        cy = src_h / 2 + free_y - 2 * free_y * e
    elif move == "shake":
        rng = np.random.default_rng(seed * 100003 + frame_index)
        strength = max(0.0, 1.0 - t * 1.5)  # strong at the start, settles down
        cx += rng.uniform(-1, 1) * free_x * 0.8 * strength
        cy += rng.uniform(-1, 1) * free_y * 0.8 * strength
    left = min(max(0.0, cx - w / 2), src_w - w)
    top = min(max(0.0, cy - h / 2), src_h - h)
    return left, top, left + w, top + h


def still_frame(image: Image.Image, move: str, t: float, size: tuple[int, int], frame_index: int = 0, seed: int = 0):
    box = crop_box(move, t, image.width, image.height, size[0] / size[1], frame_index, seed)
    return np.asarray(image.resize(size, Image.BICUBIC, box=box))


def cover_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a video frame to fill `size`, cropping the edges if needed."""
    img = Image.fromarray(frame).convert("RGB")
    box = crop_box("static", 0.0, img.width, img.height, size[0] / size[1])
    return np.asarray(img.resize(size, Image.BICUBIC, box=box))
