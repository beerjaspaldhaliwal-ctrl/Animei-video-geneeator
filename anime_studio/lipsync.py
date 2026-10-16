"""Anime-style lip sync ("kuchipaku").

Like TV anime, the talking character's mouth switches between a few drawn
shapes (closed, half open, open) in time with the voice:

1. find the face in the keyframe (small anime face detector, runs on CPU),
2. redraw only the mouth area three times with SDXL inpainting,
3. while rendering, pick the mouth picture for each frame from the loudness
   of the voice, changing at most every 2 frames ("animating on twos").
"""

import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .media import load_audio

log = logging.getLogger(__name__)

MOUTH_STATES = ("closed", "half", "open")
MOUTH_TAGS = {
    "closed": ("closed mouth", "open mouth, teeth, tongue"),
    "half": ("parted lips, talking", "closed mouth, wide open mouth"),
    "open": ("open mouth, talking", "closed mouth"),
}
FACE_REPO = "deepghs/anime_face_detection"
FACE_MODEL = "face_detect_v1.4_s/model.onnx"
FACE_THRESHOLD = 0.307
CROP_SIZE = 768

_session = None


# ---------------------------------------------------------------- face detection
def _detector():
    global _session
    if _session is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        _session = ort.InferenceSession(hf_hub_download(FACE_REPO, FACE_MODEL), providers=["CPUExecutionProvider"])
    return _session


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.6) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        overlap = inter / (area[i] + area[order[1:]] - inter + 1e-6)
        order = order[1:][overlap < iou]
    return keep


def detect_faces(image: Image.Image) -> list[tuple[float, float, float, float]]:
    """Anime face boxes (x0, y0, x1, y1), biggest first."""
    size = 640
    scale = size / max(image.size)
    resized = image.convert("RGB").resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (0, 0))
    x = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = _detector().run(None, {"images": x})[0][0]
    if out.shape[0] < out.shape[1]:  # (4 + classes, anchors) -> (anchors, 4 + classes)
        out = out.T
    scores = out[:, 4:].max(axis=1)
    mask = scores > FACE_THRESHOLD
    if not mask.any():
        return []
    cx, cy, w, h = out[mask, 0], out[mask, 1], out[mask, 2], out[mask, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) / scale
    keep = _nms(boxes, scores[mask])
    faces = [tuple(float(v) for v in boxes[i]) for i in keep]
    return sorted(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


def pick_face(faces: list, choice: str = "auto"):
    if not faces:
        return None
    if choice == "left":
        return min(faces, key=lambda b: b[0] + b[2])
    if choice == "right":
        return max(faces, key=lambda b: b[0] + b[2])
    return faces[0]  # biggest


# ---------------------------------------------------------------- mouth pictures
def mouth_ellipse(face) -> tuple[float, float, float, float]:
    """Where the mouth is inside an anime face box (anime proportions)."""
    x0, y0, x1, y1 = face
    fw, fh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, y0 + 0.765 * fh
    return cx - 0.17 * fw, cy - 0.085 * fh, cx + 0.17 * fw, cy + 0.085 * fh


def _crop_box(face, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = face
    side = min(max(x1 - x0, y1 - y0) * 1.7, width, height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 0.1 * (y1 - y0)
    left = int(min(max(0, cx - side / 2), width - side))
    top = int(min(max(0, cy - side / 2), height - side))
    return left, top, left + int(side), top + int(side)


def mouth_variants(engine, image: Image.Image, face, prompt: str, negative: str, seed: int, **inpaint_kwargs) -> dict:
    """Return {state: full keyframe with that mouth}. Only the mouth area changes."""
    image = image.convert("RGB")
    box = _crop_box(face, image.width, image.height)
    side = box[2] - box[0]
    crop = image.crop(box).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
    k = CROP_SIZE / side
    ex0, ey0, ex1, ey1 = mouth_ellipse(face)
    ellipse = [(ex0 - box[0]) * k, (ey0 - box[1]) * k, (ex1 - box[0]) * k, (ey1 - box[1]) * k]
    hard = Image.new("L", (CROP_SIZE, CROP_SIZE), 0)
    ImageDraw.Draw(hard).ellipse(ellipse, fill=255)
    soft = hard.filter(ImageFilter.GaussianBlur(CROP_SIZE * 0.015))
    paste_mask = soft.resize((side, side), Image.LANCZOS)
    variants = {}
    for state in MOUTH_STATES:
        want, avoid = MOUTH_TAGS[state]
        patch = engine.inpaint(
            crop, hard, f"{want}, {prompt}", f"{avoid}, {negative}", seed, strength=0.75, **inpaint_kwargs
        )
        result = image.copy()
        result.paste(patch.convert("RGB").resize((side, side), Image.LANCZOS), box[:2], paste_mask)
        variants[state] = result
    return variants


# ---------------------------------------------------------------- timing
def mouth_track(lines, n_frames: int, fps: int) -> np.ndarray:
    """Mouth state index (0 closed, 1 half, 2 open) for every frame of a shot."""
    track = np.zeros(n_frames, dtype=np.int8)
    sr = 16000
    hop = sr / fps
    for line in lines:
        if not getattr(line, "lipsync", False) or not line.audio:
            continue
        audio = load_audio(line.audio, sr)
        n = int(len(audio) / hop)
        if n <= 0:
            continue
        edges = (np.arange(n + 1) * hop).astype(int)
        level = np.sqrt(np.add.reduceat(audio[: edges[-1]] ** 2, edges[:-1]) / np.diff(edges))
        level = level / (np.percentile(level, 90) or 1.0)
        states = np.where(level < 0.15, 0, np.where(level < 0.55, 1, 2)).astype(np.int8)
        states = np.repeat(states[::2], 2)[:n]  # change at most every 2 frames, like anime
        start = int(round(line.start * fps))
        end = min(n_frames, start + n)
        if end > start:
            track[start:end] = states[: end - start]
    return track
