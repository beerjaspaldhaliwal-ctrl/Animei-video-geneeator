"""Episode editor: puts shots, voices, music, sound effects and subtitles together."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .camera import cover_frame, still_frame
from .fonts import font_for, wrap_text
from .lipsync import MOUTH_STATES, mouth_track
from .media import SAMPLE_RATE, ffmpeg_exe, load_audio, read_video_frames, run_ffmpeg, save_wav

log = logging.getLogger(__name__)

FADE = 0.4  # seconds of fade from/to black between scenes
LEAD_IN = 0.35  # silence before the first line of a shot
LINE_GAP = 0.3  # pause between lines
TAIL = 0.5  # time after the last line before the next shot
MIN_SHOT = 2.5


@dataclass
class TimedLine:
    start: float
    end: float
    text: str
    speaker: str = ""
    narration: bool = False
    audio: Path | None = None
    lipsync: bool = False  # move the on-screen mouth with this line


@dataclass
class ShotPlan:
    shot_id: str
    keyframe: Path | None
    camera: str
    duration: float = 0.0
    clip: Path | None = None
    lines: list[TimedLine] = field(default_factory=list)
    sfx: list[tuple[float, Path]] = field(default_factory=list)
    seed: int = 0
    mouths: dict[str, Path] | None = None  # lip sync pictures: closed / half / open


@dataclass
class ScenePlan:
    music: Path | None
    shots: list[ShotPlan] = field(default_factory=list)


@dataclass
class EpisodePlan:
    series_title: str
    episode_title: str
    scenes: list[ScenePlan] = field(default_factory=list)
    theme_music: Path | None = None


def layout_shot(plan: ShotPlan, events: list, min_seconds: float | None) -> None:
    """Place dialogue and sound effects in time and set the shot duration.

    events: list of ("line", TimedLine with audio, duration) or ("sfx", path, duration)
    """
    t = LEAD_IN
    for kind, item, dur in events:
        if kind == "line":
            item.start, item.end = t, t + dur
            plan.lines.append(item)
            t += dur + LINE_GAP
        else:
            plan.sfx.append((t, item))
            t += min(dur, 1.0)
    needed = t + TAIL if events else MIN_SHOT + 1.0
    plan.duration = max(needed, min_seconds or MIN_SHOT)


# ---------------------------------------------------------------- pictures
class _Overlay:
    """Pre-rendered RGBA text block blended onto frames with numpy."""

    def __init__(self, rgba: Image.Image, x: int, y: int):
        arr = np.asarray(rgba).astype(np.float32)
        self.rgb = arr[..., :3]
        self.alpha = arr[..., 3:4] / 255.0
        self.x, self.y = x, y

    def blend(self, frame: np.ndarray) -> np.ndarray:
        h, w = self.rgb.shape[:2]
        region = frame[self.y : self.y + h, self.x : self.x + w].astype(np.float32)
        frame[self.y : self.y + h, self.x : self.x + w] = (region * (1 - self.alpha) + self.rgb * self.alpha).astype(
            np.uint8
        )
        return frame


def subtitle_overlay(text: str, size: tuple[int, int], narration: bool = False) -> _Overlay:
    width, height = size
    font = font_for(text, max(18, height // 20))
    scratch = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = wrap_text(scratch, text, font, int(width * 0.84))[:3]
    line_h = int(font.size * 1.25)
    stroke = max(2, font.size // 12)
    box_h = line_h * len(lines) + stroke * 2 + 4
    img = Image.new("RGBA", (width, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = (255, 240, 180, 255) if narration else (255, 255, 255, 255)
    for i, line in enumerate(lines):
        tw = draw.textlength(line, font=font)
        draw.text(
            ((width - tw) / 2, stroke + i * line_h),
            line,
            font=font,
            fill=color,
            stroke_width=stroke,
            stroke_fill=(0, 0, 0, 255),
        )
    return _Overlay(img, 0, height - box_h - int(height * 0.05))


def card_image(background: Image.Image | None, title: str, subtitle: str, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if background is not None:
        bg = background.convert("RGB")
        box_aspect = width / height
        if bg.width / bg.height > box_aspect:
            new_w = int(bg.height * box_aspect)
            bg = bg.crop(((bg.width - new_w) // 2, 0, (bg.width + new_w) // 2, bg.height))
        else:
            new_h = int(bg.width / box_aspect)
            bg = bg.crop((0, (bg.height - new_h) // 2, bg.width, (bg.height + new_h) // 2))
        bg = bg.resize(size, Image.BICUBIC).filter(ImageFilter.GaussianBlur(8))
        bg = Image.blend(bg, Image.new("RGB", size, (0, 0, 0)), 0.55)
    else:
        bg = Image.new("RGB", size, (12, 12, 24))
    draw = ImageDraw.Draw(bg)
    big = font_for(title, max(28, height // 9))
    small = font_for(subtitle, max(18, height // 20))
    y = height * 0.36
    for line in wrap_text(draw, title, big, int(width * 0.85))[:3]:
        tw = draw.textlength(line, font=big)
        draw.text(((width - tw) / 2, y), line, font=big, fill="white", stroke_width=3, stroke_fill="black")
        y += big.size * 1.2
    if subtitle:
        tw = draw.textlength(subtitle, font=small)
        draw.text(
            ((width - tw) / 2, y + small.size * 0.6),
            subtitle,
            font=small,
            fill=(255, 220, 140),
            stroke_width=2,
            stroke_fill="black",
        )
    return np.asarray(bg)


def _fade_factor(t: float, duration: float, fade_in: bool, fade_out: bool) -> float:
    f = 1.0
    if fade_in and t < FADE:
        f = min(f, t / FADE)
    if fade_out and t > duration - FADE:
        f = min(f, max(0.0, (duration - t) / FADE))
    return f


class _ShotFrames:
    """Yields the frames of one shot (camera move over a still, or an AI clip)."""

    def __init__(self, plan: ShotPlan, size, fps):
        self.plan, self.size, self.fps = plan, size, fps
        self.clip_frames, self.clip_fps = None, fps
        self.still = None
        if plan.clip and Path(plan.clip).exists():
            self.clip_frames, self.clip_fps = read_video_frames(plan.clip)
        if plan.keyframe and Path(plan.keyframe).exists():
            self.still = Image.open(plan.keyframe).convert("RGB")
        if self.still is None and not self.clip_frames:
            self.still = Image.new("RGB", size, (20, 20, 30))
        self.mouths, self.track = None, None
        if not self.clip_frames and plan.mouths and all(Path(plan.mouths[s]).exists() for s in MOUTH_STATES):
            self.mouths = [Image.open(plan.mouths[s]).convert("RGB") for s in MOUTH_STATES]

    def frame(self, i: int, n: int) -> np.ndarray:
        if self.clip_frames:
            clip_seconds = len(self.clip_frames) / self.clip_fps
            # slow the clip down a little if the shot is longer, then hold
            stretch = min(1.5, max(1.0, (n / self.fps) / clip_seconds))
            idx = min(len(self.clip_frames) - 1, int(i / self.fps * self.clip_fps / stretch))
            return cover_frame(self.clip_frames[idx], self.size).copy()
        still = self.still
        if self.mouths:
            if self.track is None or len(self.track) != n:
                self.track = mouth_track(self.plan.lines, n, self.fps)
            still = self.mouths[self.track[i]]
        t = i / max(1, n - 1)
        return still_frame(still, self.plan.camera, t, self.size, i, self.plan.seed).copy()


# ---------------------------------------------------------------- audio
def _loop_to_length(audio: np.ndarray, n: int, crossfade: int) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(audio) >= n:
        return audio[:n].copy()
    crossfade = min(crossfade, len(audio) // 3)
    out = audio.copy()
    ramp = np.linspace(0, 1, crossfade, dtype=np.float32)
    while len(out) < n:
        if crossfade:
            out[-crossfade:] = out[-crossfade:] * (1 - ramp) + audio[:crossfade] * ramp
            out = np.concatenate([out, audio[crossfade:]])
        else:
            out = np.concatenate([out, audio])
    return out[:n]


def _add(track: np.ndarray, audio: np.ndarray, start_s: float, gain: float = 1.0) -> None:
    start = int(start_s * SAMPLE_RATE)
    if start >= len(track):
        return
    end = min(len(track), start + len(audio))
    track[start:end] += audio[: end - start] * gain


def _music_bed(path: Path, seconds: float) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    audio = _loop_to_length(load_audio(path), n, int(1.5 * SAMPLE_RATE))
    fade_in, fade_out = min(n // 4, SAMPLE_RATE), min(n // 3, int(1.2 * SAMPLE_RATE))
    if fade_in:
        audio[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    if fade_out:
        audio[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
    return audio


def _ducking(voice: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """Music gain curve that dips while people talk."""
    hop = SAMPLE_RATE // 50
    frames = len(voice) // hop + 1
    padded = np.zeros(frames * hop, dtype=np.float32)
    padded[: len(voice)] = np.abs(voice)
    level = padded.reshape(frames, hop).max(axis=1)
    active = (level > 0.02).astype(np.float32)
    kernel = np.ones(15, dtype=np.float32) / 15  # ~0.3 s smoothing
    active = np.clip(np.convolve(active, kernel, mode="same") * 2, 0, 1)
    gain = 1 - amount * np.repeat(active, hop)[: len(voice)]
    return gain


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


# ---------------------------------------------------------------- render
def render_episode(plan: EpisodePlan, settings: dict, out_dir: Path, progress=None) -> dict:
    import imageio_ffmpeg

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = int(settings["fps"])
    size = (int(settings["output_width"]) // 2 * 2, int(settings["output_height"]) // 2 * 2)
    title_seconds = 5.0 if settings.get("title_card") else 0.0
    end_seconds = 4.0 if settings.get("end_card") else 0.0

    # ---- timeline
    segments = []  # (kind, start, duration, payload, fade_in, fade_out)
    t = 0.0
    if title_seconds:
        segments.append(("card", t, title_seconds, "title", True, True))
        t += title_seconds
    scene_spans = []
    for scene in plan.scenes:
        scene_start = t
        for i, shot in enumerate(scene.shots):
            segments.append(("shot", t, shot.duration, shot, i == 0, i == len(scene.shots) - 1))
            t += shot.duration
        scene_spans.append((scene, scene_start, t - scene_start))
    if end_seconds:
        segments.append(("card", t, end_seconds, "end", True, True))
        t += end_seconds
    total = t
    total_frames = int(round(total * fps))

    # ---- audio mix
    n = int(total * SAMPLE_RATE) + SAMPLE_RATE
    voice = np.zeros(n, dtype=np.float32)
    sfx = np.zeros(n, dtype=np.float32)
    music = np.zeros(n, dtype=np.float32)
    subtitles = []
    for kind, start, duration, payload, _, _ in segments:
        if kind != "shot":
            continue
        for line in payload.lines:
            if line.audio and Path(line.audio).exists():
                _add(voice, load_audio(line.audio), start + line.start)
            subtitles.append((start + line.start, start + line.end, line))
        for offset, path in payload.sfx:
            if Path(path).exists():
                _add(sfx, load_audio(path), start + offset)
    for scene, start, duration in scene_spans:
        if scene.music and Path(scene.music).exists():
            _add(music, _music_bed(scene.music, duration), start)
    if plan.theme_music and Path(plan.theme_music).exists():
        for kind, start, duration, payload, _, _ in segments:
            if kind == "card":
                _add(music, _music_bed(plan.theme_music, duration + 0.5), start, 1.6)
    mix = (
        voice * float(settings["voice_volume"])
        + sfx * float(settings["sfx_volume"])
        + music * _ducking(voice) * float(settings["music_volume"])
    )
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak
    mix_path = out_dir / "mix.wav"
    save_wav(mix_path, mix[: int(total * SAMPLE_RATE)])

    # ---- subtitles file
    srt_path = out_dir / "episode.srt"
    show_names = settings.get("subtitle_speaker_names")
    entries = []
    for i, (s, e, line) in enumerate(subtitles, start=1):
        text = f"{line.speaker.title()}: {line.text}" if show_names and not line.narration else line.text
        entries.append(f"{i}\n{_srt_time(s)} --> {_srt_time(e)}\n{text}\n")
    srt_path.write_text("\n".join(entries), "utf-8")

    # ---- video frames
    first_keyframe = next(
        (sh.keyframe for sc in plan.scenes for sh in sc.shots if sh.keyframe and Path(sh.keyframe).exists()), None
    )
    bg = Image.open(first_keyframe) if first_keyframe else None
    cards = {
        "title": card_image(bg, plan.series_title, plan.episode_title, size),
        "end": card_image(bg, "To Be Continued...", plan.series_title, size),
    }
    burn = settings.get("burn_subtitles", True)
    sub_cache = {}

    video_path = out_dir / "video_only.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(video_path),
        size,
        fps=fps,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "18", "-preset", "medium"],
    )
    writer.send(None)
    written = 0
    try:
        for kind, start, duration, payload, fade_in, fade_out in segments:
            seg_frames = int(round((start + duration) * fps)) - int(round(start * fps))
            source = _ShotFrames(payload, size, fps) if kind == "shot" else None
            for i in range(seg_frames):
                local_t = i / fps
                frame = source.frame(i, seg_frames) if source else cards[payload].copy()
                if burn and kind == "shot":
                    for line in payload.lines:
                        if line.start <= local_t < line.end:
                            text = (
                                f"{line.speaker.title()}: {line.text}"
                                if show_names and not line.narration
                                else line.text
                            )
                            key = (text, line.narration)
                            if key not in sub_cache:
                                sub_cache[key] = subtitle_overlay(text, size, line.narration)
                            frame = sub_cache[key].blend(frame)
                f = _fade_factor(local_t, duration, fade_in, fade_out)
                if f < 1.0:
                    frame = (frame.astype(np.float32) * f).astype(np.uint8)
                writer.send(np.ascontiguousarray(frame))
                written += 1
                if progress and written % 24 == 0:
                    progress(written / max(1, total_frames), f"Rendering video {written}/{total_frames} frames")
    finally:
        writer.close()

    # ---- final mux
    final_path = out_dir / "episode.mp4"
    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-i",
            str(mix_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(final_path),
        ]
    )
    video_path.unlink(missing_ok=True)
    log.info("Episode written to %s (%s)", final_path, ffmpeg_exe())
    return {"video": final_path, "subtitles": srt_path, "audio": mix_path, "seconds": total}
