"""The studio pipeline: script -> keyframes -> animation -> voices/music/SFX -> episode."""

import logging
import zlib
from pathlib import Path

from PIL import Image

from .camera import auto_camera
from .config import APP_DIR
from .device import MODELS, content_hash
from .editor import EpisodePlan, ScenePlan, ShotPlan, TimedLine, layout_shot, render_episode
from .generators.audio import MusicEngine, SfxEngine, synth_music, synth_sfx
from .generators.image import ImageEngine
from .generators.video import VideoEngine
from .generators.voice import VoiceEngine
from .lipsync import MOUTH_STATES, detect_faces, mouth_variants, pick_face
from .media import audio_duration, save_frames_as_video, save_wav
from .project import Character, Project
from .prompts import character_sheet_prompt, shot_characters, shot_prompt
from .script_parser import Line, Script, Shot, parse_script

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".mp3", ".wav", ".ogg", ".flac", ".m4a")


def _file_tag(path: Path | None) -> str:
    if not path or not Path(path).exists():
        return ""
    st = Path(path).stat()
    return f"{Path(path).name}:{st.st_size}:{int(st.st_mtime)}"


class Studio:
    def __init__(self, project: Project, settings: dict):
        self.project = project
        self.settings = settings
        self.images = ImageEngine(settings)
        self.video = VideoEngine(settings)
        self.voices = VoiceEngine(settings)
        self.music = MusicEngine(settings)
        self.sfx = SfxEngine(settings)
        self.warnings: list[str] = []

    @property
    def series(self):
        return self.project.series

    def _warn(self, message: str) -> None:
        log.warning(message)
        if message not in self.warnings:
            self.warnings.append(message)

    # ------------------------------------------------------------ script
    def parse(self, episode: str) -> Script:
        return parse_script(self.project.read_script(episode))

    def scene_of(self, script: Script, shot: Shot):
        return script.scenes[shot.scene_index - 1]

    def shot_seed(self, shot: Shot) -> int:
        if shot.seed is not None:
            return shot.seed
        return (self.series.seed + zlib.crc32(shot.description.encode())) % 2**31

    # ------------------------------------------------------------ keyframes
    def _lora_path(self, char: Character) -> Path | None:
        if not char.lora_path:
            return None
        for candidate in (Path(char.lora_path), APP_DIR / "loras" / char.lora_path, self.project.root / char.lora_path):
            if candidate.exists():
                return candidate
        self._warn(f"LoRA file for {char.name} not found: {char.lora_path}")
        return None

    def keyframe_spec(self, script: Script, shot: Shot, seed: int | None = None) -> dict:
        prompt, negative = shot_prompt(self.series, self.scene_of(script, shot), shot)
        chars = shot_characters(self.series, shot)
        refs = [self.project.abs_path(c.reference_image) for c in chars]
        # A reference picture only helps when one character is on screen.
        reference = refs[0] if len(chars) == 1 and refs[0] else None
        loras = [(str(p), c.lora_weight) for c in chars if (p := self._lora_path(c))]
        return {
            "prompt": prompt,
            "negative": negative,
            "seed": self.shot_seed(shot) if seed is None else int(seed),
            "reference": str(reference) if reference else None,
            "loras": loras,
            "label": shot.id,
        }

    def keyframe_path(self, spec: dict) -> Path:
        key = content_hash(
            self.images.cache_key(),
            spec["prompt"],
            spec["negative"],
            spec["seed"],
            self.settings["image_width"],
            self.settings["image_height"],
            _file_tag(spec["reference"]),
            [(_file_tag(Path(p)), w) for p, w in spec["loras"]],
        )
        return self.project.cache_dir("keyframes") / f"{key}.png"

    def make_keyframe(self, script: Script, shot: Shot, seed: int | None = None, force: bool = False) -> Path:
        spec = self.keyframe_spec(script, shot, seed)
        path = self.keyframe_path(spec)
        if force or not path.exists():
            image = self.images.generate(
                spec["prompt"],
                spec["negative"],
                spec["seed"],
                reference=spec["reference"],
                loras=spec["loras"],
                label=spec["label"],
            )
            image.save(path)
        return path

    def existing_keyframe(self, script: Script, shot: Shot) -> Path | None:
        path = self.keyframe_path(self.keyframe_spec(script, shot))
        return path if path.exists() else None

    # ------------------------------------------------------------ animation
    def animate_mode(self, shot: Shot) -> str:
        mode = shot.animate or self.settings.get("default_animation", "motion")
        if mode == "ai" and self.settings.get("draft_mode"):
            return "motion"
        return mode

    def clip_path(self, keyframe: Path, shot: Shot) -> Path:
        motion = self.motion_prompt(shot)
        key = content_hash(self.video.cache_key(), _file_tag(keyframe), motion, self.shot_seed(shot))
        return self.project.cache_dir("clips") / f"{key}.mp4"

    def motion_prompt(self, shot: Shot) -> str:
        base = shot.motion or shot.description
        if self.on_screen_speaker(shot):
            base += ", the character is talking"
        return f"anime style animation, {base}, smooth natural motion, consistent character, high quality"

    def make_clip(self, script: Script, shot: Shot, force: bool = False) -> Path:
        keyframe = self.make_keyframe(script, shot)
        path = self.clip_path(keyframe, shot)
        if force or not path.exists():
            frames, fps = self.video.generate(
                str(keyframe), self.motion_prompt(shot), self.shot_seed(shot), shot.seconds
            )
            save_frames_as_video(frames, path, fps)
        return path

    def existing_clip(self, script: Script, shot: Shot) -> Path | None:
        keyframe = self.existing_keyframe(script, shot)
        if not keyframe:
            return None
        path = self.clip_path(keyframe, shot)
        return path if path.exists() else None

    # ------------------------------------------------------------ lip sync
    def on_screen_speaker(self, shot: Shot) -> Character | None:
        """The first character who both appears in the shot and speaks in it."""
        on_screen = shot_characters(self.series, shot)
        for line in shot.lines:
            if line.kind == "dialogue" and not line.is_narration:
                char = self.series.character(line.speaker)
                if char and any(char is c for c in on_screen):
                    return char
        return None

    def lipsync_character(self, shot: Shot) -> Character | None:
        if (
            self.settings.get("draft_mode")
            or not self.settings.get("lip_sync", True)
            or shot.lipsync == "off"
            or self.animate_mode(shot) == "ai"
        ):
            return None
        return self.on_screen_speaker(shot)

    def make_mouths(self, script: Script, shot: Shot, keyframe: Path) -> dict[str, Path] | None:
        """Closed / half / open mouth versions of the keyframe, or None if there's nothing to lip sync."""
        char = self.lipsync_character(shot)
        if char is None:
            return None
        style = (self.series.style_prompt, self.series.quality_prompt)
        key = content_hash(
            "mouths-v1", self.images.cache_key(), _file_tag(keyframe), shot.lipsync or "auto", char.appearance, style
        )
        folder = self.project.cache_dir("mouths")
        paths = {state: folder / f"{key}_{state}.png" for state in MOUTH_STATES}
        no_face = folder / f"{key}_noface"
        if all(p.exists() for p in paths.values()):
            return paths
        if no_face.exists():
            return None
        try:
            image = Image.open(keyframe).convert("RGB")
            face = pick_face(detect_faces(image), shot.lipsync or "auto")
            if face is None:
                no_face.touch()  # e.g. seen from behind; don't look again
                return None
            spec = self.keyframe_spec(script, shot)
            ref = self.project.abs_path(char.reference_image)
            lora = self._lora_path(char)
            prompt = ", ".join(p for p in (char.lora_trigger if lora else "", char.appearance, "face", *style) if p)
            variants = mouth_variants(
                self.images,
                image,
                face,
                prompt,
                spec["negative"],
                spec["seed"],
                reference=str(ref) if ref else None,
                loras=[(str(lora), char.lora_weight)] if lora else [],
            )
        except Exception as exc:
            self._warn(f"Lip sync skipped for {shot.id} ({exc})")
            return None
        for state, img in variants.items():
            img.save(paths[state])
        return paths

    # ------------------------------------------------------------ audio
    def voice_for(self, line: Line) -> tuple[str, float]:
        if line.is_narration:
            return self.series.narrator_voice or "bf_emma", 1.0
        char = self.series.character(line.speaker)
        if char:
            return char.voice, char.voice_speed
        self._warn(f"'{line.speaker}' is not in the character list, using a default voice")
        return "af_sky", 1.0

    def make_line_audio(self, line: Line) -> Path:
        voice, speed = self.voice_for(line)
        key = content_hash(self.voices.cache_key(line.text, voice, speed, line.emotion), line.speaker)
        path = self.project.cache_dir("voices") / f"{key}.wav"
        if not path.exists():
            try:
                self.voices.synthesize(line.text, voice, path, speed=speed, emotion=line.emotion, speaker=line.speaker)
            except Exception as exc:
                self._warn(f"Voice '{voice}' failed ({exc}); used beep voice instead for: {line.text[:40]}")
                from .generators.voice import _beep_voice

                save_wav(path.with_suffix(".beep.wav"), _beep_voice(line.text, line.speaker, speed))
                return path.with_suffix(".beep.wav")
        return path

    def user_audio(self, text: str) -> Path | None:
        """Your own sound file named in the script, e.g. `SFX: door.wav` or `music: theme.mp3`."""
        name = text.strip().strip('"')
        for candidate in (Path(name), self.project.root / "audio" / name, APP_DIR / "audio" / name):
            if candidate.is_file():
                return candidate
        self._warn(f"Audio file not found: {name}. Put it in {self.project.root / 'audio'}")
        return None

    def make_sfx(self, text: str) -> Path:
        if text.strip().strip('"').lower().endswith(AUDIO_EXTENSIONS):
            found = self.user_audio(text)
            if found:
                return found
            path = self.project.cache_dir("sfx") / f"missing-{content_hash(text)}.wav"
            save_wav(path, synth_sfx(text))
            return path
        path = self.project.cache_dir("sfx") / f"{content_hash(self.sfx.cache_key(text))}.wav"
        if not path.exists():
            try:
                self.sfx.generate(text, path, seed=zlib.crc32(text.encode()))
            except Exception as exc:
                self._warn(f"Sound effect generator failed ({exc}); used a simple placeholder sound")
                save_wav(path, synth_sfx(text))
        return path

    def make_music(self, prompt: str, seconds: float) -> Path | None:
        if prompt.strip().strip('"').lower().endswith(AUDIO_EXTENSIONS):
            return self.user_audio(prompt)
        path = self.project.cache_dir("music") / f"{content_hash(self.music.cache_key(prompt, seconds))}.wav"
        if not path.exists():
            try:
                self.music.generate(prompt, seconds, path, seed=zlib.crc32(prompt.encode()))
            except Exception as exc:
                self._warn(f"Music generator failed ({exc}); used simple synth music instead")
                save_wav(path, synth_music(prompt, min(30, max(5, seconds))))
        return path

    # ------------------------------------------------------------ characters
    def character_sheet(self, char: Character, count: int = 4, start_seed: int | None = None) -> list[tuple[Path, int]]:
        prompt, negative = character_sheet_prompt(self.series, char)
        lora = self._lora_path(char)
        loras = [(str(lora), char.lora_weight)] if lora else []
        base = self.series.seed + zlib.crc32(char.name.encode()) if start_seed is None else start_seed
        results = []
        for i in range(count):
            seed = (base + i * 7919) % 2**31
            key = content_hash(
                "sheet", self.images.cache_key(), prompt, negative, seed, [(_file_tag(Path(p)), w) for p, w in loras]
            )
            path = self.project.cache_dir("characters") / f"{key}.png"
            if not path.exists():
                # portrait-ish canvas for full-body designs
                img = self.images.generate(prompt, negative, seed, width=832, height=1216, loras=loras, label=char.name)
                img.save(path)
            results.append((path, seed))
        return results

    # ------------------------------------------------------------ whole episode
    def build_episode(self, episode: str, progress=None) -> dict:
        """Generate everything that's missing, then render the episode video."""
        self.warnings = []

        def report(fraction, message):
            if not message.startswith("Rendering video"):
                log.info(message)
            if progress:
                progress(min(1.0, max(0.0, fraction)), message)

        script = self.parse(episode)
        for warning in script.warnings:
            self._warn(warning)
        shots = script.shots
        if not shots:
            raise ValueError("The script has no shots yet.")

        # 1. keyframes (+ mouth shapes for lip sync)
        mouths = {}
        for i, shot in enumerate(shots):
            report(0.02 + 0.40 * i / len(shots), f"Drawing keyframe {i + 1}/{len(shots)} ({shot.id})")
            keyframe = self.make_keyframe(script, shot)
            if self.lipsync_character(shot):
                report(0.02 + 0.40 * (i + 0.5) / len(shots), f"Drawing mouth shapes for lip sync ({shot.id})")
                mouths[shot.id] = self.make_mouths(script, shot, keyframe)

        # 2. AI animation (only shots set to "ai")
        ai_shots = [s for s in shots if self.animate_mode(s) == "ai"]
        clips = {}
        for i, shot in enumerate(ai_shots):
            report(
                0.42 + 0.25 * i / max(1, len(ai_shots)), f"Animating shot {i + 1}/{len(ai_shots)} with AI ({shot.id})"
            )
            try:
                clips[shot.id] = self.make_clip(script, shot)
            except Exception as exc:
                self._warn(f"AI animation failed for {shot.id} ({exc}); used a camera move instead")

        # 3. voices and sound effects, then shot timing
        MODELS.unload()  # free VRAM from the picture/video models
        all_lines = [line for shot in shots for line in shot.lines]
        done = 0
        plans: dict[str, ShotPlan] = {}
        for shot in shots:
            events = []
            lip_char = self.lipsync_character(shot) if mouths.get(shot.id) else None
            for line in shot.lines:
                done += 1
                report(
                    0.67 + 0.13 * done / max(1, len(all_lines)), f"Recording voices and sounds {done}/{len(all_lines)}"
                )
                if line.kind == "dialogue":
                    path = self.make_line_audio(line)
                    timed = TimedLine(0, 0, line.text, line.speaker, line.is_narration, path)
                    timed.lipsync = lip_char is not None and self.series.character(line.speaker) is lip_char
                    events.append(("line", timed, audio_duration(path)))
                else:
                    path = self.make_sfx(line.text)
                    events.append(("sfx", path, audio_duration(path)))
            mode = self.animate_mode(shot)
            plan = ShotPlan(
                shot_id=shot.id,
                keyframe=self.make_keyframe(script, shot),
                camera="still" if mode == "still" else auto_camera(shot),
                clip=clips.get(shot.id),
                seed=self.shot_seed(shot),
                mouths=mouths.get(shot.id),
            )
            layout_shot(plan, events, shot.seconds)
            plans[shot.id] = plan

        # 4. music per scene
        episode_plan = EpisodePlan(series_title=self.series.title, episode_title=script.title)
        for i, scene in enumerate(script.scenes):
            report(0.80 + 0.08 * i / len(script.scenes), f"Composing music for scene {i + 1}/{len(script.scenes)}")
            scene_plans = [plans[s.id] for s in scene.shots]
            seconds = sum(p.duration for p in scene_plans)
            music = None
            if scene.music and scene.music.lower() not in ("none", "no", "off", "silence"):
                music = self.make_music(scene.music, seconds)
            episode_plan.scenes.append(ScenePlan(music=music, shots=scene_plans))
        if (self.settings.get("title_card") or self.settings.get("end_card")) and self.series.theme_music:
            report(0.88, "Composing the theme music")
            episode_plan.theme_music = self.make_music(self.series.theme_music, 12)
        MODELS.unload()

        # 5. edit it all together
        report(0.90, "Editing the episode")
        result = render_episode(
            episode_plan,
            self.settings,
            self.project.output_dir(episode),
            progress=lambda f, m: report(0.90 + 0.10 * f, m),
        )
        result["warnings"] = list(self.warnings)
        report(1.0, "Done!")
        return result
