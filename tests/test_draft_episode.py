"""End-to-end test in draft mode (no AI models needed)."""

import subprocess

from anime_studio.config import DEFAULTS
from anime_studio.media import ffmpeg_exe
from anime_studio.pipeline import Studio
from anime_studio.project import Project
from anime_studio.prompts import shot_characters, shot_prompt


def _settings(tmp_path):
    settings = dict(DEFAULTS)
    settings.update(draft_mode=True, projects_dir=str(tmp_path), output_width=640, output_height=360, fps=12)
    return settings


def test_prompt_uses_character_and_location(tmp_path):
    project = Project.create(str(tmp_path), "Test Show")
    studio = Studio(project, _settings(tmp_path))
    script = studio.parse("ep01")
    first = script.shots[0]
    assert [c.name for c in shot_characters(project.series, first)] == ["Aiko"]
    prompt, negative = shot_prompt(project.series, script.scenes[0], first)
    assert "long silver hair" in prompt and "chain-link fence" in prompt and "masterpiece" in prompt
    assert "worst quality" in negative
    # two characters on screen: no conflicting "solo" tags
    both = script.shots[3]
    prompt, _ = shot_prompt(project.series, script.scenes[1], both)
    assert "solo" not in prompt and "1girl" in prompt and "1boy" in prompt


def test_build_draft_episode(tmp_path):
    project = Project.create(str(tmp_path), "Test Show")
    studio = Studio(project, _settings(tmp_path))
    result = studio.build_episode("ep01")
    assert result["video"].exists()
    assert result["warnings"] == []
    srt = result["subtitles"].read_text("utf-8")
    assert "I always wondered what's beyond those mountains." in srt
    probe = subprocess.run([ffmpeg_exe(), "-i", str(result["video"])], capture_output=True, text=True).stderr
    assert "Video: h264" in probe and "640x360" in probe and "Audio: aac" in probe

    # Second build reuses the cache: no new keyframes are drawn.
    keyframes = sorted(project.cache_dir("keyframes").iterdir())
    studio.build_episode("ep01")
    assert sorted(project.cache_dir("keyframes").iterdir()) == keyframes


def test_own_audio_files(tmp_path):
    import numpy as np

    from anime_studio.media import save_wav

    project = Project.create(str(tmp_path), "Audio Show", with_example=False)
    save_wav(project.root / "audio" / "song.wav", np.zeros(44100, dtype=np.float32))
    save_wav(project.root / "audio" / "boom.wav", np.zeros(4410, dtype=np.float32))
    project.write_script(
        "ep01", "SCENE: field | music: song.wav\nSHOT: a big explosion\nSFX: boom.wav\nSFX: missing.mp3\n"
    )
    studio = Studio(project, _settings(tmp_path))
    assert studio.make_music("song.wav", 10) == project.root / "audio" / "song.wav"
    assert studio.make_sfx("boom.wav") == project.root / "audio" / "boom.wav"
    result = studio.build_episode("ep01")
    assert result["video"].exists()
    assert len(result["warnings"]) == 1 and "missing.mp3" in result["warnings"][0]


def test_editor_uses_ai_clip(tmp_path):
    from PIL import Image

    from anime_studio.editor import EpisodePlan, ScenePlan, ShotPlan, render_episode
    from anime_studio.media import read_video_frames, save_frames_as_video

    clip = tmp_path / "clip.mp4"
    save_frames_as_video([Image.new("RGB", (64, 48), (255, 0, 0))] * 24, clip, 24)  # 1 s of red
    plan = EpisodePlan(
        "Show", "Ep", [ScenePlan(None, [ShotPlan("sc01_sh01", None, "static", duration=2.0, clip=clip)])]
    )
    settings = dict(DEFAULTS, output_width=320, output_height=180, fps=12, title_card=False, end_card=False)
    result = render_episode(plan, settings, tmp_path / "out")
    frames, fps = read_video_frames(result["video"])
    assert len(frames) == 24 and frames[0].shape == (180, 320, 3)
    middle = frames[12].astype(int)
    assert middle[..., 0].mean() > 200 and middle[..., 1].mean() < 60  # the red clip is shown
