import numpy as np
from PIL import Image

from anime_studio import lipsync
from anime_studio.config import DEFAULTS
from anime_studio.editor import EpisodePlan, ScenePlan, ShotPlan, TimedLine, render_episode
from anime_studio.generators.image import _draft_inpaint, placeholder_image
from anime_studio.media import SAMPLE_RATE, read_video_frames, save_wav
from anime_studio.script_parser import parse_script


def _voice(path, silence=0.5, talk=1.0):
    t = np.arange(int(talk * SAMPLE_RATE)) / SAMPLE_RATE
    audio = np.concatenate([np.zeros(int(silence * SAMPLE_RATE)), 0.8 * np.sin(2 * np.pi * 220 * t)])
    save_wav(path, audio.astype(np.float32))


def test_lipsync_option_parses():
    script = parse_script("SHOT: a | lipsync: off\nSHOT: b | lips: left\nSHOT: c | lipsync: maybe\n")
    assert [s.lipsync for s in script.shots] == ["off", "left", ""]
    assert any("lipsync must be one of" in w for w in script.warnings)


def test_mouth_track_follows_voice(tmp_path):
    _voice(tmp_path / "v.wav")
    line = TimedLine(1.0, 2.5, "hi", "AIKO", audio=tmp_path / "v.wav", lipsync=True)
    track = lipsync.mouth_track([line], n_frames=96, fps=24)
    assert (track[:24] == 0).all()  # before the line starts: closed
    assert (track[24:36] == 0).all()  # first 0.5 s of the line is silence
    assert (track[40:58] == 2).all()  # loud voice: open
    assert (track[70:] == 0).all()  # after the line: closed
    # mouth changes only on even frames ("on twos")
    changes = np.nonzero(np.diff(track))[0] + 1
    assert all((c - 24) % 2 == 0 for c in changes)
    # lines that are not lip-synced (e.g. off-screen speaker) don't move the mouth
    line.lipsync = False
    assert not lipsync.mouth_track([line], 96, 24).any()


class _FakeEngine:
    def inpaint(self, image, mask, prompt, negative, seed, strength=0.75, **kwargs):
        return _draft_inpaint(image, mask, prompt)


def test_mouth_variants_only_change_the_mouth():
    image = placeholder_image("a face", 832, 832, 1)
    face = (300.0, 200.0, 500.0, 420.0)
    variants = lipsync.mouth_variants(_FakeEngine(), image, face, "1girl", "bad", 5)
    assert set(variants) == {"closed", "half", "open"}
    closed, opened = (np.asarray(variants[s]).astype(int) for s in ("closed", "open"))
    diff = np.abs(closed - opened).sum(axis=2)
    ys, xs = np.nonzero(diff > 30)
    assert len(ys) > 50
    ex0, ey0, ex1, ey1 = lipsync.mouth_ellipse(face)
    assert xs.min() >= ex0 - 5 and xs.max() <= ex1 + 5 and ys.min() >= ey0 - 5 and ys.max() <= ey1 + 5


def test_editor_switches_mouth_pictures(tmp_path):
    _voice(tmp_path / "v.wav", silence=0.5, talk=1.0)
    mouths = {}
    for state, value in (("closed", 0), ("half", 128), ("open", 255)):
        mouths[state] = tmp_path / f"{state}.png"
        Image.new("RGB", (320, 180), (value, value, value)).save(mouths[state])
    # line starts 0.5 s into the shot, and its audio has 0.5 s of silence first: talking from 1.0 s to 2.0 s
    line = TimedLine(0.5, 2.0, "hi", "AIKO", audio=tmp_path / "v.wav", lipsync=True)
    shot = ShotPlan("sc01_sh01", None, "still", duration=3.0, lines=[line], mouths=mouths)
    settings = dict(
        DEFAULTS, output_width=320, output_height=180, fps=12, title_card=False, end_card=False, burn_subtitles=False
    )
    result = render_episode(EpisodePlan("S", "E", [ScenePlan(None, [shot])]), settings, tmp_path / "out")
    frames, _ = read_video_frames(result["video"])
    brightness = [f.mean() for f in frames]
    assert brightness[8] < 40  # 0.67 s, silent: closed mouth (black)
    assert brightness[15] > 200  # 1.25 s, talking: open mouth (white)
    assert brightness[30] < 40  # 2.5 s, finished talking: closed again


def test_pipeline_wires_lipsync(tmp_path, monkeypatch):
    from anime_studio import pipeline
    from anime_studio.pipeline import Studio
    from anime_studio.project import Project

    project = Project.create(str(tmp_path), "Lips", with_example=True)
    for char in project.series.characters:
        char.voice = "beep"
    project.series.theme_music = ""
    project.save()
    project.write_script(
        "ep01",
        "SCENE: rooftop | music: none\n"
        "SHOT: close-up, Aiko smiling\nAIKO: Hello there!\n"
        "SHOT: close-up, Aiko listening\nREN: I'm not on screen.\n"
        "SHOT: Aiko waving | lipsync: off\nAIKO: Bye!\n",
    )
    settings = dict(DEFAULTS, projects_dir=str(tmp_path), output_width=320, output_height=180, fps=12)
    settings.update(title_card=False, end_card=False)
    studio = Studio(project, settings)

    class Images(_FakeEngine):
        def cache_key(self):
            return ["fake"]

        def generate(self, prompt, negative, seed, width=None, height=None, label="", **kwargs):
            return placeholder_image(prompt, 832, 480, seed, label)

    studio.images = Images()
    monkeypatch.setattr(pipeline, "detect_faces", lambda image: [(300.0, 100.0, 450.0, 280.0)])
    script = studio.parse("ep01")
    talker, listener, off = script.shots
    assert studio.lipsync_character(talker).name == "Aiko"
    assert studio.lipsync_character(listener) is None  # Ren speaks but isn't on screen
    assert studio.lipsync_character(off) is None
    result = studio.build_episode("ep01")
    assert result["warnings"] == []
    assert len(list(project.cache_dir("mouths").glob("*_open.png"))) == 1
