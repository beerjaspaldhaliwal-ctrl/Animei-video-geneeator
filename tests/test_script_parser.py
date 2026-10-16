from anime_studio.project import EXAMPLE_SCRIPT
from anime_studio.script_parser import parse_script, set_shot_option


def test_example_script_parses():
    script = parse_script(EXAMPLE_SCRIPT)
    assert script.title == "The Wind Beyond the Mountains"
    assert len(script.scenes) == 2
    assert [s.id for s in script.shots] == ["sc01_sh01", "sc01_sh02", "sc01_sh03", "sc02_sh01", "sc02_sh02"]
    assert script.scenes[0].location == "Rooftop"
    assert script.scenes[0].music.startswith("gentle hopeful piano")
    assert script.warnings == []


def test_dialogue_emotion_sfx_and_options():
    script = parse_script(
        "SCENE: park | music: none\n"
        "SHOT: Aiko runs | camera: pan_left | seconds: 5 | seed: 42 | animate: ai | motion: hair flowing\n"
        "AIKO (excited): Look!\n"
        "SFX: birds chirping\n"
        "NARRATOR: It was spring.\n"
    )
    shot = script.shots[0]
    assert (shot.camera, shot.seconds, shot.seed, shot.animate, shot.motion) == (
        "pan_left",
        5.0,
        42,
        "ai",
        "hair flowing",
    )
    kinds = [(line.kind, line.speaker, line.emotion) for line in shot.lines]
    assert kinds == [("dialogue", "AIKO", "excited"), ("sfx", "", ""), ("dialogue", "NARRATOR", "")]
    assert shot.lines[2].is_narration
    assert script.speakers() == ["AIKO"]


def test_implicit_shots_and_warnings():
    script = parse_script(
        "# a comment\n"
        "AIKO: Hello before any scene\n"
        "SCENE: classroom\n"
        "REN: Dialogue before any shot\n"
        "Ren looks out of the window\n"
        "SHOT: x | camera: spin | bogus: 1\n"
        "SCENE: empty scene\n"
    )
    assert len(script.scenes) == 3
    assert script.scenes[1].shots[0].description == "classroom"  # implicit from the scene
    assert script.scenes[1].shots[1].description == "Ren looks out of the window"
    assert script.scenes[2].shots[0].description == "empty scene"
    assert any("unknown camera" in w for w in script.warnings)
    assert any("unknown option" in w for w in script.warnings)


def test_set_shot_option_rewrites_and_inserts():
    text = "SCENE: park\nSHOT: Aiko runs | seed: 1\nAIKO: hi\nRen waves\n"
    script = parse_script(text)
    text2 = set_shot_option(text, script.shots[0], "seed", "99")
    assert "SHOT: Aiko runs | seed: 99" in text2
    assert "seed: 1\n" not in text2
    text3 = set_shot_option(text2, parse_script(text2).shots[1], "seed", "7")
    assert "SHOT: Ren waves | seed: 7" in text3
    assert parse_script(text3).shots[1].seed == 7

    implicit = "SCENE: park\nAIKO: hi\n"
    shot = parse_script(implicit).shots[0]
    fixed = set_shot_option(implicit, shot, "animate", "ai")
    assert fixed.splitlines()[1] == "SHOT: park | animate: ai"
    reparsed = parse_script(fixed)
    assert len(reparsed.shots) == 1 and reparsed.shots[0].animate == "ai"
    assert reparsed.shots[0].lines[0].text == "hi"
