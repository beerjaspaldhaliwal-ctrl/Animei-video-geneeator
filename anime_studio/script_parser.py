"""Parser for the simple anime screenplay format.

    TITLE: The Girl Who Chased the Wind

    SCENE: School rooftop at sunset | location: Rooftop | music: hopeful piano
    SHOT: wide shot, Aiko stands at the railing, wind in her hair | camera: pan_right
    AIKO: I always wondered what's beyond those mountains.
    SHOT: close-up, Ren opens the rooftop door, surprised | seconds: 3
    REN (surprised): Aiko? What are you doing up here?
    SFX: metal door creaking
    NARRATOR: Neither of them knew tomorrow would change everything.

A plain line without a colon becomes a new shot. Lines starting with # are
comments. Everything after "|" on SCENE/SHOT lines are options.
"""

import re
from dataclasses import dataclass, field

CAMERA_MOVES = ["static", "zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down", "shake"]
ANIMATE_MODES = ["motion", "ai", "still"]
LIPSYNC_MODES = {
    "auto": "auto",
    "on": "auto",
    "yes": "auto",
    "off": "off",
    "no": "off",
    "none": "off",
    "left": "left",
    "right": "right",
}
NARRATOR_NAMES = {"NARRATOR", "NARRATION", "VOICEOVER", "V.O."}

_KEYWORD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _.'\-]{0,40}?)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")

_OPTION_ALIASES = {
    "seconds": "seconds",
    "secs": "seconds",
    "sec": "seconds",
    "duration": "seconds",
    "time": "seconds",
    "camera": "camera",
    "cam": "camera",
    "seed": "seed",
    "animate": "animate",
    "animation": "animate",
    "anim": "animate",
    "motion": "motion",
    "characters": "characters",
    "chars": "characters",
    "cast": "characters",
    "location": "location",
    "loc": "location",
    "place": "location",
    "music": "music",
    "bgm": "music",
    "lipsync": "lipsync",
    "lip_sync": "lipsync",
    "lips": "lipsync",
}


@dataclass
class Line:
    kind: str  # "dialogue" | "sfx"
    text: str
    speaker: str = ""
    emotion: str = ""
    line_no: int = 0

    @property
    def is_narration(self) -> bool:
        return self.speaker.upper() in NARRATOR_NAMES


@dataclass
class Shot:
    id: str
    description: str
    scene_index: int
    camera: str = ""
    seconds: float | None = None
    seed: int | None = None
    animate: str = ""
    motion: str = ""
    characters: list[str] | None = None
    lipsync: str = ""  # "" (auto) | auto | off | left | right
    lines: list[Line] = field(default_factory=list)
    line_no: int = 0  # 1-based line in the script, 0 when created implicitly
    from_action: bool = False  # written as a plain line without "SHOT:"
    scene_line_no: int = 0


@dataclass
class Scene:
    index: int
    description: str
    location: str = ""
    music: str = ""
    shots: list[Shot] = field(default_factory=list)
    line_no: int = 0


@dataclass
class Script:
    title: str = "Untitled Episode"
    scenes: list[Scene] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def shots(self) -> list[Shot]:
        return [shot for scene in self.scenes for shot in scene.shots]

    def find_shot(self, shot_id: str) -> Shot | None:
        return next((s for s in self.shots if s.id == shot_id), None)

    def speakers(self) -> list[str]:
        names = []
        for shot in self.shots:
            for line in shot.lines:
                if line.kind == "dialogue" and not line.is_narration and line.speaker not in names:
                    names.append(line.speaker)
        return names


def _split_options(text: str, line_no: int, warnings: list[str]) -> tuple[str, dict]:
    parts = text.split("|")
    options = {}
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition(":")
        if not sep:
            key, sep, value = part.partition("=")
        key = key.strip().lower()
        if not sep or key not in _OPTION_ALIASES:
            warnings.append(f"Line {line_no}: unknown option '{part}' (ignored)")
            continue
        options[_OPTION_ALIASES[key]] = value.strip()
    return parts[0].strip(), options


def _apply_shot_options(shot: Shot, options: dict, line_no: int, warnings: list[str]) -> None:
    if "camera" in options:
        cam = options["camera"].lower().replace(" ", "_").replace("-", "_")
        if cam in CAMERA_MOVES:
            shot.camera = cam
        else:
            warnings.append(
                f"Line {line_no}: unknown camera '{options['camera']}'. Use one of: {', '.join(CAMERA_MOVES)}"
            )
    if "seconds" in options:
        try:
            shot.seconds = max(0.5, float(options["seconds"].rstrip("s ")))
        except ValueError:
            warnings.append(f"Line {line_no}: seconds must be a number")
    if "seed" in options:
        try:
            shot.seed = int(options["seed"])
        except ValueError:
            warnings.append(f"Line {line_no}: seed must be a whole number")
    if "animate" in options:
        mode = options["animate"].lower()
        if mode in ANIMATE_MODES:
            shot.animate = mode
        else:
            warnings.append(f"Line {line_no}: animate must be one of: {', '.join(ANIMATE_MODES)}")
    if "motion" in options:
        shot.motion = options["motion"]
    if "lipsync" in options:
        mode = LIPSYNC_MODES.get(options["lipsync"].strip().lower())
        if mode:
            shot.lipsync = mode
        else:
            warnings.append(f"Line {line_no}: lipsync must be one of: auto, off, left, right")
    if "characters" in options:
        value = options["characters"].strip()
        shot.characters = (
            [] if value.lower() in ("none", "nobody", "") else [c.strip() for c in value.split(",") if c.strip()]
        )
    for key in ("location", "music"):
        if key in options:
            warnings.append(f"Line {line_no}: '{key}' belongs on the SCENE line (ignored on SHOT)")


def parse_script(text: str) -> Script:
    script = Script()
    warnings = script.warnings
    scene: Scene | None = None
    shot: Shot | None = None

    def new_scene(description: str, line_no: int) -> Scene:
        nonlocal shot
        sc = Scene(index=len(script.scenes) + 1, description=description, line_no=line_no)
        script.scenes.append(sc)
        shot = None
        return sc

    def new_shot(description: str, line_no: int, from_action: bool = False) -> Shot:
        nonlocal scene
        if scene is None:
            scene = new_scene(description, 0)
        sh = Shot(
            id=f"sc{scene.index:02d}_sh{len(scene.shots) + 1:02d}",
            description=description,
            scene_index=scene.index,
            line_no=line_no,
            from_action=from_action,
            scene_line_no=scene.line_no,
        )
        scene.shots.append(sh)
        return sh

    def current_shot() -> Shot:
        nonlocal shot
        if shot is None:
            # Dialogue before any SHOT: show the scene itself.
            shot = new_shot(scene.description if scene else "", 0)
        return shot

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        match = _KEYWORD_RE.match(line)
        if not match:
            shot = new_shot(_split_options(line, line_no, warnings)[0], line_no, from_action=True)
            continue
        name, emotion, rest = match.group(1).strip(), (match.group(2) or "").strip(), match.group(3).strip()
        keyword = name.upper()

        if keyword == "TITLE":
            script.title = rest or script.title
        elif keyword == "NOTE":
            continue
        elif keyword == "SCENE":
            description, options = _split_options(rest, line_no, warnings)
            scene = new_scene(description, line_no)
            scene.location = options.get("location", "")
            scene.music = options.get("music", "")
            for key in options:
                if key not in ("location", "music"):
                    warnings.append(f"Line {line_no}: '{key}' belongs on a SHOT line (ignored on SCENE)")
        elif keyword == "SHOT":
            description, options = _split_options(rest, line_no, warnings)
            shot = new_shot(description, line_no)
            _apply_shot_options(shot, options, line_no, warnings)
        elif keyword == "MUSIC":
            if scene is None:
                scene = new_scene("", 0)
            scene.music = rest
        elif keyword == "SFX":
            if rest:
                current_shot().lines.append(Line(kind="sfx", text=rest, line_no=line_no))
        else:
            if not rest:
                warnings.append(f"Line {line_no}: '{name}' has no dialogue after the colon")
                continue
            current_shot().lines.append(
                Line(kind="dialogue", text=rest, speaker=name, emotion=emotion, line_no=line_no)
            )

    for sc in script.scenes:
        if not sc.shots:
            sc.shots.append(
                Shot(
                    id=f"sc{sc.index:02d}_sh01",
                    description=sc.description,
                    scene_index=sc.index,
                    scene_line_no=sc.line_no,
                )
            )
    if not script.scenes:
        warnings.append("The script is empty. Add a SCENE: line and some SHOT: lines.")
    return script


def set_shot_option(text: str, shot: Shot, key: str, value: str) -> str:
    """Return the script text with `key: value` set on the given shot's line.

    Shots that were created implicitly get a real SHOT line inserted.
    """
    lines = text.splitlines()
    option = f"{key}: {value}"
    if shot.line_no and shot.line_no <= len(lines):
        idx = shot.line_no - 1
        current = lines[idx]
        indent = current[: len(current) - len(current.lstrip())]
        body = current.strip()
        if shot.from_action:
            body = "SHOT: " + body
        parts = [p.strip() for p in body.split("|")]
        parts = [p for p in parts[:1]] + [
            p for p in parts[1:] if _OPTION_ALIASES.get(p.partition(":")[0].strip().lower()) != key
        ]
        lines[idx] = indent + " | ".join(parts + [option])
    else:
        # Implicit shot: insert an explicit SHOT line before its first dialogue,
        # or right after its SCENE line.
        if shot.lines:
            at = shot.lines[0].line_no - 1
        elif shot.scene_line_no:
            at = shot.scene_line_no
        else:
            at = len(lines)
        lines.insert(at, f"SHOT: {shot.description} | {option}")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
