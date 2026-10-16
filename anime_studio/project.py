"""Series projects on disk.

projects/<series>/
    series.json            title, premise, art style, characters, locations
    characters/            character reference pictures
    audio/                 your own music and sound files (optional)
    episodes/ep01/script.txt
    episodes/ep01/output/  finished episode video + subtitles
    cache/                 generated keyframes, clips, voices, music (reused)
"""

import json
import re
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_STYLE = "anime screenshot, anime coloring, detailed background, cinematic lighting"
DEFAULT_QUALITY = "masterpiece, high score, great score, absurdres"
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, "
    "cropped, worst quality, low quality, low score, bad score, average score, signature, "
    "watermark, username, blurry, nsfw"
)

EXAMPLE_SCRIPT = """TITLE: The Wind Beyond the Mountains

# Lines starting with # are notes for you. They are ignored.
# SCENE starts a new place/time. SHOT is one picture on screen.
# NAME: text is dialogue. NARRATOR: text is narration. SFX: is a sound effect.

SCENE: school rooftop at sunset, orange sky, city in the distance | location: Rooftop | music: gentle hopeful piano, emotional anime soundtrack
SHOT: wide shot, Aiko standing at the railing, looking at the distant mountains, wind blowing her hair | camera: pan_right
NARRATOR: Every day after school, Aiko came to the rooftop to watch the mountains.
AIKO: I always wondered what's beyond those mountains.
SHOT: Ren opening the rooftop door, surprised expression | camera: zoom_in
SFX: metal door creaking open
REN (surprised): Aiko? What are you doing up here?
SHOT: close-up, Aiko turning around and smiling | camera: zoom_in
AIKO (happy): Ren! Come look. The sky is on fire today.

SCENE: quiet street at night, glowing street lamps, cherry blossom petals | location: Street | music: mysterious ambient synth, soft
SHOT: Aiko and Ren walking side by side under street lamps | camera: pan_left
REN: Do you really want to leave this town?
AIKO (determined): Someday. I want to see the whole world.
SHOT: a strange blue light glowing in the sky above the mountains | camera: zoom_out
SFX: low magical humming
NARRATOR: Neither of them knew that tomorrow would change everything.
"""


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "untitled"


@dataclass
class Character:
    name: str
    appearance: str = ""  # anime tags, e.g. "1girl, long silver hair, blue eyes, school uniform"
    personality: str = ""
    aliases: str = ""  # other names used in the script, comma separated
    voice: str = "af_heart"
    voice_speed: float = 1.0
    reference_image: str = ""  # path relative to the project folder
    lora_path: str = ""
    lora_weight: float = 0.8
    lora_trigger: str = ""

    def names(self) -> list[str]:
        return [self.name] + [a.strip() for a in self.aliases.split(",") if a.strip()]


@dataclass
class Location:
    name: str
    description: str = ""


@dataclass
class Series:
    title: str
    premise: str = ""
    style_prompt: str = DEFAULT_STYLE
    quality_prompt: str = DEFAULT_QUALITY
    negative_prompt: str = DEFAULT_NEGATIVE
    seed: int = 1234
    theme_music: str = "upbeat energetic anime opening theme, j-rock, electric guitar, drums"
    narrator_voice: str = "bf_emma"
    characters: list[Character] = field(default_factory=list)
    locations: list[Location] = field(default_factory=list)

    def character(self, name: str) -> Character | None:
        key = name.strip().lower()
        for char in self.characters:
            if key in (n.lower() for n in char.names()):
                return char
        return None

    def location(self, name: str) -> Location | None:
        key = name.strip().lower()
        return next((loc for loc in self.locations if loc.name.lower() == key), None)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Series":
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        char_fields = {f.name for f in fields(Character)}
        values["characters"] = [
            Character(**{k: v for k, v in c.items() if k in char_fields}) for c in data.get("characters", [])
        ]
        values["locations"] = [Location(**c) for c in data.get("locations", [])]
        return cls(**values)


class Project:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.series = Series.from_dict(json.loads((self.root / "series.json").read_text("utf-8")))

    # ---- creation / listing ----
    @staticmethod
    def list_all(projects_dir: str) -> list[tuple[str, str]]:
        """Return (title, folder name) pairs."""
        base = Path(projects_dir)
        result = []
        if base.exists():
            for path in sorted(base.iterdir()):
                meta = path / "series.json"
                if meta.exists():
                    try:
                        title = json.loads(meta.read_text("utf-8")).get("title", path.name)
                    except ValueError:
                        title = path.name
                    result.append((title, path.name))
        return result

    @staticmethod
    def create(projects_dir: str, title: str, with_example: bool = True) -> "Project":
        base = Path(projects_dir)
        slug = slugify(title)
        root = base / slug
        n = 2
        while root.exists():
            root = base / f"{slug}-{n}"
            n += 1
        root.mkdir(parents=True)
        (root / "audio").mkdir()  # your own music / sound effect files
        series = Series(title=title.strip() or "My Anime")
        if with_example:
            series.premise = "Two high school friends discover a mysterious light beyond the mountains."
            series.characters = [
                Character(
                    name="Aiko",
                    appearance="1girl, solo, long silver hair, blue eyes, school uniform, red ribbon",
                    personality="curious, brave, dreamy",
                    voice="af_heart",
                ),
                Character(
                    name="Ren",
                    appearance="1boy, solo, short black hair, brown eyes, school uniform, open blazer",
                    personality="calm, kind, a little shy",
                    voice="am_michael",
                ),
            ]
            series.locations = [
                Location("Rooftop", "school rooftop, chain-link fence, water tank, sunset sky"),
                Location("Street", "japanese residential street at night, street lamps, cherry blossom trees"),
            ]
        (root / "series.json").write_text(json.dumps(series.to_dict(), indent=2), "utf-8")
        project = Project(root)
        project.new_episode(EXAMPLE_SCRIPT if with_example else "TITLE: Episode 1\n\nSCENE: \nSHOT: \n")
        return project

    def save(self) -> None:
        (self.root / "series.json").write_text(json.dumps(self.series.to_dict(), indent=2, ensure_ascii=False), "utf-8")

    def delete(self) -> None:
        shutil.rmtree(self.root)

    # ---- episodes ----
    @property
    def episodes_dir(self) -> Path:
        return self.root / "episodes"

    def episodes(self) -> list[str]:
        if not self.episodes_dir.exists():
            return []
        return sorted(p.name for p in self.episodes_dir.iterdir() if (p / "script.txt").exists())

    def new_episode(self, script: str = "") -> str:
        existing = self.episodes()
        number = 1
        while f"ep{number:02d}" in existing:
            number += 1
        episode = f"ep{number:02d}"
        (self.episodes_dir / episode).mkdir(parents=True, exist_ok=True)
        self.write_script(episode, script or f"TITLE: Episode {number}\n\nSCENE: \nSHOT: \n")
        return episode

    def script_path(self, episode: str) -> Path:
        return self.episodes_dir / episode / "script.txt"

    def read_script(self, episode: str) -> str:
        path = self.script_path(episode)
        return path.read_text("utf-8") if path.exists() else ""

    def write_script(self, episode: str, text: str) -> None:
        path = self.script_path(episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, "utf-8")

    def output_dir(self, episode: str) -> Path:
        path = self.episodes_dir / episode / "output"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---- files ----
    def cache_dir(self, kind: str) -> Path:
        path = self.root / "cache" / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def abs_path(self, relative: str) -> Path | None:
        if not relative:
            return None
        path = Path(relative)
        path = path if path.is_absolute() else self.root / path
        return path if path.exists() else None

    def store_character_image(self, name: str, source_path: str) -> str:
        folder = self.root / "characters"
        folder.mkdir(exist_ok=True)
        target = folder / f"{slugify(name)}.png"
        from PIL import Image

        Image.open(source_path).convert("RGB").save(target)
        return str(target.relative_to(self.root))
