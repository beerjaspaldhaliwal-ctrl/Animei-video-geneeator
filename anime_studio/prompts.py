"""Turn a script shot plus the series bible into an image prompt."""

import re

from .project import Character, Series
from .script_parser import Scene, Shot


def _mentions(text: str, name: str) -> bool:
    return re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", text, re.IGNORECASE) is not None


def shot_characters(series: Series, shot: Shot) -> list[Character]:
    """Characters that appear on screen in a shot.

    Explicit `| characters: A, B` wins. Otherwise characters named in the shot
    description, otherwise the first speaker of the shot.
    """
    if shot.characters is not None:
        return [c for c in (series.character(n) for n in shot.characters) if c]
    found = [c for c in series.characters if any(_mentions(shot.description, n) for n in c.names())]
    if found:
        return found
    for line in shot.lines:
        if line.kind == "dialogue" and not line.is_narration:
            char = series.character(line.speaker)
            if char:
                return [char]
    return []


def location_text(series: Series, scene: Scene) -> str:
    if scene.location:
        loc = series.location(scene.location)
        if loc and loc.description:
            parts = [loc.description]
            if scene.description:
                parts.append(scene.description)
            return ", ".join(parts)
        return ", ".join(p for p in (scene.location, scene.description) if p)
    return scene.description


def _join(parts) -> str:
    cleaned = [p.strip().strip(",").strip() for p in parts if p and p.strip().strip(",").strip()]
    return ", ".join(cleaned)


def shot_prompt(series: Series, scene: Scene, shot: Shot) -> tuple[str, str]:
    chars = shot_characters(series, shot)
    if len(chars) > 1:
        # Multi-character shots: drop per-character "solo" tags that fight each other.
        char_parts = [re.sub(r"(^|,)\s*solo\s*(?=,|$)", r"\1", c.appearance) for c in chars]
    else:
        char_parts = [c.appearance for c in chars]
    triggers = [c.lora_trigger for c in chars if c.lora_path and c.lora_trigger]
    prompt = _join(
        triggers
        + char_parts
        + [shot.description, location_text(series, scene), series.style_prompt, series.quality_prompt]
    )
    return prompt, series.negative_prompt


def character_sheet_prompt(series: Series, char: Character) -> tuple[str, str]:
    prompt = _join(
        [
            char.lora_trigger if char.lora_path else "",
            char.appearance,
            "full body, standing, looking at viewer, simple white background, character design",
            series.style_prompt.replace("detailed background", ""),
            series.quality_prompt,
        ]
    )
    return prompt, series.negative_prompt
