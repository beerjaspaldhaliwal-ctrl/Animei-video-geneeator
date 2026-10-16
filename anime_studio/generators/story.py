"""Optional story assistant using a free local LLM through Ollama (https://ollama.com)."""

import re

import requests

from ..project import Series

FORMAT_RULES = """Write the script in EXACTLY this format, nothing else:

TITLE: <episode title>

SCENE: <place and time, visual description> | location: <location name> | music: <music style for this scene>
SHOT: <what we see: shot type, who is there, what they do, expression> | camera: <static|zoom_in|zoom_out|pan_left|pan_right|pan_up|pan_down|shake>
NAME: <spoken line>
NAME (emotion): <spoken line>
NARRATOR: <narration>
SFX: <sound effect>

Rules:
- Use 3 to 6 SCENEs. Each SCENE has 2 to 6 SHOTs. Each SHOT has 0 to 3 lines.
- SHOT descriptions are visual and concrete (they become anime pictures). Name the characters who appear.
- Only use character names from the cast list. Write names in CAPITALS before dialogue.
- Keep spoken lines short and natural, like anime dialogue.
- Do not add explanations, markdown, or anything outside the format."""


def _cast_text(series: Series) -> str:
    lines = []
    for c in series.characters:
        lines.append(f"- {c.name}: looks like ({c.appearance}); personality: {c.personality or 'n/a'}")
    return "\n".join(lines) or "- (no characters yet, invent 2 main characters)"


def _locations_text(series: Series) -> str:
    return "\n".join(f"- {loc.name}: {loc.description}" for loc in series.locations) or "- (invent locations)"


def list_models(url: str) -> list[str]:
    resp = requests.get(url.rstrip("/") + "/api/tags", timeout=5)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def _generate(settings: dict, prompt: str) -> str:
    url = settings["ollama_url"].rstrip("/") + "/api/generate"
    try:
        resp = requests.post(
            url,
            json={
                "model": settings["ollama_model"],
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.8, "num_ctx": 8192},
            },
            timeout=900,
        )
    except requests.ConnectionError:
        raise RuntimeError(
            f"Can't reach Ollama. Install it from https://ollama.com, then run:  ollama pull {settings['ollama_model']}"
        ) from None
    if resp.status_code == 404:
        raise RuntimeError(f"Ollama model not found. Run:  ollama pull {settings['ollama_model']}")
    resp.raise_for_status()
    text = resp.json().get("response", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # reasoning models
    text = re.sub(r"^```[a-z]*\s*|```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip() + "\n"


def write_episode(settings: dict, series: Series, synopsis: str, previous: str = "") -> str:
    prompt = f"""You are a professional anime screenwriter.

SERIES: {series.title}
PREMISE: {series.premise}

CAST:
{_cast_text(series)}

LOCATIONS:
{_locations_text(series)}

{"PREVIOUSLY: " + previous if previous else ""}
THIS EPISODE: {synopsis}

{FORMAT_RULES}
"""
    return _generate(settings, prompt)


def brainstorm(settings: dict, series: Series, count: int = 5) -> str:
    prompt = f"""You are an anime series writer. Suggest {count} exciting episode ideas for this series.
For each idea give a short title and a 2-3 sentence synopsis. Keep the cast consistent.

SERIES: {series.title}
PREMISE: {series.premise}
CAST:
{_cast_text(series)}
"""
    return _generate(settings, prompt)
