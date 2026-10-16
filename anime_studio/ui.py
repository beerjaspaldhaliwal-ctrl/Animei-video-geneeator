"""The Anime Studio app window (runs in your web browser, fully local)."""

import json
import logging
import random
import traceback
from pathlib import Path

import gradio as gr

from .config import DEFAULTS, load_settings, save_settings
from .device import content_hash, gpu_info
from .generators import story
from .generators.video import ENGINES
from .generators.voice import ALL_VOICES, VoiceEngine
from .pipeline import Studio
from .project import Character, Location, Project
from .prompts import shot_characters
from .script_parser import parse_script, set_shot_option

log = logging.getLogger(__name__)

SETTINGS = load_settings()
NEW_CHAR = "➕ New character"
NEW_LOC = "➕ New location"

SCRIPT_HELP = """
### Script format

```
TITLE: Episode title

SCENE: where and when, what it looks like | location: Rooftop | music: calm piano
SHOT: what we see, who is there, what they do | camera: zoom_in | seconds: 4
AIKO: Spoken line.
REN (angry): A line with an emotion.
NARRATOR: Narration.
SFX: door slamming
```

* **SCENE** starts a new place or time. `location:` picks one of your saved locations. `music:` describes the background music (or `none`).
* **SHOT** is one picture on screen. A plain line without a colon is also a shot.
* Shot options: `camera:` static, zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down, shake
  · `seconds:` minimum length · `animate:` motion / ai / still · `motion:` what moves (for AI animation)
  · `characters:` who is on screen (default: names mentioned in the shot) · `seed:` picture variation
  · `lipsync:` auto / off / left / right (which face talks when two people are on screen)
* **NAME: text** is dialogue. Add an emotion in brackets: `(happy)`, `(angry)`, `(whisper)`, `(excited)`, `(sad)`.
* **SFX:** makes a sound effect. Lines starting with `#` are notes and are ignored.
* **Your own audio:** put files in the series' `audio` folder and write `music: my_song.mp3` or `SFX: door.wav`.
"""

HELP = """
## How to make an episode

1. **Series** – give your anime a title, story premise and art style.
2. **Characters** – describe each character with anime tags (hair, eyes, clothes), pick a voice,
   then click *Design character* and choose the picture you like as their **reference**. That keeps
   them looking the same in every shot.
3. **Locations** – save the places your story happens in.
4. **Script** – write the episode (or let the Story Assistant draft it). Click *Check script*.
5. **Storyboard** – draw all the keyframes. Don't like a picture? *Try 4 new versions* and pick one.
   Want real movement? *Animate with AI* (slow but cool).
6. **Make Episode** – voices, music, sound effects, subtitles and editing happen automatically.
   Tick *Draft mode* first to check timing in seconds, then make the real one.

Everything is saved in the `projects` folder. Generated pictures and sounds are cached, so
re-making an episode after small script changes only redoes what changed.

### Tips
* Use **Danbooru-style tags** for looks: `1girl, long silver hair, blue eyes, school uniform`.
* Keep one character per shot when you can: reference pictures work best that way.
* For a fixed character look you can also load a **LoRA** file (put it in the `loras` folder).
* **Japanese voices:** use `ja-JP-NanamiNeural` / `ja-JP-KeitaNeural` (free, needs internet), or
  Kokoro `jf_alpha` after installing `misaki[ja]`.
"""


# ------------------------------------------------------------------ helpers
def _projects():
    return Project.list_all(SETTINGS["projects_dir"])


def _project_choices():
    return [(title, folder) for title, folder in _projects()]


def _load(folder) -> Project | None:
    if not folder:
        return None
    path = Path(SETTINGS["projects_dir"]) / folder
    return Project(path) if (path / "series.json").exists() else None


def _need(folder) -> Project:
    project = _load(folder)
    if project is None:
        raise gr.Error("Create or pick a series first (top of the page).")
    return project


def _episode_choices(project: Project | None):
    if not project:
        return []
    choices = []
    for ep in project.episodes():
        title = parse_script(project.read_script(ep)).title
        choices.append((f"{ep} · {title}", ep))
    return choices


def _shot_choices(project: Project | None, episode):
    if not project or not episode:
        return []
    script = parse_script(project.read_script(episode))
    return [(f"{s.id} · {s.description[:70]}", s.id) for s in script.shots]


def _studio(project: Project, **overrides) -> Studio:
    settings = dict(SETTINGS)
    settings.update(overrides)
    return Studio(project, settings)


def _progress_adapter(progress):
    return lambda fraction, message: progress(fraction, desc=message)


def _friendly(exc: Exception) -> gr.Error:
    log.error("%s", traceback.format_exc())
    text = str(exc)
    if "out of memory" in text.lower():
        text += "  → Turn on 'Low VRAM mode' in Settings, or close other GPU apps."
    return gr.Error(text[:600])


# ------------------------------------------------------------------ series
def series_fields(folder):
    project = _load(folder)
    if not project:
        return ["", "", "", "", "", 1234, "", "bf_emma"]
    s = project.series
    return [
        s.title,
        s.premise,
        s.style_prompt,
        s.quality_prompt,
        s.negative_prompt,
        s.seed,
        s.theme_music,
        s.narrator_voice,
    ]


def save_series(folder, title, premise, style, quality, negative, seed, theme, narrator):
    project = _need(folder)
    s = project.series
    s.title, s.premise, s.style_prompt, s.quality_prompt = title.strip() or s.title, premise, style, quality
    s.negative_prompt, s.seed, s.theme_music, s.narrator_voice = negative, int(seed), theme, narrator
    project.save()
    gr.Info("Series saved")
    return gr.update(choices=_project_choices(), value=folder)


def create_series(title, with_example):
    if not title.strip():
        raise gr.Error("Type a title for your new series first.")
    project = Project.create(SETTINGS["projects_dir"], title, with_example=with_example)
    gr.Info(f"Created '{project.series.title}'")
    return gr.update(choices=_project_choices(), value=project.root.name), ""


def preview_theme(folder, theme, progress=gr.Progress()):
    project = _need(folder)
    progress(0.1, desc="Composing music (first time downloads the music model)...")
    try:
        return str(_studio(project).make_music(theme, 15))
    except Exception as exc:
        raise _friendly(exc)


# ------------------------------------------------------------------ characters
def char_choices(project):
    return [c.name for c in project.series.characters] + [NEW_CHAR] if project else [NEW_CHAR]


def char_fields(folder, name):
    project = _load(folder)
    char = project.series.character(name) if project and name != NEW_CHAR else None
    if not char:
        return ["", "", "1girl, solo, ", "", "af_heart", 1.0, None, "", 0.8, ""]
    ref = project.abs_path(char.reference_image)
    return [
        char.name,
        char.aliases,
        char.appearance,
        char.personality,
        char.voice,
        char.voice_speed,
        str(ref) if ref else None,
        char.lora_path,
        char.lora_weight,
        char.lora_trigger,
    ]


def save_char(folder, selected, name, aliases, appearance, personality, voice, speed, ref, lora, lora_w, trigger):
    project = _need(folder)
    name = name.strip()
    if not name:
        raise gr.Error("The character needs a name.")
    existing = project.series.character(selected) if selected != NEW_CHAR else None
    clash = project.series.character(name)
    if clash and clash is not existing:
        raise gr.Error(f"There is already a character called {clash.name}.")
    char = existing or Character(name=name)
    if not existing:
        project.series.characters.append(char)
    char.name, char.aliases, char.appearance, char.personality = name, aliases, appearance, personality
    char.voice, char.voice_speed = voice or "af_heart", float(speed)
    char.lora_path, char.lora_weight, char.lora_trigger = lora.strip(), float(lora_w), trigger.strip()
    if ref:
        stored = project.abs_path(char.reference_image)
        if not stored or Path(ref).resolve() != stored.resolve():
            char.reference_image = project.store_character_image(name, ref)
    else:
        char.reference_image = ""
    project.save()
    gr.Info(f"Saved {name}")
    return gr.update(choices=char_choices(project), value=name)


def delete_char(folder, selected):
    project = _need(folder)
    char = project.series.character(selected) if selected != NEW_CHAR else None
    if char:
        project.series.characters.remove(char)
        project.save()
        gr.Info(f"Deleted {char.name}")
    return gr.update(choices=char_choices(project), value=NEW_CHAR)


def test_voice(folder, name, voice, speed):
    project = _need(folder)
    line = f"Hi! I'm {name or 'your new character'}. Nice to meet you!"
    out = project.cache_dir("voices") / f"test_{content_hash(line, voice, speed)}.wav"
    try:
        VoiceEngine(SETTINGS).synthesize(line, voice, out, speed=float(speed), speaker=name)
    except Exception as exc:
        raise _friendly(exc)
    return str(out)


def design_char(folder, selected, name, appearance, lora, lora_w, trigger, progress=gr.Progress()):
    project = _need(folder)
    temp = Character(
        name=name or "character",
        appearance=appearance,
        lora_path=lora.strip(),
        lora_weight=float(lora_w),
        lora_trigger=trigger.strip(),
    )
    progress(0.05, desc="Drawing 4 designs (first time downloads the image model, ~7 GB)...")
    try:
        results = _studio(project).character_sheet(temp, 4, start_seed=random.randint(0, 2**31 - 1))
    except Exception as exc:
        raise _friendly(exc)
    return [(str(p), f"seed {s}") for p, s in results], [str(p) for p, _ in results]


def pick_design(paths, evt: gr.SelectData):
    return paths[evt.index] if paths and evt.index is not None else None


def use_design(path):
    if not path:
        raise gr.Error("Click a design first.")
    gr.Info("Now click 'Save character' to keep it.")
    return path


def pick_index(evt: gr.SelectData):
    return evt.index


# ------------------------------------------------------------------ locations
def loc_choices(project):
    return [loc.name for loc in project.series.locations] + [NEW_LOC] if project else [NEW_LOC]


def loc_fields(folder, name):
    project = _load(folder)
    loc = project.series.location(name) if project and name != NEW_LOC else None
    return [loc.name, loc.description] if loc else ["", ""]


def save_loc(folder, selected, name, description):
    project = _need(folder)
    name = name.strip()
    if not name:
        raise gr.Error("The location needs a name.")
    loc = project.series.location(selected) if selected != NEW_LOC else None
    if not loc:
        loc = Location(name=name)
        project.series.locations.append(loc)
    loc.name, loc.description = name, description
    project.save()
    gr.Info(f"Saved {name}")
    return gr.update(choices=loc_choices(project), value=name)


def delete_loc(folder, selected):
    project = _need(folder)
    loc = project.series.location(selected) if selected != NEW_LOC else None
    if loc:
        project.series.locations.remove(loc)
        project.save()
    return gr.update(choices=loc_choices(project), value=NEW_LOC)


# ------------------------------------------------------------------ script
def load_script(folder, episode):
    project = _load(folder)
    return project.read_script(episode) if project and episode else ""


def save_script(folder, episode, text, quiet=False):
    project = _need(folder)
    if not episode:
        raise gr.Error("Pick or create an episode first.")
    project.write_script(episode, text)
    if not quiet:
        gr.Info("Script saved")
    return gr.update(choices=_episode_choices(project), value=episode), gr.update(
        choices=_shot_choices(project, episode)
    )


def new_episode(folder):
    project = _need(folder)
    episode = project.new_episode()
    return gr.update(choices=_episode_choices(project), value=episode)


def check_script(folder, text):
    project = _need(folder)
    script = parse_script(text)
    series = project.series
    lines = [f"**{script.title}** · {len(script.scenes)} scenes · {len(script.shots)} shots"]
    total_lines = 0
    for scene in script.scenes:
        loc = f" · location *{scene.location}*" if scene.location else ""
        if scene.location and not series.location(scene.location):
            loc += " ⚠️ (not in your locations, used as plain text)"
        lines.append(
            f"\n**Scene {scene.index}:** {scene.description}{loc}" + (f" · 🎵 {scene.music}" if scene.music else "")
        )
        for shot in scene.shots:
            cast = ", ".join(c.name for c in shot_characters(series, shot)) or "no characters"
            n = sum(1 for line in shot.lines if line.kind == "dialogue")
            total_lines += n
            lines.append(f"- `{shot.id}` {shot.description[:80]} — 👥 {cast}, 💬 {n} lines")
    unknown = [s for s in script.speakers() if not series.character(s)]
    if unknown:
        lines.append(f"\n⚠️ Speakers not in your characters (they get a default voice): {', '.join(unknown)}")
    for w in script.warnings:
        lines.append(f"\n⚠️ {w}")
    lines.append(f"\n✅ About {total_lines} spoken lines.")
    return "\n".join(lines)


def ai_write(folder, synopsis, progress=gr.Progress()):
    project = _need(folder)
    if not synopsis.strip():
        raise gr.Error("Describe what happens in this episode first.")
    progress(0.2, desc=f"Writing with {SETTINGS['ollama_model']} (can take a minute)...")
    try:
        return story.write_episode(SETTINGS, project.series, synopsis)
    except Exception as exc:
        raise _friendly(exc)


def ai_ideas(folder, progress=gr.Progress()):
    project = _need(folder)
    progress(0.2, desc="Brainstorming...")
    try:
        return story.brainstorm(SETTINGS, project.series)
    except Exception as exc:
        raise _friendly(exc)


# ------------------------------------------------------------------ storyboard
def storyboard_gallery(folder, episode):
    project = _load(folder)
    if not project or not episode:
        return []
    studio = _studio(project)
    script = studio.parse(episode)
    items = []
    for shot in script.shots:
        path = studio.existing_keyframe(script, shot)
        if path:
            items.append((str(path), f"{shot.id} · {shot.description[:50]}"))
    return items


def draw_all(folder, episode, text, draft, progress=gr.Progress()):
    project = _need(folder)
    save_script(folder, episode, text, quiet=True)
    studio = _studio(project, draft_mode=bool(draft))
    script = studio.parse(episode)
    items = []
    try:
        for i, shot in enumerate(script.shots):
            progress(i / max(1, len(script.shots)), desc=f"Drawing {shot.id} ({i + 1}/{len(script.shots)})")
            path = studio.make_keyframe(script, shot)
            items.append((str(path), f"{shot.id} · {shot.description[:50]}"))
            yield items
    except Exception as exc:
        raise _friendly(exc)


def shot_details(folder, episode, shot_id):
    project = _load(folder)
    if not project or not episode or not shot_id:
        return "", None, None
    studio = _studio(project)
    script = studio.parse(episode)
    shot = script.find_shot(shot_id)
    if not shot:
        return "Shot not found. Save the script and pick again.", None, None
    spec = studio.keyframe_spec(script, shot)
    cast = ", ".join(c.name for c in shot_characters(project.series, shot)) or "none"
    ref = "yes" if spec["reference"] else "no"
    talker = studio.lipsync_character(shot)
    lips = talker.name if talker else "off"
    info = (
        f"**Characters:** {cast} · **reference picture used:** {ref} · **seed:** {spec['seed']} · "
        f"**animation:** {studio.animate_mode(shot)} · **lip sync:** {lips}\n\n**Prompt:** {spec['prompt']}"
    )
    keyframe = studio.existing_keyframe(script, shot)
    clip = studio.existing_clip(script, shot)
    return info, str(keyframe) if keyframe else None, str(clip) if clip else None


def lipsync_preview(folder, episode, shot_id, progress=gr.Progress()):
    project = _need(folder)
    studio = _studio(project)
    script = studio.parse(episode)
    shot = script.find_shot(shot_id or "")
    if not shot:
        raise gr.Error("Pick a shot first.")
    talker = studio.lipsync_character(shot)
    if talker is None:
        raise gr.Error(
            "No lip sync in this shot: nobody who is on screen speaks (or lip sync is off / it's an AI clip)."
        )
    progress(0.05, desc="Drawing closed, half-open and open mouths...")
    try:
        mouths = studio.make_mouths(script, shot, studio.make_keyframe(script, shot))
    except Exception as exc:
        raise _friendly(exc)
    if not mouths:
        raise gr.Error(f"Couldn't find {talker.name}'s face in this picture. Try another version of the keyframe.")
    return [(str(mouths[s]), f"{talker.name}: {s} mouth") for s in ("closed", "half", "open")]


def shot_variants(folder, episode, shot_id, draft, progress=gr.Progress()):
    project = _need(folder)
    studio = _studio(project, draft_mode=bool(draft))
    script = studio.parse(episode)
    shot = script.find_shot(shot_id or "")
    if not shot:
        raise gr.Error("Pick a shot first.")
    items, seeds = [], []
    try:
        for i in range(4):
            seed = random.randint(0, 2**31 - 1)
            progress(i / 4, desc=f"Drawing version {i + 1}/4")
            path = studio.make_keyframe(script, shot, seed=seed)
            items.append((str(path), f"seed {seed}"))
            seeds.append(seed)
    except Exception as exc:
        raise _friendly(exc)
    return items, seeds


def use_variant(folder, episode, shot_id, text, seeds, evt_index):
    project = _need(folder)
    if evt_index is None or not seeds:
        raise gr.Error("Click one of the 4 versions first.")
    script = parse_script(text)
    shot = script.find_shot(shot_id or "")
    if not shot:
        raise gr.Error("Pick a shot first.")
    new_text = set_shot_option(text, shot, "seed", str(seeds[evt_index]))
    project.write_script(episode, new_text)
    gr.Info(f"{shot_id} now uses seed {seeds[evt_index]} (saved in the script)")
    return new_text


def animate_shot(folder, episode, shot_id, text, progress=gr.Progress()):
    project = _need(folder)
    script = parse_script(text)
    shot = script.find_shot(shot_id or "")
    if not shot:
        raise gr.Error("Pick a shot first.")
    if SETTINGS.get("draft_mode"):
        raise gr.Error("AI animation is off in draft mode (Settings).")
    if shot.animate != "ai":
        text = set_shot_option(text, shot, "animate", "ai")
        project.write_script(episode, text)
        script = parse_script(text)
        shot = script.find_shot(shot_id)
    studio = _studio(project)
    progress(0.05, desc=f"Animating with {SETTINGS['video_engine']} (first time downloads the video model)...")
    try:
        clip = studio.make_clip(script, shot)
    except Exception as exc:
        raise _friendly(exc)
    gr.Info("Done! This shot will use the AI animation in the episode.")
    return str(clip), text


# ------------------------------------------------------------------ episode
def make_episode(
    folder, episode, text, draft, animate_all, burn, title_card, end_card, lip_sync, progress=gr.Progress()
):
    project = _need(folder)
    if not episode:
        raise gr.Error("Pick an episode on the Script tab.")
    project.write_script(episode, text)
    studio = _studio(
        project,
        draft_mode=bool(draft),
        default_animation="ai" if animate_all else SETTINGS.get("default_animation", "motion"),
        burn_subtitles=bool(burn),
        title_card=bool(title_card),
        end_card=bool(end_card),
        lip_sync=bool(lip_sync),
    )
    try:
        (project.root / "audio").mkdir(exist_ok=True)
        result = studio.build_episode(episode, progress=_progress_adapter(progress))
    except Exception as exc:
        raise _friendly(exc)
    minutes, seconds = divmod(int(result["seconds"]), 60)
    log_lines = [f"✅ Finished: {minutes}m {seconds}s of anime.", f"📁 Saved in: {result['video'].parent}"]
    log_lines += [f"⚠️ {w}" for w in result["warnings"]]
    return str(result["video"]), [str(result["video"]), str(result["subtitles"])], "\n".join(log_lines)


# ------------------------------------------------------------------ settings
SETTING_FIELDS = [
    ("low_vram", "Low VRAM mode (slower, use if you get 'out of memory')"),
    ("lip_sync", "Lip sync (mouths move when characters talk)"),
    ("image_model", "Image model (Hugging Face id or path to an SDXL .safetensors file)"),
    ("image_steps", "Image steps (more = better & slower)"),
    ("image_guidance", "Image guidance (CFG)"),
    ("ip_adapter", "Use character reference pictures (IP-Adapter)"),
    ("ip_adapter_scale", "Reference strength (0.3-0.8)"),
    ("default_animation", "Default animation for shots"),
    ("video_engine", "AI video engine"),
    ("video_steps", "AI video steps"),
    ("video_seconds", "AI clip length (seconds)"),
    ("fps", "Frames per second"),
    ("output_width", "Video width"),
    ("output_height", "Video height"),
    ("subtitle_speaker_names", "Show speaker names in subtitles"),
    ("voice_volume", "Voice volume"),
    ("music_volume", "Music volume"),
    ("sfx_volume", "Sound effects volume"),
    ("ollama_url", "Ollama address"),
    ("ollama_model", "Ollama model for the story assistant"),
]


def _setting_component(key, label):
    value = SETTINGS.get(key, DEFAULTS[key])
    if key == "default_animation":
        return gr.Radio(["motion", "ai", "still"], value=value, label=label)
    if key == "video_engine":
        return gr.Radio([(v, k) for k, v in ENGINES.items()], value=value, label=label)
    if isinstance(DEFAULTS[key], bool):
        return gr.Checkbox(value=bool(value), label=label)
    if isinstance(DEFAULTS[key], (int, float)):
        return gr.Number(value=value, label=label)
    return gr.Textbox(value=value, label=label)


def save_settings_ui(*values):
    for (key, _), value in zip(SETTING_FIELDS, values):
        default = DEFAULTS[key]
        if isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, float):
            value = float(value)
        SETTINGS[key] = value
    save_settings(SETTINGS)
    gr.Info("Settings saved")
    return json.dumps(SETTINGS, indent=2)


def save_advanced(text):
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise gr.Error(f"Not valid JSON: {exc}")
    SETTINGS.update({k: v for k, v in data.items() if k in DEFAULTS})
    save_settings(SETTINGS)
    gr.Info("Settings saved (restart the app if you changed models_dir)")


def gpu_text():
    info = gpu_info()
    return f"**Device:** {info['name']} · **Backend:** {info['backend']} · **VRAM:** {info['vram_gb']} GB"


# ------------------------------------------------------------------ layout
def build_ui() -> gr.Blocks:
    if not _projects():
        # First launch: start with a ready-to-run example series.
        Project.create(SETTINGS["projects_dir"], "Sky Chasers (example)")
    projects = _project_choices()
    first = projects[0][1] if projects else None
    project = _load(first)
    episodes = _episode_choices(project)
    first_ep = episodes[0][1] if episodes else None

    with gr.Blocks(title="Anime Studio", theme=gr.themes.Soft(primary_hue="pink", secondary_hue="violet")) as demo:
        gr.Markdown("# 🎌 Anime Studio\nMake your own anime series, free, on your own computer.")
        with gr.Row(equal_height=True):
            project_dd = gr.Dropdown(projects, value=first, label="📺 Your series", scale=3)
            new_title = gr.Textbox(label="New series title", placeholder="e.g. Sky Chasers", scale=2)
            with_example = gr.Checkbox(value=True, label="Start with example", scale=1)
            new_btn = gr.Button("➕ Create series", variant="primary", scale=1)

        with gr.Tabs():
            # ---------------- series
            with gr.Tab("1 · Series"):
                s_title = gr.Textbox(label="Title")
                s_premise = gr.Textbox(label="Premise (what the series is about)", lines=3)
                with gr.Row():
                    s_style = gr.Textbox(label="Art style tags", lines=2)
                    s_quality = gr.Textbox(label="Quality tags", lines=2)
                s_negative = gr.Textbox(label="Things to avoid in pictures (negative prompt)", lines=2)
                with gr.Row():
                    s_seed = gr.Number(label="Series seed", precision=0)
                    s_narrator = gr.Dropdown(ALL_VOICES, label="Narrator voice", allow_custom_value=True)
                s_theme = gr.Textbox(label="Theme music (plays on title and end cards)")
                with gr.Row():
                    save_series_btn = gr.Button("💾 Save series", variant="primary")
                    theme_btn = gr.Button("🎵 Preview theme music")
                theme_audio = gr.Audio(label="Theme music", interactive=False)

            # ---------------- characters
            with gr.Tab("2 · Characters"):
                char_dd = gr.Dropdown(char_choices(project), value=NEW_CHAR, label="Character")
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            c_name = gr.Textbox(label="Name")
                            c_aliases = gr.Textbox(label="Other names in the script (comma separated)")
                        c_appearance = gr.Textbox(
                            label="Appearance (anime tags)",
                            lines=3,
                            placeholder="1girl, solo, long silver hair, blue eyes, school uniform, red ribbon",
                        )
                        c_personality = gr.Textbox(label="Personality (for the story assistant)")
                        with gr.Row():
                            c_voice = gr.Dropdown(ALL_VOICES, label="Voice", allow_custom_value=True)
                            c_speed = gr.Slider(0.6, 1.5, value=1.0, step=0.05, label="Voice speed")
                        with gr.Accordion("LoRA (optional, for an exact character look)", open=False):
                            c_lora = gr.Textbox(label="LoRA file (path, or file name inside the loras folder)")
                            with gr.Row():
                                c_lora_w = gr.Slider(0, 1.5, value=0.8, step=0.05, label="LoRA strength")
                                c_trigger = gr.Textbox(label="LoRA trigger word")
                    with gr.Column(scale=2):
                        c_ref = gr.Image(
                            label="Reference picture (keeps the face consistent)", type="filepath", height=360
                        )
                with gr.Row():
                    save_char_btn = gr.Button("💾 Save character", variant="primary")
                    voice_btn = gr.Button("🔊 Test voice")
                    design_btn = gr.Button("🎨 Design character (4 options)")
                    delete_char_btn = gr.Button("🗑️ Delete", variant="stop")
                voice_audio = gr.Audio(label="Voice test", interactive=False)
                design_gallery = gr.Gallery(label="Click a design, then 'Use as reference'", columns=4, height=420)
                design_paths = gr.State([])
                picked_design = gr.State(None)
                use_design_btn = gr.Button("👍 Use selected design as reference picture")

            # ---------------- locations
            with gr.Tab("3 · Locations"):
                loc_dd = gr.Dropdown(loc_choices(project), value=NEW_LOC, label="Location")
                l_name = gr.Textbox(label="Name (use it in the script as  | location: Name)")
                l_desc = gr.Textbox(
                    label="What it looks like (tags)",
                    lines=3,
                    placeholder="school rooftop, chain-link fence, water tank, sunset sky",
                )
                with gr.Row():
                    save_loc_btn = gr.Button("💾 Save location", variant="primary")
                    delete_loc_btn = gr.Button("🗑️ Delete", variant="stop")

            # ---------------- script
            with gr.Tab("4 · Script"):
                with gr.Row(equal_height=True):
                    ep_dd = gr.Dropdown(episodes, value=first_ep, label="Episode", scale=3)
                    new_ep_btn = gr.Button("➕ New episode", scale=1)
                script_box = gr.Textbox(label="Script", lines=24, max_lines=60, value=load_script(first, first_ep))
                with gr.Row():
                    save_script_btn = gr.Button("💾 Save script", variant="primary")
                    check_btn = gr.Button("🔍 Check script")
                check_out = gr.Markdown()
                with gr.Accordion("✍️ Story assistant (free, local AI writer via Ollama)", open=False):
                    gr.Markdown(
                        "Install [Ollama](https://ollama.com), then run `ollama pull "
                        f"{SETTINGS['ollama_model']}` once. The draft replaces the script box (not saved until you click Save)."
                    )
                    synopsis = gr.Textbox(label="What happens in this episode?", lines=3)
                    with gr.Row():
                        write_btn = gr.Button("✍️ Write the script for me")
                        ideas_btn = gr.Button("💡 Give me episode ideas")
                    ideas_out = gr.Markdown()
                with gr.Accordion("📖 Script format help", open=False):
                    gr.Markdown(SCRIPT_HELP)

            # ---------------- storyboard
            with gr.Tab("5 · Storyboard"):
                with gr.Row():
                    sb_draft = gr.Checkbox(
                        value=bool(SETTINGS.get("draft_mode")), label="Draft pictures (instant, no AI)"
                    )
                    draw_all_btn = gr.Button("🖌️ Draw all keyframes", variant="primary")
                    refresh_sb_btn = gr.Button("🔄 Refresh")
                sb_gallery = gr.Gallery(label="Storyboard", columns=4, height=420)
                shot_dd = gr.Dropdown(_shot_choices(project, first_ep), label="Shot")
                shot_info = gr.Markdown()
                with gr.Row():
                    shot_image = gr.Image(label="Keyframe", interactive=False, height=320)
                    shot_video = gr.Video(label="AI animation", interactive=False, height=320)
                with gr.Row():
                    variants_btn = gr.Button("🎲 Try 4 new versions")
                    animate_btn = gr.Button("🎞️ Animate this shot with AI (slow)")
                    lips_btn = gr.Button("👄 Preview lip sync mouths")
                lips_gallery = gr.Gallery(label="Lip sync mouth shapes", columns=3, height=260)
                variants_gallery = gr.Gallery(label="Click the version you like", columns=4, height=300)
                variant_seeds = gr.State([])
                variant_pick = gr.State(None)
                use_variant_btn = gr.Button("👍 Use selected version")

            # ---------------- episode
            with gr.Tab("6 · Make Episode"):
                with gr.Row():
                    ep_draft = gr.Checkbox(
                        value=bool(SETTINGS.get("draft_mode")), label="Draft mode (fast preview, no AI)"
                    )
                    ep_ai = gr.Checkbox(
                        value=SETTINGS.get("default_animation") == "ai", label="Animate every shot with AI (very slow)"
                    )
                    ep_burn = gr.Checkbox(value=bool(SETTINGS.get("burn_subtitles")), label="Subtitles on video")
                    ep_title = gr.Checkbox(value=bool(SETTINGS.get("title_card")), label="Title card")
                    ep_end = gr.Checkbox(value=bool(SETTINGS.get("end_card")), label="'To be continued' card")
                    ep_lips = gr.Checkbox(value=bool(SETTINGS.get("lip_sync", True)), label="Lip sync")
                make_btn = gr.Button("🎬 Make episode", variant="primary", size="lg")
                episode_video = gr.Video(label="Your episode", interactive=False)
                episode_files = gr.File(label="Download", file_count="multiple", interactive=False)
                episode_log = gr.Textbox(label="Report", lines=6, interactive=False)

            # ---------------- settings
            with gr.Tab("⚙️ Settings"):
                gpu_md = gr.Markdown(gpu_text())
                setting_inputs = [_setting_component(k, label) for k, label in SETTING_FIELDS]
                save_settings_btn = gr.Button("💾 Save settings", variant="primary")
                with gr.Accordion("Advanced (all settings as JSON)", open=False):
                    adv_json = gr.Code(json.dumps(SETTINGS, indent=2), language="json")
                    adv_btn = gr.Button("💾 Save advanced settings")

            with gr.Tab("❓ Help"):
                gr.Markdown(HELP)

        # ---------------- events
        series_outputs = [s_title, s_premise, s_style, s_quality, s_negative, s_seed, s_theme, s_narrator]

        def on_project(folder):
            p = _load(folder)
            eps = _episode_choices(p)
            ep = eps[0][1] if eps else None
            return (
                *series_fields(folder),
                gr.update(choices=char_choices(p), value=NEW_CHAR),
                gr.update(choices=loc_choices(p), value=NEW_LOC),
                gr.update(choices=eps, value=ep),
                load_script(folder, ep),
            )

        project_outputs = series_outputs + [char_dd, loc_dd, ep_dd, script_box]
        project_dd.change(on_project, project_dd, project_outputs)

        def on_page_load(folder):
            # Re-read from disk so a browser refresh shows the latest series.
            choices = _project_choices()
            if folder not in [c[1] for c in choices]:
                folder = choices[0][1] if choices else None
            return (gr.update(choices=choices, value=folder), *on_project(folder))

        demo.load(on_page_load, project_dd, [project_dd] + project_outputs)
        new_btn.click(create_series, [new_title, with_example], [project_dd, new_title])
        save_series_btn.click(save_series, [project_dd] + series_outputs, project_dd)
        theme_btn.click(preview_theme, [project_dd, s_theme], theme_audio)

        char_outputs = [
            c_name,
            c_aliases,
            c_appearance,
            c_personality,
            c_voice,
            c_speed,
            c_ref,
            c_lora,
            c_lora_w,
            c_trigger,
        ]
        char_dd.change(char_fields, [project_dd, char_dd], char_outputs)
        save_char_btn.click(save_char, [project_dd, char_dd] + char_outputs, char_dd)
        delete_char_btn.click(delete_char, [project_dd, char_dd], char_dd)
        voice_btn.click(test_voice, [project_dd, c_name, c_voice, c_speed], voice_audio)
        design_btn.click(
            design_char,
            [project_dd, char_dd, c_name, c_appearance, c_lora, c_lora_w, c_trigger],
            [design_gallery, design_paths],
        )
        design_gallery.select(pick_design, design_paths, picked_design)
        use_design_btn.click(use_design, picked_design, c_ref)

        loc_dd.change(loc_fields, [project_dd, loc_dd], [l_name, l_desc])
        save_loc_btn.click(save_loc, [project_dd, loc_dd, l_name, l_desc], loc_dd)
        delete_loc_btn.click(delete_loc, [project_dd, loc_dd], loc_dd)

        def on_episode(folder, ep):
            p = _load(folder)
            return (
                load_script(folder, ep),
                gr.update(choices=_shot_choices(p, ep), value=None),
                storyboard_gallery(folder, ep),
            )

        ep_dd.change(on_episode, [project_dd, ep_dd], [script_box, shot_dd, sb_gallery])
        new_ep_btn.click(new_episode, project_dd, ep_dd)
        save_script_btn.click(save_script, [project_dd, ep_dd, script_box], [ep_dd, shot_dd])
        check_btn.click(check_script, [project_dd, script_box], check_out)
        write_btn.click(ai_write, [project_dd, synopsis], script_box)
        ideas_btn.click(ai_ideas, project_dd, ideas_out)

        draw_all_btn.click(draw_all, [project_dd, ep_dd, script_box, sb_draft], sb_gallery).then(
            lambda f, e: gr.update(choices=_shot_choices(_load(f), e)), [project_dd, ep_dd], shot_dd
        )
        refresh_sb_btn.click(storyboard_gallery, [project_dd, ep_dd], sb_gallery).then(
            lambda f, e: gr.update(choices=_shot_choices(_load(f), e)), [project_dd, ep_dd], shot_dd
        )
        shot_dd.change(shot_details, [project_dd, ep_dd, shot_dd], [shot_info, shot_image, shot_video])
        variants_btn.click(shot_variants, [project_dd, ep_dd, shot_dd, sb_draft], [variants_gallery, variant_seeds])
        variants_gallery.select(pick_index, None, variant_pick)
        use_variant_btn.click(
            use_variant, [project_dd, ep_dd, shot_dd, script_box, variant_seeds, variant_pick], script_box
        ).then(shot_details, [project_dd, ep_dd, shot_dd], [shot_info, shot_image, shot_video])
        animate_btn.click(animate_shot, [project_dd, ep_dd, shot_dd, script_box], [shot_video, script_box])
        lips_btn.click(lipsync_preview, [project_dd, ep_dd, shot_dd], lips_gallery)

        make_btn.click(
            make_episode,
            [project_dd, ep_dd, script_box, ep_draft, ep_ai, ep_burn, ep_title, ep_end, ep_lips],
            [episode_video, episode_files, episode_log],
        )

        save_settings_btn.click(save_settings_ui, setting_inputs, adv_json)
        adv_btn.click(save_advanced, adv_json, None)
        demo.load(gpu_text, None, gpu_md)
    return demo
