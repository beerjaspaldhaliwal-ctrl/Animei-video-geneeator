# 🎌 Anime Studio

**Make your own anime series, free, on your own computer.**

Write a script, design your characters, and Anime Studio turns it into finished anime episodes
with pictures, animation, voices, music, sound effects and subtitles. No subscriptions, no
paid APIs. Everything runs on your PC with free, open-source AI models.

| Step | What Anime Studio does | AI model (free) |
|---|---|---|
| ✍️ Story | Writes episode scripts and ideas from your synopsis *(optional)* | Any local LLM through [Ollama](https://ollama.com) |
| 🧑‍🎨 Characters | Designs characters and keeps their faces consistent across shots | Animagine XL 4.0 + IP-Adapter, optional LoRA |
| 🏞️ Storyboard | Draws every shot as an anime keyframe | Animagine XL 4.0 (any SDXL anime model works) |
| 🎞️ Animation | Camera moves (pan, zoom, shake) **or** real AI motion | LTX-Video or Wan 2.2 |
| 🗣️ Voice acting | A different voice per character, with emotions | Kokoro (offline) or free Edge voices (incl. Japanese) |
| 👄 Lip sync | The talking character's mouth opens and closes with the voice, anime style | anime face finder + Animagine inpainting |
| 🎵 Music | Background music for each scene, plus an opening theme | MusicGen |
| 💥 Sound effects | Door creaks, explosions, wind... | AudioLDM |
| 🎬 Editing | Timing, subtitles, fades, title and "To be continued" cards, music ducking | built in (ffmpeg) |

You get an `episode.mp4` (720p by default) and an `episode.srt` subtitle file for every episode.

---

## What you need

* **Graphics card:** AMD Radeon RX 7600 XT 16 GB ✅ (any RX 7000/9000 card or NVIDIA RTX with 8 GB+ works;
  16 GB is recommended)
* **Windows 10/11** or **Linux**
* **RAM:** 16 GB minimum, 32 GB recommended (needed for the AI video models)
* **Disk space:** about 15 GB for the basic models, plus ~28 GB for LTX-Video and ~34 GB for Wan 2.2 if you use AI animation
* **Latest AMD Adrenalin driver**: update it from [amd.com/support](https://www.amd.com/en/support) before installing
* **Python 3.12** from [python.org](https://www.python.org/downloads/windows/). Tick **"Add python.exe to PATH"**
  while installing. (Python 3.13 doesn't work yet because of the Kokoro voice library.)

## Install (Windows)

1. Click the green **Code** button on GitHub → **Download ZIP**, and unzip it somewhere simple like `C:\AnimeStudio`.
2. Double-click **`install_windows.bat`**.
3. When it asks for your graphics card, type **`1`** (AMD Radeon RX 7000 series, which includes the RX 7600 XT) and press Enter.
4. Wait. It downloads PyTorch for AMD (ROCm) and the rest (a few GB). At the end it should list
   `AMD Radeon RX 7600 XT` under "GPUs found".
5. Double-click **`run_windows.bat`**. Your browser opens Anime Studio at `http://127.0.0.1:7860`.

Keep the black window open while you use the app; close it to quit.

## Install (Linux)

```bash
git clone https://github.com/beerjaspaldhaliwal-ctrl/Animei-video-geneeator.git
cd Animei-video-geneeator
./install_linux.sh     # choose 1 for RX 7000 series
./run_linux.sh
```

If the GPU isn't found on Linux: `sudo usermod -aG render,video $USER`, then log out and back in.

> **First run:** each AI model downloads the first time you use it, into the `models` folder
> (image model ~7 GB, IP-Adapter ~3.5 GB, voices ~0.4 GB, music ~2.4 GB, sound effects ~1.7 GB,
> LTX-Video ~28 GB, Wan 2.2 ~34 GB). After that everything works offline, except Edge voices,
> which always need internet.

---

## Make your first episode

The app opens with an example series, so you can try it immediately.

1. **Make Episode tab** → tick **Draft mode** → **🎬 Make episode**.
   Draft mode uses placeholder pictures and beep voices and finishes in seconds. It's for checking timing.
2. Untick Draft mode and click **Make episode** again for the real thing.

Then make it yours:

1. **Series**: title, premise, art style and theme music.
2. **Characters**: describe each one with anime tags, e.g.
   `1girl, solo, long silver hair, blue eyes, school uniform, red ribbon`.
   Pick a voice and click **🔊 Test voice**. Click **🎨 Design character**, pick your favourite of 4
   designs, then **Use selected design as reference picture** → **Save**.
   The reference picture keeps the character's face the same in every shot.
3. **Locations**: save the places your story happens in.
4. **Script**: write the episode (format below), or open the **Story assistant**.
5. **Storyboard**: **Draw all keyframes**. Don't like a shot? Pick it → **🎲 Try 4 new versions** →
   click the best one → **Use selected version**. Want real movement in a shot? **🎞️ Animate this shot with AI**.
6. **Make Episode** → **🎬 Make episode**. Your video and subtitles appear for download.

Everything is cached, so if you change one line of the script, only that line's voice is redone.

## Script format

```
TITLE: The Wind Beyond the Mountains

SCENE: school rooftop at sunset, orange sky | location: Rooftop | music: gentle hopeful piano
SHOT: wide shot, Aiko standing at the railing, wind in her hair | camera: pan_right
NARRATOR: Every day after school, Aiko came to the rooftop.
AIKO: I always wondered what's beyond those mountains.
SHOT: Ren opening the rooftop door, surprised | camera: zoom_in
SFX: metal door creaking open
REN (surprised): Aiko? What are you doing up here?
```

| Line | Meaning |
|---|---|
| `TITLE:` | Episode title (shown on the title card) |
| `SCENE:` | A new place or time. Options: `location:` (a saved location), `music:` (describe the music, a file name like `song.mp3`, or `none`) |
| `SHOT:` | One picture on screen. A plain line without a colon is also a shot |
| `NAME: text` | Dialogue. Add an emotion: `REN (angry):`, `(happy)`, `(sad)`, `(whisper)`, `(shouting)`, `(excited)` |
| `NARRATOR:` | Narration (different subtitle colour) |
| `SFX:` | Sound effect: a description, or your own file like `boom.wav` |
| `# ...` | A note to yourself; ignored |

Shot options (after `|`):

* `camera:` `static`, `zoom_in`, `zoom_out`, `pan_left`, `pan_right`, `pan_up`, `pan_down`, `shake`
* `seconds:` minimum shot length
* `animate:` `motion` (camera move), `ai` (AI animation), `still` (no movement)
* `motion:` what should move in the AI animation, e.g. `hair and skirt blowing in the wind`
* `characters:` who's on screen (default: characters named in the shot)
* `seed:` which picture variation to use (the Storyboard tab sets this for you)
* `lipsync:` `auto` (default), `off`, or `left` / `right` to choose which face talks when two people are on screen

## Lip sync

When a character who is **on screen** speaks, Anime Studio finds their face, draws three mouth
shapes (closed, half open, open) into the keyframe, and switches between them in time with the
voice, changing every 2 frames like real TV anime ("kuchipaku"). The rest of the picture never changes.

* It's automatic. Turn it off for the whole episode with the **Lip sync** tick box, or per shot with `| lipsync: off`.
* Storyboard tab → pick a shot → **👄 Preview lip sync mouths** shows the three mouth shapes.
* It works on picture shots (camera moves). AI-animated shots make their own movement, so
  Anime Studio asks the video model to show the character talking instead.
* Characters seen from behind or too small to find get no lip sync. Narration and off-screen
  voices never move anyone's mouth.
* In two-person shots the biggest face talks. Use `| lipsync: left` or `| lipsync: right` to pick the other one,
  or better, give each speaker their own close-up shot.
* Lip sync costs about 1.5× the drawing time of a keyframe for each talking shot (done once, then cached).

## Tips for a good series

* **Consistent characters:** use the same tags every time (they're stored in the character, so
  you don't repeat them in the script), set a reference picture, and keep **one character per shot**
  when you can. For a perfectly fixed look, use a **LoRA** file: put it in the `loras` folder and
  type its name in Characters → LoRA.
* **Better pictures:** shot descriptions work best as anime tags plus a short action:
  `close-up, Aiko smiling, tears in eyes, cherry blossoms`.
* **Animation:** camera moves are fast and look like classic limited TV animation. AI animation
  looks amazing on key moments (action, emotional close-ups) but takes several minutes per shot.
  Use `| animate: ai` only on the shots that need it.
* **Japanese voice acting:** pick `ja-JP-NanamiNeural` / `ja-JP-KeitaNeural` (free Microsoft
  Edge voices, need internet) and write the dialogue in Japanese. Offline Kokoro Japanese voices
  (`jf_alpha`, ...) need one extra command: `venv\Scripts\pip install "misaki[ja]"`.
* **Your own music:** put files in `projects/<your series>/audio/` and write `music: my_song.mp3`.

## Story assistant (optional)

1. Install [Ollama](https://ollama.com) (it works with AMD GPUs).
2. Open a terminal and run `ollama pull llama3.1:8b` once.
3. Script tab → **Story assistant** → describe the episode → **Write the script for me**.

You can choose another Ollama model in Settings (for example `qwen3:8b` or `gemma3:12b`).

## Where your files are

```
projects/<your series>/
    series.json                  your series, characters and locations
    characters/                  character reference pictures
    audio/                       your own music and sound files
    episodes/ep01/script.txt     the script
    episodes/ep01/output/        episode.mp4 and episode.srt  ← your finished episode
    cache/                       generated pictures, clips and sounds (safe to delete)
models/                          downloaded AI models (safe to delete, re-downloads when needed)
```

## Troubleshooting

* **"GPUs found: NONE":** update the AMD Adrenalin driver, restart, and run
  `install_windows.bat` again (choose `1`). Check that you have Python **3.12**.
* **Ryzen 7000/9000 CPUs (e.g. Ryzen 7 9700X) have built-in graphics** called "AMD Radeon(TM) Graphics".
  Anime Studio automatically uses your graphics card instead (the black window prints which GPU it uses).
  If you still get errors like `HIP error` or `invalid device function`, turn the built-in graphics off
  in the BIOS (usually *Advanced → AMD CBS → NBIO → GFX Configuration → iGPU: Disabled*; the exact
  place depends on your motherboard). Keep your monitor plugged into the graphics card.
* **Missing DLL errors on Windows:** install the latest
  [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist).
* **Out of memory:** Settings → tick **Low VRAM mode**, close games and browsers using the GPU,
  or lower the video size.
* **Slow:** the first run of each step is slow because models download. Use Draft mode while
  you write; draw keyframes once, then only redo the shots you change.
* **Voices don't work:** Kokoro needs internet the first time (it downloads the voice files).
  As a backup, pick an Edge voice like `en-US-AnaNeural`.
* **Wrong number of people in a shot:** add `| characters: Aiko` (or `none`) to the SHOT line.

## Licenses (read before you publish)

Anime Studio's own code is yours to use. The AI models have their own licenses:

| Model | License | Commercial use (e.g. monetized YouTube) |
|---|---|---|
| Animagine XL 4.0 | CreativeML Open RAIL++-M | ✅ allowed, with its use restrictions |
| IP-Adapter, Kokoro, Wan 2.2 | Apache 2.0 | ✅ |
| Anime face detector (deepghs, for lip sync) | MIT | ✅ |
| LTX-Video | LTX-Video open weights license | ⚠️ read its license first |
| Edge voices | Microsoft service terms | ⚠️ check Microsoft's terms |
| **MusicGen** | **CC-BY-NC 4.0** | ❌ **non-commercial only** |
| **AudioLDM** | **CC-BY-NC-SA 4.0** | ❌ **non-commercial only** |

If you plan to earn money from your anime, use your own or royalty-free music and sound effects
(`music: my_song.mp3`, `SFX: door.wav`) instead of the generated ones.

## For developers

```bash
pip install pytest
python -m pytest tests        # runs in draft mode, no GPU or model downloads needed
```

The code lives in `anime_studio/`: `script_parser.py` (screenplay format), `pipeline.py`
(script → keyframes → clips → audio → episode), `editor.py` (timeline, mixing, subtitles,
rendering), `generators/` (image, video, voice, music/SFX, story) and `ui.py` (Gradio app).
