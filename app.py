"""Start Anime Studio:  python app.py   (then open http://127.0.0.1:7860)"""

import argparse
import logging
import os
import sys
from pathlib import Path

from anime_studio.config import load_settings


def main():
    parser = argparse.ArgumentParser(description="Anime Studio: make your own anime series")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a temporary public link")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    args = parser.parse_args()

    settings = load_settings()
    # Keep all downloaded AI models inside the app folder (easy to find and delete).
    os.environ.setdefault("HF_HOME", settings["models_dir"])
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    Path(settings["projects_dir"]).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "httpcore", "gradio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from anime_studio.device import gpu_info, isolate_discrete_gpu

    isolate_discrete_gpu()  # before torch loads: skip the CPU's built-in graphics
    from anime_studio.ui import build_ui

    info = gpu_info()
    print(f"\n  Anime Studio  |  {info['name']}  |  {info['backend']}  |  VRAM {info['vram_gb']} GB\n", flush=True)
    if info["backend"] == "CPU":
        print("  WARNING: no GPU found. Drawing will be VERY slow. See README 'Troubleshooting'.\n", flush=True)

    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        allowed_paths=[settings["projects_dir"]],
    )


if __name__ == "__main__":
    sys.exit(main())
