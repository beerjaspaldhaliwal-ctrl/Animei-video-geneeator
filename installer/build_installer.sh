#!/usr/bin/env bash
# Build AnimeStudio-Setup.exe with NSIS.  Linux: sudo apt install nsis
set -euo pipefail
cd "$(dirname "$0")/.."

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
# Only the files the app needs (tracked files, as they are in the working tree).
git ls-files -z app.py requirements.txt install_windows.bat run_windows.bat README.md anime_studio loras \
    | xargs -0 tar -cf - | tar -xf - -C "$stage"

makensis -V2 -DSRC="$stage" -DOUTFILE="$PWD/AnimeStudio-Setup.exe" installer/AnimeStudio.nsi
echo "Built $PWD/AnimeStudio-Setup.exe"
