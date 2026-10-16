#!/usr/bin/env bash
# Anime Studio installer for Linux.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
    echo "Python 3.10-3.12 is required (Kokoro voices don't support 3.13 yet)."
    echo "Ubuntu: sudo apt install python3.12 python3.12-venv"
    exit 1
fi

echo "Which graphics card do you have?"
echo "  1 = AMD Radeon RX 7000 series (RX 7600, 7600 XT, 7700 XT, 7800 XT, 7900 ...)"
echo "  2 = AMD Radeon RX 9000 series (RX 9060 XT, 9070, 9070 XT)"
echo "  3 = NVIDIA GeForce"
echo "  4 = No graphics card (CPU only, very slow)"
read -rp "Type 1, 2, 3 or 4 and press Enter: " GPU
case "$GPU" in
    1) TORCH_INDEX="https://repo.amd.com/rocm/whl/gfx110X-all/" ;;
    2) TORCH_INDEX="https://repo.amd.com/rocm/whl/gfx120X-all/" ;;
    3) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
    4) TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
    *) echo "Please type 1, 2, 3 or 4."; exit 1 ;;
esac

[ -x venv/bin/python ] || "$PY" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url "$TORCH_INDEX" torch torchvision torchaudio
python - > constraints.txt <<'EOF'
import importlib.metadata as m
for p in ("torch", "torchvision", "torchaudio"):
    print(f"{p}=={m.version(p)}")
EOF
python -m pip install -r requirements.txt -c constraints.txt

python -c "import torch; ok = torch.cuda.is_available(); print('GPU ready:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND (running on CPU)')"
if [ "$GPU" = "1" ] || [ "$GPU" = "2" ]; then
    if ! id -nG | grep -qw render; then
        echo
        echo "Tip: if the GPU is not found, add yourself to the GPU groups and log in again:"
        echo "  sudo usermod -aG render,video \$USER"
    fi
fi
echo
echo "Done! Start Anime Studio with:  ./run_linux.sh"
