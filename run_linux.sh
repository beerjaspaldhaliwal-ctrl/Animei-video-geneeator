#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "Anime Studio is not installed yet. Run ./install_linux.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source venv/bin/activate
exec python app.py "$@"
