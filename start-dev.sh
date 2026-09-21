#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${PYUGEN_VENV:-.venv}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Error: $PYTHON was not found."
    echo "Install Python 3 and try again."
    exit 1
fi

# Keep development dependencies isolated from the system Python.
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating local Python environment: $VENV_DIR"
    if ! "$PYTHON" -m venv "$VENV_DIR"; then
        echo
        echo "Could not create a virtual environment."
        echo "On Debian/Ubuntu, install it with:"
        echo "  sudo apt install python3-venv"
        exit 1
    fi
fi

VENV_PY="$VENV_DIR/bin/python"

# Install dependencies on first launch (or if the environment is incomplete).
if ! "$VENV_PY" -c 'import fastapi, uvicorn, multipart, redis, rq, numpy, cv2, PIL, pydantic' >/dev/null 2>&1; then
    echo "Installing 180pyugen web dependencies..."
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install -r requirements.txt
fi

export PYUGEN_INLINE_JOBS=1
export PYUGEN_DATA_ROOT="${PYUGEN_DATA_ROOT:-/tmp/180pyugen}"

echo
echo "Starting 180pyugen Web development server..."
echo "Open: http://127.0.0.1:8080"
echo

exec "$VENV_PY" -m uvicorn backend.app:app \
    --host 127.0.0.1 \
    --port 8080 \
    --reload
