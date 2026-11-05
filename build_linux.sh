#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is for Linux only."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi

VENV_DIR=".venv-linux-build"
PYTHON_BIN="$VENV_DIR/bin/python"
PYINSTALLER_BIN="$VENV_DIR/bin/pyinstaller"

if [[ ! -x "$PYTHON_BIN" ]]; then
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Failed to create virtualenv python at $PYTHON_BIN"
  echo "Install the venv package for your distro (example: sudo apt install python3-venv) and rerun."
  exit 1
fi

"$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install pyinstaller numpy pillow websocket-client

if command -v g++ >/dev/null 2>&1; then
  g++ -O2 -std=c++17 -o src/xdc1 xdc1/xdc1.cpp
else
  echo "Warning: g++ not found. XDC1 will require XDC1_BIN_PATH or Wine with src/xdc1.exe at runtime."
fi

ADD_DATA_ARGS=()

if [[ -f src/icon.png ]]; then
  ADD_DATA_ARGS+=(--add-data "src/icon.png:.")
fi
if [[ -f src/icon.ico ]]; then
  ADD_DATA_ARGS+=(--add-data "src/icon.ico:.")
fi
if [[ -f src/xdc1 ]]; then
  chmod +x src/xdc1
  ADD_DATA_ARGS+=(--add-data "src/xdc1:.")
fi
if [[ -f src/xdc1.exe ]]; then
  ADD_DATA_ARGS+=(--add-data "src/xdc1.exe:.")
fi

"$PYINSTALLER_BIN" \
  --noconfirm \
  --clean \
  --onefile \
  --name Daniel-linux \
  --collect-all numpy \
  --collect-all PIL \
  --hidden-import websocket \
  "${ADD_DATA_ARGS[@]}" \
  src/daniel.py

echo "Build complete: dist/Daniel-linux"
