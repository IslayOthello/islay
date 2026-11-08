#!/usr/bin/env bash
set -e

# Change directory to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Build islay engine if missing
if [[ ! -f "build/islay" && ! -f "_build/islay" && ! -f "islay" ]]; then
  echo "==> Building islay engine..."
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
fi

# Ensure native macOS desktop app is compiled if on Darwin
if [[ "$(uname)" == "Darwin" && ! -f "gui/bin/islay_gui" ]]; then
  echo "==> Compiling native desktop GUI application..."
  mkdir -p gui/bin
  clang -O2 -fobjc-arc -framework Cocoa -framework WebKit gui/src/app.m -o gui/bin/islay_gui
fi

MODE="${1:-default}"

if [[ "$MODE" == "--love" ]]; then
  if command -v love >/dev/null 2>&1; then
    echo "==> Starting Islay GUI with LÖVE 2D (White Theme)..."
    exec love gui
  elif [[ -f "/Applications/love.app/Contents/MacOS/love" ]]; then
    echo "==> Starting Islay GUI with /Applications/love.app..."
    exec /Applications/love.app/Contents/MacOS/love gui
  else
    echo "Error: 'love' not found. Launching native desktop app instead."
    MODE="default"
  fi
fi

if [[ "$MODE" == "--web" ]]; then
  echo "==> Starting Islay Desktop Web GUI..."
  exec python3 gui/server.py
fi

# Default launcher
if [[ "$(uname)" == "Darwin" && -f "gui/bin/islay_gui" ]]; then
  echo "==> Starting Islay Professional Desktop GUI (WZebra Edition, White Theme)..."
  exec ./gui/bin/islay_gui
elif command -v python3 >/dev/null 2>&1; then
  echo "==> Starting Islay Desktop GUI via local bridge..."
  exec python3 gui/server.py
elif command -v love >/dev/null 2>&1; then
  echo "==> Starting Islay GUI with LÖVE 2D..."
  exec love gui
else
  echo "Error: Neither native GUI, python3, nor love was found."
  exit 1
fi
