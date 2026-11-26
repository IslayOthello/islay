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

MODE="${1:-gtk}"

if [[ "$MODE" == "--love" ]]; then
  if command -v love >/dev/null 2>&1; then
    echo "==> Starting Islay GUI with LÖVE 2D..."
    exec love gui
  elif [[ -f "/Applications/love.app/Contents/MacOS/love" ]]; then
    exec /Applications/love.app/Contents/MacOS/love gui
  else
    echo "Error: 'love' not found."
    exit 1
  fi
elif [[ "$MODE" == "--web" ]]; then
  echo "==> Starting Islay Desktop Web GUI..."
  exec python3 gui/server.py
elif [[ "$MODE" == "--native" ]]; then
  if [[ "$(uname)" == "Darwin" && ! -f "gui/bin/islay_gui" ]]; then
    mkdir -p gui/bin
    clang -O2 -fobjc-arc -framework Cocoa -framework WebKit gui/src/app.m -o gui/bin/islay_gui
  fi
  exec ./gui/bin/islay_gui
fi

# Default: Pure Lua + GTK3 Desktop GUI (WZebra + AlphaGo MCTS)
if command -v luajit >/dev/null 2>&1; then
  if command -v luarocks >/dev/null 2>&1; then
    eval "$(luarocks --lua-version=5.1 path)"
  fi
  if [[ -d "/opt/homebrew/lib/girepository-1.0" ]]; then
    export DYLD_LIBRARY_PATH="/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export GI_TYPELIB_PATH="/opt/homebrew/lib/girepository-1.0${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"
  fi
  echo "==> Starting Islay Pure Lua GTK3 Studio (WZebra + AlphaGo MCTS)..."
  exec luajit gui/gtk_app.lua
fi

echo "Error: luajit not found."
exit 1
