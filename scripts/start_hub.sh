#!/usr/bin/env sh
# Start the OllamaPower hub. Uses ./.venv if it exists (a source checkout), else the
# `ollamapower` on PATH (pipx). Extra arguments go to `ollamapower hub`, e.g. --no-tui.
set -e
here="$(cd "$(dirname "$0")/.." && pwd)"
if [ -x "$here/.venv/bin/ollamapower" ]; then
    exec "$here/.venv/bin/ollamapower" hub "$@"
fi
exec ollamapower hub "$@"
