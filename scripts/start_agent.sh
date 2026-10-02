#!/usr/bin/env sh
# Start the OllamaPower helper agent: scripts/start_agent.sh --hub 192.168.1.10
# (or set hub.address in agent.toml). Uses ./.venv if it exists, else `ollamapower` on PATH.
set -e
here="$(cd "$(dirname "$0")/.." && pwd)"
if [ -x "$here/.venv/bin/ollamapower" ]; then
    exec "$here/.venv/bin/ollamapower" agent "$@"
fi
exec ollamapower agent "$@"
