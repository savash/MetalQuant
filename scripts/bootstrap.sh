#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it with: curl -L https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install -U pip setuptools wheel
python -m pip install -e .

echo "MetalQuant environment ready. Activate with: source .venv/bin/activate"
