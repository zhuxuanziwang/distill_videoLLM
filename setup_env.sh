#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_VENV="$ROOT_DIR/.venv"
SHARED_VENV="$ROOT_DIR/../parallel_minimal/.venv"

if [ ! -x "$LOCAL_VENV/bin/python" ]; then
  python3 -m venv "$LOCAL_VENV"
fi

if "$LOCAL_VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
  echo "Using local env: $LOCAL_VENV"
  exit 0
fi

if [ -x "$SHARED_VENV/bin/python" ] && "$SHARED_VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
  echo "Local env does not have torch."
  echo "Use shared env for now: $SHARED_VENV"
  echo "Run: source $SHARED_VENV/bin/activate"
  exit 0
fi

echo "No usable environment with torch found."
echo "If network is available, install with:"
echo "  source $LOCAL_VENV/bin/activate && pip install -r requirements.txt"
exit 1
