#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/setup_venv.sh        # create .venv + upgrade pip
#   source .venv/bin/activate     # then continue

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
  echo "$VENV_DIR already exists. Activate it with: source $VENV_DIR/bin/activate"
  exit 0
fi

$PYTHON -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# upgrade packaging
pip install --upgrade pip setuptools wheel

echo "Created venv at $VENV_DIR. Activate with: source $VENV_DIR/bin/activate"
