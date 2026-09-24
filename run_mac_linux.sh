#!/usr/bin/env bash
# Starts the icon2svg web app (creates a virtual env on first run).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up (1-2 minutes)..."
  "$PY" -m venv .venv
fi
# (re)install when requirements.txt changed since the last install
if ! cmp -s requirements.txt .venv/installed.ok; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv/installed.ok
fi
if [ "$1" != "" ]; then
  .venv/bin/python convert.py "$@"   # ./run_mac_linux.sh Input/  -> batch convert
else
  .venv/bin/python app.py
fi
