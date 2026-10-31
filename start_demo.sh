#!/usr/bin/env sh
# RunwayEye on macOS / Linux: first run installs everything, then starts the server.
cd "$(dirname "$0")" || exit 1

if [ ! -x venv/bin/python ]; then
  echo "[RunwayEye] Prima pornire: instalez mediul Python (câteva minute)."
  python3 backend/scripts/setup.py || { echo "[RunwayEye] E nevoie de Python 3.11+ (python3)."; exit 1; }
fi
[ -f .env ] || cp .env.example .env
export PYTHONPATH="$PWD/backend"  # the Python code lives in backend/ (package "app")
[ -f data/videos/synthetic_demo.mp4 ] || venv/bin/python backend/scripts/make_demo_clip.py

PORT=$(venv/bin/python -c "from app.config import get_settings as g; print(g().port)" 2>/dev/null || echo 8000)
# Open the dashboard a few seconds after the server starts.
( sleep 4; (command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT/") || open "http://localhost:$PORT/" ) >/dev/null 2>&1 &
exec venv/bin/python -m app
