"""`python -m app` starts the server on HOST:PORT from .env (0.0.0.0:8000 by default)."""

import sys

import uvicorn

from app.access import load_key
from app.config import get_settings
from app.services.team_alerts import team_urls


def main() -> None:
    # A console that isn't UTF-8 (cp1252, cp852, redirected output) would crash on the diacritics.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    s = get_settings()
    print()
    print("  RunwayEye")
    print(f"  Panou:           http://localhost:{s.port}/")
    for i, a in enumerate(team_urls(s.port, load_key(s))):
        print(f"  {'Telefon echipă:' if i == 0 else '':16s} {a['url']}  ({a['label']})")
    print(flush=True)
    # No access log: each open dashboard asks for /api/status every 2 s and would bury the useful lines.
    uvicorn.run("app.main:app", host=s.host, port=s.port, timeout_graceful_shutdown=2, log_level="info",
                access_log=False)


if __name__ == "__main__":
    main()
