"""`python -m app` starts the server on HOST:PORT from .env (0.0.0.0:8000 by default)."""

import sys

import uvicorn

from app.config import get_settings
from app.services.team_alerts import team_url


def main() -> None:
    # A console that isn't UTF-8 (cp1252, cp852, redirected output) would crash on the diacritics.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    s = get_settings()
    print()
    print("  RunwayEye")
    print(f"  Panou:           http://localhost:{s.port}/")
    print(f"  Telefon echipă:  {team_url(s.port)}")
    print(flush=True)
    uvicorn.run("app.main:app", host=s.host, port=s.port, timeout_graceful_shutdown=2, log_level="info")


if __name__ == "__main__":
    main()
