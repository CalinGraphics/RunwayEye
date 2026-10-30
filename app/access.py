"""Access key for other devices on the network.

The laptop itself (localhost) never needs it. Phones get it from the QR code: the first
visit with ?key=... sets a cookie, so the team page and the dashboard keep working after that.
"""

import hmac
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import Settings

COOKIE = "runwayeye_key"
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
OPEN_PATHS = ("/static/", "/health")


def load_key(s: Settings) -> str:
    if s.access_key:
        return s.access_key
    f = s.data_path / "access_key.txt"
    try:
        key = f.read_text(encoding="utf-8").strip()
    except OSError:
        key = ""
    if not key:
        key = secrets.token_urlsafe(9)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(key, encoding="utf-8")
    return key


def _matches(value: str | None, key: str) -> bool:
    return bool(value) and hmac.compare_digest(value, key)


async def require_key(request: Request, call_next):
    key = getattr(request.app.state, "access_key", "")
    host = request.client.host if request.client else ""
    if not key or host in LOCAL_HOSTS or request.url.path.startswith(OPEN_PATHS):
        return await call_next(request)

    if _matches(request.query_params.get("key"), key):
        params = [(k, v) for k, v in request.query_params.multi_items() if k != "key"]
        url = request.url.replace_query_params(**dict(params)) if params else request.url.replace(query="")
        resp = RedirectResponse(str(url), status_code=303)
        resp.set_cookie(COOKIE, key, max_age=30 * 24 * 3600, httponly=True, samesite="lax")
        return resp
    if _matches(request.cookies.get(COOKIE), key):
        return await call_next(request)

    message = "Scanați codul QR de pe panoul RunwayEye ca să deschideți pagina pe acest dispozitiv."
    if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
        from app.templating import templates

        return templates.TemplateResponse(
            request, "error.html", {"status_code": 401, "title": "Acces doar cu codul QR", "message": message},
            status_code=401,
        )
    return JSONResponse({"detail": message}, status_code=401)
