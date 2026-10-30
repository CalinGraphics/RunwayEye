from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi.templating import Jinja2Templates

from app.config import ROOT_DIR, get_settings

templates = Jinja2Templates(directory=ROOT_DIR / "app" / "templates")

try:
    LOCAL_TZ = ZoneInfo("Europe/Bucharest")
except Exception:  # tzdata missing
    LOCAL_TZ = timezone.utc


def _local(dt) -> datetime | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LOCAL_TZ)


def fmt_time(dt) -> str:
    d = _local(dt)
    return d.strftime("%H:%M:%S") if d else "—"


def fmt_datetime(dt) -> str:
    d = _local(dt)
    return d.strftime("%d.%m %H:%M:%S") if d else "—"


def fmt_seconds(secs: float) -> str:
    m, s = divmod(max(0, int(secs)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_duration(start, end) -> str:
    a, b = _local(start), _local(end)
    if not a or not b:
        return "—"
    return fmt_seconds((b - a).total_seconds())


def asset_version() -> str:
    css = ROOT_DIR / "app" / "static" / "css" / "app.css"
    return str(int(css.stat().st_mtime)) if css.exists() else "0"


templates.env.filters["time"] = fmt_time
templates.env.filters["datetime"] = fmt_datetime
templates.env.globals["duration"] = fmt_duration
templates.env.globals["settings"] = get_settings()
templates.env.globals["asset_version"] = asset_version
