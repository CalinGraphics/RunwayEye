"""Team (phone) page helpers: its address on the local network and the QR code."""

import io
import socket
import time

import qrcode
import qrcode.image.svg


# Windows "Mobile hotspot" always gives the laptop this network: the phone talks straight to the
# laptop, with no router in between that could isolate the devices (common on campus/office Wi-Fi).
HOTSPOT_PREFIX = "192.168.137."


def _default_route_ip() -> str:
    """IP of the network that carries the internet route (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.254.254.254", 1))
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


def lan_ips() -> list[str]:
    """Every IPv4 address of this laptop a phone could reach, the most likely first:
    the laptop's own hotspot, then the network with the internet route, then the rest."""
    main = _default_route_ip()
    try:
        others = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        others = []
    ips = [ip for ip in dict.fromkeys([main, *others]) if ip and not ip.startswith(("127.", "169.254."))]
    ips.sort(key=lambda ip: (not ip.startswith(HOTSPOT_PREFIX), ip != main))
    return ips or ["127.0.0.1"]


def address_label(ip: str) -> str:
    if ip.startswith(HOTSPOT_PREFIX):
        return "Hotspotul laptopului"
    if ip == _default_route_ip():
        return "Rețeaua Wi-Fi / cablu"
    return "Altă rețea"


_cache: tuple[list[str], float] = ([], 0.0)


def cached_ips() -> list[str]:
    # Polled every 2 s by each open dashboard; a hotspot turned on mid-demo shows up within 10 s.
    global _cache
    ips, at = _cache
    if not ips or time.monotonic() - at > 10:
        _cache = (lan_ips(), time.monotonic())
    return _cache[0]


def team_url(port: int, key: str = "", ip: str | None = None) -> str:
    ips = cached_ips()
    host = ip if ip in ips else ips[0]
    return f"http://{host}:{port}/team" + (f"?key={key}" if key else "")


def team_urls(port: int, key: str = "") -> list[dict]:
    return [{"ip": ip, "label": address_label(ip), "url": team_url(port, key, ip)} for ip in cached_ips()]


def qr_svg(text: str) -> bytes:
    img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathFillImage, box_size=12, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue()

