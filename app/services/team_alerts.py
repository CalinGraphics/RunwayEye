"""Team (phone) page helpers: its address on the local network and the QR code."""

import io
import socket

import qrcode
import qrcode.image.svg


def lan_ip() -> str:
    """IP of this laptop on the local network (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
    except OSError:
        ip = ""
    finally:
        s.close()
    if not ip or ip.startswith("127."):
        try:
            candidates = socket.gethostbyname_ex(socket.gethostname())[2]
            ip = next((c for c in candidates if not c.startswith("127.")), "127.0.0.1")
        except OSError:
            ip = "127.0.0.1"
    return ip


def team_url(port: int) -> str:
    return f"http://{lan_ip()}:{port}/team"


def qr_svg(text: str) -> bytes:
    img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathFillImage, box_size=12, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue()

