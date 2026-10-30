"""Drive a headless Edge/Chrome through the DevTools protocol: open pages, run JS, save screenshots.

Used to check the pages and to capture slides without extra dependencies
(`websockets` already comes with uvicorn[standard]). Pages with live MJPEG/SSE never finish
loading, so plain `--screenshot` hangs; this waits a fixed time instead.

    python scripts/browser_check.py http://127.0.0.1:8000/ panel.png --wait 4
    python scripts/browser_check.py URL OUT.png --width 412 --height 915 --mobile --eval "document.title"
"""

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


class Browser:
    def __init__(self, width: int, height: int, port: int = 9333):
        exe = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("msedge") or shutil.which("chrome")
        if not exe:
            raise SystemExit("Edge/Chrome not found")
        self.port = port
        self.profile = tempfile.mkdtemp(prefix="re-cdp-")
        self.proc = subprocess.Popen(
            [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--mute-audio",
             "--autoplay-policy=no-user-gesture-required", f"--remote-debugging-port={port}",
             f"--user-data-dir={self.profile}", f"--window-size={width},{height}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._id = 0

    def ws_url(self) -> str:
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list") as r:
                    pages = [p for p in json.load(r) if p.get("type") == "page"]
                if pages:
                    return pages[0]["webSocketDebuggerUrl"]
            except OSError:
                pass
            time.sleep(0.2)
        raise SystemExit("DevTools did not start")

    async def __aenter__(self):
        self.ws = await websockets.connect(self.ws_url(), max_size=50_000_000)
        await self.cmd("Page.enable")
        await self.cmd("Runtime.enable")
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()
        self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)

    async def cmd(self, method: str, **params):
        self._id += 1
        my_id = self._id
        await self.ws.send(json.dumps({"id": my_id, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == my_id:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    async def goto(self, url: str, wait: float) -> None:
        await self.cmd("Page.navigate", url=url)
        await asyncio.sleep(wait)

    async def eval(self, expression: str):
        r = await self.cmd(
            "Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True, userGesture=True
        )
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r["exceptionDetails"]))
        return r.get("result", {}).get("value")

    async def screenshot(self, out: Path) -> None:
        r = await self.cmd("Page.captureScreenshot", format="png")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(r["data"]))

    async def mobile(self, width: int, height: int) -> None:
        await self.cmd("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=2, mobile=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--wait", type=float, default=3.0)
    ap.add_argument("--mobile", action="store_true")
    ap.add_argument("--eval", action="append", default=[], help="JS to run after load (printed)")
    args = ap.parse_args()
    async with Browser(args.width, args.height) as b:
        if args.mobile:
            await b.mobile(args.width, args.height)
        await b.goto(args.url, args.wait)
        for expr in args.eval:
            print(json.dumps(await b.eval(expr), ensure_ascii=False))
        if args.out:
            await b.screenshot(Path(args.out))
            print(f"saved {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
