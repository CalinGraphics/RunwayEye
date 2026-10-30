"""Generates drawn test scenarios with their exact answers, for evaluate.py.

Every clip is a mock-up runway drawn by code, so we know when and where each object lands.
They test the detection logic under hard conditions (light changes, shadows, a vehicle,
camera noise, tiny or dark objects). They are NOT the numbers for the jury: those come
from real clips of the mock-up (see "Cum demonstrăm că merge" in README.md).

    python scripts/make_eval_set.py                     # writes data/eval_synthetic/
    python scripts/evaluate.py --dir data/eval_synthetic --no-yolo
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from make_demo_clip import FPS, H, ROOT, W, background, draw_bolt, draw_clip, draw_hand, hand_path


def draw_nut(img, cx, cy):
    pts = np.array([[cx + 9 * np.cos(a), cy + 9 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 7)[:-1]], np.int32)
    cv2.fillPoly(img, [pts], (175, 178, 182))
    cv2.circle(img, (cx, cy), 3, (70, 72, 75), -1)


def draw_wrench(img, cx, cy):
    cv2.line(img, (cx - 26, cy + 8), (cx + 18, cy - 6), (190, 195, 200), 5)
    cv2.circle(img, (cx + 22, cy - 8), 8, (190, 195, 200), -1)
    cv2.circle(img, (cx + 25, cy - 11), 4, (72, 74, 78), -1)


def draw_washer(img, cx, cy):
    cv2.circle(img, (cx, cy), 6, (200, 200, 205), -1)
    cv2.circle(img, (cx, cy), 2, (72, 74, 78), -1)


def draw_rubber(img, cx, cy):
    # Dark piece of tyre rubber: low contrast on asphalt.
    cv2.ellipse(img, (cx, cy), (16, 9), 20, 0, 360, (38, 38, 40), -1)


DRAW = {"surub": draw_bolt, "plastic": draw_clip, "piulita": draw_nut, "cheie": draw_wrench,
        "metal": draw_washer, "necunoscut": draw_rubber}
SIZE = {"surub": (40, 18), "plastic": (36, 16), "piulita": (20, 20), "cheie": (60, 30), "metal": (14, 14), "necunoscut": (34, 22)}


@dataclass
class Drop:
    label: str
    at: float
    pos: tuple[int, int]
    picked_up: float | None = None

    @property
    def released(self) -> float:
        # The hand reaches the runway 1.5 s after it appears, lets go and stays above it for 1 s.
        return self.at + 1.5

    def zone(self) -> dict:
        w, h = SIZE[self.label]
        return {"x": self.pos[0] - w // 2, "y": self.pos[1] - h // 2, "w": w, "h": h}


@dataclass
class Scenario:
    name: str
    about: str
    seconds: int
    drops: list[Drop] = field(default_factory=list)
    brightness: Callable[[float], float] = lambda t: 0.0
    noise: float = 1.5
    jitter: int = 0
    extras: list[Callable] = field(default_factory=list)


def shadow(start: float, duration: float):
    def draw(frame, t):
        k = (t - start) / duration
        if 0 <= k <= 1:
            x = int(-150 + (W + 300) * k)
            mask = np.zeros((H, W), np.float32)
            cv2.ellipse(mask, (x, 280), (130, 70), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (61, 61), 0)[:, :, None]
            frame[:] = (frame * (1 - 0.35 * mask)).astype(frame.dtype)
    return draw


def vehicle(start: float, stop_at: float = 6.0, stop_for: float = 5.0):
    def draw(frame, t):
        local = t - start
        if local < 0:
            return
        travel = 60.0
        if local < stop_at:
            x = local * travel
        elif local < stop_at + stop_for:
            x = stop_at * travel
        else:
            x = stop_at * travel + (local - stop_at - stop_for) * travel
        x = int(x) - 160
        if x > W:
            return
        cv2.rectangle(frame, (x, 230), (x + 150, 300), (40, 150, 235), -1)
        cv2.rectangle(frame, (x + 100, 238), (x + 140, 292), (30, 40, 50), -1)
    return draw


def hand_only(start: float, pos: tuple[int, int]):
    def draw(frame, t):
        p = hand_path(t, start, pos)
        if p is not None:
            draw_hand(frame, *p)
    return draw


def scenarios() -> list[Scenario]:
    return [
        Scenario("s01_baza", "două obiecte lăsate de mână, lumină constantă", 45,
                 [Drop("surub", 8, (330, 230)), Drop("plastic", 25, (640, 330))]),
        Scenario("s02_obiecte_mici", "piuliță și șaibă de ~15 px", 45,
                 [Drop("piulita", 8, (420, 210)), Drop("metal", 25, (700, 340))]),
        Scenario("s03_contrast_mic", "bucată de cauciuc închisă la culoare pe asfalt", 40,
                 [Drop("necunoscut", 10, (500, 250))]),
        Scenario("s04_lumina_lenta", "lumina crește lent cu ~30 de niveluri", 60,
                 [Drop("cheie", 30, (450, 320))], brightness=lambda t: 30 * t / 60),
        Scenario("s05_lumina_brusca", "se aprinde o lampă (salt de lumină), apoi cade un obiect", 50,
                 [Drop("surub", 30, (380, 330))], brightness=lambda t: 40.0 if t >= 15 else 0.0),
        Scenario("s06_umbra", "o umbră traversează pista, apoi cade un obiect", 45,
                 [Drop("piulita", 28, (600, 220))], extras=[shadow(8, 10)]),
        Scenario("s07_vehicul", "un vehicul traversează și oprește 5 s pe pistă", 50,
                 [Drop("plastic", 35, (300, 360))], extras=[vehicle(5)]),
        Scenario("s08_multe", "patru obiecte la momente diferite, unul ridicat după 10 s", 70,
                 [Drop("surub", 6, (200, 220)), Drop("piulita", 18, (480, 360), picked_up=32),
                  Drop("cheie", 40, (650, 230)), Drop("plastic", 55, (800, 320))]),
        Scenario("s09_zgomot", "cameră cu zgomot mare și tremur de 1 px", 45,
                 [Drop("surub", 10, (400, 250)), Drop("cheie", 28, (700, 330))], noise=5.0, jitter=1),
        Scenario("s10_fara_obiecte", "2 minute fără obiecte; o mână trece de două ori fără să lase nimic", 120,
                 [], extras=[hand_only(20, (450, 280)), hand_only(70, (650, 250))]),
    ]


def render(sc: Scenario, out_dir: Path, rng: np.random.Generator) -> None:
    path = out_dir / f"{sc.name}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        sys.exit(f"Nu pot scrie {path}")
    base = background(rng)
    for i in range(sc.seconds * FPS):
        t = i / FPS
        img = base + sc.brightness(t) + rng.normal(0, sc.noise, base.shape).astype(np.float32)
        frame = img.clip(0, 255).astype(np.uint8)
        for d in sc.drops:
            draw = DRAW[d.label]
            on_runway = t >= d.released and (d.picked_up is None or t < d.picked_up)
            if on_runway:
                draw(frame, *d.pos)
            p = hand_path(t, d.at, d.pos)
            if p is not None:
                if t < d.released:
                    draw(frame, int(p[0]), int(p[1]))
                draw_hand(frame, *p)
            if d.picked_up is not None:
                p = hand_path(t, d.picked_up - 2.0, d.pos)
                if p is not None:
                    draw_hand(frame, *p)
        for extra in sc.extras:
            extra(frame, t)
        if sc.jitter:
            dx, dy = rng.integers(-sc.jitter, sc.jitter + 1, 2)
            frame = np.roll(frame, (int(dy), int(dx)), axis=(0, 1))
        writer.write(frame)
    writer.release()

    ann = {
        "clip": path.name,
        "synthetic": True,
        "about": sc.about,
        "frame_width": W,
        # Runway corners in calibration order: threshold 09 left, 09 right, 27 right, 27 left.
        "corners": [[90, 150], [90, 400], [870, 400], [870, 150]],
        "objects": [{"label": d.label, "appear_s": d.released, **({"disappear_s": d.picked_up} if d.picked_up else {}),
                     "zone": d.zone()} for d in sc.drops],
    }
    (out_dir / f"{sc.name}.json").write_text(json.dumps(ann, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {path.name}  {sc.seconds} s · {len(sc.drops)} obiecte · {sc.about}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval_synthetic")
    ap.add_argument("--only", help="doar scenariile care conțin acest text, ex. s05")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    chosen = [s for s in scenarios() if not args.only or args.only in s.name]
    print(f"Scriu {len(chosen)} scenarii în {out}:")
    for sc in chosen:
        render(sc, out, rng)


if __name__ == "__main__":
    main()
