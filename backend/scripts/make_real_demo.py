"""Real-scale demo clip: a real runway photo, real FOD photos laid flat on its asphalt with the
photo's own perspective, and a real person who walks on, stands, and walks off.

The camera comes from the photo itself. On the 1280x720 frame every line along the runway meets
at the vanishing point (620, 322). The centreline runs through the centre markings, the right edge
line is the long solid one, and the runway is 45 m wide, which puts the camera about 4.7 m above
the asphalt; with a ~64 degree lens the bottom of the frame is 12 m away. Objects are placed in
metres and projected, so a 33 cm hammer at 13 m gets the pixels a real camera would give it,
squashed along the runway the way a low camera squashes it. The can and the bolt are there on
purpose: at this height they are a few pixels, near the limit of what can be seen.

Next to the clip it writes:
  real_runway_demo.scene.json  the stretch of runway 12-72 m ahead (60 m x 45 m) and its corners: picking
                               the clip on "Sursă video" switches runway and calibration on its own
  real_runway_demo.truth.json  where each object really is and how big, to check the detector

    python backend/scripts/make_real_demo.py
Needs data/videos/assets/runway_magda_ehlers_9873573.jpg (pexels.com/photo/9873573) and
person_muneeb_babar_22427879.jpg (pexels.com/photo/22427879), both under the free Pexels licence.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.vision.calibration import CORNER_NAMES, RunwayGeometry  # noqa: E402

ASSETS = ROOT / "data/videos/assets"
SAMPLES = ROOT / "samples/fod-a-test"
W, H, FPS, SECONDS = 1280, 720, 15, 48

# Camera, measured on the photo (see the module docstring).
VP = (620.0, 322.0)
# x pixels per y pixel below the horizon: the right edge line, and the centreline through the
# centre markings. (The long line on the left is a parallel strip, not the runway edge.)
RIGHT_SLOPE, CENTER_SLOPE = 5.23, 0.43
WIDTH_M = 45.0
K = 2 * (RIGHT_SLOPE - CENTER_SLOPE) / WIDTH_M  # lateral px per metre, per pixel below the horizon
FOCAL = 1024.0  # ~64 degree horizontal field of view; sets the distances, not the widths

# The stretch of runway the scene calls "the runway": from 12 m (the bottom of the frame) to 72 m
# ahead of the camera, in 10 m sectors. Further away a camera this low sees only large things.
NEAR_M, LENGTH_M, SECTORS_ALONG = 12.0, 60.0, 6
GEOMETRY = RunwayGeometry(length_m=LENGTH_M, width_m=WIDTH_M, sectors_along=SECTORS_ALONG, sectors_across=3)

# (sample photo, label, real size in cm, metres ahead, metres right of the centreline, rotation, appears at s)
# The can and the bolt sit at the limit of what 3 m of height can show, on purpose.
OBJECTS = [
    ("metal_sheet_013763.jpg", "tablă", 40, 15.0, -8.0, 20, 6.0),
    ("hammer_019119.jpg", "ciocan", 33, 13.5, 2.0, 60, 11.0),
    ("luggage_part_020940.jpg", "bucată de bagaj", 25, 14.0, 5.0, -15, 16.0),
    ("soda_can_029822.jpg", "doză", 12, 13.0, -4.0, 0, 28.0),
    ("bolt_012456.jpg", "șurub", 8, 13.0, 3.5, 35, 32.0),
]
PERSON_HEIGHT_M = 1.75
PERSON_Z, PERSON_STAND_X, PERSON_OFF_X = 35.0, 7.0, 25.0
PERSON_IN, PERSON_STAND, PERSON_OUT, PERSON_GONE = 19.0, 25.0, 35.0, 41.0


def project(z: float, x: float) -> tuple[float, float]:
    """Point on the runway `z` m ahead and `x` m right of the centreline -> pixel."""
    dy = FOCAL / (K * z)
    return VP[0] + (CENTER_SLOPE + K * x) * dy, VP[1] + dy


def px_per_m(z: float) -> float:
    return FOCAL / z


def matte(photo: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """BGRA cut-out of the object inside `rect`, from GrabCut, cropped to the object."""
    mask = np.zeros(photo.shape[:2], np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(photo, mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    alpha = cv2.GaussianBlur(cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), (3, 3), 0)
    ys, xs = np.nonzero(alpha > 128)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return np.dstack([photo[y0:y1, x0:x1], alpha[y0:y1, x0:x1]])


def fod_cutout(name: str, boxes: dict) -> np.ndarray:
    photo = cv2.imread(str(SAMPLES / name))
    b = boxes[name]
    pad = 6
    x, y = max(0, b["x"] - pad), max(0, b["y"] - pad)
    w = min(photo.shape[1] - x, b["w"] + 2 * pad)
    h = min(photo.shape[0] - y, b["h"] + 2 * pad)
    return matte(photo, (x, y, w, h))


def person_cutout() -> np.ndarray:
    photo = cv2.imread(str(ASSETS / "person_muneeb_babar_22427879.jpg"))
    return matte(photo, (905, 510, 120, 285))


def blend(frame: np.ndarray, layer: np.ndarray, x0: int, y0: int) -> None:
    """Alpha-blend a BGRA layer whose top-left corner is at (x0, y0), clipped to the frame."""
    h, w = layer.shape[:2]
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    part = layer[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    a = part[:, :, 3:4].astype(np.float32) / 255
    roi = frame[fy0:fy1, fx0:fx1]
    roi[:] = roi * (1 - a) + part[:, :, :3] * a


def lay_flat(frame: np.ndarray, cut: np.ndarray, size_cm: float, z: float, x: float, angle: float) -> np.ndarray:
    """Lay the cut-out on the runway: its long side is `size_cm`, centred at (z, x). Returns the image quad."""
    h, w = cut.shape[:2]
    long_m = size_cm / 100
    a_m, b_m = (long_m, long_m * h / w) if w >= h else (long_m * w / h, long_m)
    t = math.radians(angle)
    # Cut-out corners in metres around the centre (x right, z ahead), rotated on the ground.
    pts = []
    for dx, dz in [(-a_m / 2, b_m / 2), (a_m / 2, b_m / 2), (a_m / 2, -b_m / 2), (-a_m / 2, -b_m / 2)]:
        rx, rz = dx * math.cos(t) - dz * math.sin(t), dx * math.sin(t) + dz * math.cos(t)
        pts.append(project(z + rz, x + rx))
    quad = np.array(pts, np.float32)
    x0, y0 = np.floor(quad.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int) + 1
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    m = cv2.getPerspectiveTransform(src, (quad - np.array([x0, y0], np.float32)).astype(np.float32))
    layer = cv2.warpPerspective(cut, m, (x1 - x0, y1 - y0), flags=cv2.INTER_AREA)
    # The photos were taken in brighter light than this overcast runway.
    layer[:, :, :3] = (layer[:, :, :3] * 0.8).astype(np.uint8)
    blend(frame, layer, x0, y0)
    return quad


def stand(frame: np.ndarray, cut: np.ndarray, z: float, x: float, bob: float = 0.0) -> tuple[int, int, int, int]:
    """Stand the person with their feet at (z, x). Returns their box (x, y, w, h)."""
    scale = PERSON_HEIGHT_M * px_per_m(z) / cut.shape[0]
    layer = cv2.resize(cut, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    fx, fy = project(z, x)
    h, w = layer.shape[:2]
    x0, y0 = int(round(fx - w / 2)), int(round(fy - h + bob))
    # A soft shadow under the feet, so they touch the ground.
    shadow = np.zeros((max(4, h // 6), max(6, int(w * 1.6)), 4), np.uint8)
    cv2.ellipse(shadow, (shadow.shape[1] // 2, shadow.shape[0] // 2),
                (shadow.shape[1] // 2 - 1, shadow.shape[0] // 2 - 1), 0, 0, 360, (20, 20, 20, 90), -1)
    shadow = cv2.GaussianBlur(shadow, (5, 5), 0)
    blend(frame, shadow, int(fx - shadow.shape[1] / 2 + w * 0.15), int(fy - shadow.shape[0] / 2))
    blend(frame, layer, x0, y0)
    return x0, y0, w, h


def person_x(t: float) -> float | None:
    if t < PERSON_IN or t >= PERSON_GONE:
        return None
    if t < PERSON_STAND:
        f = (t - PERSON_IN) / (PERSON_STAND - PERSON_IN)
        return PERSON_OFF_X + (PERSON_STAND_X - PERSON_OFF_X) * f
    if t < PERSON_OUT:
        return PERSON_STAND_X
    f = (t - PERSON_OUT) / (PERSON_GONE - PERSON_OUT)
    return PERSON_STAND_X + (PERSON_OFF_X - PERSON_STAND_X) * f


def runway_point(z: float, x: float) -> dict:
    u, v = z - NEAR_M, x + WIDTH_M / 2
    return {"distance_m": round(u, 1), "offset_m": round(x, 1), "sector": GEOMETRY.sector(u, v)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/videos/real_runway_demo.mp4")
    args = ap.parse_args()
    out = ROOT / args.out

    photo = cv2.imread(str(ASSETS / "runway_magda_ehlers_9873573.jpg"))
    if photo is None:
        sys.exit(f"Missing {ASSETS / 'runway_magda_ehlers_9873573.jpg'} (pexels.com/photo/9873573)")
    background = cv2.resize(photo[photo.shape[0] - 1080:], (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    boxes = json.loads((SAMPLES / "boxes.json").read_text(encoding="utf-8"))
    cutouts = {name: fod_cutout(name, boxes) for name, *_ in OBJECTS}
    person = person_cutout()

    rng = np.random.default_rng(7)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".part.mp4")
    writer = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(SECONDS * FPS):
        t = i / FPS
        frame = background.copy()
        for name, _, size, z, x, angle, start in OBJECTS:
            if t >= start:
                lay_flat(frame, cutouts[name], size, z, x, angle)
        px = person_x(t)
        if px is not None:
            walking = t < PERSON_STAND or t >= PERSON_OUT
            stand(frame, person, PERSON_Z, px, bob=1.5 * math.sin(t * 9) if walking else 0.0)
        # Sensor noise: a photo is perfectly still, a camera is not.
        frame += rng.normal(0, 1.2, frame.shape).astype(np.float32)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()
    part.replace(out)

    corners = [project(NEAR_M, -WIDTH_M / 2), project(NEAR_M, WIDTH_M / 2),
               project(NEAR_M + LENGTH_M, WIDTH_M / 2), project(NEAR_M + LENGTH_M, -WIDTH_M / 2)]
    # Near the camera the runway is wider than the frame: the watched area is the runway cut at
    # the frame's edges, while the corners (off the picture) still give the metres.
    frame_rect = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    _, visible = cv2.intersectConvexConvex(np.array(corners, np.float32), frame_rect)
    outline = visible.reshape(-1, 2).tolist()
    # Same direction as the corners (from the near-left corner around), whatever OpenCV returned.
    near_left = min(range(len(outline)), key=lambda i: (outline[i][0] - corners[0][0]) ** 2 + (outline[i][1] - corners[0][1]) ** 2)
    outline = outline[near_left:] + outline[:near_left]
    if cv2.contourArea(np.array(outline, np.float32), oriented=True) * cv2.contourArea(np.array(corners, np.float32), oriented=True) < 0:
        outline = [outline[0]] + outline[1:][::-1]
    scene = {
        "about": "Pexels photo 9873573 seen from ~4.7 m: the runway here is the stretch 12-72 m ahead of the camera",
        "runway": {"length_m": LENGTH_M, "width_m": WIDTH_M, "sectors_along": SECTORS_ALONG, "sectors_across": 3},
        "corners": [[round(x, 1), round(y, 1)] for x, y in corners],
        "outline": [[round(x, 1), round(y, 1)] for x, y in outline],
        "corner_names": CORNER_NAMES,
        "frame_size": [W, H],
        # Objects here are a few pixels across; the mock-up's thresholds (60 px, contrast 30) miss them.
        "tuning": {"min_area_px": 10, "diff_threshold": 15},
    }
    out.with_suffix(".scene.json").write_text(json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8")
    truth = [{"label": label, "size_cm": size, "appears_s": start, **runway_point(z, x)}
             for _, label, size, z, x, _, start in OBJECTS]
    truth.append({"label": "persoană", "size_cm": PERSON_HEIGHT_M * 100, "appears_s": PERSON_STAND,
                  **runway_point(PERSON_Z, PERSON_STAND_X)})
    out.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{out.relative_to(ROOT)}: {SECONDS} s, {W}x{H} @ {FPS} fps, with .scene.json and .truth.json")


if __name__ == "__main__":
    main()
