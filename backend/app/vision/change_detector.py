"""Change detection against a clean-runway reference image.

Catches any new object, including classes the YOLO model was never trained on.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.vision.types import BBox


@dataclass
class ChangeParams:
    reference_frames: int = 15
    diff_threshold: int = 30
    blur_ksize: int = 3
    min_area_px: int = 60
    max_area_px: int = 6000
    learning_rate: float = 0.02
    global_change_frac: float = 0.45


@dataclass
class Region:
    bbox: BBox
    center: tuple[int, int]
    area: int
    large: bool
    score: float
    edge: bool = False


@dataclass
class ChangeResult:
    regions: list[Region] = field(default_factory=list)
    learning: bool = False
    global_change: bool = False
    fg_fraction: float = 0.0
    intrusion: bool = False


_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_BAND_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_GHOST_FRAMES = 3
_PROTECT_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
_SHIFT_KERNEL = np.ones((3, 3), np.uint8)
_RING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
# Small changes this close to a large one belong to it: a hand at the end of an arm,
# a piece of a vehicle. The sleeve over dark asphalt often doesn't show, so the hand looks detached.
INTRUSION_PX = 60
# A shadow or a lamp scales all colours by about the same factor inside and around a region.
SHADING_LOG_RATIO = 0.22


class ChangeDetector:
    def __init__(self, params: ChangeParams | None = None, mask: np.ndarray | None = None):
        self.params = params or ChangeParams()
        self.mask = mask
        self._reference: np.ndarray | None = None
        self._buffer: list[np.ndarray] = []
        self._last_img: np.ndarray | None = None
        self._ghosts: dict[tuple[int, int], int] = {}
        self._seen_ghosts: set[tuple[int, int]] = set()
        self._absorbed: list[BBox] = []
        self._mask_edge: np.ndarray | None = None
        if mask is not None:
            self.set_mask(mask)

    @property
    def ready(self) -> bool:
        return self._reference is not None

    @property
    def learned_frames(self) -> int:
        return self.params.reference_frames if self.ready else len(self._buffer)

    def reset_reference(self) -> None:
        """'Pista e curată acum': relearn the reference from the next frames."""
        self._reference = None
        self._buffer = []
        self._absorbed = []
        self._ghosts = {}

    def set_mask(self, mask: np.ndarray | None) -> None:
        self.mask = mask
        self._mask_edge = None if mask is None else cv2.subtract(mask, cv2.erode(mask, _SHIFT_KERNEL, iterations=2))

    def absorb(self, bbox: BBox) -> None:
        """Learn the current look of `bbox` as background (after a false alarm).

        The spot is remembered: if whatever was there is removed later, the empty
        asphalt would look like a new object (a 'ghost'), so it is checked for that.
        """
        if self._copy_to_reference(bbox):
            self._absorbed.append(bbox)

    def _copy_to_reference(self, bbox: BBox, pad: int = 6) -> bool:
        if self._reference is None or self._last_img is None:
            return False
        h, w = self._reference.shape[:2]
        x1, y1 = max(0, bbox.x - pad), max(0, bbox.y - pad)
        x2, y2 = min(w, bbox.x2 + pad), min(h, bbox.y2 + pad)
        self._reference[y1:y2, x1:x2] = self._last_img[y1:y2, x1:x2]
        return True

    def _in_absorbed(self, b: BBox, pad: int = 10) -> BBox | None:
        for a in self._absorbed:
            if b.x < a.x2 + pad and a.x - pad < b.x2 and b.y < a.y2 + pad and a.y - pad < b.y2:
                return a
        return None

    def reference_image(self) -> np.ndarray | None:
        return None if self._reference is None else cv2.convertScaleAbs(self._reference)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Colour is kept: a red clip on grey asphalt differs much more in one channel than in grey.
        k = max(1, self.params.blur_ksize) | 1
        return cv2.GaussianBlur(frame, (k, k), 0)

    @staticmethod
    def _gradient(patch: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    def _is_ghost(self, img: np.ndarray, contour: np.ndarray, b: BBox, pad: int = 4) -> bool:
        """Is this change a removed object ('ghost') rather than a new one?

        The outline of a new object shows up as an edge in the current frame; the outline
        of a removed one only exists in the reference. Comparing only along the outline
        keeps a uniform object lying over textured markings from looking like a ghost.
        """
        h, w = img.shape[:2]
        x1, y1 = max(0, b.x - pad), max(0, b.y - pad)
        x2, y2 = min(w, b.x2 + pad), min(h, b.y2 + pad)
        region = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.drawContours(region, [contour], -1, 255, -1, offset=(-x1, -y1))
        band = cv2.morphologyEx(region, cv2.MORPH_GRADIENT, _BAND_KERNEL) > 0
        if not band.any():
            return False
        cur = float(self._gradient(img[y1:y2, x1:x2])[band].mean())
        ref = float(self._gradient(cv2.convertScaleAbs(self._reference[y1:y2, x1:x2]))[band].mean())
        return ref > 1.8 * cur and ref - cur > 20

    def _shift_only(self, img: np.ndarray, inside: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """The camera moved by a pixel: edges of the markings shift, nothing new appears.

        Each pixel is compared with the range of the reference in its 3×3 neighbourhood,
        so a shifted edge still falls inside that range while a real object does not.
        """
        cur = img[y1:y2, x1:x2].astype(np.float32)
        ref = self._reference[y1:y2, x1:x2]
        out = np.maximum(cur - cv2.dilate(ref, _SHIFT_KERNEL), cv2.erode(ref, _SHIFT_KERNEL) - cur).clip(0)
        if out.ndim == 3:
            out = out.max(axis=2)
        return float(out[inside].mean()) < 0.5 * self.params.diff_threshold

    def _shading_only(self, img: np.ndarray, inside: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """A shadow or a light change, not an object.

        Light multiplies brightness: under a shadow the region and the asphalt around it
        darken by the same factor, in every colour channel. An object changes the region
        differently from its surroundings (or changes one colour more than the others).
        """
        ring = (cv2.dilate(inside.astype(np.uint8), _RING_KERNEL) > 0) & ~inside
        if ring.sum() < 20:
            return False
        ratio = (img[y1:y2, x1:x2].astype(np.float32) + 4) / (self._reference[y1:y2, x1:x2] + 4)
        if ratio.ndim == 2:
            ratio = ratio[:, :, None]
        k_in = np.median(ratio[inside], axis=0)
        k_ring = np.median(ratio[ring], axis=0)
        return bool(np.abs(np.log(k_in) - np.log(k_ring)).max() < SHADING_LOG_RATIO)

    def _arm_from_outside(self, fg: np.ndarray, mask: np.ndarray) -> list[BBox]:
        """Something big reaching from the edge of the picture up to the runway (an arm, a vehicle).

        It is looked for in the whole picture, because the part outside the runway is cut
        away below: on the runway only the hand may be left, with no sign of the arm.
        """
        h, w = fg.shape[:2]
        x, y, bw, bh = cv2.boundingRect(mask)
        x1, y1 = x - INTRUSION_PX, y - INTRUSION_PX
        x2, y2 = x + bw + INTRUSION_PX, y + bh + INTRUSION_PX
        small = cv2.resize(fg, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        n, _, stats, _ = cv2.connectedComponentsWithStats(small)
        found = []
        for cx, cy, cw, ch, area in stats[1:]:
            # Too small to be an arm, or so big it is the light changing on the whole table.
            if area * 16 < self.params.max_area_px // 2 or area > 0.2 * small.size:
                continue
            border = cx <= 0 or cy <= 0 or cx + cw >= small.shape[1] or cy + ch >= small.shape[0]
            bx1, by1, bx2, by2 = cx * 4, cy * 4, (cx + cw) * 4, (cy + ch) * 4
            if border and bx1 < x2 and x1 < bx2 and by1 < y2 and y1 < by2:
                found.append(BBox(x=int(bx1), y=int(by1), w=int(bx2 - bx1), h=int(by2 - by1)))
        return found

    def _confirm_ghost(self, center: tuple[int, int]) -> bool:
        """Absorb a ghost only after it was seen in the same place for a few frames."""
        for key in list(self._ghosts):
            if abs(key[0] - center[0]) <= 10 and abs(key[1] - center[1]) <= 10:
                self._ghosts[key] += 1
                self._seen_ghosts.add(key)
                if self._ghosts[key] >= _GHOST_FRAMES:
                    del self._ghosts[key]
                    return True
                return False
        self._ghosts[center] = 1
        self._seen_ghosts.add(center)
        return False

    def detect(self, frame: np.ndarray, protect: list[BBox] | None = None) -> ChangeResult:
        p = self.params
        img = self.preprocess(frame)
        self._last_img = img

        if self._reference is not None and self._reference.shape != img.shape:
            self.reset_reference()
        if self._reference is None:
            self._buffer.append(img)
            if len(self._buffer) >= p.reference_frames:
                self._reference = np.median(np.stack(self._buffer), axis=0).astype(np.float32)
                self._buffer = []
            return ChangeResult(learning=True)

        mask = self.mask
        if mask is not None and mask.shape != img.shape[:2]:
            mask = None

        diff = cv2.absdiff(img, cv2.convertScaleAbs(self._reference))
        if diff.ndim == 3:
            diff = diff.max(axis=2)
        _, fg = cv2.threshold(diff, p.diff_threshold, 255, cv2.THRESH_BINARY)
        arms: list[BBox] = []
        if mask is not None:
            arms = self._arm_from_outside(fg, mask)
            fg = cv2.bitwise_and(fg, mask)
        # Closing joins the pieces of one object. No opening: it would erase thin objects
        # (a paper clip is 1–2 px wide); speckle noise is removed by MIN_AREA_PX and persistence.
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _CLOSE_KERNEL)

        mask_area = cv2.countNonZero(mask) if mask is not None else diff.size
        fg_fraction = cv2.countNonZero(fg) / max(1, mask_area)
        if fg_fraction > p.global_change_frac:
            # Lighting or camera change, not an object: don't report, don't learn it as background.
            return ChangeResult(global_change=True, fg_fraction=fg_fraction)

        regions: list[Region] = []
        self._seen_ghosts = set()
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            if mask is not None and mask[cy, cx] == 0:
                continue
            roi = fg[y : y + h, x : x + w]
            area = cv2.countNonZero(roi)
            if area < p.min_area_px:
                continue
            bbox = BBox(x=x, y=y, w=w, h=h)
            fh, fw = fg.shape[:2]
            # Cut by the frame edge: can't be measured, and it's usually something entering the picture.
            border = x <= 1 or y <= 1 or x + w >= fw - 1 or y + h >= fh - 1
            large = area > p.max_area_px or border
            # Touching the edge of the calibrated runway: an arm or a vehicle coming onto it.
            edge = border or (
                mask is not None and self._mask_edge is not None
                and cv2.countNonZero(cv2.bitwise_and(roi, self._mask_edge[y : y + h, x : x + w])) > 0
            )
            if not large:
                pad = 8
                x1, y1, x2, y2 = max(0, x - pad), max(0, y - pad), min(fw, x + w + pad), min(fh, y + h + pad)
                inside = np.zeros((y2 - y1, x2 - x1), np.uint8)
                cv2.drawContours(inside, [contour], -1, 1, -1, offset=(-x1, -y1))
                inside = inside > 0
                if self._shift_only(img, inside, x1, y1, x2, y2) or self._shading_only(img, inside, x1, y1, x2, y2):
                    continue
            absorbed = self._in_absorbed(bbox) if not large else None
            # Ghost check only where something was learned into the reference on purpose:
            # elsewhere an asphalt-coloured object on the markings would look the same, and
            # a missed object is worse than a false alarm.
            if absorbed is not None and self._is_ghost(img, contour, bbox):
                if self._confirm_ghost((cx, cy)):
                    self._copy_to_reference(bbox)
                    self._absorbed.remove(absorbed)
                continue
            mean_diff = float(cv2.mean(diff[y : y + h, x : x + w], mask=roi)[0])
            regions.append(
                Region(
                    bbox=bbox,
                    center=(cx, cy),
                    area=area,
                    large=large,
                    score=round(min(1.0, mean_diff / (p.diff_threshold * 3)), 3),
                    edge=edge,
                )
            )

        big = [r.bbox for r in regions if r.large]
        if big:
            regions = [r for r in regions if r.large or not any(_near(r.bbox, b, INTRUSION_PX) for b in big)]

        # Ghost candidates not seen in this frame are forgotten.
        self._ghosts = {k: v for k, v in self._ghosts.items() if k in self._seen_ghosts}

        # Slow reference update, only outside changed areas and known objects,
        # so lighting drift is absorbed but an object never fades into the background.
        keep = cv2.dilate(fg, _PROTECT_KERNEL)
        # The arm is kept out of the reference too, or it would leave a trace on the table.
        for b in [*(protect or []), *arms]:
            cv2.rectangle(keep, (b.x - 8, b.y - 8), (b.x2 + 8, b.y2 + 8), 255, -1)
        update_mask = cv2.bitwise_not(keep)
        cv2.accumulateWeighted(img, self._reference, p.learning_rate, mask=update_mask)

        return ChangeResult(regions=regions, fg_fraction=fg_fraction, intrusion=bool(arms))


def _near(a: BBox, b: BBox, gap: int) -> bool:
    return a.x < b.x2 + gap and b.x - gap < a.x2 and a.y < b.y2 + gap and b.y - gap < a.y2
