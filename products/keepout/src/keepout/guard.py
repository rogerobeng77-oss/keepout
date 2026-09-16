"""Is the fixed guard still there? Answered by comparing the guard region to a
reference frame captured when the operator confirmed the guard was in place.

Machine Guarding (29 CFR 1910.212) is on OSHA's top-ten most-cited list every
year, and the HSE conveyor case that this product is named for turns on the
sentence "There was no guard in place". A removed or swung-open guard is a static
change in a fixed camera's view, which is one of the few things a classical
method does genuinely well.

Two independent comparisons, because each fails differently:

* **Structure** — normalised correlation of Canny edge maps. A guard is mostly
  straight edges: mesh, a frame, a hinge line. Remove it and the edge structure
  in that region changes completely.

  Edges are computed **after CLAHE local contrast equalisation**, and that is not
  a flourish. A test in `tests/test_guard.py` dimmed the scene by 38% with the
  guard still bolted in place: raw Canny edge correlation collapsed from 1.00 to
  0.19, because the gradients fell under the fixed Canny thresholds, and the
  checker called a present guard missing. The received wisdom that edges are
  lighting-invariant is only true if the contrast survives the thresholding.
  CLAHE restores local contrast first, so the same test now passes.
* **Appearance** — zero-mean normalised cross-correlation of the greyscale patch,
  the same quantity `cv2.matchTemplate` computes for `TM_CCOEFF_NORMED`, done
  here over an arbitrary mask so occluded pixels can be excluded rather than
  averaged in. This catches a guard that is present but swung open, where the
  edges may still be busy but are in the wrong place.

They are combined by taking the **weaker** of the two. A guard is called present
only when both agree it is there. Asymmetric on purpose: a false "guard missing"
costs somebody walking over to look; a false "guard present" costs an arm.

Occlusion is handled explicitly. A worker standing in front of the guard makes
both scores collapse, and calling that "guard removed" every time anybody walks
past would destroy trust in the alert. When the person boxes cover more than
`max_occluded_fraction` of the guard region, the check returns `UNKNOWN` and the
last confident state is held.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

from .zones import Zone


class GuardStatus(str, Enum):  # noqa: UP042 - the str mixin is what puts these straight into JSON
    PRESENT = "present"
    MISSING = "missing"
    UNKNOWN = "unknown"  # occluded, or no reference learned yet


@dataclass
class GuardConfig:
    edge_threshold: float = 0.55  # below this the edge structure no longer matches
    appearance_threshold: float = 0.55
    max_occluded_fraction: float = 0.35
    min_state_ms: float = 1500.0  # a change must persist this long before we report it
    canny_low: int = 60
    canny_high: int = 160
    blur_ksize: int = 3
    work_width: int = 480
    clahe_clip: float = 2.5
    clahe_tiles: int = 8


@dataclass
class GuardReference:
    """What a guard looked like when somebody confirmed it was in place."""

    zone: Zone
    edges: np.ndarray
    grey: np.ndarray
    size: tuple[int, int]  # (width, height) of the work image the reference was built at
    captured_ms: float = 0.0
    edge_density: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone.to_dict(),
            "size": list(self.size),
            "captured_ms": self.captured_ms,
            "edge_density": round(self.edge_density, 4),
        }


@dataclass
class GuardObservation:
    status: GuardStatus
    edge_score: float
    appearance_score: float
    score: float
    occluded_fraction: float
    since_ms: float | None = None
    note: str = ""

    @property
    def missing(self) -> bool:
        return self.status is GuardStatus.MISSING

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "edge_score": round(self.edge_score, 4),
            "appearance_score": round(self.appearance_score, 4),
            "score": round(self.score, 4),
            "occluded_fraction": round(self.occluded_fraction, 3),
            "since_ms": self.since_ms,
            "note": self.note,
        }


def _work(frame: np.ndarray, width: int, blur: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(1.0, width / float(w)) if w > 0 else 1.0
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur > 1:
        grey = cv2.GaussianBlur(grey, (blur, blur), 0)
    return grey, scale


def _edges(grey: np.ndarray, cfg: GuardConfig) -> np.ndarray:
    """Canny over a locally contrast-equalised image, so a dimmer is not a removed guard."""
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tiles,) * 2)
    return cv2.Canny(clahe.apply(grey), cfg.canny_low, cfg.canny_high)


def _normalised_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Zero-mean normalised correlation over the masked pixels, in [-1, 1]."""
    sel = mask > 0
    if int(sel.sum()) < 32:
        return 0.0
    x = a[sel].astype(np.float64)
    y = b[sel].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-9:
        # Both patches are flat. Identical flatness is a match; one flat, one not is not.
        return 1.0 if np.allclose(a[sel], b[sel], atol=2.0) else 0.0
    return float(np.dot(x, y) / denom)


class GuardChecker:
    """Compares a guard region against a learned reference, frame by frame."""

    def __init__(self, config: GuardConfig | None = None) -> None:
        self.config = config or GuardConfig()
        self.reference: GuardReference | None = None
        self.status = GuardStatus.UNKNOWN
        self.since_ms: float | None = None
        self._pending: tuple[GuardStatus, float] | None = None

    # ---- learning ---------------------------------------------------------
    def learn(self, frame: np.ndarray, zone: Zone, timestamp_ms: float = 0.0) -> GuardReference:
        """Capture the reference. The caller is asserting the guard is in place."""
        cfg = self.config
        grey, _ = _work(frame, cfg.work_width, cfg.blur_ksize)
        h, w = grey.shape[:2]
        mask = zone.mask((w, h))
        edges = cv2.bitwise_and(_edges(grey, cfg), mask)
        density = float(np.count_nonzero(edges)) / max(1.0, float(np.count_nonzero(mask)))
        self.reference = GuardReference(
            zone=zone, edges=edges, grey=grey, size=(w, h),
            captured_ms=timestamp_ms, edge_density=density,
        )
        self.status = GuardStatus.PRESENT
        self.since_ms = timestamp_ms
        self._pending = None
        return self.reference

    # ---- checking ---------------------------------------------------------
    def check(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        *,
        person_boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> GuardObservation:
        cfg = self.config
        if self.reference is None:
            return GuardObservation(
                GuardStatus.UNKNOWN, 0.0, 0.0, 0.0, 0.0,
                note="no guard reference captured; confirm the guard is in place and learn it",
            )

        grey, scale = _work(frame, cfg.work_width, cfg.blur_ksize)
        if grey.shape[:2] != self.reference.grey.shape[:2]:
            grey = cv2.resize(grey, self.reference.size, interpolation=cv2.INTER_AREA)
            scale = self.reference.size[0] / float(frame.shape[1])
        h, w = grey.shape[:2]
        mask = self.reference.zone.mask((w, h))
        region_px = max(1, int(np.count_nonzero(mask)))

        occl = np.zeros((h, w), dtype=np.uint8)
        for box in person_boxes or []:
            x1, y1, x2, y2 = (v * scale for v in box)
            cv2.rectangle(occl, (int(max(0, x1)), int(max(0, y1))),
                          (int(min(w, x2)), int(min(h, y2))), 255, -1)
        occluded = float(np.count_nonzero(cv2.bitwise_and(mask, occl))) / region_px
        visible = cv2.bitwise_and(mask, cv2.bitwise_not(occl))

        if occluded > cfg.max_occluded_fraction:
            return GuardObservation(
                GuardStatus.UNKNOWN, 0.0, 0.0, 0.0, occluded,
                since_ms=self.since_ms,
                note=f"{occluded * 100:.0f}% of the guard is blocked by someone standing "
                     "in front of it; holding the previous state",
            )

        edges = cv2.bitwise_and(_edges(grey, cfg), mask)
        edge_score = _normalised_correlation(self.reference.edges, edges, visible)
        appearance_score = _normalised_correlation(self.reference.grey, grey, visible)
        score = min(edge_score, appearance_score)

        observed = (
            GuardStatus.PRESENT
            if edge_score >= cfg.edge_threshold and appearance_score >= cfg.appearance_threshold
            else GuardStatus.MISSING
        )

        if observed != self.status:
            if self._pending is None or self._pending[0] != observed:
                self._pending = (observed, timestamp_ms)
            elif timestamp_ms - self._pending[1] >= cfg.min_state_ms:
                self.status = observed
                self.since_ms = timestamp_ms
                self._pending = None
        else:
            self._pending = None

        note = ""
        if self.status is GuardStatus.MISSING:
            note = ("the guard region no longer matches the reference: the guard is open, "
                    "removed, or something is covering it")
        return GuardObservation(
            self.status, edge_score, appearance_score, score, occluded,
            since_ms=self.since_ms, note=note,
        )

    def reset(self) -> None:
        self.reference = None
        self.status = GuardStatus.UNKNOWN
        self.since_ms = None
        self._pending = None
