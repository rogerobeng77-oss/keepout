"""Danger-zone geometry: where the zone is, who is standing in it, and for how long.

Three ideas, kept separate on purpose.

**The zone** is a polygon in reference-frame pixel coordinates. It is drawn once
by an operator against a specific reference frame, and it is only valid against
that frame. If the camera moves, the polygon is meaningless — see `viewcheck`.

**The contact point** is the part of a person that decides whether they are in
the zone. A bounding box centroid is the wrong answer: a person standing beside
a guarded press with an arm outstretched has a centroid outside the zone and a
hand inside it. We use the bottom-centre of the box (where the feet touch the
floor plane) for floor zones, because a floor polygon drawn on a floor is a
statement about where the feet may go. `ContactPoint.BOX_OVERLAP` covers the
reach case: any part of the box intersecting the polygon counts.

**Dwell** is how long a track has continuously satisfied the test. Hysteresis is
mandatory. A person whose feet oscillate across the boundary by two pixels must
not generate forty incidents, so entry and exit use different distance
thresholds and exit additionally requires a grace period.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np


class ContactPoint(str, Enum):  # noqa: UP042 - the str mixin is what puts these straight into JSON
    """Which part of a detection box is tested against the polygon."""

    FEET = "feet"  # bottom-centre: the floor-plane assumption
    CENTROID = "centroid"  # box centre: for zones drawn on a vertical plane
    BOX_OVERLAP = "box_overlap"  # any overlap: the reach case, most conservative


@dataclass(frozen=True)
class Zone:
    """A named polygon in reference-frame pixels.

    `points` are (x, y) in the coordinate system of the reference frame the
    operator drew against. `reference_size` is that frame's (width, height); it
    lets us scale the polygon if the analysis runs at a different resolution,
    which is a scale change, not a camera move.
    """

    name: str
    points: tuple[tuple[float, float], ...]
    reference_size: tuple[int, int]
    kind: str = "danger"  # danger | machine | guard
    contact: ContactPoint = ContactPoint.FEET

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(f"zone {self.name!r} needs at least 3 points, got {len(self.points)}")
        w, h = self.reference_size
        if w <= 0 or h <= 0:
            raise ValueError(f"zone {self.name!r} has a non-positive reference size")

    # ---- geometry ---------------------------------------------------------
    def scaled_to(self, size: tuple[int, int]) -> np.ndarray:
        """The polygon as an (N, 2) float array in a frame of `size` = (width, height)."""
        rw, rh = self.reference_size
        sw, sh = size
        fx, fy = sw / rw, sh / rh
        return np.array([(x * fx, y * fy) for x, y in self.points], dtype=np.float64)

    def as_int(self, size: tuple[int, int] | None = None) -> np.ndarray:
        poly = self.scaled_to(size) if size else np.array(self.points, dtype=np.float64)
        return np.round(poly).astype(np.int32)

    @property
    def area_px(self) -> float:
        return abs(float(cv2.contourArea(np.array(self.points, dtype=np.float32))))

    def signed_distance(self, point: tuple[float, float], size: tuple[int, int] | None = None
                        ) -> float:
        """Distance from `point` to the polygon boundary, positive inside.

        `cv2.pointPolygonTest` with measureDist=True gives exactly this, and gives
        it in pixels, which is what the hysteresis thresholds are expressed in.
        """
        poly = (self.scaled_to(size) if size else np.array(self.points, dtype=np.float64))
        return float(cv2.pointPolygonTest(poly.astype(np.float32),
                                          (float(point[0]), float(point[1])), True))

    def contains(self, point: tuple[float, float], size: tuple[int, int] | None = None) -> bool:
        return self.signed_distance(point, size) >= 0.0

    def mask(self, size: tuple[int, int]) -> np.ndarray:
        """A uint8 mask, 255 inside. `size` is (width, height)."""
        w, h = size
        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(canvas, [self.as_int(size)], 255)
        return canvas

    def box_overlap_fraction(self, box: tuple[float, float, float, float],
                             size: tuple[int, int] | None = None) -> float:
        """Fraction of the box's area that lies inside the polygon, in [0, 1].

        Computed by rasterising both into the box's own bounding rectangle, which
        is small, so this stays cheap even with a 20-vertex polygon.
        """
        x1, y1, x2, y2 = box
        bw, bh = math.ceil(x2 - x1), math.ceil(y2 - y1)
        if bw <= 0 or bh <= 0:
            return 0.0
        poly = (self.scaled_to(size) if size else np.array(self.points, dtype=np.float64))
        shifted = np.round(poly - np.array([x1, y1])).astype(np.int32)
        canvas = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(canvas, [shifted], 1)
        return float(canvas.sum()) / float(bw * bh)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "contact": self.contact.value,
            "points": [[round(x, 2), round(y, 2)] for x, y in self.points],
            "reference_size": list(self.reference_size),
            "area_px": round(self.area_px, 1),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Zone:
        return cls(
            name=data["name"],
            points=tuple((float(p[0]), float(p[1])) for p in data["points"]),
            reference_size=(int(data["reference_size"][0]), int(data["reference_size"][1])),
            kind=data.get("kind", "danger"),
            contact=ContactPoint(data.get("contact", "feet")),
        )


def contact_point(box: tuple[float, float, float, float], mode: ContactPoint
                  ) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    if mode is ContactPoint.CENTROID:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return ((x1 + x2) / 2.0, y2)  # FEET, and the fallback for BOX_OVERLAP reporting


# ---------------------------------------------------------------------------
# Occupancy with hysteresis
# ---------------------------------------------------------------------------


@dataclass
class DwellConfig:
    """Thresholds for entering and leaving a zone.

    `enter_margin_px` > `exit_margin_px` is the hysteresis: you must be this far
    *inside* to count as in, and this far *outside* to count as out. Between the
    two the previous state holds. `exit_grace_ms` stops a single frame of
    detector jitter from closing an incident.
    """

    enter_margin_px: float = 0.0
    exit_margin_px: float = -12.0
    min_dwell_ms: float = 0.0
    exit_grace_ms: float = 600.0
    overlap_enter: float = 0.12  # for ContactPoint.BOX_OVERLAP
    overlap_exit: float = 0.04

    def __post_init__(self) -> None:
        if self.exit_margin_px > self.enter_margin_px:
            raise ValueError("exit_margin_px must be <= enter_margin_px for hysteresis to work")
        if self.overlap_exit > self.overlap_enter:
            raise ValueError("overlap_exit must be <= overlap_enter")


@dataclass
class Occupancy:
    """Per-track dwell state inside one zone."""

    track_id: int
    zone: str
    inside: bool = False
    entered_ms: float | None = None
    last_inside_ms: float | None = None
    last_seen_ms: float | None = None
    peak_depth_px: float = 0.0
    frames_inside: int = 0

    def dwell_ms(self, now_ms: float) -> float:
        # Note the explicit `is None` checks. Timestamps are genuinely 0.0 at the
        # start of a clip, and an `or` chain treats that as missing.
        if self.entered_ms is None:
            return 0.0
        if self.inside:
            end = now_ms
        elif self.last_inside_ms is not None:
            end = self.last_inside_ms
        else:
            end = self.entered_ms
        return max(0.0, end - self.entered_ms)

    def to_dict(self, now_ms: float | None = None) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "zone": self.zone,
            "inside": self.inside,
            "entered_ms": self.entered_ms,
            "dwell_ms": round(self.dwell_ms(
                now_ms if now_ms is not None
                else (self.last_seen_ms if self.last_seen_ms is not None else 0.0)
            ), 1),
            "peak_depth_px": round(self.peak_depth_px, 1),
            "frames_inside": self.frames_inside,
        }


@dataclass
class ZoneMonitor:
    """Tracks who is inside one zone over time, with hysteresis.

    Feed it `update(track_id, box, timestamp_ms)` once per track per frame, then
    `sweep(timestamp_ms)` to close out anyone who stopped being detected.
    """

    zone: Zone
    config: DwellConfig = field(default_factory=DwellConfig)
    frame_size: tuple[int, int] | None = None
    states: dict[int, Occupancy] = field(default_factory=dict)

    def _score(self, box: tuple[float, float, float, float]) -> tuple[float, tuple[float, float]]:
        """Returns (test_value, reported_point). Higher test_value = further inside."""
        if self.zone.contact is ContactPoint.BOX_OVERLAP:
            return self.zone.box_overlap_fraction(box, self.frame_size), contact_point(
                box, ContactPoint.FEET
            )
        point = contact_point(box, self.zone.contact)
        return self.zone.signed_distance(point, self.frame_size), point

    def _thresholds(self) -> tuple[float, float]:
        if self.zone.contact is ContactPoint.BOX_OVERLAP:
            return self.config.overlap_enter, self.config.overlap_exit
        return self.config.enter_margin_px, self.config.exit_margin_px

    def update(self, track_id: int, box: tuple[float, float, float, float],
               timestamp_ms: float) -> Occupancy:
        value, _point = self._score(box)
        enter_at, exit_at = self._thresholds()
        state = self.states.setdefault(track_id, Occupancy(track_id, self.zone.name))
        state.last_seen_ms = timestamp_ms

        if state.inside:
            if value <= exit_at:
                # Below the exit threshold: only really leave after the grace period.
                if state.last_inside_ms is None or (
                    timestamp_ms - state.last_inside_ms >= self.config.exit_grace_ms
                ):
                    state.inside = False
            else:
                state.last_inside_ms = timestamp_ms
                state.frames_inside += 1
                state.peak_depth_px = max(state.peak_depth_px, value)
        elif value >= enter_at:
            state.inside = True
            state.entered_ms = timestamp_ms
            state.last_inside_ms = timestamp_ms
            state.frames_inside = 1
            state.peak_depth_px = max(state.peak_depth_px, value)
        return state

    def sweep(self, timestamp_ms: float) -> list[Occupancy]:
        """Close out tracks that have not been seen for longer than the grace period."""
        closed: list[Occupancy] = []
        for state in self.states.values():
            if not state.inside:
                continue
            last = state.last_inside_ms
            if last is None:
                last = state.entered_ms
            if last is None:
                last = timestamp_ms
            if timestamp_ms - last >= self.config.exit_grace_ms:
                state.inside = False
                closed.append(state)
        return closed

    def occupants(self) -> list[Occupancy]:
        return [s for s in self.states.values() if s.inside]

    def qualified_occupants(self, timestamp_ms: float) -> list[Occupancy]:
        """Occupants whose dwell has passed `min_dwell_ms`."""
        return [s for s in self.occupants()
                if s.dwell_ms(timestamp_ms) >= self.config.min_dwell_ms]

    def forget(self, track_ids: set[int]) -> None:
        for tid in track_ids:
            self.states.pop(tid, None)


# ---------------------------------------------------------------------------
# Zone proposal from machine motion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneProposal:
    """A zone the system suggests, plus everything needed to distrust it."""

    zone: Zone | None
    confidence: float
    motion_pixels: int
    frames_used: int
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone.to_dict() if self.zone else None,
            "confidence": round(self.confidence, 3),
            "motion_pixels": self.motion_pixels,
            "frames_used": self.frames_used,
            "reason": self.reason,
        }


def propose_zone_from_motion(
    frames: list[np.ndarray],
    *,
    name: str = "proposed",
    dilate_px: int = 35,
    min_area_fraction: float = 0.002,
    max_area_fraction: float = 0.60,
    diff_threshold: int = 18,
) -> ZoneProposal:
    """Propose a danger zone from where the scene moves, with a margin around it.

    The method: accumulate absolute frame differences over the clip, threshold,
    take the largest connected component, dilate it by `dilate_px` (the standoff
    margin — a danger zone is the machine *plus* the reach around it), and return
    its convex hull.

    This is a **proposal**, never an answer. A conveyor that is stopped during the
    learning clip produces no motion and therefore no zone, and the reported
    confidence must make that obvious rather than returning an empty polygon that
    silently never alerts.
    """
    if len(frames) < 3:
        return ZoneProposal(None, 0.0, 0, len(frames), "need at least 3 frames to see motion")

    grays = [f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = grays[0].shape[:2]
    accum = np.zeros((h, w), dtype=np.float32)
    for prev, cur in itertools.pairwise(grays):
        if cur.shape[:2] != prev.shape[:2]:
            return ZoneProposal(None, 0.0, 0, len(frames), "frames differ in size")
        cv2.accumulate(cv2.absdiff(prev, cur), accum)

    peak = float(accum.max())
    if peak <= 0:
        return ZoneProposal(None, 0.0, 0, len(frames),
                            "no motion at all in the learning clip; is the machine running?")

    scaled = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(scaled, diff_threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return ZoneProposal(None, 0.0, 0, len(frames),
                            "motion was too sparse to form a region")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    frame_area = float(w * h)

    if area < min_area_fraction * frame_area:
        return ZoneProposal(None, area / frame_area, area, len(frames),
                            "the moving region is too small to be a machine")
    if area > max_area_fraction * frame_area:
        return ZoneProposal(None, area / frame_area, area, len(frames),
                            "most of the frame is moving; the camera is probably not fixed")

    component = (labels == largest).astype(np.uint8) * 255
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        component = cv2.dilate(
            component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ZoneProposal(None, 0.0, area, len(frames), "no contour after dilation")
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    approx = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(approx) < 3:
        return ZoneProposal(None, 0.0, area, len(frames), "degenerate hull")

    # Confidence: how concentrated the motion is. A single tight blob that holds
    # most of the thresholded motion is a machine; motion smeared everywhere is not.
    concentration = area / max(1.0, float((binary > 0).sum()))
    confidence = float(np.clip(concentration, 0.0, 1.0))

    zone = Zone(
        name=name,
        points=tuple((float(x), float(y)) for x, y in approx),
        reference_size=(w, h),
        kind="danger",
    )
    return ZoneProposal(
        zone=zone,
        confidence=confidence,
        motion_pixels=area,
        frames_used=len(frames),
        reason=f"largest moving region, dilated by {dilate_px} px as a standoff margin",
    )


def draw_zone(image: np.ndarray, zone: Zone, *, colour: tuple[int, int, int] = (60, 90, 240),
              alpha: float = 0.22, label: bool = True) -> np.ndarray:
    """A filled, outlined zone overlay. Used for both the live view and evidence."""
    out = image.copy()
    h, w = out.shape[:2]
    poly = zone.as_int((w, h))
    overlay = out.copy()
    cv2.fillPoly(overlay, [poly], colour)
    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0, out)
    cv2.polylines(out, [poly], True, colour, 2, cv2.LINE_AA)
    if label:
        anchor = poly[poly[:, 1].argmin()]
        _put(out, zone.name, (int(anchor[0]), max(14, int(anchor[1]) - 8)), colour)
    return out


def _put(image: np.ndarray, text: str, org: tuple[int, int],
         colour: tuple[int, int, int], size: int = 16) -> None:
    """OpenCV 5's FontFace path, with the legacy call as a fallback."""
    if hasattr(cv2, "FontFace"):
        cv2.putText(image, text, org, colour, cv2.FontFace("sans"), size)
    else:  # pragma: no cover - OpenCV 5 always has FontFace
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
