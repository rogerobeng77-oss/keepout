"""Staged clip generation: real people, a rendered machine cell, exact ground truth.

Why this module exists
----------------------
Evaluating this product needs clips where we know, frame by frame, when a person
was inside the zone, when the machine was running, and when the guard was open.
No public dataset carries those labels, and we have no factory to film in.

So the clips are **composites**, and the document says so everywhere:

* The **people are real pixels** — person crops matted out of `vtest.avi`, the
  pedestrian clip that ships in OpenCV's own repository under Apache-2.0. They are
  real human shapes, real clothing, real motion blur, and the detector has to earn
  every detection. A rendered rectangle would not test anything.
* The **machine cell is rendered** — concrete floor, hazard stripes, a conveyor
  whose belt texture genuinely scrolls when it is running, and a guard panel that
  can be removed at a scripted frame. Because it is rendered we know the truth.

What a composite cannot tell you is written down in `docs/evaluation.md`: it does
not test against real factory lighting, steam, coolant spray, high-vis clothing,
or the shapes of real machinery. The courtyard clip, which is entirely unmodified
real footage, is what we use to measure the false-alert rate.

Matting: `vtest.avi` has a near-static background, so the per-pixel median over
the clip is a clean plate. Subtracting it inside a detection box gives a usable
person matte with two lines of morphology.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CELL_SIZE = (960, 540)  # (width, height)
DEFAULT_FPS = 12.0

SOURCE_NOTE = (
    "Person imagery matted from samples/data/vtest.avi in the OpenCV repository "
    "(Apache-2.0). The machine cell, conveyor, guard panel and all degradations "
    "are rendered by keepout.synth, so every label is exact by construction."
)


# ---------------------------------------------------------------------------
# Sprites
# ---------------------------------------------------------------------------


@dataclass
class Sprite:
    """A matted person: BGR pixels plus an 8-bit alpha of the same size."""

    bgr: np.ndarray
    alpha: np.ndarray

    @property
    def size(self) -> tuple[int, int]:
        return self.bgr.shape[1], self.bgr.shape[0]

    def scaled(self, height_px: float) -> Sprite:
        h, w = self.bgr.shape[:2]
        factor = height_px / max(1.0, float(h))
        size = (max(2, round(w * factor)), max(2, round(h * factor)))
        return Sprite(
            cv2.resize(self.bgr, size, interpolation=cv2.INTER_CUBIC),
            cv2.resize(self.alpha, size, interpolation=cv2.INTER_LINEAR),
        )

    def rotated(self, degrees: float) -> Sprite:
        if abs(degrees) < 0.01:
            return self
        h, w = self.bgr.shape[:2]
        diag = int(math.hypot(h, w)) + 4
        pad_bgr = np.zeros((diag, diag, 3), np.uint8)
        pad_a = np.zeros((diag, diag), np.uint8)
        oy, ox = (diag - h) // 2, (diag - w) // 2
        pad_bgr[oy:oy + h, ox:ox + w] = self.bgr
        pad_a[oy:oy + h, ox:ox + w] = self.alpha
        m = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), degrees, 1.0)
        return Sprite(
            cv2.warpAffine(pad_bgr, m, (diag, diag)),
            cv2.warpAffine(pad_a, m, (diag, diag)),
        )

    def flipped(self) -> Sprite:
        return Sprite(cv2.flip(self.bgr, 1), cv2.flip(self.alpha, 1))


def content_box(sprite: Sprite, threshold: int = 8) -> tuple[int, int, int, int]:
    """The sprite's actual occupied rectangle, ignoring the transparent padding.

    Rotation pads the sprite onto a square canvas, so anchoring by the canvas puts
    a person who has rotated 80 degrees a hundred pixels above the floor. Anchoring
    by the content box puts the body where a body would be.
    """
    ys, xs = np.nonzero(sprite.alpha > threshold)
    if ys.size == 0:
        h, w = sprite.alpha.shape[:2]
        return (0, 0, w, h)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def composite(canvas: np.ndarray, sprite: Sprite, top_left: tuple[int, int]) -> None:
    """Alpha-blend a sprite into `canvas` in place, clipped at the frame edge."""
    x, y = int(top_left[0]), int(top_left[1])
    h, w = sprite.bgr.shape[:2]
    ch, cw = canvas.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + w), min(ch, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    sx1, sy1 = x1 - x, y1 - y
    patch = sprite.bgr[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
    alpha = sprite.alpha[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)].astype(np.float32) / 255.0
    alpha = alpha[..., None]
    region = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = (region * (1.0 - alpha) + patch.astype(np.float32) * alpha).astype(
        np.uint8
    )


def clean_plate(video: Path | str, samples: int = 40) -> np.ndarray:
    """Per-pixel median over evenly spaced frames: the empty scene."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max(1, samples))
    stack: list[np.ndarray] = []
    try:
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                stack.append(frame)
            if len(stack) >= samples:
                break
    finally:
        cap.release()
    if not stack:
        raise ValueError(f"no frames read from {video}")
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def matte(frame: np.ndarray, plate: np.ndarray, box: tuple[float, float, float, float],
          *, threshold: int = 26) -> Sprite | None:
    """Cut a person out of `frame` by differencing against the clean plate."""
    x1, y1, x2, y2 = (round(v) for v in box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    back = plate[y1:y2, x1:x2]
    diff = cv2.absdiff(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(back, cv2.COLOR_BGR2GRAY))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 0.12 * mask.size:
        return None
    mask = ((labels == largest).astype(np.uint8)) * 255
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return Sprite(crop.copy(), mask)


def harvest_sprites(video: Path | str, detector, *, wanted: int = 6,
                    start_ms: float = 5000.0, stride: int = 23) -> list[Sprite]:
    """Matte the tallest, most confident people out of a pedestrian clip."""
    from visioncore import iter_video

    plate = clean_plate(video)
    out: list[Sprite] = []
    for frame in iter_video(video, stride=stride, start_ms=start_ms, max_frames=60):
        people = [d for d in detector(frame.image) if (d[0][3] - d[0][1]) >= 60]
        people.sort(key=lambda d: -(d[0][3] - d[0][1]))
        for box, score in people[:2]:
            if score < 0.6:
                continue
            sprite = matte(frame.image, plate, box)
            if sprite is not None:
                out.append(sprite)
            if len(out) >= wanted:
                return out
    return out


# ---------------------------------------------------------------------------
# The rendered cell
# ---------------------------------------------------------------------------


def render_cell(size: tuple[int, int] = CELL_SIZE, *, belt_phase: float = 0.0,
                guard: bool = True, seed: int = 7) -> np.ndarray:
    """A machine cell: concrete floor, hazard stripes, a conveyor, a guard panel.

    `belt_phase` scrolls the belt texture. When the caller advances it frame to
    frame the conveyor really moves, so `machine.MotionEnergy` measures genuine
    motion rather than a flag we set.
    """
    w, h = size
    rng = np.random.default_rng(seed)
    frame = np.full((h, w, 3), 96, np.uint8)

    # Concrete floor. Low-amplitude, low-frequency mottling only: heavy per-pixel
    # grain looked like a real floor in a still and turned into coloured mush once
    # the clip was encoded to H.264, which is what the browser and the detector
    # actually see. Slab joints give the encoder something real to hold on to.
    coarse = rng.normal(0, 7, (h // 24 + 1, w // 24 + 1)).astype(np.float32)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    coarse = cv2.GaussianBlur(coarse, (0, 0), 6)
    floor = np.clip(118 + coarse, 92, 148).astype(np.uint8)
    frame = cv2.cvtColor(floor, cv2.COLOR_GRAY2BGR)

    # Slab joints, receding with the floor
    for jy in (int(h * 0.62), int(h * 0.80), int(h * 0.97)):
        cv2.line(frame, (0, jy), (w, jy), (96, 99, 103), 2, cv2.LINE_AA)
    for jx in (int(w * 0.16), int(w * 0.50), int(w * 0.84)):
        cv2.line(frame, (jx, int(h * 0.44)), (jx + int((jx - w / 2) * 0.35), h),
                 (98, 101, 105), 2, cv2.LINE_AA)

    # Back wall
    cv2.rectangle(frame, (0, 0), (w, int(h * 0.26)), (74, 78, 82), -1)
    cv2.line(frame, (0, int(h * 0.26)), (w, int(h * 0.26)), (52, 55, 58), 3)

    # Hazard stripes marking the floor around the machine
    stripe_top, stripe_bottom = int(h * 0.40), int(h * 0.44)
    for x in range(int(w * 0.18), int(w * 0.94), 34):
        pts = np.array([[x, stripe_bottom], [x + 17, stripe_top],
                        [x + 34, stripe_top], [x + 17, stripe_bottom]], np.int32)
        cv2.fillPoly(frame, [pts], (40, 190, 235))

    # Conveyor frame
    cx1, cy1, cx2, cy2 = int(w * 0.22), int(h * 0.20), int(w * 0.90), int(h * 0.40)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (108, 112, 118), -1)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (58, 60, 64), 3)

    # The belt: diagonal cleats that scroll with belt_phase
    belt_top, belt_bottom = cy1 + 12, cy2 - 12
    pitch = 46
    offset = int(belt_phase) % pitch
    for x in range(cx1 - pitch, cx2 + pitch, pitch):
        a = (x + offset, belt_bottom)
        b = (x + offset + 24, belt_top)
        cv2.line(frame, a, b, (66, 70, 76), 7, cv2.LINE_AA)
        cv2.line(frame, (a[0] + 10, a[1]), (b[0] + 10, b[1]), (142, 146, 152), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (cx1, cy1), (cx2, belt_top), (86, 90, 96), -1)
    cv2.rectangle(frame, (cx1, belt_bottom), (cx2, cy2), (86, 90, 96), -1)

    # Legs
    for lx in (int(w * 0.28), int(w * 0.56), int(w * 0.84)):
        cv2.rectangle(frame, (lx - 7, cy2), (lx + 7, int(h * 0.56)), (92, 96, 102), -1)

    # The fixed guard: a mesh panel bolted across the drive end
    gx1, gy1, gx2, gy2 = int(w * 0.30), int(h * 0.40), int(w * 0.54), int(h * 0.60)
    if guard:
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (118, 150, 124), -1)
        for x in range(gx1 + 6, gx2, 12):
            cv2.line(frame, (x, gy1 + 4), (x, gy2 - 4), (56, 92, 62), 1)
        for y in range(gy1 + 6, gy2, 12):
            cv2.line(frame, (gx1 + 4, y), (gx2 - 4, y), (56, 92, 62), 1)
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (40, 70, 46), 4)
        for bx, by in ((gx1 + 8, gy1 + 8), (gx2 - 8, gy1 + 8),
                       (gx1 + 8, gy2 - 8), (gx2 - 8, gy2 - 8)):
            cv2.circle(frame, (bx, by), 4, (30, 52, 36), -1)
    else:
        # Guard removed: the open drive end and its shadow are what is left.
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (58, 60, 66), -1)
        cv2.rectangle(frame, (gx1 + 14, gy1 + 10), (gx2 - 14, gy2 - 16), (34, 36, 40), -1)
    return frame


def cell_zones() -> dict[str, Any]:
    """The polygons that match `render_cell`, in its own pixel coordinates."""
    w, h = CELL_SIZE
    return {
        "danger": [[w * 0.18, h * 0.40], [w * 0.94, h * 0.40],
                   [w * 0.98, h * 0.98], [w * 0.12, h * 0.98]],
        "machine": [[w * 0.22, h * 0.20], [w * 0.90, h * 0.20],
                    [w * 0.90, h * 0.40], [w * 0.22, h * 0.40]],
        "guard": [[w * 0.30, h * 0.40], [w * 0.54, h * 0.40],
                  [w * 0.54, h * 0.60], [w * 0.30, h * 0.60]],
    }


# ---------------------------------------------------------------------------
# Scripting
# ---------------------------------------------------------------------------


@dataclass
class Walk:
    """One person's scripted path, as timed waypoints.

    One person is one `Walk`, always. An earlier version of this module modelled a
    person who walked in, stood still, then walked out as two `Walk` objects, and
    the label generator dutifully emitted a zone exit and a zone entry at the same
    instant. The evaluation then scored a correct single incident as a missed
    entry. Waypoints exist so the staging cannot invent events the scene does not
    contain.

    `path` is ((t_seconds, x, y), ...) in ascending time, with x, y the foot point.
    `start`/`end` remain as a shorthand for the two-waypoint case.
    """

    enter_s: float
    exit_s: float
    start: tuple[float, float] | None = None
    end: tuple[float, float] | None = None
    path: tuple[tuple[float, float, float], ...] = ()
    height_px: float = 150.0
    sprite_index: int = 0
    collapse_at_s: float | None = None  # rotate to `collapse_angle` and stop
    collapse_angle: float = 78.0
    flip: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            if self.start is None or self.end is None:
                raise ValueError("a Walk needs either `path` or both `start` and `end`")
            stop = self.collapse_at_s if self.collapse_at_s is not None else self.exit_s
            self.path = ((self.enter_s, *self.start), (stop, *self.end))
        if len(self.path) < 2:
            raise ValueError("a Walk path needs at least two waypoints")
        times = [w[0] for w in self.path]
        if times != sorted(times):
            raise ValueError("Walk waypoints must be in ascending time order")

    def position(self, t: float) -> tuple[float, float] | None:
        if t < self.enter_s or t > self.exit_s:
            return None
        points = self.path
        if t <= points[0][0]:
            return (points[0][1], points[0][2])
        if t >= points[-1][0]:
            return (points[-1][1], points[-1][2])
        for (t0, x0, y0), (t1, x1, y1) in itertools.pairwise(points):
            if t0 <= t <= t1:
                u = (t - t0) / max(1e-6, t1 - t0)
                return (x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)
        return (points[-1][1], points[-1][2])

    def angle(self, t: float) -> float:
        if self.collapse_at_s is None or t < self.collapse_at_s:
            return 0.0
        fall = min(1.0, (t - self.collapse_at_s) / 0.9)
        return self.collapse_angle * fall


@dataclass
class Degradation:
    """A scripted view problem, so the honesty rails have something to catch."""

    kind: str  # camera_moved | lens_blocked | too_dark | frozen_feed
    start_s: float
    end_s: float
    magnitude: float = 1.0


@dataclass
class StageScript:
    """A whole clip: how long, what runs, who walks, what breaks."""

    name: str
    duration_s: float = 30.0
    fps: float = DEFAULT_FPS
    # Real sensors are never silent. Without a little per-frame noise a rendered
    # clip produces byte-identical frames whenever nothing moves, and the frozen
    # feed check correctly calls that a dead stream. The noise is the camera, not
    # a fudge: `viewcheck.frozen_max_absdiff` is calibrated against exactly this.
    sensor_noise: float = 1.4
    running_spans: tuple[tuple[float, float], ...] = ((0.0, 1e9),)
    guard_removed_from_s: float | None = None
    walks: list[Walk] = field(default_factory=list)
    degradations: list[Degradation] = field(default_factory=list)
    seed: int = 7
    note: str = ""

    def running_at(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.running_spans)

    def guard_at(self, t: float) -> bool:
        return self.guard_removed_from_s is None or t < self.guard_removed_from_s


@dataclass
class GroundTruth:
    """Exactly what the clip contains, written next to it as JSON."""

    clip: str
    fps: float
    size: tuple[int, int]
    frames: list[dict[str, Any]]
    zones: dict[str, Any]
    events: list[dict[str, Any]]
    note: str = SOURCE_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip": self.clip,
            "fps": self.fps,
            "size": list(self.size),
            "zones": self.zones,
            "events": self.events,
            "frames": self.frames,
            "note": self.note,
        }

    def save(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def _point_in(points: list[list[float]], p: tuple[float, float]) -> bool:
    poly = np.array(points, np.float32)
    return cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0


def render_script(script: StageScript, sprites: list[Sprite], out_path: Path,
                  *, size: tuple[int, int] = CELL_SIZE) -> GroundTruth:
    """Render a scripted clip to `out_path` and return its exact ground truth."""
    w, h = size
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), script.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a writer for {out_path}")

    zones = cell_zones()
    rng_noise = np.random.default_rng(script.seed + 1000)
    total = round(script.duration_s * script.fps)
    belt_phase = 0.0
    frames_truth: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    last_good: np.ndarray | None = None
    prev_in_zone: set[int] = set()

    try:
        for i in range(total):
            t = i / script.fps
            running = script.running_at(t)
            guard_on = script.guard_at(t)
            if running:
                belt_phase += 9.0

            frame = render_cell(size, belt_phase=belt_phase, guard=guard_on, seed=script.seed)

            in_zone: set[int] = set()
            people: list[dict[str, Any]] = []
            for wi, walk in enumerate(script.walks):
                pos = walk.position(t)
                if pos is None:
                    continue
                base = sprites[walk.sprite_index % len(sprites)]
                sprite = base.flipped() if walk.flip else base
                sprite = sprite.scaled(walk.height_px)
                angle = walk.angle(t)
                if angle:
                    sprite = sprite.rotated(angle)
                # Anchor by the sprite's occupied pixels, not its padded canvas, so a
                # rotated body lies on the floor instead of floating above it.
                cx1, cy1, cx2, cy2 = content_box(sprite)
                top_left = (int(pos[0] - (cx1 + cx2) / 2.0), int(pos[1] - cy2))
                composite(frame, sprite, top_left)
                box = [top_left[0] + cx1, top_left[1] + cy1,
                       top_left[0] + cx2, top_left[1] + cy2]
                feet = (pos[0], pos[1])
                inside = _point_in(zones["danger"], feet)
                if inside:
                    in_zone.add(wi)
                people.append({
                    "walk": wi, "bbox": [round(v, 1) for v in box],
                    "feet": [round(feet[0], 1), round(feet[1], 1)],
                    "in_danger_zone": inside, "body_angle_deg": round(angle, 1),
                    "collapsed": angle >= walk.collapse_angle * 0.95,
                })

            if script.sensor_noise > 0:
                noise = rng_noise.normal(0, script.sensor_noise, frame.shape)
                frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            frame, applied = _degrade(frame, script, t, last_good)
            if "frozen_feed" not in applied:
                last_good = frame.copy()

            writer.write(frame)
            frames_truth.append({
                "index": i,
                "timestamp_ms": round(t * 1000.0, 1),
                "machine_running": running,
                "guard_present": guard_on,
                "people": people,
                "people_in_danger_zone": len(in_zone),
                "view_problems": applied,
                "usable": not applied,
            })

            for wi in in_zone - prev_in_zone:
                events.append({
                    "ts_ms": round(t * 1000.0, 1), "event": "zone_entry", "walk": wi,
                    "machine_running": running,
                    "expected_level": "alert" if running else "note",
                })
            for wi in prev_in_zone - in_zone:
                events.append({"ts_ms": round(t * 1000.0, 1), "event": "zone_exit", "walk": wi})
            prev_in_zone = in_zone
    finally:
        writer.release()

    for walk in script.walks:
        if walk.collapse_at_s is not None:
            events.append({
                "ts_ms": round(walk.collapse_at_s * 1000.0, 1),
                "event": "person_collapses",
                "expected_level": "critical",
            })
    if script.guard_removed_from_s is not None:
        events.append({
            "ts_ms": round(script.guard_removed_from_s * 1000.0, 1),
            "event": "guard_removed", "expected_level": "guard",
        })
    for span in script.running_spans:
        events.append({"ts_ms": round(span[0] * 1000.0, 1), "event": "machine_starts"})
        if span[1] < script.duration_s:
            events.append({"ts_ms": round(span[1] * 1000.0, 1), "event": "machine_stops"})
    for deg in script.degradations:
        events.append({
            "ts_ms": round(deg.start_s * 1000.0, 1), "event": f"view_{deg.kind}",
            "until_ms": round(deg.end_s * 1000.0, 1),
        })
    events.sort(key=lambda e: e["ts_ms"])

    return GroundTruth(
        clip=out_path.name, fps=script.fps, size=size,
        frames=frames_truth, zones=zones, events=events,
        note=(script.note + " " + SOURCE_NOTE).strip(),
    )


def _degrade(frame: np.ndarray, script: StageScript, t: float, last_good: np.ndarray | None
             ) -> tuple[np.ndarray, list[str]]:
    applied: list[str] = []
    for deg in script.degradations:
        if not (deg.start_s <= t < deg.end_s):
            continue
        applied.append(deg.kind)
        if deg.kind == "camera_moved":
            shift = deg.magnitude
            m = np.float32([[1, 0, shift], [0, 1, shift * 0.6]])
            frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]),
                                   borderMode=cv2.BORDER_REPLICATE)
        elif deg.kind == "lens_blocked":
            k = max(31, int(deg.magnitude) | 1)
            frame = cv2.GaussianBlur(frame, (k, k), 0)
            frame = cv2.addWeighted(frame, 0.35, np.full_like(frame, 118), 0.65, 0)
        elif deg.kind == "too_dark":
            frame = (frame.astype(np.float32) * max(0.01, deg.magnitude)).astype(np.uint8)
        elif deg.kind == "frozen_feed" and last_good is not None:
            frame = last_good.copy()
    return frame, applied
