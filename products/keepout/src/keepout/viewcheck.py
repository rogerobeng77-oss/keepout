"""Honesty rails: when the camera cannot see, say so and stop pretending.

Every result in this product is expressed in the coordinate system of a polygon
an operator drew on one reference frame. That polygon means nothing if the camera
has been nudged, the lens is covered in coolant, the lights are off, or the video
feed has frozen on the last good frame. In all four cases a zone monitor keeps
running happily and keeps reporting "nobody in the zone", which is the worst
possible failure: silence that looks like safety.

So the pipeline asks four questions of every frame before it asks anything else.

**Has the camera moved?** Estimated by ECC alignment of a downscaled grey frame
against the reference the zone was drawn on (`cv2.findTransformECC`, Euclidean
motion model). It returns the actual translation and rotation, so we can report
"the view has shifted about 40 pixels" rather than a bare boolean. When ECC fails
to converge at all, that is itself evidence the scene no longer resembles the
reference, and we fall back to phase correlation for a translation estimate.

We deliberately do **not** try to warp the zone into the new view. A homography
fitted to a scene that has partly changed is exactly the silent-confident-nonsense
failure that FINDINGS.md §5.0 warns about. The zone must be re-drawn by a human.

**Is the lens blocked?** A covered or heavily fouled lens has almost no edge
content and low local contrast, everywhere, at once. Measured as the fraction of
pixels carrying Canny edges plus the standard deviation of a Laplacian response.

**Is it too dark?** Median luminance below a floor, with the additional check that
the usable dynamic range has collapsed: night footage with working lights is fine,
a black frame is not.

**Is the feed frozen?** This one had to be rebuilt after measuring it.

The obvious test is byte-identical consecutive frames, keyed on the *maximum*
absolute difference, on the reasoning that a real sensor always has some noise
and a duplicated frame has none. That test fails on every real camera, because
real cameras deliver compressed video. Re-encoding a genuinely frozen feed to
H.264 and decoding it back gave differences of up to **7 grey levels** frame to
frame, purely from the codec, so a max-difference threshold of 2 never fired.

Measured on the sample clips (320 px wide, mean absolute difference per frame):

| condition | mean | p95 | max |
|---|---|---|---|
| frozen feed, after H.264 round trip | 0.005 | 0.029 | 7 |
| live, machine stopped, empty room | 0.075 | 0.409 | 7 |
| live, very dark | 0.108 | 0.136 | 7 |
| live, machine running | 2.639 | 2.921 | 86 |
| live, real pedestrian footage | 1.556 | 2.551 | 250 |

The **mean** separates frozen from live-but-static by about fifteen times where
the maximum does not separate them at all. So the test is a low mean sustained
over `frozen_frames` consecutive frames.

The honest limit: a scene that is genuinely motionless in front of a very
low-noise sensor is not distinguishable from a frozen feed by pixels alone. In a
real deployment the answer is the stream's own timestamps or sequence numbers,
which a video file does not carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np


class ViewProblem(str, Enum):  # noqa: UP042 - the str mixin is what puts these straight into JSON
    CAMERA_MOVED = "camera_moved"
    LENS_BLOCKED = "lens_blocked"
    TOO_DARK = "too_dark"
    FROZEN_FEED = "frozen_feed"
    SIZE_CHANGED = "size_changed"


HUMAN_TEXT: dict[ViewProblem, str] = {
    ViewProblem.CAMERA_MOVED:
        "The camera has moved since the zone was drawn. The zone no longer marks the "
        "same floor. Re-draw it against a fresh reference frame before trusting any alert.",
    ViewProblem.LENS_BLOCKED:
        "The lens looks blocked or fouled. Almost no edge detail is reaching the sensor.",
    ViewProblem.TOO_DARK:
        "The scene is too dark to detect anybody reliably. Alerts are suppressed.",
    ViewProblem.FROZEN_FEED:
        "The video feed is not advancing. Frames are arriving identical, so nothing "
        "that happens in the room would be seen.",
    ViewProblem.SIZE_CHANGED:
        "The frame size changed mid-stream, so the zone cannot be placed.",
}


@dataclass
class ViewConfig:
    work_width: int = 320
    # camera moved
    max_shift_px: float = 12.0  # measured at reference resolution
    max_rotation_deg: float = 1.5
    ecc_iterations: int = 60
    ecc_epsilon: float = 1e-4
    # lens blocked
    min_edge_fraction: float = 0.004
    min_laplacian_std: float = 3.0
    # too dark
    min_median_luma: float = 22.0
    min_luma_range: float = 24.0  # p95 - p5
    # frozen - see the table in the module docstring for where these come from
    frozen_mean_absdiff: float = 0.05
    frozen_max_absdiff: int = 12
    frozen_frames: int = 12


@dataclass
class ViewStatus:
    """What we think of this frame. `usable` is the only thing the pipeline branches on."""

    usable: bool = True
    problems: list[ViewProblem] = field(default_factory=list)
    shift_px: float = 0.0
    rotation_deg: float = 0.0
    edge_fraction: float = 0.0
    laplacian_std: float = 0.0
    median_luma: float = 0.0
    luma_range: float = 0.0
    identical_frames: int = 0
    ecc_converged: bool = True
    messages: list[str] = field(default_factory=list)

    def has(self, problem: ViewProblem) -> bool:
        return problem in self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "problems": [p.value for p in self.problems],
            "messages": list(self.messages),
            "shift_px": round(self.shift_px, 2),
            "rotation_deg": round(self.rotation_deg, 3),
            "edge_fraction": round(self.edge_fraction, 5),
            "laplacian_std": round(self.laplacian_std, 3),
            "median_luma": round(self.median_luma, 1),
            "luma_range": round(self.luma_range, 1),
            "identical_frames": self.identical_frames,
            "ecc_converged": self.ecc_converged,
        }


def _work(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, width / float(w)) if w > 0 else 1.0
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def estimate_shift(reference: np.ndarray, frame: np.ndarray, config: ViewConfig
                   ) -> tuple[float, float, bool]:
    """(translation px, rotation degrees, converged) at the *work* resolution.

    ECC with a Euclidean model. It is iterative and it can fail to converge, which
    we report rather than swallow, because failure to align is itself a signal.
    """
    ref = np.float32(reference) / 255.0
    cur = np.float32(frame) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                config.ecc_iterations, config.ecc_epsilon)
    try:
        cv2.findTransformECC(ref, cur, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        converged = True
    except cv2.error:
        converged = False
        (dx, dy), _response = cv2.phaseCorrelate(np.float64(reference), np.float64(frame))
        return float(np.hypot(dx, dy)), 0.0, converged
    shift = float(np.hypot(warp[0, 2], warp[1, 2]))
    rotation = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
    return shift, rotation, converged


class ViewGuard:
    """Holds the reference frame and judges every subsequent frame against it."""

    def __init__(self, config: ViewConfig | None = None) -> None:
        self.config = config or ViewConfig()
        self._reference: np.ndarray | None = None
        self._reference_size: tuple[int, int] | None = None  # full-res (w, h)
        self._prev: np.ndarray | None = None
        self._identical = 0

    @property
    def has_reference(self) -> bool:
        return self._reference is not None

    def set_reference(self, frame: np.ndarray) -> None:
        """Adopt this frame as the one the zone was drawn against."""
        h, w = frame.shape[:2]
        self._reference = _work(frame, self.config.work_width)
        self._reference_size = (w, h)
        self._prev = None
        self._identical = 0

    def check(self, frame: np.ndarray) -> ViewStatus:
        cfg = self.config
        status = ViewStatus()
        h, w = frame.shape[:2]
        work = _work(frame, cfg.work_width)

        # ---- brightness ---------------------------------------------------
        status.median_luma = float(np.median(work))
        p5, p95 = (float(v) for v in np.percentile(work, [5, 95]))
        status.luma_range = p95 - p5
        if status.median_luma < cfg.min_median_luma and status.luma_range < cfg.min_luma_range:
            status.problems.append(ViewProblem.TOO_DARK)

        # ---- lens blockage ------------------------------------------------
        edges = cv2.Canny(work, 50, 150)
        status.edge_fraction = float(np.count_nonzero(edges)) / float(edges.size)
        status.laplacian_std = float(cv2.Laplacian(work, cv2.CV_64F).std())
        if (status.edge_fraction < cfg.min_edge_fraction
                and status.laplacian_std < cfg.min_laplacian_std):
            status.problems.append(ViewProblem.LENS_BLOCKED)

        # ---- frozen feed --------------------------------------------------
        if self._prev is not None and self._prev.shape == work.shape:
            difference = cv2.absdiff(work, self._prev)
            still = (float(difference.mean()) <= cfg.frozen_mean_absdiff
                     and int(difference.max()) <= cfg.frozen_max_absdiff)
            self._identical = self._identical + 1 if still else 0
        status.identical_frames = self._identical
        if self._identical >= cfg.frozen_frames:
            status.problems.append(ViewProblem.FROZEN_FEED)
        self._prev = work

        # ---- camera moved -------------------------------------------------
        if self._reference is not None:
            if self._reference_size != (w, h):
                status.problems.append(ViewProblem.SIZE_CHANGED)
            elif work.shape == self._reference.shape:
                # A blocked or black frame carries no alignable structure, so any
                # alignment against it is noise. Report what we already know is
                # wrong and do not add a second, invented problem on top.
                blind = (ViewProblem.LENS_BLOCKED in status.problems
                         or ViewProblem.TOO_DARK in status.problems)
                if not blind:
                    shift, rotation, converged = estimate_shift(self._reference, work, cfg)
                    to_full = w / float(work.shape[1])
                    status.shift_px = shift * to_full
                    status.rotation_deg = rotation
                    status.ecc_converged = converged
                    moved = (status.shift_px > cfg.max_shift_px
                             or abs(rotation) > cfg.max_rotation_deg)
                    # A non-converging ECC on an otherwise well-lit, detailed frame
                    # means the scene stopped resembling the reference at all.
                    if moved or not converged:
                        status.problems.append(ViewProblem.CAMERA_MOVED)

        status.usable = not status.problems
        status.messages = [HUMAN_TEXT[p] for p in status.problems]
        return status

    def reset(self) -> None:
        self._prev = None
        self._identical = 0
