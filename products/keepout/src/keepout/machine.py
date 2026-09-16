"""Is the machine running? Answered from motion energy inside the machine region.

Why this matters more than it sounds: the escalation ladder turns on it. A person
stepping into a danger zone around a *stopped* conveyor during a changeover is
routine work and must produce a note, not a siren. The same step while the belt
is turning is the incident the HSE conveyor case describes. Get this wrong in the
alarming direction and the system is switched off within a week.

The signal is mean absolute frame difference inside the machine polygon, measured
on a blurred, downscaled grey image so that sensor noise and compression mosquito
noise do not read as motion.

Two pieces of discipline:

**Person exclusion.** A worker walking past a stopped machine moves pixels inside
the machine polygon. We zero out the detected person boxes before measuring, or a
stopped machine reads as running exactly when somebody is near it, which is
precisely the wrong time.

**Hysteresis plus dwell.** Machinery has dead spots: an indexing table is still
between indexes, a press is still at the top of its stroke. A bare threshold
flickers. The state changes only after the signal has been on the far side of the
relevant threshold for `min_state_ms`, so a two-second pause in a stroke cycle
does not register as a stop.

The absolute value of the motion score is meaningless across cameras. Calibrate
per installation with `MotionEnergy.suggest_thresholds`, which watches a clip the
operator labels as running and one labelled as stopped, and reports whether the
two are separable at all.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .zones import Zone


@dataclass
class MachineConfig:
    """Thresholds in mean-absolute-difference units on an 8-bit blurred grey image."""

    running_threshold: float = 1.6
    stopped_threshold: float = 0.8  # hysteresis: must fall below this to be called stopped
    min_state_ms: float = 1200.0  # how long a new state must hold before we believe it
    blur_ksize: int = 5
    work_width: int = 480  # analysis width; None-equivalent if the frame is smaller
    history: int = 90
    dilate_person_px: int = 12  # grow person boxes before excluding them

    def __post_init__(self) -> None:
        if self.stopped_threshold > self.running_threshold:
            raise ValueError("stopped_threshold must be <= running_threshold")
        if self.blur_ksize % 2 == 0:
            raise ValueError("blur_ksize must be odd")


@dataclass
class MachineState:
    running: bool = False
    score: float = 0.0
    baseline: float = 0.0
    since_ms: float | None = None
    confidence: float = 0.0
    samples: int = 0
    excluded_fraction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "score": round(self.score, 4),
            "baseline": round(self.baseline, 4),
            "since_ms": self.since_ms,
            "confidence": round(self.confidence, 3),
            "samples": self.samples,
            "excluded_fraction": round(self.excluded_fraction, 3),
        }


class MotionEnergy:
    """Rolling motion-energy estimate inside one region, with state hysteresis."""

    def __init__(self, zone: Zone | None, config: MachineConfig | None = None) -> None:
        self.zone = zone
        self.config = config or MachineConfig()
        self.state = MachineState()
        self.scores: deque[tuple[float, float]] = deque(maxlen=self.config.history)
        self._prev: np.ndarray | None = None
        self._prev_ms: float | None = None
        self._mask_cache: tuple[tuple[int, int], np.ndarray] | None = None
        self._pending: tuple[bool, float] | None = None

    # ---- internals --------------------------------------------------------
    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Downscale + grey + blur. Returns (image, scale) where scale maps full->work."""
        h, w = frame.shape[:2]
        scale = min(1.0, self.config.work_width / float(w)) if w > 0 else 1.0
        if scale < 1.0:
            frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k = self.config.blur_ksize
        return cv2.GaussianBlur(grey, (k, k), 0), scale

    def _region_mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        if self.zone is None:
            return None
        h, w = shape
        if self._mask_cache and self._mask_cache[0] == (w, h):
            return self._mask_cache[1]
        mask = self.zone.mask((w, h))
        self._mask_cache = ((w, h), mask)
        return mask

    # ---- public -----------------------------------------------------------
    def update(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        *,
        person_boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> MachineState:
        work, scale = self._prepare(frame)
        region = self._region_mask(work.shape[:2])

        exclude = np.zeros(work.shape[:2], dtype=np.uint8)
        for box in person_boxes or []:
            x1, y1, x2, y2 = (v * scale for v in box)
            pad = self.config.dilate_person_px * scale
            cv2.rectangle(
                exclude,
                (int(max(0, x1 - pad)), int(max(0, y1 - pad))),
                (int(min(work.shape[1], x2 + pad)), int(min(work.shape[0], y2 + pad))),
                255,
                -1,
            )

        valid = np.full(work.shape[:2], 255, dtype=np.uint8) if region is None else region.copy()
        region_px = int(np.count_nonzero(valid))
        valid = cv2.bitwise_and(valid, cv2.bitwise_not(exclude))
        usable_px = int(np.count_nonzero(valid))
        self.state.excluded_fraction = (
            0.0 if region_px == 0 else 1.0 - usable_px / float(region_px)
        )

        if self._prev is None or self._prev.shape != work.shape:
            self._prev, self._prev_ms = work, timestamp_ms
            return self.state

        diff = cv2.absdiff(work, self._prev)
        self._prev, self._prev_ms = work, timestamp_ms

        if usable_px < 64:
            # The machine region is almost entirely occluded by people. We cannot
            # measure; hold the last state and say so rather than guess.
            self.state.confidence = 0.0
            return self.state

        score = float(cv2.mean(diff, mask=valid)[0])
        self.scores.append((timestamp_ms, score))
        self.state.score = score
        self.state.samples = len(self.scores)
        values = np.array([s for _, s in self.scores], dtype=np.float64)
        self.state.baseline = float(np.percentile(values, 20)) if len(values) >= 5 else score

        cfg = self.config
        want = self.state.running
        if score >= cfg.running_threshold:
            want = True
        elif score <= cfg.stopped_threshold:
            want = False

        if want != self.state.running:
            if self._pending is None or self._pending[0] != want:
                self._pending = (want, timestamp_ms)
            elif timestamp_ms - self._pending[1] >= cfg.min_state_ms:
                self.state.running = want
                self.state.since_ms = timestamp_ms
                self._pending = None
        else:
            self._pending = None

        span = max(1e-6, cfg.running_threshold - cfg.stopped_threshold)
        margin = (score - cfg.running_threshold) if self.state.running else (
            cfg.stopped_threshold - score
        )
        self.state.confidence = float(np.clip(0.5 + margin / (2.0 * span), 0.0, 1.0))
        return self.state

    def series(self) -> list[tuple[float, float]]:
        return list(self.scores)


@dataclass
class ThresholdSuggestion:
    """The output of calibrating against labelled running and stopped clips."""

    running_threshold: float
    stopped_threshold: float
    separable: bool
    running_p10: float
    stopped_p90: float
    margin: float
    note: str = ""
    samples: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "running_threshold": round(self.running_threshold, 4),
            "stopped_threshold": round(self.stopped_threshold, 4),
            "separable": self.separable,
            "running_p10": round(self.running_p10, 4),
            "stopped_p90": round(self.stopped_p90, 4),
            "margin": round(self.margin, 4),
            "note": self.note,
            "samples": dict(self.samples),
        }


def suggest_thresholds(running_scores: list[float], stopped_scores: list[float]
                       ) -> ThresholdSuggestion:
    """Pick thresholds from labelled score samples, and say when they do not separate.

    The honest case is the failure case. If the tenth percentile of the running
    clip is not above the ninetieth percentile of the stopped clip, the two states
    are not distinguishable by motion energy from this camera angle, and the
    product must say so rather than emit a threshold that is really a coin flip.
    """
    if not running_scores or not stopped_scores:
        return ThresholdSuggestion(
            running_threshold=MachineConfig.running_threshold,
            stopped_threshold=MachineConfig.stopped_threshold,
            separable=False,
            running_p10=0.0,
            stopped_p90=0.0,
            margin=0.0,
            note="need score samples from both a running and a stopped clip",
            samples={"running": len(running_scores), "stopped": len(stopped_scores)},
        )
    run = np.asarray(running_scores, dtype=np.float64)
    stop = np.asarray(stopped_scores, dtype=np.float64)
    run_p10 = float(np.percentile(run, 10))
    stop_p90 = float(np.percentile(stop, 90))
    margin = run_p10 - stop_p90
    separable = margin > 0.0

    if separable:
        running_threshold = stop_p90 + 0.66 * margin
        stopped_threshold = stop_p90 + 0.20 * margin
        note = "thresholds set inside the gap between the two labelled states"
    else:
        # Keep something usable but flag it loudly; the caller must not silently ship this.
        running_threshold = float(np.percentile(run, 50))
        stopped_threshold = float(np.percentile(stop, 50))
        if stopped_threshold > running_threshold:
            stopped_threshold = running_threshold
        note = (
            "running and stopped clips overlap: motion energy cannot tell these apart "
            "from this view. Re-frame the machine region, or wire in the machine's own "
            "run signal instead of inferring it."
        )
    return ThresholdSuggestion(
        running_threshold=running_threshold,
        stopped_threshold=stopped_threshold,
        separable=separable,
        running_p10=run_p10,
        stopped_p90=stop_p90,
        margin=margin,
        note=note,
        samples={"running": int(run.size), "stopped": int(stop.size)},
    )
