"""Person-down detection from box posture and stillness. No pose model, no identity.

The highest escalation level in this product is "a person is down and not
moving". That is the state the HSE conveyor case ends in: a man alone, injured,
with nobody watching. It is also the state most likely to be missed, because the
person stops generating the motion that a naive motion alarm keys on.

We use two signals that a person detector already gives us for free:

**Posture** — the aspect ratio of the detection box. Standing people are tall
boxes; a person lying on the floor is a wide box. The ratio width/height crosses
1.0 somewhere in between. This is crude and it is honest about being crude: a
person crouching to clear a jam also produces a wide box, which is why posture
alone never raises anything.

**Stillness** — how far the box centre and its dimensions have moved over a
window. A crouching worker fidgets; an unconscious one does not. Stillness is
measured as the maximum centre displacement over the window in units of the box
diagonal, so it is scale-invariant and a person far from the camera is judged the
same as one near it.

The state only fires when posture **and** stillness agree for `min_duration_ms`.
That delay is the cost of not screaming every time somebody kneels down, and it
is stated in the UI so nobody believes the alarm is instant.

What this is not: it is not fall detection, because we do not observe the fall
event itself, only the resulting state. A person who was already lying down when
the camera started is detected the same as one who fell. For the purpose — get a
human to look at a frame — that distinction does not matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .track import Track


@dataclass
class PostureConfig:
    """All thresholds are dimensionless so they transfer between cameras."""

    down_aspect: float = 0.95  # width/height at or above this reads as non-upright
    upright_aspect: float = 0.70  # below this reads as upright (hysteresis band between)
    stillness_window_ms: float = 2500.0
    still_motion_ratio: float = 0.06  # max centre travel as a fraction of box diagonal
    min_duration_ms: float = 4000.0  # posture + stillness must hold this long
    min_samples: int = 6

    def __post_init__(self) -> None:
        if self.upright_aspect > self.down_aspect:
            raise ValueError("upright_aspect must be <= down_aspect")


@dataclass
class PostureState:
    track_id: int
    aspect: float = 0.0
    motion_ratio: float = 1.0
    non_upright: bool = False
    still: bool = False
    down: bool = False
    since_ms: float | None = None
    duration_ms: float = 0.0
    samples: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "aspect": round(self.aspect, 3),
            "motion_ratio": round(self.motion_ratio, 4),
            "non_upright": self.non_upright,
            "still": self.still,
            "down": self.down,
            "since_ms": self.since_ms,
            "duration_ms": round(self.duration_ms, 1),
            "samples": self.samples,
            "reason": self.reason,
        }


def box_aspect(box: tuple[float, float, float, float]) -> float:
    w = max(1e-6, box[2] - box[0])
    h = max(1e-6, box[3] - box[1])
    return float(w / h)


def motion_ratio(history: list[tuple[float, tuple[float, float, float, float]]]) -> float:
    """Peak centre displacement over the window, in units of the median box diagonal.

    Peak, not mean: a person who is still for two seconds and then rolls over is
    not still, and averaging hides that. Using the median diagonal rather than the
    latest one keeps a single bad box from rescaling the answer.
    """
    if len(history) < 2:
        return 1.0
    centres = np.array(
        [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for _, b in history], dtype=np.float64
    )
    diagonals = np.array(
        [np.hypot(b[2] - b[0], b[3] - b[1]) for _, b in history], dtype=np.float64
    )
    scale = float(np.median(diagonals))
    if scale <= 1e-6:
        return 1.0
    reference = centres.mean(axis=0)
    peak = float(np.max(np.linalg.norm(centres - reference, axis=1)))
    return peak / scale


class DownDetector:
    """Per-track person-down state. Feed it confirmed tracks once per frame."""

    def __init__(self, config: PostureConfig | None = None) -> None:
        self.config = config or PostureConfig()
        self.states: dict[int, PostureState] = {}
        self._candidate_since: dict[int, float] = {}

    def update(self, tracks: list[Track], timestamp_ms: float) -> list[PostureState]:
        cfg = self.config
        live = {t.track_id for t in tracks}
        for gone in set(self.states) - live:
            self.states.pop(gone, None)
            self._candidate_since.pop(gone, None)

        out: list[PostureState] = []
        for track in tracks:
            state = self.states.setdefault(track.track_id, PostureState(track.track_id))
            aspect = box_aspect(track.box)
            window = track.recent(cfg.stillness_window_ms, timestamp_ms)
            ratio = motion_ratio(window)

            state.aspect = aspect
            state.motion_ratio = ratio
            state.samples = len(window)

            if aspect >= cfg.down_aspect:
                state.non_upright = True
            elif aspect <= cfg.upright_aspect:
                state.non_upright = False
            state.still = (
                len(window) >= cfg.min_samples
                and (timestamp_ms - window[0][0]) >= cfg.stillness_window_ms * 0.6
                and ratio <= cfg.still_motion_ratio
            )

            candidate = state.non_upright and state.still
            if candidate:
                since = self._candidate_since.setdefault(track.track_id, timestamp_ms)
                state.duration_ms = timestamp_ms - since
                if state.duration_ms >= cfg.min_duration_ms and not state.down:
                    state.down = True
                    state.since_ms = timestamp_ms
            else:
                self._candidate_since.pop(track.track_id, None)
                state.duration_ms = 0.0
                state.down = False
                state.since_ms = None

            state.reason = _reason(state, cfg)
            out.append(state)
        return out

    def get(self, track_id: int) -> PostureState | None:
        return self.states.get(track_id)


def _reason(state: PostureState, cfg: PostureConfig) -> str:
    if state.down:
        return (
            f"box is wider than tall (w/h {state.aspect:.2f}) and has not moved more than "
            f"{state.motion_ratio * 100:.0f}% of its own size for "
            f"{state.duration_ms / 1000:.0f} s"
        )
    if state.non_upright and not state.still:
        return "not upright, but still moving: probably crouching or kneeling"
    if state.still and not state.non_upright:
        return "standing still"
    if state.samples < cfg.min_samples:
        return "not enough history yet to judge stillness"
    return "upright and moving"
