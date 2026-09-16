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

**Upright first.** A track only becomes eligible for "down" after it has been seen
upright (`min_upright_samples` boxes at or below `upright_aspect`). This came from
real footage: on a construction clip from Wikimedia Commons (Malta, 2013) a bag or
a stack of roof tiles on the bottom edge of the frame was detected as a person,
its box was wider than tall, it never moved, and after four seconds it raised a
critical. Nothing about it had ever been a standing person. A fall needs somebody
who was standing, so now it needs a track that was.

A fall often breaks the track, because the box changes shape too fast for the
tracker to match it. So a new track inherits "was upright" from a track that was
upright, stopped being seen within `inherit_window_ms`, and was last seen within
one box height of where the new track starts.

**Edges.** A box touching the frame edge is a person cut off by the edge of the
picture, and its aspect ratio says nothing about posture. It is never judged not
upright.

What this is not: it is not fall detection from the fall event itself, only from
the state after it, with the upright history as the one piece of before. The cost
of that rule is stated plainly: **a person who is already lying down when the
camera starts, and was never seen standing, does not raise `person_down`.**
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
    min_upright_samples: int = 3  # upright boxes seen before "down" is possible
    inherit_window_ms: float = 3000.0  # a fall that broke the track keeps its history
    edge_margin_px: float = 3.0  # boxes this close to the frame edge are truncated

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
    was_upright: bool = False
    at_edge: bool = False
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
            "was_upright": self.was_upright,
            "at_edge": self.at_edge,
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
        # Survives a missed frame, unlike `states`: track id -> upright boxes seen.
        self._upright_count: dict[int, int] = {}
        # Tracks that were upright: track id -> (last seen ms, last box).
        self._upright_last: dict[int, tuple[float, tuple[float, float, float, float]]] = {}

    def _inherit(self, track: Track, live: set[int]) -> bool:
        cfg = self.config
        box = track.box
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        for other_id, (seen_ms, other) in self._upright_last.items():
            if other_id in live or other_id == track.track_id:
                continue
            if not 0.0 <= track.first_ms - seen_ms <= cfg.inherit_window_ms:
                continue
            ox, oy = (other[0] + other[2]) / 2.0, (other[1] + other[3]) / 2.0
            height = max(other[3] - other[1], box[3] - box[1])
            if np.hypot(cx - ox, cy - oy) <= height:
                return True
        return False

    def update(self, tracks: list[Track], timestamp_ms: float,
               frame_size: tuple[int, int] | None = None) -> list[PostureState]:
        cfg = self.config
        live = {t.track_id for t in tracks}
        for gone in set(self.states) - live:
            self.states.pop(gone, None)
            self._candidate_since.pop(gone, None)
        horizon = timestamp_ms - 60_000.0
        for stale in [k for k, (seen, _) in self._upright_last.items() if seen < horizon]:
            self._upright_last.pop(stale, None)
            self._upright_count.pop(stale, None)

        out: list[PostureState] = []
        for track in tracks:
            state = self.states.setdefault(track.track_id, PostureState(track.track_id))
            aspect = box_aspect(track.box)
            window = track.recent(cfg.stillness_window_ms, timestamp_ms)
            ratio = motion_ratio(window)

            state.aspect = aspect
            state.motion_ratio = ratio
            state.samples = len(window)
            state.at_edge = frame_size is not None and _touches_edge(
                track.box, frame_size, cfg.edge_margin_px)

            if aspect <= cfg.upright_aspect and not state.at_edge:
                count = self._upright_count.get(track.track_id, 0) + 1
                self._upright_count[track.track_id] = count
            count = self._upright_count.get(track.track_id, 0)
            if count >= cfg.min_upright_samples:
                self._upright_last[track.track_id] = (timestamp_ms, track.box)
            elif (count == 0 and track.track_id not in self._upright_last
                  and self._inherit(track, live)):
                self._upright_count[track.track_id] = cfg.min_upright_samples
                self._upright_last[track.track_id] = (timestamp_ms, track.box)
            state.was_upright = (
                self._upright_count.get(track.track_id, 0) >= cfg.min_upright_samples)

            if state.at_edge:
                state.non_upright = False
            elif aspect >= cfg.down_aspect:
                state.non_upright = True
            elif aspect <= cfg.upright_aspect:
                state.non_upright = False
            state.still = (
                len(window) >= cfg.min_samples
                and (timestamp_ms - window[0][0]) >= cfg.stillness_window_ms * 0.6
                and ratio <= cfg.still_motion_ratio
            )

            candidate = state.non_upright and state.still and state.was_upright
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


def _touches_edge(box: tuple[float, float, float, float], size: tuple[int, int],
                  margin: float) -> bool:
    w, h = size
    return box[0] <= margin or box[1] <= margin or box[2] >= w - margin or box[3] >= h - margin


def _reason(state: PostureState, cfg: PostureConfig) -> str:
    if state.at_edge:
        return "box touches the frame edge, so its shape says nothing about posture"
    if state.non_upright and state.still and not state.was_upright:
        return "not upright and still, but never seen standing: not treated as a fall"
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
