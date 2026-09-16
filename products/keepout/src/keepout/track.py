"""Multi-object tracking without OpenCV's trackers, because OpenCV 5 removed them.

`cv2.TrackerCSRT`, `cv2.TrackerKCF` and the whole `cv2.legacy` namespace are gone
from the main `opencv-python` wheel in 5.0. Rather
than pull in `opencv-contrib-python`, which conflicts with the main wheel and
doubles the image, we associate detections ourselves.

The method is detection-to-track assignment by minimum total cost, solved
exactly with the Hungarian algorithm, with a constant-velocity predictor so a
person who is missed for a frame or two keeps their identity. Cost is a blend of
IoU distance and centre distance: IoU alone loses a fast-moving person whose
boxes stop overlapping between frames, and centre distance alone swaps two
people who pass each other.

Why exact assignment rather than greedy: greedy matching fails in exactly the
case this product exists for. Two workers near a machine, one steps into the
zone, and a greedy pass can hand the entering box to the wrong track, moving the
incident onto a person who never entered. That is a wrong accusation with an
evidence frame attached. The Hungarian solve is O(n^3) on a matrix that is never
bigger than about 20x20, so it costs nothing worth saving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

Box = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment on a rectangular matrix. Returns (row, col) pairs.

    A direct implementation of the O(n^3) Jonker-Volgenant shortest-augmenting-path
    form of the Hungarian algorithm, which is what `scipy.optimize.linear_sum_assignment`
    runs. We implement it rather than depend on SciPy so the container stays small,
    and it is tested against brute force on random matrices in `tests/test_track.py`.

    Rows are padded implicitly: if there are more rows than columns the matrix is
    transposed and the result flipped back, so the shorter side always drives.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.size == 0:
        return []
    if matrix.ndim != 2:
        raise ValueError("cost must be 2-D")
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    n_rows, n_cols = matrix.shape

    # u, v are the dual potentials; `assignment[j]` is the row matched to column j.
    u = np.zeros(n_rows + 1)
    v = np.zeros(n_cols + 1)
    assignment = np.full(n_cols + 1, -1, dtype=int)
    way = np.zeros(n_cols + 1, dtype=int)

    for i in range(1, n_rows + 1):
        assignment[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = assignment[j0]
            delta, j1 = np.inf, -1
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 < 0:  # pragma: no cover - only reachable on a malformed matrix
                break
            for j in range(n_cols + 1):
                if used[j]:
                    u[assignment[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if assignment[j0] == -1:
                break
        while j0:
            j1 = way[j0]
            assignment[j0] = assignment[j1]
            j0 = j1

    pairs = [(assignment[j] - 1, j - 1) for j in range(1, n_cols + 1) if assignment[j] > 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _centre(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _diagonal(box: Box) -> float:
    w, h = box[2] - box[0], box[3] - box[1]
    return float(np.hypot(w, h)) or 1.0


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


@dataclass
class Track:
    """One person, followed across frames.

    `velocity` is pixels per millisecond, estimated by exponential smoothing. It
    exists to predict where a missed detection would have been, and to feed the
    stillness test that `posture.py` uses for the person-down state.
    """

    track_id: int
    box: Box
    score: float
    first_ms: float
    last_ms: float
    hits: int = 1
    misses: int = 0
    age: int = 1
    velocity: tuple[float, float] = (0.0, 0.0)
    history: list[tuple[float, Box]] = field(default_factory=list)
    confirmed: bool = False
    first_box: Box | None = None  # where the track was born; history gets trimmed

    HISTORY_LIMIT = 240

    def box_at(self, timestamp_ms: float, tolerance_ms: float) -> Box | None:
        """The detected box nearest to a moment, if there is one close enough."""
        best, gap = None, tolerance_ms
        for t, b in self.history:
            if abs(t - timestamp_ms) <= gap:
                best, gap = b, abs(t - timestamp_ms)
        return best

    def predict(self, timestamp_ms: float) -> Box:
        dt = max(0.0, timestamp_ms - self.last_ms)
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        x1, y1, x2, y2 = self.box
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def update(self, box: Box, score: float, timestamp_ms: float, smoothing: float = 0.5) -> None:
        dt = timestamp_ms - self.last_ms
        if dt > 0:
            old, new = _centre(self.box), _centre(box)
            vx, vy = (new[0] - old[0]) / dt, (new[1] - old[1]) / dt
            a = smoothing
            self.velocity = (a * vx + (1 - a) * self.velocity[0],
                             a * vy + (1 - a) * self.velocity[1])
        self.box = box
        self.score = score
        self.last_ms = timestamp_ms
        self.hits += 1
        self.misses = 0
        self.history.append((timestamp_ms, box))
        if len(self.history) > self.HISTORY_LIMIT:
            del self.history[: len(self.history) - self.HISTORY_LIMIT]

    def mark_missed(self, timestamp_ms: float) -> None:
        self.misses += 1
        self.age += 1
        self.last_ms = timestamp_ms

    def speed_px_per_s(self) -> float:
        return float(np.hypot(*self.velocity) * 1000.0)

    def recent(self, window_ms: float, now_ms: float) -> list[tuple[float, Box]]:
        return [(t, b) for t, b in self.history if now_ms - t <= window_ms]

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.box
        return {
            "track_id": self.track_id,
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "score": round(self.score, 3),
            "hits": self.hits,
            "misses": self.misses,
            "age": self.age,
            "confirmed": self.confirmed,
            "speed_px_per_s": round(self.speed_px_per_s(), 2),
            "first_ms": round(self.first_ms, 1),
            "last_ms": round(self.last_ms, 1),
        }


@dataclass
class TrackerConfig:
    iou_weight: float = 0.6
    max_cost: float = 0.72  # above this an assignment is rejected as a coincidence
    max_misses: int = 12  # frames a track survives unmatched
    min_hits: int = 3  # detections before a track is 'confirmed' and can raise an incident
    centre_gate: float = 2.5  # reject pairs further apart than this many box diagonals
    velocity_smoothing: float = 0.5


class BoxTracker:
    """Detection-to-track association with a constant-velocity predictor.

    Deliberately not a Kalman filter. A Kalman filter buys smoother boxes and
    costs a covariance model we have no data to tune. Everything downstream needs
    identity continuity and a rough speed, both of which this gives.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1
        self.removed_ids: list[int] = []

    def cost_matrix(self, boxes: list[Box], timestamp_ms: float) -> np.ndarray:
        """Rows are tracks, columns are detections. Lower is better."""
        cfg = self.config
        n_t, n_d = len(self.tracks), len(boxes)
        cost = np.full((n_t, n_d), 1e6, dtype=np.float64)
        for i, track in enumerate(self.tracks):
            predicted = track.predict(timestamp_ms)
            pc = _centre(predicted)
            gate = cfg.centre_gate * _diagonal(predicted)
            for j, det in enumerate(boxes):
                dc = _centre(det)
                distance = float(np.hypot(pc[0] - dc[0], pc[1] - dc[1]))
                if distance > gate:
                    continue
                iou_cost = 1.0 - iou(predicted, det)
                dist_cost = min(1.0, distance / gate)
                cost[i, j] = cfg.iou_weight * iou_cost + (1.0 - cfg.iou_weight) * dist_cost
        return cost

    def update(self, boxes: list[Box], scores: list[float], timestamp_ms: float) -> list[Track]:
        """One frame of association. Returns the live tracks, confirmed or not."""
        if len(boxes) != len(scores):
            raise ValueError("boxes and scores must be the same length")
        cfg = self.config
        self.removed_ids = []

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        if self.tracks and boxes:
            cost = self.cost_matrix(boxes, timestamp_ms)
            for i, j in hungarian(cost):
                if cost[i, j] > cfg.max_cost:
                    continue
                self.tracks[i].update(boxes[j], scores[j], timestamp_ms, cfg.velocity_smoothing)
                matched_tracks.add(i)
                matched_dets.add(j)

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.mark_missed(timestamp_ms)
            if track.hits >= cfg.min_hits:
                track.confirmed = True

        for j, box in enumerate(boxes):
            if j in matched_dets:
                continue
            track = Track(
                track_id=self._next_id,
                box=box,
                score=scores[j],
                first_ms=timestamp_ms,
                last_ms=timestamp_ms,
                first_box=box,
            )
            track.history.append((timestamp_ms, box))
            track.confirmed = cfg.min_hits <= 1
            self.tracks.append(track)
            self._next_id += 1

        survivors = []
        for track in self.tracks:
            if track.misses > cfg.max_misses:
                self.removed_ids.append(track.track_id)
            else:
                survivors.append(track)
        self.tracks = survivors
        return list(self.tracks)

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed and t.misses == 0]

    def get(self, track_id: int) -> Track | None:
        return next((t for t in self.tracks if t.track_id == track_id), None)
