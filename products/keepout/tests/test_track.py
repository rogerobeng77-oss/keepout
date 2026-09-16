"""The tracker we had to write because OpenCV 5 removed the trackers.

The Hungarian solve is checked against brute-force enumeration on random small
matrices, which is the only way to know an assignment implementation is right.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from keepout.track import BoxTracker, Track, TrackerConfig, hungarian, iou


def brute_force(cost: np.ndarray) -> float:
    rows, cols = cost.shape
    k = min(rows, cols)
    best = float("inf")
    for row_pick in itertools.combinations(range(rows), k):
        for col_pick in itertools.permutations(range(cols), k):
            total = sum(cost[r, c] for r, c in zip(row_pick, col_pick, strict=True))
            best = min(best, total)
    return best


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3), (4, 4), (2, 4), (4, 2), (3, 5)])
def test_hungarian_matches_brute_force(shape):
    rng = np.random.default_rng(shape[0] * 100 + shape[1])
    for _ in range(12):
        cost = rng.uniform(0, 10, shape)
        pairs = hungarian(cost)
        assert len(pairs) == min(shape)
        assert len({r for r, _ in pairs}) == len(pairs), "a row was assigned twice"
        assert len({c for _, c in pairs}) == len(pairs), "a column was assigned twice"
        total = sum(cost[r, c] for r, c in pairs)
        assert total == pytest.approx(brute_force(cost), abs=1e-9)


def test_hungarian_handles_empty():
    assert hungarian(np.zeros((0, 0))) == []
    assert hungarian(np.zeros((0, 3))) == []


def test_hungarian_prefers_the_globally_cheaper_pairing():
    # Greedy takes (0,0)=1 then is forced into (1,1)=100, total 101.
    # The optimal answer is (0,1)+(1,0) = 2+3 = 5.
    cost = np.array([[1.0, 2.0], [3.0, 100.0]])
    assert hungarian(cost) == [(0, 1), (1, 0)]


def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_track_keeps_identity_through_a_straight_walk():
    tracker = BoxTracker(TrackerConfig(min_hits=2))
    ids = []
    for i in range(12):
        x = 50.0 + i * 9.0
        tracks = tracker.update([(x, 100, x + 40, 220)], [0.9], timestamp_ms=i * 100.0)
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1, "a straight-line walk must not change identity"
    assert tracker.tracks[0].confirmed
    assert tracker.tracks[0].hits == 12


def test_two_people_crossing_keep_their_own_identities():
    """The case that matters: a wrong swap here puts an incident on the wrong person."""
    tracker = BoxTracker(TrackerConfig(min_hits=2))
    left_id = right_id = None
    for i in range(20):
        a_x = 40.0 + i * 12.0  # left to right
        b_x = 280.0 - i * 12.0  # right to left, different height so they are separable
        boxes = [(a_x, 100, a_x + 36, 220), (b_x, 90, b_x + 40, 240)]
        tracks = tracker.update(boxes, [0.9, 0.9], timestamp_ms=i * 100.0)
        by_x = sorted(tracks, key=lambda t: t.box[0])
        if i == 0:
            left_id, right_id = by_x[0].track_id, by_x[1].track_id
        if i == 19:
            # They have swapped sides; the ids must have travelled with them.
            moving_right = next(t for t in tracks if t.velocity[0] > 0)
            moving_left = next(t for t in tracks if t.velocity[0] < 0)
            assert moving_right.track_id == left_id
            assert moving_left.track_id == right_id
    assert len(tracker.tracks) == 2


def test_track_survives_a_short_gap_then_dies():
    cfg = TrackerConfig(min_hits=2, max_misses=3)
    tracker = BoxTracker(cfg)
    for i in range(5):
        x = 50.0 + i * 8
        tracker.update([(x, 100, x + 40, 220)], [0.9], timestamp_ms=i * 100.0)
    original = tracker.tracks[0].track_id

    for i in range(5, 8):  # three missed frames, still inside max_misses
        tracker.update([], [], timestamp_ms=i * 100.0)
    assert tracker.tracks and tracker.tracks[0].track_id == original

    x = 50.0 + 8 * 8
    tracker.update([(x, 100, x + 40, 220)], [0.9], timestamp_ms=800.0)
    assert tracker.tracks[0].track_id == original, "the track should have been recovered"

    for i in range(9, 20):
        tracker.update([], [], timestamp_ms=i * 100.0)
    assert tracker.tracks == []
    assert original in tracker.removed_ids or tracker.get(original) is None


def test_far_apart_detection_starts_a_new_track_rather_than_teleporting():
    tracker = BoxTracker(TrackerConfig(min_hits=1, centre_gate=1.5))
    tracker.update([(10, 100, 50, 220)], [0.9], timestamp_ms=0.0)
    first = tracker.tracks[0].track_id
    tracker.update([(600, 100, 640, 220)], [0.9], timestamp_ms=100.0)
    ids = {t.track_id for t in tracker.tracks}
    assert first in ids
    assert len(ids) == 2, "a box on the other side of the frame is a different person"


def test_velocity_points_the_right_way():
    tracker = BoxTracker(TrackerConfig(min_hits=1))
    for i in range(6):
        x = 100.0 + i * 20.0
        tracker.update([(x, 50, x + 40, 170)], [0.9], timestamp_ms=i * 100.0)
    track = tracker.tracks[0]
    assert track.velocity[0] > 0
    assert track.speed_px_per_s() == pytest.approx(200.0, rel=0.25)


def test_predict_extrapolates_forwards():
    track = Track(track_id=1, box=(0, 0, 10, 10), score=0.9, first_ms=0.0, last_ms=0.0)
    track.velocity = (0.1, 0.0)  # px per ms
    assert track.predict(100.0)[0] == pytest.approx(10.0)


def test_mismatched_inputs_are_rejected():
    tracker = BoxTracker()
    with pytest.raises(ValueError):
        tracker.update([(0, 0, 1, 1)], [], timestamp_ms=0.0)
