"""Person-down posture and stillness, and the crouching case it must not fire on."""

from __future__ import annotations

import pytest

from keepout import DownDetector, PostureConfig, Track, box_aspect, motion_ratio

from .conftest import lying_box, person_box


def make_track(track_id: int, boxes: list[tuple[float, float, float, float]],
               step_ms: float = 200.0) -> Track:
    track = Track(track_id=track_id, box=boxes[0], score=0.9, first_ms=0.0, last_ms=0.0)
    track.history = [(i * step_ms, b) for i, b in enumerate(boxes)]
    track.box = boxes[-1]
    track.last_ms = (len(boxes) - 1) * step_ms
    track.confirmed = True
    return track


def fall(n_upright: int, n_lying: int) -> list[tuple[float, float, float, float]]:
    """Standing, then on the floor in the same place. A fall needs the standing part."""
    return [person_box(300, 420) for _ in range(n_upright)] + [
        lying_box(300, 400) for _ in range(n_lying)]


def feed(detector, track, boxes, start: int = 0, step_ms: float = 200.0, frame_size=None):
    state = None
    for i, box in enumerate(boxes):
        t = (start + i) * step_ms
        track.box = box
        track.history.append((t, box))
        state = detector.update([track], timestamp_ms=t, frame_size=frame_size)[0]
    return state


def test_aspect_separates_standing_from_lying():
    assert box_aspect(person_box(100, 300)) < 0.5
    assert box_aspect(lying_box(100, 300)) > 1.0


def test_motion_ratio_is_scale_invariant():
    """A person near the camera and one far away, moving the same fraction of their
    own size, must produce the same stillness number."""
    near = [(0.0, (0.0, 0.0, 100.0, 300.0)), (1.0, (10.0, 0.0, 110.0, 300.0))]
    far = [(0.0, (0.0, 0.0, 50.0, 150.0)), (1.0, (5.0, 0.0, 55.0, 150.0))]
    assert motion_ratio(near) == pytest.approx(motion_ratio(far), rel=0.02)


def test_motion_ratio_needs_history():
    assert motion_ratio([]) == 1.0
    assert motion_ratio([(0.0, (0, 0, 10, 10))]) == 1.0


def test_a_motionless_person_on_the_floor_is_reported_down():
    cfg = PostureConfig(stillness_window_ms=2000.0, min_duration_ms=3000.0, min_samples=5)
    detector = DownDetector(cfg)
    boxes = fall(5, 60)
    track = make_track(1, boxes, step_ms=200.0)

    state = None
    for i in range(len(boxes)):
        track.box = boxes[i]
        track.history = [(j * 200.0, boxes[j]) for j in range(i + 1)]
        state = detector.update([track], timestamp_ms=i * 200.0)[0]
    assert state.down is True
    assert state.non_upright and state.still
    assert "not moved" in state.reason


def test_the_alarm_waits_the_configured_time():
    cfg = PostureConfig(stillness_window_ms=1000.0, min_duration_ms=4000.0, min_samples=4)
    detector = DownDetector(cfg)
    boxes = fall(5, 40)
    track = make_track(1, boxes)
    fired_at = None
    for i in range(len(boxes)):
        track.box = boxes[i]
        track.history = [(j * 200.0, boxes[j]) for j in range(i + 1)]
        state = detector.update([track], timestamp_ms=i * 200.0)[0]
        if state.down and fired_at is None:
            fired_at = i * 200.0
    assert fired_at is not None
    # Stillness needs its window, then the duration on top; never sooner than the duration.
    assert fired_at - 5 * 200.0 >= cfg.min_duration_ms


def test_a_crouching_worker_who_keeps_moving_is_not_reported_down():
    """The false-positive case that decides whether anyone keeps the system on."""
    cfg = PostureConfig(stillness_window_ms=1500.0, min_duration_ms=2500.0, min_samples=4)
    detector = DownDetector(cfg)
    boxes = []
    for i in range(50):
        # Wide box (crouched) but the centre shuffles back and forth constantly.
        shift = 22.0 if i % 2 == 0 else -22.0
        boxes.append(lying_box(300 + shift, 400))
    track = make_track(1, boxes)
    ever_down = False
    for i in range(len(boxes)):
        track.box = boxes[i]
        track.history = [(j * 200.0, boxes[j]) for j in range(i + 1)]
        state = detector.update([track], timestamp_ms=i * 200.0)[0]
        ever_down = ever_down or state.down
    assert ever_down is False
    assert "still moving" in state.reason


def test_a_person_standing_perfectly_still_is_not_reported_down():
    cfg = PostureConfig(stillness_window_ms=1000.0, min_duration_ms=2000.0, min_samples=4)
    detector = DownDetector(cfg)
    boxes = [person_box(300, 400) for _ in range(40)]
    track = make_track(1, boxes)
    for i in range(len(boxes)):
        track.box = boxes[i]
        track.history = [(j * 200.0, boxes[j]) for j in range(i + 1)]
        state = detector.update([track], timestamp_ms=i * 200.0)[0]
    assert state.down is False
    assert state.still is True
    assert state.non_upright is False
    assert state.reason == "standing still"


def test_getting_up_clears_the_state():
    cfg = PostureConfig(stillness_window_ms=800.0, min_duration_ms=1200.0, min_samples=3)
    detector = DownDetector(cfg)
    lying = fall(5, 25)
    track = make_track(1, lying)
    for i in range(len(lying)):
        track.box = lying[i]
        track.history = [(j * 200.0, lying[j]) for j in range(i + 1)]
        detector.update([track], timestamp_ms=i * 200.0)
    assert detector.get(1).down is True

    standing = [person_box(320 + i * 6, 400) for i in range(15)]
    for i, box in enumerate(standing):
        t = (30 + i) * 200.0
        track.box = box
        track.history.append((t, box))
        state = detector.update([track], timestamp_ms=t)[0]
    assert state.down is False


def test_states_are_dropped_when_a_track_disappears():
    detector = DownDetector()
    track = make_track(9, [person_box(100, 200)])
    detector.update([track], timestamp_ms=0.0)
    assert detector.get(9) is not None
    detector.update([], timestamp_ms=1000.0)
    assert detector.get(9) is None


def test_config_rejects_inverted_aspect_band():
    with pytest.raises(ValueError):
        PostureConfig(down_aspect=0.5, upright_aspect=0.9)


# ---------------------------------------------------------------------------
# Found on real footage: a static object on the frame edge became "a person who fell"
# ---------------------------------------------------------------------------


def test_a_wide_still_detection_never_seen_upright_is_not_a_fall():
    """Malta shot 4, 15.6 s: a bag or a stack of roof tiles, detected as a person,
    wider than tall, motionless for four seconds, raised a critical. It had never
    been a standing person, so it cannot have fallen."""
    cfg = PostureConfig(stillness_window_ms=1000.0, min_duration_ms=2000.0, min_samples=4)
    detector = DownDetector(cfg)
    boxes = [lying_box(300, 300) for _ in range(60)]
    track = make_track(1, boxes[:1])
    track.history = []
    state = feed(detector, track, boxes, frame_size=(960, 540))
    assert state.non_upright and state.still
    assert state.was_upright is False
    assert state.down is False
    assert "never seen standing" in state.reason


def test_a_box_touching_the_frame_edge_is_not_judged_for_posture():
    """A person cut off by the bottom of the picture is a wide box too. Even a track
    that was upright first cannot be called down from a truncated box."""
    cfg = PostureConfig(stillness_window_ms=1000.0, min_duration_ms=2000.0, min_samples=4)
    detector = DownDetector(cfg)
    upright = [person_box(300, 400) for _ in range(5)]
    truncated = [(240.0, 480.0, 360.0, 540.0) for _ in range(60)]  # bottom edge at 540
    track = make_track(1, upright[:1])
    track.history = []
    state = feed(detector, track, upright + truncated, frame_size=(960, 540))
    assert state.at_edge is True
    assert state.down is False


def test_a_fall_that_breaks_the_track_still_counts_as_upright_first():
    """The box changes shape too fast for the tracker, so the person on the floor is
    a new track id. It inherits "was upright" from the track that just went."""
    cfg = PostureConfig(stillness_window_ms=1000.0, min_duration_ms=2000.0, min_samples=4)
    detector = DownDetector(cfg)
    standing = make_track(1, [person_box(300, 420)])
    standing.history = []
    feed(detector, standing, [person_box(300, 420) for _ in range(8)])  # until 1.4 s

    lying = [lying_box(300, 400) for _ in range(40)]
    floor = Track(track_id=2, box=lying[0], score=0.9, first_ms=1800.0, last_ms=1800.0)
    floor.confirmed = True
    state = feed(detector, floor, lying, start=9, frame_size=(960, 540))
    assert state.was_upright is True
    assert state.down is True


def test_a_new_still_wide_box_far_from_anyone_who_stood_does_not_inherit():
    cfg = PostureConfig(stillness_window_ms=1000.0, min_duration_ms=2000.0, min_samples=4)
    detector = DownDetector(cfg)
    standing = make_track(1, [person_box(100, 420)])
    standing.history = []
    feed(detector, standing, [person_box(100, 420) for _ in range(8)])

    lying = [lying_box(700, 300) for _ in range(40)]
    far = Track(track_id=2, box=lying[0], score=0.9, first_ms=1800.0, last_ms=1800.0)
    far.confirmed = True
    state = feed(detector, far, lying, start=9, frame_size=(960, 540))
    assert state.was_upright is False
    assert state.down is False
