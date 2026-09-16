"""Guard-present checking, including the occlusion case that would otherwise
generate a false 'guard removed' every time somebody walks in front of it."""

from __future__ import annotations

import cv2
import numpy as np

from keepout import GuardChecker, GuardConfig, GuardStatus, Zone

from .conftest import make_scene

SIZE = (480, 320)
GUARD_ZONE = Zone(name="guard", points=((140, 120), (340, 120), (340, 250), (140, 250)),
                  reference_size=SIZE, kind="guard")


def scene(*, guard: bool, seed: int = 4, lighting: float = 1.0) -> np.ndarray:
    frame = make_scene(SIZE, seed=seed)
    if guard:
        cv2.rectangle(frame, (140, 120), (340, 250), (120, 150, 126), -1)
        for x in range(146, 340, 12):
            cv2.line(frame, (x, 124), (x, 246), (56, 92, 62), 1)
        for y in range(126, 250, 12):
            cv2.line(frame, (144, y), (336, y), (56, 92, 62), 1)
        cv2.rectangle(frame, (140, 120), (340, 250), (40, 70, 46), 4)
    else:
        cv2.rectangle(frame, (140, 120), (340, 250), (58, 60, 66), -1)
        cv2.rectangle(frame, (154, 130), (326, 234), (34, 36, 40), -1)
    if lighting != 1.0:
        frame = np.clip(frame.astype(np.float32) * lighting, 0, 255).astype(np.uint8)
    return frame


def test_without_a_reference_the_answer_is_unknown_not_present():
    checker = GuardChecker()
    obs = checker.check(scene(guard=True), timestamp_ms=0.0)
    assert obs.status is GuardStatus.UNKNOWN
    assert "no guard reference" in obs.note


def test_the_guard_still_being_there_reads_as_present():
    checker = GuardChecker(GuardConfig(min_state_ms=100.0))
    checker.learn(scene(guard=True), GUARD_ZONE, timestamp_ms=0.0)
    obs = checker.check(scene(guard=True), timestamp_ms=1000.0)
    assert obs.status is GuardStatus.PRESENT
    assert obs.edge_score > 0.9
    assert obs.appearance_score > 0.9


def test_removing_the_guard_is_detected():
    checker = GuardChecker(GuardConfig(min_state_ms=300.0))
    checker.learn(scene(guard=True), GUARD_ZONE, timestamp_ms=0.0)
    obs = None
    for t in (500.0, 900.0, 1400.0, 2000.0):
        obs = checker.check(scene(guard=False), timestamp_ms=t)
    assert obs.status is GuardStatus.MISSING
    assert obs.missing is True
    assert obs.score < GuardConfig().edge_threshold
    assert "guard is open, removed" in obs.note


def test_a_single_bad_frame_does_not_flip_the_state():
    checker = GuardChecker(GuardConfig(min_state_ms=1500.0))
    checker.learn(scene(guard=True), GUARD_ZONE, timestamp_ms=0.0)
    obs = checker.check(scene(guard=False), timestamp_ms=200.0)
    assert obs.status is GuardStatus.PRESENT, "one frame must not change the state"


def test_the_guard_coming_back_clears_the_state():
    checker = GuardChecker(GuardConfig(min_state_ms=200.0))
    checker.learn(scene(guard=True), GUARD_ZONE, timestamp_ms=0.0)
    for t in (400.0, 800.0, 1200.0):
        checker.check(scene(guard=False), timestamp_ms=t)
    assert checker.status is GuardStatus.MISSING
    for t in (1600.0, 2000.0, 2400.0):
        obs = checker.check(scene(guard=True), timestamp_ms=t)
    assert obs.status is GuardStatus.PRESENT


def test_somebody_standing_in_front_of_the_guard_returns_unknown_not_missing():
    """Without this rule the product cries wolf whenever anybody walks past."""
    checker = GuardChecker(GuardConfig(min_state_ms=100.0, max_occluded_fraction=0.35))
    checker.learn(scene(guard=True), GUARD_ZONE, timestamp_ms=0.0)

    blocked = scene(guard=True)
    cv2.rectangle(blocked, (150, 60), (330, 260), (250, 250, 250), -1)  # a body in the way
    person = (150.0, 60.0, 330.0, 260.0)

    obs = checker.check(blocked, timestamp_ms=1000.0, person_boxes=[person])
    assert obs.status is GuardStatus.UNKNOWN
    assert obs.occluded_fraction > 0.35
    assert "blocked by someone" in obs.note
    assert checker.status is GuardStatus.PRESENT, "the previous state must be held"


def test_partial_occlusion_still_answers_from_the_visible_part():
    checker = GuardChecker(GuardConfig(min_state_ms=100.0, max_occluded_fraction=0.5))
    checker.learn(scene(guard=True), GUARD_ZONE, timestamp_ms=0.0)
    frame = scene(guard=True)
    cv2.rectangle(frame, (140, 120), (200, 250), (250, 250, 250), -1)  # ~30% covered
    obs = checker.check(frame, timestamp_ms=1000.0, person_boxes=[(140.0, 120.0, 200.0, 250.0)])
    assert obs.status is GuardStatus.PRESENT
    assert 0.0 < obs.occluded_fraction < 0.5


def test_a_lighting_change_alone_does_not_read_as_a_removed_guard():
    """Edge correlation exists precisely so a dimmer does not raise an alert."""
    checker = GuardChecker(GuardConfig(min_state_ms=100.0))
    checker.learn(scene(guard=True), GUARD_ZONE, timestamp_ms=0.0)
    for t in (500.0, 1000.0, 1500.0):
        obs = checker.check(scene(guard=True, lighting=0.62), timestamp_ms=t)
    assert obs.status is GuardStatus.PRESENT, (
        f"a 38% dim was read as a removed guard (edge {obs.edge_score:.2f}, "
        f"appearance {obs.appearance_score:.2f})"
    )


def test_reset_forgets_the_reference():
    checker = GuardChecker()
    checker.learn(scene(guard=True), GUARD_ZONE)
    checker.reset()
    assert checker.reference is None
    assert checker.status is GuardStatus.UNKNOWN
