"""The honesty rails. Each of these is a way the product can be blind while
looking like it is working, which is the failure mode we care most about."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from keepout import ViewConfig, ViewGuard, ViewProblem

from .conftest import make_scene

SIZE = (480, 320)


def noisy(frame: np.ndarray, sigma: float = 1.2, seed: int = 0) -> np.ndarray:
    """Real sensors always add a little noise. A frozen feed does not."""
    rng = np.random.default_rng(seed)
    out = frame.astype(np.float32) + rng.normal(0, sigma, frame.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def test_a_normal_frame_is_usable():
    guard = ViewGuard()
    reference = make_scene(SIZE)
    guard.set_reference(reference)
    status = guard.check(noisy(reference, seed=1))
    assert status.usable is True
    assert status.problems == []
    assert status.shift_px < 3.0


def test_a_shifted_camera_is_detected_and_the_shift_is_reported():
    guard = ViewGuard(ViewConfig(max_shift_px=10.0))
    reference = make_scene(SIZE)
    guard.set_reference(reference)

    m = np.float32([[1, 0, 28], [0, 1, 16]])
    moved = cv2.warpAffine(reference, m, SIZE, borderMode=cv2.BORDER_REPLICATE)
    status = guard.check(noisy(moved, seed=2))

    assert status.usable is False
    assert status.has(ViewProblem.CAMERA_MOVED)
    assert status.shift_px > 10.0
    assert "Re-draw it" in " ".join(status.messages)


def test_a_small_nudge_within_tolerance_is_not_flagged():
    guard = ViewGuard(ViewConfig(max_shift_px=14.0))
    reference = make_scene(SIZE)
    guard.set_reference(reference)
    m = np.float32([[1, 0, 3], [0, 1, 2]])
    nudged = cv2.warpAffine(reference, m, SIZE, borderMode=cv2.BORDER_REPLICATE)
    status = guard.check(noisy(nudged, seed=3))
    assert not status.has(ViewProblem.CAMERA_MOVED), f"shift measured {status.shift_px:.1f}"


def test_a_rotated_camera_is_detected():
    guard = ViewGuard(ViewConfig(max_shift_px=40.0, max_rotation_deg=1.5))
    reference = make_scene(SIZE)
    guard.set_reference(reference)
    m = cv2.getRotationMatrix2D((SIZE[0] / 2, SIZE[1] / 2), 6.0, 1.0)
    rotated = cv2.warpAffine(reference, m, SIZE, borderMode=cv2.BORDER_REPLICATE)
    status = guard.check(noisy(rotated, seed=4))
    assert status.has(ViewProblem.CAMERA_MOVED)


def test_a_blocked_lens_is_detected():
    guard = ViewGuard()
    guard.set_reference(make_scene(SIZE))
    blocked = np.full((SIZE[1], SIZE[0], 3), 118, np.uint8)
    status = guard.check(noisy(blocked, sigma=0.4, seed=5))
    assert status.has(ViewProblem.LENS_BLOCKED)
    assert status.usable is False
    assert status.edge_fraction < 0.01


def test_darkness_is_detected():
    guard = ViewGuard()
    reference = make_scene(SIZE)
    guard.set_reference(reference)
    dark = (reference.astype(np.float32) * 0.04).astype(np.uint8)
    status = guard.check(dark)
    assert status.has(ViewProblem.TOO_DARK)
    assert status.median_luma < 22.0


def test_a_dim_but_workable_scene_is_not_called_too_dark():
    guard = ViewGuard()
    reference = make_scene(SIZE)
    guard.set_reference(reference)
    dim = (reference.astype(np.float32) * 0.45).astype(np.uint8)
    status = guard.check(noisy(dim, seed=6))
    assert not status.has(ViewProblem.TOO_DARK), f"median luma {status.median_luma}"


def test_a_frozen_feed_is_detected_only_after_enough_identical_frames():
    guard = ViewGuard(ViewConfig(frozen_frames=8))
    reference = make_scene(SIZE)
    guard.set_reference(reference)
    frame = noisy(reference, seed=7)
    for i in range(7):
        status = guard.check(frame)
        assert not status.has(ViewProblem.FROZEN_FEED), f"fired after {i + 1} frames"
    for _ in range(3):
        status = guard.check(frame)
    assert status.has(ViewProblem.FROZEN_FEED)
    assert status.identical_frames >= 8


def test_live_noise_does_not_read_as_frozen():
    guard = ViewGuard(ViewConfig(frozen_frames=6))
    reference = make_scene(SIZE)
    guard.set_reference(reference)
    for i in range(25):
        status = guard.check(noisy(reference, sigma=2.0, seed=100 + i))
    assert not status.has(ViewProblem.FROZEN_FEED)


def test_a_size_change_is_reported_rather_than_silently_rescaled():
    guard = ViewGuard()
    guard.set_reference(make_scene(SIZE))
    other = make_scene((640, 480))
    status = guard.check(other)
    assert status.has(ViewProblem.SIZE_CHANGED)


def test_without_a_reference_only_the_frame_level_checks_run():
    guard = ViewGuard()
    status = guard.check(make_scene(SIZE))
    assert status.usable is True
    assert status.shift_px == 0.0


def test_camera_move_and_blocked_lens_do_not_double_report():
    """A covered lens cannot be aligned. It must be reported as blocked, not moved."""
    guard = ViewGuard()
    guard.set_reference(make_scene(SIZE))
    blocked = np.full((SIZE[1], SIZE[0], 3), 118, np.uint8)
    status = guard.check(noisy(blocked, sigma=0.4, seed=9))
    assert status.has(ViewProblem.LENS_BLOCKED)
    assert not status.has(ViewProblem.CAMERA_MOVED)


@pytest.mark.parametrize("problem", list(ViewProblem))
def test_every_problem_has_operator_facing_text(problem):
    from keepout.viewcheck import HUMAN_TEXT

    assert problem in HUMAN_TEXT
    assert len(HUMAN_TEXT[problem]) > 40
