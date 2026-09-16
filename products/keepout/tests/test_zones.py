"""Zone geometry, hysteresis, and the motion-based zone proposal."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from keepout import ContactPoint, DwellConfig, Zone, ZoneMonitor, propose_zone_from_motion
from keepout.zones import contact_point

from .conftest import person_box

SQUARE = Zone(
    name="square",
    points=((100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)),
    reference_size=(400, 400),
)


def test_zone_needs_three_points():
    with pytest.raises(ValueError):
        Zone(name="bad", points=((0, 0), (1, 1)), reference_size=(10, 10))


def test_area_and_containment_are_exact_for_a_square():
    assert SQUARE.area_px == pytest.approx(200 * 200)
    assert SQUARE.contains((200, 200))
    assert not SQUARE.contains((50, 200))
    # Signed distance is positive inside and equals the distance to the nearest edge.
    assert SQUARE.signed_distance((200, 200)) == pytest.approx(100.0)
    assert SQUARE.signed_distance((110, 200)) == pytest.approx(10.0)
    assert SQUARE.signed_distance((90, 200)) == pytest.approx(-10.0)


def test_scaling_to_a_different_resolution_keeps_the_same_relative_shape():
    scaled = SQUARE.scaled_to((800, 800))
    assert scaled[0].tolist() == [200.0, 200.0]
    assert SQUARE.contains((150, 150), size=(800, 800)) is False  # outside the scaled square
    assert SQUARE.contains((400, 400), size=(800, 800)) is True


def test_mask_has_the_expected_number_of_pixels():
    mask = SQUARE.mask((400, 400))
    assert mask.shape == (400, 400)
    assert int(np.count_nonzero(mask)) == pytest.approx(200 * 200, rel=0.02)


def test_contact_point_modes():
    box = (100.0, 50.0, 140.0, 250.0)
    assert contact_point(box, ContactPoint.FEET) == (120.0, 250.0)
    assert contact_point(box, ContactPoint.CENTROID) == (120.0, 150.0)


def test_box_overlap_fraction_is_a_real_fraction():
    # Box exactly half inside the square: x from 200 to 400, all inside vertically.
    assert SQUARE.box_overlap_fraction((200, 150, 400, 250)) == pytest.approx(0.5, abs=0.02)
    assert SQUARE.box_overlap_fraction((0, 0, 50, 50)) == 0.0
    assert SQUARE.box_overlap_fraction((150, 150, 250, 250)) == pytest.approx(1.0, abs=0.02)


def test_feet_contact_catches_someone_standing_on_the_line():
    """A person whose centroid is outside but whose feet are inside is inside."""
    zone = Zone(name="floor", points=((0, 200), (400, 200), (400, 400), (0, 400)),
                reference_size=(400, 400), contact=ContactPoint.FEET)
    box = person_box(cx=200, feet_y=210, height=150)  # centroid at y=135, feet at 210
    assert zone.contains(contact_point(box, ContactPoint.CENTROID)) is False
    assert zone.contains(contact_point(box, ContactPoint.FEET)) is True


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


def test_hysteresis_rejects_a_bad_config():
    with pytest.raises(ValueError):
        DwellConfig(enter_margin_px=0.0, exit_margin_px=5.0)


def test_a_person_jittering_on_the_boundary_produces_one_entry_not_forty():
    """The boundary-jitter case, which is what destroys trust in a real install."""
    monitor = ZoneMonitor(SQUARE, DwellConfig(enter_margin_px=0.0, exit_margin_px=-12.0,
                                              exit_grace_ms=600.0))
    entries = 0
    was_inside = False
    # Feet oscillate between x=98 and x=104, i.e. +/- a few px across the x=100 edge.
    for i in range(60):
        x = 104.0 if i % 2 == 0 else 98.0
        box = (x - 20, 100, x + 20, 200)
        state = monitor.update(1, box, timestamp_ms=i * 100.0)
        if state.inside and not was_inside:
            entries += 1
        was_inside = state.inside
    assert entries == 1, f"boundary jitter produced {entries} entries"
    assert monitor.states[1].inside is True


def test_a_real_exit_is_recognised_after_the_grace_period():
    monitor = ZoneMonitor(SQUARE, DwellConfig(exit_margin_px=-12.0, exit_grace_ms=500.0))
    monitor.update(1, (180, 180, 220, 220), timestamp_ms=0.0)
    assert monitor.states[1].inside

    # Step well outside. The exit is not immediate.
    monitor.update(1, (20, 180, 60, 220), timestamp_ms=100.0)
    assert monitor.states[1].inside, "exit must not fire on the first frame outside"

    monitor.update(1, (20, 180, 60, 220), timestamp_ms=1000.0)
    assert not monitor.states[1].inside


def test_dwell_time_is_measured_from_entry():
    monitor = ZoneMonitor(SQUARE, DwellConfig(min_dwell_ms=1000.0))
    for t in (0.0, 500.0, 1200.0, 2000.0):
        monitor.update(7, (180, 180, 220, 220), timestamp_ms=t)
    occ = monitor.states[7]
    assert occ.dwell_ms(2000.0) == pytest.approx(2000.0)
    assert [o.track_id for o in monitor.qualified_occupants(2000.0)] == [7]
    assert monitor.qualified_occupants(500.0) == []


def test_sweep_closes_a_track_that_stopped_being_detected():
    monitor = ZoneMonitor(SQUARE, DwellConfig(exit_grace_ms=800.0))
    monitor.update(3, (180, 180, 220, 220), timestamp_ms=0.0)
    assert monitor.occupants()
    closed = monitor.sweep(timestamp_ms=2000.0)
    assert [c.track_id for c in closed] == [3]
    assert monitor.occupants() == []


def test_forget_removes_state():
    monitor = ZoneMonitor(SQUARE)
    monitor.update(5, (180, 180, 220, 220), timestamp_ms=0.0)
    monitor.forget({5})
    assert monitor.states == {}


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------


def _moving_block_frames(n: int = 14, size=(400, 300)) -> list[np.ndarray]:
    w, h = size
    rng = np.random.default_rng(11)
    base = rng.integers(70, 130, (h, w), np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    frames = []
    for i in range(n):
        f = base.copy()
        # A bar that slides back and forth inside a fixed 120x60 region.
        x = 150 + (i % 4) * 8
        cv2.rectangle(f, (x, 120), (x + 60, 170), (240, 240, 240), -1)
        frames.append(f)
    return frames


def test_proposal_finds_the_moving_region_and_pads_it():
    frames = _moving_block_frames()
    proposal = propose_zone_from_motion(frames, dilate_px=20, min_area_fraction=0.0005)
    assert proposal.zone is not None, proposal.reason
    zone = proposal.zone
    assert zone.reference_size == (400, 300)
    # The moving bar's own centre must be inside the proposed zone.
    assert zone.contains((190, 145))
    # And the padding must push the boundary out past the bar itself.
    assert zone.area_px > 60 * 50
    assert proposal.confidence > 0.0


def test_proposal_refuses_a_still_scene_instead_of_returning_an_empty_polygon():
    still = [np.full((200, 200, 3), 100, np.uint8) for _ in range(8)]
    proposal = propose_zone_from_motion(still)
    assert proposal.zone is None
    assert "no motion" in proposal.reason


def test_proposal_refuses_when_the_whole_frame_moves():
    rng = np.random.default_rng(5)
    frames = [rng.integers(0, 255, (200, 200, 3), dtype=np.uint8) for _ in range(8)]
    proposal = propose_zone_from_motion(frames, min_area_fraction=0.0001)
    assert proposal.zone is None
    assert "camera" in proposal.reason or "too small" in proposal.reason


def test_proposal_needs_enough_frames():
    assert propose_zone_from_motion([np.zeros((10, 10, 3), np.uint8)]).zone is None
