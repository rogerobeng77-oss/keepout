"""Machine-running detection, including the case where it cannot be decided."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from keepout import MachineConfig, MotionEnergy, Zone, suggest_thresholds

from .conftest import make_scene

SIZE = (480, 320)
MACHINE = Zone(name="machine", points=((100, 60), (380, 60), (380, 180), (100, 180)),
               reference_size=SIZE, kind="machine")


def running_frames(n: int, size=SIZE, amplitude: int = 9) -> list[np.ndarray]:
    """A belt whose cleats scroll inside the machine region, and nothing else moves."""
    base = make_scene(size, seed=4)
    frames = []
    for i in range(n):
        f = base.copy()
        offset = (i * amplitude) % 40
        for x in range(100 - 40, 380 + 40, 40):
            cv2.line(f, (x + offset, 70), (x + offset + 18, 170), (235, 235, 235), 6)
        frames.append(f)
    return frames


def stopped_frames(n: int, size=SIZE) -> list[np.ndarray]:
    """The same scene, belt still, with the faint sensor noise a real camera has."""
    base = running_frames(1, size)[0]
    rng = np.random.default_rng(2)
    out = []
    for _ in range(n):
        noise = rng.normal(0, 0.9, base.shape).astype(np.float32)
        out.append(np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return out


def test_config_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        MachineConfig(running_threshold=1.0, stopped_threshold=2.0)
    with pytest.raises(ValueError):
        MachineConfig(blur_ksize=4)


def test_a_scrolling_belt_reads_as_running():
    energy = MotionEnergy(MACHINE, MachineConfig(min_state_ms=200.0))
    state = None
    for i, frame in enumerate(running_frames(24)):
        state = energy.update(frame, timestamp_ms=i * 100.0)
    assert state.running is True
    assert state.score > energy.config.running_threshold


def test_a_still_belt_reads_as_stopped():
    energy = MotionEnergy(MACHINE, MachineConfig(min_state_ms=200.0))
    state = None
    for i, frame in enumerate(stopped_frames(24)):
        state = energy.update(frame, timestamp_ms=i * 100.0)
    assert state.running is False
    assert state.score < energy.config.stopped_threshold


def test_a_stop_is_seen_but_only_after_the_state_has_held():
    cfg = MachineConfig(min_state_ms=900.0)
    energy = MotionEnergy(MACHINE, cfg)
    t = 0.0
    for frame in running_frames(20):
        energy.update(frame, timestamp_ms=t)
        t += 100.0
    assert energy.state.running is True

    stopped = stopped_frames(4)
    for frame in stopped:
        energy.update(frame, timestamp_ms=t)
        t += 100.0
    assert energy.state.running is True, "a 400 ms pause must not read as a stop"

    for frame in stopped_frames(16):
        energy.update(frame, timestamp_ms=t)
        t += 100.0
    assert energy.state.running is False


def test_a_person_walking_past_a_stopped_machine_does_not_make_it_read_as_running():
    """The exclusion rule. Without it, a stopped machine reads as running exactly
    when somebody is next to it, which is the worst possible moment."""
    cfg = MachineConfig(min_state_ms=200.0)
    with_person = MotionEnergy(MACHINE, cfg)
    without = MotionEnergy(MACHINE, cfg)

    base = stopped_frames(30)
    for i, frame in enumerate(base):
        moving = frame.copy()
        x = 110 + i * 8  # a bright figure crossing the machine region
        cv2.rectangle(moving, (x, 70), (x + 46, 175), (250, 250, 250), -1)
        box = (float(x), 70.0, float(x + 46), 175.0)
        with_person.update(moving, timestamp_ms=i * 100.0, person_boxes=[box])
        without.update(moving, timestamp_ms=i * 100.0)

    assert with_person.state.running is False, "person exclusion failed"
    assert without.state.running is True, (
        "the test scene is not actually exercising the exclusion path"
    )


def test_a_fully_occluded_machine_reports_no_confidence_rather_than_guessing():
    energy = MotionEnergy(MACHINE, MachineConfig(min_state_ms=200.0))
    frames = running_frames(10)
    whole = (0.0, 0.0, float(SIZE[0]), float(SIZE[1]))
    for i, frame in enumerate(frames):
        state = energy.update(frame, timestamp_ms=i * 100.0, person_boxes=[whole])
    assert state.confidence == 0.0
    assert state.excluded_fraction == pytest.approx(1.0, abs=0.02)


def test_no_zone_means_the_whole_frame_is_the_machine():
    energy = MotionEnergy(None, MachineConfig(min_state_ms=200.0))
    for i, frame in enumerate(running_frames(20)):
        energy.update(frame, timestamp_ms=i * 100.0)
    assert energy.state.running is True


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_suggested_thresholds_sit_inside_a_clean_gap():
    suggestion = suggest_thresholds(running_scores=[5.0, 5.5, 6.0, 4.8],
                                    stopped_scores=[0.4, 0.5, 0.6, 0.3])
    assert suggestion.separable is True
    assert suggestion.stopped_p90 < suggestion.stopped_threshold
    assert suggestion.stopped_threshold < suggestion.running_threshold
    assert suggestion.running_threshold < suggestion.running_p10


def test_overlapping_clips_are_reported_as_not_separable():
    suggestion = suggest_thresholds(running_scores=[1.0, 1.2, 0.9],
                                    stopped_scores=[0.9, 1.1, 1.3])
    assert suggestion.separable is False
    assert "cannot tell these apart" in suggestion.note


def test_missing_samples_are_reported_rather_than_defaulted_silently():
    suggestion = suggest_thresholds([], [1.0])
    assert suggestion.separable is False
    assert "need score samples" in suggestion.note


def test_calibration_round_trip_on_real_rendered_clips():
    """Measure both clips, fit thresholds, and check they classify the clips back."""
    cfg = MachineConfig(min_state_ms=200.0)
    run_scores, stop_scores = [], []
    for source, sink in ((running_frames(30), run_scores), (stopped_frames(30), stop_scores)):
        energy = MotionEnergy(MACHINE, cfg)
        for i, frame in enumerate(source):
            energy.update(frame, timestamp_ms=i * 100.0)
        sink.extend(s for _, s in energy.series())

    suggestion = suggest_thresholds(run_scores, stop_scores)
    assert suggestion.separable, suggestion.note

    tuned = MachineConfig(running_threshold=suggestion.running_threshold,
                          stopped_threshold=suggestion.stopped_threshold,
                          min_state_ms=200.0)
    energy = MotionEnergy(MACHINE, tuned)
    for i, frame in enumerate(running_frames(20)):
        energy.update(frame, timestamp_ms=i * 100.0)
    assert energy.state.running is True
