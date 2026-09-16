"""End-to-end pipeline behaviour on sequences whose answer is known by construction.

These tests inject a scripted detector rather than running YOLOX, so what is
under test is the logic that turns detections into incidents: the zone rules, the
escalation ladder, the moved-camera path, the vanish rule, and privacy. The real
detector gets its own measurement in `eval/` and in `docs/evaluation.md`.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from keepout import ContactPoint, Keepout, KeepoutConfig, Level, Zone
from keepout.machine import MachineConfig
from keepout.posture import PostureConfig
from keepout.track import TrackerConfig
from keepout.viewcheck import ViewConfig, ViewProblem
from keepout.zones import DwellConfig

from .conftest import lying_box, make_scene, person_box

SIZE = (480, 320)
MACHINE_ZONE = Zone(name="conveyor", points=((80, 40), (400, 40), (400, 130), (80, 130)),
                    reference_size=SIZE, kind="machine")
DANGER_ZONE = Zone(name="conveyor danger zone",
                   points=((60, 140), (420, 140), (420, 310), (60, 310)),
                   reference_size=SIZE, contact=ContactPoint.FEET)


def scripted(boxes_per_frame: list[list[tuple[float, float, float, float]]]):
    """A detector that returns whatever the script says for the current frame."""
    state = {"i": 0}

    def detect(_frame):
        i = min(state["i"], len(boxes_per_frame) - 1)
        state["i"] += 1
        return [(b, 0.9) for b in boxes_per_frame[i]]

    return detect


def belt_frame(i: int, *, running: bool, base: np.ndarray | None = None) -> np.ndarray:
    frame = (base if base is not None else make_scene(SIZE, seed=6)).copy()
    offset = (i * 11) % 40 if running else 0
    for x in range(40, 420, 40):
        cv2.line(frame, (x + offset, 50), (x + offset + 18, 120), (235, 235, 235), 6)
    if not running:
        rng = np.random.default_rng(i)
        frame = np.clip(frame.astype(np.float32) + rng.normal(0, 0.9, frame.shape),
                        0, 255).astype(np.uint8)
    return frame


def base_config(**kwargs) -> KeepoutConfig:
    defaults = dict(
        danger_zone=DANGER_ZONE,
        machine_zone=MACHINE_ZONE,
        dwell=DwellConfig(exit_grace_ms=400.0),
        machine=MachineConfig(min_state_ms=300.0),
        tracker=TrackerConfig(min_hits=2),
        view=ViewConfig(max_shift_px=12.0),
    )
    defaults.update(kwargs)
    return KeepoutConfig(**defaults)


def run(config: KeepoutConfig, frames: list[np.ndarray], boxes, on_evidence=None):
    keeper = Keepout(config, scripted(boxes), on_evidence=on_evidence)
    results = keeper.run((f, i * 100.0) for i, f in enumerate(frames))
    return keeper, results


# ---------------------------------------------------------------------------
# The ladder, end to end
# ---------------------------------------------------------------------------


def test_entry_while_the_machine_is_stopped_is_a_note():
    n = 40
    frames = [belt_frame(i, running=False) for i in range(n)]
    boxes = [[person_box(240, 280)] for _ in range(n)]
    keeper, _ = run(base_config(), frames, boxes)

    assert keeper.machine.state.running is False
    incidents = keeper.log.incidents
    assert len(incidents) == 1
    assert incidents[0].level is Level.NOTE
    assert incidents[0].kind == "zone_entry"


def test_entry_while_the_machine_is_running_is_an_alert():
    n = 40
    frames = [belt_frame(i, running=True) for i in range(n)]
    # Nobody for the first 12 frames, so the machine state settles first.
    boxes = [[] for _ in range(12)] + [[person_box(240, 280)] for _ in range(n - 12)]
    keeper, _ = run(base_config(), frames, boxes)

    assert keeper.machine.state.running is True
    assert len(keeper.log.incidents) == 1
    assert keeper.log.incidents[0].level is Level.ALERT


def test_a_note_upgrades_to_an_alert_when_the_machine_starts_under_someone():
    stopped = [belt_frame(i, running=False) for i in range(20)]
    running = [belt_frame(i, running=True) for i in range(40)]
    frames = stopped + running
    boxes = [[person_box(240, 280)] for _ in range(len(frames))]
    keeper, _ = run(base_config(), frames, boxes)

    assert len(keeper.log.incidents) == 1, "it must stay one story, not become two"
    incident = keeper.log.incidents[0]
    assert incident.level is Level.ALERT
    assert incident.peak_level is Level.ALERT
    assert any(h.get("to") == "alert" for h in incident.history)


def test_someone_walking_past_outside_the_zone_raises_nothing():
    n = 40
    frames = [belt_frame(i, running=True) for i in range(n)]
    # Feet at y=120, which is above the zone's top edge at y=140.
    boxes = [[person_box(60 + i * 8, 120)] for i in range(n)]
    keeper, _ = run(base_config(), frames, boxes)
    assert keeper.log.incidents == []


def test_leaving_the_zone_closes_the_incident():
    frames = [belt_frame(i, running=False) for i in range(50)]
    boxes = [[person_box(240, 280)] for _ in range(20)]
    boxes += [[person_box(240, 100)] for _ in range(30)]  # steps back out, above the zone
    keeper, _ = run(base_config(), frames, boxes)
    incident = keeper.log.incidents[0]
    assert incident.open is False
    assert incident.ended_ms is not None


def test_two_people_produce_two_incidents_with_their_own_tracks():
    n = 40
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[] for _ in range(12)]
    boxes += [[person_box(150, 280), person_box(340, 300, height=140)] for _ in range(n - 12)]
    keeper, _ = run(base_config(), frames, boxes)
    assert len(keeper.log.incidents) == 2
    assert len({i.track_id for i in keeper.log.incidents}) == 2


# ---------------------------------------------------------------------------
# The two critical paths
# ---------------------------------------------------------------------------


def test_a_person_who_lies_down_and_stops_moving_raises_a_latching_critical():
    n = 90
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[] for _ in range(10)]
    boxes += [[person_box(240, 280)] for _ in range(10)]
    boxes += [[lying_box(240, 280)] for _ in range(n - 20)]  # collapses and stays put

    config = base_config(posture=PostureConfig(stillness_window_ms=1200.0,
                                               min_duration_ms=2000.0, min_samples=5))
    keeper, _ = run(config, frames, boxes)

    critical = [i for i in keeper.log.incidents if i.level is Level.CRITICAL]
    assert critical, [i.to_dict() for i in keeper.log.incidents]
    incident = critical[0]
    assert incident.latched is True
    assert incident.open is True

    # And it stays open even after the condition would have cleared.
    keeper.log.clear(incident.kind, incident.zone, incident.track_id, 99999.0)
    assert incident.open is True
    assert keeper.log.acknowledge(incident.incident_id, 100000.0, by="supervisor")
    assert incident.open is False


def test_a_person_who_vanishes_deep_inside_the_zone_raises_a_critical():
    """The measured answer to the measured detector limit: past ~40 degrees of body
    rotation YOLOX stops returning a box at all, so the disappearance is the signal."""
    n = 80
    frames = [belt_frame(i, running=True) for i in range(n)]
    # Feet at y=250: 60 px inside the zone and well clear of every frame edge.
    boxes = [[person_box(240, 250)] for _ in range(25)]
    boxes += [[] for _ in range(n - 25)]  # stops being detected, in the middle of the zone

    config = base_config(vanish_grace_ms=1200.0, vanish_min_track_ms=1500.0,
                         tracker=TrackerConfig(min_hits=2, max_misses=6))
    keeper, _ = run(config, frames, boxes)

    unaccounted = [i for i in keeper.log.incidents if i.kind == "person_unaccounted"]
    assert unaccounted, [i.to_dict() for i in keeper.log.incidents]
    assert unaccounted[0].level is Level.CRITICAL
    assert unaccounted[0].latched is True
    assert unaccounted[0].detail["depth_inside_zone_px"] > 0


def test_someone_who_walks_out_of_shot_does_not_raise_unaccounted():
    """The rule must not fire for the ordinary way people stop being detected."""
    n = 80
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = []
    for i in range(30):  # walks right, off the edge of the frame
        boxes.append([person_box(200 + i * 10, 280)])
    boxes += [[] for _ in range(n - 30)]

    config = base_config(vanish_grace_ms=1200.0, vanish_min_track_ms=1500.0,
                         tracker=TrackerConfig(min_hits=2, max_misses=6))
    keeper, _ = run(config, frames, boxes)
    assert [i for i in keeper.log.incidents if i.kind == "person_unaccounted"] == []


def test_someone_who_vanishes_just_inside_the_boundary_does_not_raise_unaccounted():
    n = 70
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[person_box(240, 155, height=60)] for _ in range(25)]  # 15 px inside the top edge
    boxes += [[] for _ in range(n - 25)]
    config = base_config(vanish_grace_ms=1000.0, vanish_min_depth_px=18.0,
                         vanish_min_track_ms=1500.0,
                         tracker=TrackerConfig(min_hits=2, max_misses=6))
    keeper, _ = run(config, frames, boxes)
    assert [i for i in keeper.log.incidents if i.kind == "person_unaccounted"] == []


# ---------------------------------------------------------------------------
# Honesty rails
# ---------------------------------------------------------------------------


def test_a_moved_camera_stops_the_pipeline_rather_than_alerting_on_a_stale_zone():
    good = [belt_frame(i, running=True) for i in range(15)]
    m = np.float32([[1, 0, 44], [0, 1, 22]])
    moved = [cv2.warpAffine(belt_frame(i, running=True), m, SIZE,
                            borderMode=cv2.BORDER_REPLICATE) for i in range(25)]
    frames = good + moved
    boxes = [[] for _ in range(15)] + [[person_box(240, 280)] for _ in range(25)]

    keeper, results = run(base_config(), frames, boxes)

    assert any(r.view.has(ViewProblem.CAMERA_MOVED) for r in results[15:])
    assert keeper.frames_skipped >= 20
    assert keeper.log.incidents == [], (
        "a moved camera must not produce zone incidents against the old polygon"
    )
    assert keeper.summary()["view_problems"]["camera_moved"] >= 20


def test_a_blocked_lens_suppresses_detection_and_says_why():
    good = [belt_frame(i, running=False) for i in range(10)]
    blocked = [np.full((SIZE[1], SIZE[0], 3), 118, np.uint8) for _ in range(20)]
    frames = good + blocked
    boxes = [[] for _ in range(10)] + [[person_box(240, 280)] for _ in range(20)]
    keeper, results = run(base_config(), frames, boxes)
    assert keeper.log.incidents == []
    assert results[-1].view.has(ViewProblem.LENS_BLOCKED)
    assert "lens looks blocked" in " ".join(results[-1].view.messages)


def test_the_usable_fraction_is_reported():
    good = [belt_frame(i, running=False) for i in range(20)]
    dark = [(belt_frame(i, running=False).astype(np.float32) * 0.03).astype(np.uint8)
            for i in range(20)]
    keeper, _ = run(base_config(), good + dark, [[] for _ in range(40)])
    summary = keeper.summary()
    assert summary["frames_seen"] == 40
    assert summary["frames_skipped_unusable"] == 20
    assert summary["usable_fraction"] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# Evidence and privacy
# ---------------------------------------------------------------------------


def test_evidence_frames_are_written_redacted_and_referenced():
    written: list[tuple[str, np.ndarray, dict]] = []

    def on_evidence(name, image, report):
        written.append((name, image, report))
        return f"/evidence/{name}.jpg"

    n = 40
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[] for _ in range(12)] + [[person_box(240, 280)] for _ in range(n - 12)]
    keeper, _ = run(base_config(), frames, boxes, on_evidence=on_evidence)

    assert written, "an alert must come with the frame that proves it"
    _name, image, report = written[0]
    assert report["enabled"] is True
    assert report["regions"] >= 1
    assert image.shape == frames[0].shape
    incident = keeper.log.incidents[0]
    assert incident.evidence
    assert incident.evidence[0].uri.startswith("/evidence/")
    assert incident.evidence[0].redacted is True


def test_evidence_capture_is_rate_limited():
    written = []
    n = 120
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[] for _ in range(12)] + [[person_box(240, 280)] for _ in range(n - 12)]
    run(base_config(), frames, boxes,
        on_evidence=lambda name, img, rep: written.append(name) or f"/e/{name}")
    # One alert, held for ~11 seconds, must not write a hundred frames.
    assert 1 <= len(written) <= 10, len(written)


def test_the_summary_is_json_shaped_and_complete():
    frames = [belt_frame(i, running=True) for i in range(30)]
    keeper, _ = run(base_config(), frames, [[] for _ in range(30)])
    summary = keeper.summary()
    for key in ("frames_seen", "frames_usable", "view_problems", "incidents",
                "machine", "guard", "tracks_created", "timings_ms", "privacy"):
        assert key in summary
    assert summary["privacy"]["face_blur"] is True

    import json

    json.dumps(summary)  # must be serialisable with no custom encoder


def test_frame_results_serialise_for_the_live_view():
    frames = [belt_frame(i, running=True) for i in range(20)]
    boxes = [[person_box(240, 280)] for _ in range(20)]
    _, results = run(base_config(), frames, boxes)
    payload = results[-1].to_dict()
    for key in ("index", "timestamp_ms", "view", "tracks", "occupants", "machine",
                "guard", "postures", "highest_level"):
        assert key in payload
    import json

    json.dumps(payload)


def test_default_zones_fit_inside_the_frame():
    from keepout import default_zones

    zones = default_zones((1280, 720))
    for zone in zones.values():
        pts = np.array(zone.points)
        assert pts[:, 0].min() >= 0 and pts[:, 0].max() <= 1280
        assert pts[:, 1].min() >= 0 and pts[:, 1].max() <= 720


def test_a_brief_detector_blip_that_disappears_does_not_raise_unaccounted():
    """Measured on real pedestrian footage: without a minimum track lifetime the
    vanish rule fired three times in forty seconds on detector noise."""
    n = 70
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[] for _ in range(10)]
    boxes += [[person_box(240, 250)] for _ in range(6)]  # 600 ms of existence
    boxes += [[] for _ in range(n - 16)]

    config = base_config(vanish_grace_ms=1000.0, vanish_min_track_ms=3000.0,
                         tracker=TrackerConfig(min_hits=2, max_misses=4))
    keeper, _ = run(config, frames, boxes)
    assert [i for i in keeper.log.incidents if i.kind == "person_unaccounted"] == []


def test_two_people_occluding_each_other_do_not_raise_unaccounted():
    """One track dies because the other person walked in front of them. That is a
    crowd, not an emergency."""
    n = 80
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = []
    for i in range(40):  # both present and overlapping by the end
        a = person_box(240, 250)
        b = person_box(200 + i * 1.0, 250)
        boxes.append([a, b])
    for _ in range(n - 40):  # one of them stops being detected, behind the other
        boxes.append([person_box(240, 250)])

    config = base_config(vanish_grace_ms=1000.0, vanish_min_track_ms=1500.0,
                         tracker=TrackerConfig(min_hits=2, max_misses=4))
    keeper, _ = run(config, frames, boxes)
    assert [i for i in keeper.log.incidents if i.kind == "person_unaccounted"] == []


def test_a_person_who_vanishes_across_the_guard_keeps_the_guard_check_honest():
    """Somebody lying over the guard must not read as a removed guard."""
    from keepout.guard import GuardStatus

    n = 90
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[person_box(240, 250)] for _ in range(40)]
    boxes += [[] for _ in range(n - 40)]
    guard_zone = Zone(name="fixed guard", points=((180, 150), (320, 150), (320, 260),
                                                  (180, 260)),
                      reference_size=SIZE, kind="guard", contact=ContactPoint.CENTROID)
    config = base_config(guard_zone=guard_zone, vanish_grace_ms=1000.0,
                         vanish_min_track_ms=1500.0,
                         tracker=TrackerConfig(min_hits=2, max_misses=4))
    keeper, _ = run(config, frames, boxes)

    assert [i for i in keeper.log.incidents if i.kind == "person_unaccounted"], (
        "the vanish rule should have fired, or this test is not exercising the path"
    )
    assert keeper.guard.status is not GuardStatus.MISSING, (
        "a person lying across the guard was reported as a removed guard"
    )
