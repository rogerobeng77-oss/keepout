"""Regressions for the defects real footage exposed.

Keepout was run over fixed-camera construction clips from Wikimedia Commons
(listed in `eval/real_footage/README.md`). Each test below builds the
failure by construction, so the answer is known, and names the clip that showed it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from keepout import Keepout, Level
from keepout.privacy import (
    FaceBlurrer,
    FaceDetectorMissing,
    PrivacyConfig,
    find_yunet,
    head_region,
)
from keepout.track import TrackerConfig
from keepout.viewcheck import ViewConfig, ViewGuard, ViewProblem

from .conftest import make_scene, person_box
from .test_pipeline import SIZE, base_config, belt_frame

NO_YUNET = PrivacyConfig(use_face_detector=False)


def scored(frames: list[list[tuple[tuple[float, float, float, float], float]]]):
    """A scripted detector that also scripts the score of each box."""
    state = {"i": 0}

    def detect(_frame):
        i = min(state["i"], len(frames) - 1)
        state["i"] += 1
        return list(frames[i])

    return detect


def checker(image: np.ndarray, box, cell: int = 3) -> None:
    """Paint a hard black-and-white checkerboard into a region: a stand-in face whose
    sharpness is easy to measure and impossible to mistake for blur."""
    x1, y1, x2, y2 = (round(v) for v in box)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    pattern = (((yy // cell) + (xx // cell)) % 2 * 255).astype(np.uint8)
    image[y1:y2, x1:x2] = pattern[..., None]


def sharpness(image: np.ndarray, box, inset: int = 5) -> float:
    x1, y1, x2, y2 = (round(v) for v in box)
    patch = cv2.cvtColor(image[y1 + inset:y2 - inset, x1 + inset:x2 - inset],
                         cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def faces_in(boxes, shape):
    """The inner head slice of each person box: where a face would be."""
    cfg = PrivacyConfig()
    out = []
    for b in boxes:
        x1, y1, x2, y2 = head_region(b, shape, cfg)
        cx, w = (x1 + x2) / 2, (x2 - x1) * 0.6
        out.append((cx - w / 2, y1 + 2, cx + w / 2, y1 + (y2 - y1) * 0.85))
    return out


# ---------------------------------------------------------------------------
# 1. Privacy: every person in a stored frame, not just the incident's subject
# ---------------------------------------------------------------------------


def test_every_head_in_an_evidence_frame_is_blurred_not_only_the_subjects():
    """Malta shot 4, config B: the person-down evidence frame blurred one box, and
    the faces of the other workers in the zone stayed sharp."""
    subject = person_box(250, 290, height=110)  # enters the zone: the incident
    tracked_bystander = person_box(360, 290, height=110)  # also tracked, in the zone
    weak_bystander = person_box(120, 120, height=100)  # score 0.2: below tracking
    sweep_only = person_box(440, 120, height=100)  # only the privacy sweep finds it
    people = [subject, tracked_bystander, weak_bystander, sweep_only]

    n = 30
    base = make_scene(SIZE, seed=6)
    frames = []
    for i in range(n):
        f = belt_frame(i, running=True, base=base)
        for face in faces_in(people, f.shape):
            checker(f, face)
        frames.append(f)
    control = (410, 250, 470, 300)  # a sharp patch with no person on it
    for f in frames:
        checker(f, control)

    per_frame = [
        [(subject, 0.9), (tracked_bystander, 0.9), (weak_bystander, 0.2)] if i >= 12 else [(weak_bystander, 0.2)]
        for i in range(n)
    ]

    written = []
    config = base_config(privacy=NO_YUNET)
    keeper = Keepout(config, scored(per_frame),
                     on_evidence=lambda name, img, rep: written.append((name, img, rep)) or name,
                     privacy_detector=lambda _f: [(sweep_only, 0.3)])
    keeper.run((f, i * 100.0) for i, f in enumerate(frames))

    assert written, "the entries must have produced evidence"
    for name, image, report in written:
        idx = 12  # evidence is written on the entry frame; pixels are identical per frame
        raw = frames[idx]
        for face in faces_in(people, raw.shape):
            before, after = sharpness(raw, face), sharpness(image, face)
            assert after < 0.25 * before, (name, face, before, after)
        assert sharpness(image, control) > 0.6 * sharpness(raw, control), (
            "the control patch lost its detail, so this test is not measuring blur")
        assert report["enabled"] is True
        assert report["person_boxes"] >= len(people)
        assert report["face_detector"] == "unavailable"
        assert report["method"] == "person-head"
        assert "tiles" in report["person_sweep"]


def test_a_view_problem_evidence_frame_blurs_the_people_in_it():
    """An unusable frame is not watched, but it is stored, and the people in it are
    still people."""
    person = person_box(240, 280, height=110)
    good = [belt_frame(i, running=False) for i in range(15)]
    m = np.float32([[1, 0, 44], [0, 1, 22]])
    moved = [cv2.warpAffine(belt_frame(i, running=False), m, SIZE,
                            borderMode=cv2.BORDER_REPLICATE) for i in range(5)]
    for f in moved:
        for face in faces_in([person], f.shape):
            checker(f, face)
    written = []
    keeper = Keepout(base_config(privacy=NO_YUNET), scored([[]]),
                     on_evidence=lambda n, img, rep: written.append((n, img, rep)) or n,
                     privacy_detector=lambda _f: [(person, 0.8)])
    keeper.run((f, i * 100.0) for i, f in enumerate(good + moved))
    views = [w for w in written if w[0] == "view-camera_moved"]
    assert views
    _, image, report = views[0]
    face = faces_in([person], image.shape)[0]
    assert sharpness(image, face) < 0.25 * sharpness(moved[0], face)
    assert report["head_regions"] >= 1


def test_the_report_says_when_no_face_detector_ran(monkeypatch, tmp_path):
    monkeypatch.delenv("KEEPOUT_REQUIRE_FACE_DETECTOR", raising=False)
    monkeypatch.setenv("KEEPOUT_YUNET_PATH", str(tmp_path / "absent.onnx"))
    monkeypatch.setenv("OPENCV26_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    blurrer = FaceBlurrer(PrivacyConfig())
    if blurrer.model_path is not None:  # pragma: no cover - weights inside the source tree
        pytest.skip("YuNet weights are bundled in the source tree")
    _, report = blurrer.redact(make_scene(SIZE), [person_box(100, 200)])
    assert report["face_detector"] == "unavailable"
    assert report["method"] == "person-head"
    assert report["head_regions"] == 1


def test_production_refuses_to_start_without_the_face_detector(monkeypatch, tmp_path):
    monkeypatch.setenv("KEEPOUT_REQUIRE_FACE_DETECTOR", "1")
    monkeypatch.setenv("KEEPOUT_YUNET_PATH", str(tmp_path / "absent.onnx"))
    monkeypatch.setenv("OPENCV26_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    if find_yunet() is not None:  # pragma: no cover - weights inside the source tree
        pytest.skip("YuNet weights are bundled in the source tree")
    with pytest.raises(FaceDetectorMissing):
        FaceBlurrer(PrivacyConfig())


def test_a_service_request_cannot_switch_blurring_off():
    from keepout.service import config_from_params

    config = config_from_params({"blur_faces": False}, (960, 540))
    assert config.privacy.enabled is True


# ---------------------------------------------------------------------------
# 2. The reference frame
# ---------------------------------------------------------------------------


def test_a_fade_from_black_is_not_taken_as_the_reference():
    """Prefabricated_house_construction.ogv opens black. The black frame became the
    reference and 0 of 2,689 frames were watched."""
    black = [np.zeros((SIZE[1], SIZE[0], 3), np.uint8) for _ in range(4)]
    frames = black + [belt_frame(i, running=True) for i in range(40)]
    keeper = Keepout(base_config(), scored([[]]))
    results = keeper.run((f, i * 100.0) for i, f in enumerate(frames))

    assert all(r.view.has(ViewProblem.TOO_DARK) for r in results[:4])
    assert all(r.view.usable for r in results[4:]), [r.view.problems for r in results[4:]]
    summary = keeper.summary()
    assert summary["frames_usable"] == 40
    assert summary["reference"]["adopted_ms"] == 400.0
    assert summary["reference"]["replaced"] == 0
    assert "camera_moved" not in summary["view_problems"]


def test_a_reference_every_later_frame_disagrees_with_is_replaced_and_logged():
    """A reference that passes the frame checks but is still wrong (a cut, a dissolve,
    a glitch frame) must not blind the rest of the run."""
    # A frame from a different framing: the same kind of scene, well off alignment.
    wrong = cv2.warpAffine(belt_frame(0, running=True), np.float32([[1, 0, 57], [0, 1, 33]]),
                           SIZE, borderMode=cv2.BORDER_REFLECT)
    frames = [wrong] + [belt_frame(i, running=True) for i in range(80)]
    boxes = [[] for _ in range(50)] + [[(person_box(240, 280), 0.9)] for _ in range(31)]
    keeper = Keepout(base_config(), scored(boxes))
    results = keeper.run((f, i * 100.0) for i, f in enumerate(frames))

    events = keeper.summary()["reference"]["events"]
    replaced = [e for e in events if e["event"] == "reference_replaced"]
    assert len(replaced) == 1, events
    at = replaced[0]["index"]
    # ECC can land on a false optimum for the odd frame against a wrong reference;
    # those frames neither confirm the reference nor restart the run.
    assert ViewConfig().reference_recover_frames < at <= 2 * ViewConfig().reference_recover_frames
    before = results[1:at]
    assert sum(r.view.has(ViewProblem.CAMERA_MOVED) for r in before) >= 0.8 * len(before)
    assert all(r.view.usable for r in results[at:])
    assert any(r.view_events for r in results)
    assert [i.level for i in keeper.log.incidents] == [Level.ALERT], (
        "once re-established, the zone is watched again")


def test_a_real_camera_bump_that_holds_still_raises_camera_moved_for_good():
    """The case the replacement rule must not swallow: a confirmed view, then a
    sudden shift that holds for far longer than the recovery run."""
    good = [belt_frame(i, running=True) for i in range(20)]
    m = np.float32([[1, 0, 47], [0, 1, 23]])  # not a multiple of the scene's 40 px grid
    moved = [cv2.warpAffine(belt_frame(i, running=True), m, SIZE,
                            borderMode=cv2.BORDER_REPLICATE) for i in range(90)]
    boxes = [[] for _ in range(20)] + [[(person_box(240, 280), 0.9)] for _ in range(90)]
    keeper = Keepout(base_config(), scored(boxes))
    results = keeper.run((f, i * 100.0) for i, f in enumerate(good + moved))

    assert all(r.view.has(ViewProblem.CAMERA_MOVED) for r in results[20:])
    assert keeper.summary()["reference"]["replaced"] == 0
    assert keeper.log.incidents == []


def test_view_guard_confirms_a_reference_only_after_frames_agree_with_it():
    guard = ViewGuard(ViewConfig(reference_confirm_frames=3))
    reference = make_scene(SIZE)
    guard.set_reference(reference)
    assert guard.reference_confirmed is False
    for _ in range(3):
        status = guard.check(reference)
    assert status.reference_confirmed is True


# ---------------------------------------------------------------------------
# 4 and 5. Track fragmentation
# ---------------------------------------------------------------------------


def test_one_person_whose_track_breaks_is_one_incident_not_two():
    """Malta shot 4, config B: 27 alerts for about 8 people in the zone."""
    n = 80
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[] for _ in range(12)]
    boxes += [[(person_box(240, 280), 0.9)] for _ in range(20)]  # tracked
    boxes += [[] for _ in range(16)]  # lost for 1.6 s: the track is dropped
    boxes += [[(person_box(252, 282), 0.9)] for _ in range(n - 48)]  # new id, same person
    config = base_config(tracker=TrackerConfig(min_hits=2, max_misses=6))
    keeper = Keepout(config, scored(boxes))
    keeper.run((f, i * 100.0) for i, f in enumerate(frames))

    entries = [i for i in keeper.log.incidents if i.kind == "zone_entry"]
    assert len(entries) == 1, [i.to_dict() for i in entries]
    assert any(h.get("event") == "track_rejoined" for h in entries[0].history)
    assert len(entries[0].detail["track_ids"]) == 2
    assert keeper.summary()["incidents_rejoined"] == 1


def test_a_neighbour_missed_for_one_frame_keeps_their_own_incident():
    """The first version of the rejoin rule, on the Malta clip: a worker missed for a
    single frame had their incident handed to the colleague beside them, who had
    been tracked all along. A track that already existed is somebody else."""
    n = 60
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = []
    for i in range(n):
        people = []
        if i >= 12 and i != 30:
            people.append((person_box(240, 280), 0.9))  # A, in the zone, missed at 30
        if i >= 12:
            feet = 120 if i < 30 else 282  # B, tracked outside, steps in at frame 30
            people.append((person_box(275, feet), 0.9))
        boxes.append(people)
    config = base_config(tracker=TrackerConfig(min_hits=2, max_misses=6))
    keeper = Keepout(config, scored(boxes))
    keeper.run((f, i * 100.0) for i, f in enumerate(frames))
    entries = [i for i in keeper.log.incidents if i.kind == "zone_entry"]
    assert len(entries) == 2, [i.to_dict() for i in entries]
    assert all(not any(h.get("event") == "track_rejoined" for h in i.history) for i in entries)
    assert len({i.track_id for i in entries}) == 2


def test_a_track_missed_past_the_exit_grace_resumes_its_own_incident():
    """Malta, second pass: the same track id, missed for longer than the zone's exit
    grace but not long enough to be dropped, opened a second incident."""
    n = 60
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[] for _ in range(12)] + [[(person_box(240, 280), 0.9)] for _ in range(15)]
    boxes += [[] for _ in range(6)]  # 600 ms: past exit_grace_ms (400), within max_misses
    boxes += [[(person_box(242, 280), 0.9)] for _ in range(n - 33)]
    config = base_config(tracker=TrackerConfig(min_hits=2, max_misses=12))
    keeper = Keepout(config, scored(boxes))
    keeper.run((f, i * 100.0) for i, f in enumerate(frames))
    entries = [i for i in keeper.log.incidents if i.kind == "zone_entry"]
    assert len(entries) == 1, [i.to_dict() for i in entries]
    assert any(h.get("event") == "resumed" for h in entries[0].history)
    assert entries[0].open is True


def test_somebody_else_entering_elsewhere_is_still_a_new_incident():
    n = 80
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[] for _ in range(12)]
    boxes += [[(person_box(120, 280), 0.9)] for _ in range(20)]
    boxes += [[] for _ in range(10)]
    boxes += [[(person_box(380, 290), 0.9)] for _ in range(n - 42)]  # far away
    config = base_config(tracker=TrackerConfig(min_hits=2, max_misses=6))
    keeper = Keepout(config, scored(boxes))
    keeper.run((f, i * 100.0) for i, f in enumerate(frames))
    assert len([i for i in keeper.log.incidents if i.kind == "zone_entry"]) == 2


def test_a_dropped_track_ends_its_incident_instead_of_leaving_it_open():
    n = 60
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[] for _ in range(12)] + [[(person_box(240, 280), 0.9)] for _ in range(20)]
    boxes += [[] for _ in range(n - 32)]
    # max_misses shorter than exit_grace_ms: the tracker drops the track before the
    # zone monitor's grace closes it, which at 25 fps is the default (12 frames is
    # 480 ms against 600 ms). The incident used to stay open for the rest of the run.
    config = base_config(tracker=TrackerConfig(min_hits=2, max_misses=2),
                         vanish_min_track_ms=60_000.0)
    keeper = Keepout(config, scored(boxes))
    keeper.run((f, i * 100.0) for i, f in enumerate(frames))
    entry = next(i for i in keeper.log.incidents if i.kind == "zone_entry")
    assert entry.open is False
    assert entry.ended_ms == pytest.approx(3100.0)


def test_a_person_redetected_under_a_new_id_is_not_unaccounted_for():
    """Courtyard (vtest.avi): somebody passed behind a sign post inside the zone,
    their track ended, a new track picked them up on the other side, and the vanish
    rule still raised a critical for the old id."""
    n = 90
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = [[(person_box(240, 250), 0.9)] for _ in range(35)]
    boxes += [[] for _ in range(6)]
    boxes += [[(person_box(275, 252), 0.9)] for _ in range(n - 41)]
    config = base_config(vanish_grace_ms=1200.0, vanish_min_track_ms=1500.0,
                         tracker=TrackerConfig(min_hits=2, max_misses=4))
    keeper = Keepout(config, scored(boxes))
    keeper.run((f, i * 100.0) for i, f in enumerate(frames))
    assert [i for i in keeper.log.incidents if i.kind == "person_unaccounted"] == []


def test_somebody_hidden_by_a_passer_by_as_they_vanish_is_not_unaccounted_for():
    """Courtyard (vtest.avi), 16.5 s: two pedestrians crossed inside the zone, one id
    was lost while the two boxes overlapped, and the other person had walked 170 px
    away by the time the vanish rule looked 2.5 s later. The overlap has to be judged
    at the moment the track was last seen."""
    n = 90
    frames = [belt_frame(i, running=True) for i in range(n)]
    boxes = []
    for i in range(n):
        people = []
        if i <= 35:
            people.append((person_box(240, 250), 0.9))  # stands, then is hidden and lost
        walker_x = 120 + (i - 25) * 12 if i >= 25 else None  # walks through them
        if walker_x is not None and walker_x < 420:
            people.append((person_box(walker_x, 252), 0.9))
        boxes.append(people)
    config = base_config(vanish_grace_ms=1200.0, vanish_min_track_ms=1500.0,
                         tracker=TrackerConfig(min_hits=2, max_misses=4))
    keeper = Keepout(config, scored(boxes))
    keeper.run((f, i * 100.0) for i, f in enumerate(frames))
    assert [i for i in keeper.log.incidents if i.kind == "person_unaccounted"] == []

