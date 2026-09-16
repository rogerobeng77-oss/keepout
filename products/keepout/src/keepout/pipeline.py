"""The Keepout pipeline: one fixed camera in, an incident list with evidence out.

Order of operations per frame, and why it is this order:

1. **View check first.** If the camera moved, the lens is covered, it is too dark
   or the feed is frozen, nothing below this line means anything. We record the
   problem, emit a `view` incident-equivalent in the record, and skip detection
   for that frame rather than produce confident output from an unusable view.
2. **Detect people.** YOLOX-tiny through `cv2.dnn`, filtered to the COCO `person`
   class. Apache-2.0, ONNX, CPU (research/FINDINGS.md §4.4).
3. **Associate into tracks.** Our own Hungarian assignment; OpenCV 5's main wheel
   has no trackers (§1.3).
4. **Machine state**, measured with the person boxes excluded so a passer-by does
   not make a stopped machine read as running.
5. **Guard state**, measured with the person boxes excluded so somebody standing
   in front of the guard does not read as a removed guard.
6. **Posture**, over each track's own history.
7. **Escalate**, which is the only step that creates or upgrades incidents, and
   the only step that writes evidence frames.

The pipeline is deliberately free of IO. It takes frames and returns state; the
service layer decides where evidence bytes go. That is what makes the synthetic
tests in `tests/test_pipeline.py` possible: they build sequences where the answer
is known by construction and assert on the returned state.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .escalate import REASONS, EvidenceRef, IncidentLog, Level, level_for_zone_entry
from .guard import GuardChecker, GuardConfig, GuardStatus
from .machine import MachineConfig, MotionEnergy
from .posture import DownDetector, PostureConfig
from .privacy import FaceBlurrer, PrivacyConfig
from .track import BoxTracker, TrackerConfig
from .viewcheck import ViewConfig, ViewGuard, ViewProblem, ViewStatus
from .zones import ContactPoint, DwellConfig, Zone, ZoneMonitor, draw_zone

Box = tuple[float, float, float, float]
PersonDetector = Callable[[np.ndarray], list[tuple[Box, float]]]

PERSON_CLASS_ID = 0
EVIDENCE_COOLDOWN_MS = 1500.0


@dataclass
class KeepoutConfig:
    """Everything an installation needs. One object, serialisable, in the record."""

    danger_zone: Zone | None = None
    machine_zone: Zone | None = None
    guard_zone: Zone | None = None
    dwell: DwellConfig = field(default_factory=DwellConfig)
    machine: MachineConfig = field(default_factory=MachineConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    view: ViewConfig = field(default_factory=ViewConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    detection_score: float = 0.35
    min_person_height_px: float = 24.0
    # A person who stops being detected while standing well inside the zone has
    # either left through a boundary we would have seen them cross, or they are
    # on the floor. See `_check_vanished` and docs/evaluation.md.
    vanish_grace_ms: float = 2500.0
    vanish_min_depth_px: float = 18.0
    vanish_edge_margin_px: float = 40.0
    # A track has to have existed for this long before its disappearance means
    # anything. Measured on real pedestrian footage: without it, short-lived
    # detector blips on a busy walkway produced three false criticals in forty
    # seconds. See docs/evaluation.md.
    vanish_min_track_ms: float = 3000.0
    # And it must not have been standing on top of somebody else when it went.
    # Two people crossing occlude each other, one track dies, and that is not an
    # emergency.
    vanish_max_neighbour_iou: float = 0.05
    stride: int = 1
    max_frames: int | None = None
    max_side: int | None = 960
    assume_machine_running: bool | None = None  # override, for cameras with no machine view

    def to_dict(self) -> dict[str, Any]:
        return {
            "danger_zone": self.danger_zone.to_dict() if self.danger_zone else None,
            "machine_zone": self.machine_zone.to_dict() if self.machine_zone else None,
            "guard_zone": self.guard_zone.to_dict() if self.guard_zone else None,
            "dwell": {
                "enter_margin_px": self.dwell.enter_margin_px,
                "exit_margin_px": self.dwell.exit_margin_px,
                "min_dwell_ms": self.dwell.min_dwell_ms,
                "exit_grace_ms": self.dwell.exit_grace_ms,
            },
            "machine": {
                "running_threshold": self.machine.running_threshold,
                "stopped_threshold": self.machine.stopped_threshold,
                "min_state_ms": self.machine.min_state_ms,
            },
            "guard": {
                "edge_threshold": self.guard.edge_threshold,
                "appearance_threshold": self.guard.appearance_threshold,
                "max_occluded_fraction": self.guard.max_occluded_fraction,
            },
            "posture": {
                "down_aspect": self.posture.down_aspect,
                "still_motion_ratio": self.posture.still_motion_ratio,
                "min_duration_ms": self.posture.min_duration_ms,
            },
            "detection_score": self.detection_score,
            "vanish_grace_ms": self.vanish_grace_ms,
            "vanish_min_depth_px": self.vanish_min_depth_px,
            "stride": self.stride,
            "max_side": self.max_side,
            "privacy": self.privacy.to_dict(),
            "assume_machine_running": self.assume_machine_running,
        }


@dataclass
class FrameResult:
    """What the pipeline concluded about one frame. The UI's live view renders this."""

    index: int
    timestamp_ms: float
    view: ViewStatus
    tracks: list[dict[str, Any]] = field(default_factory=list)
    occupants: list[dict[str, Any]] = field(default_factory=list)
    machine: dict[str, Any] = field(default_factory=dict)
    guard: dict[str, Any] = field(default_factory=dict)
    postures: list[dict[str, Any]] = field(default_factory=list)
    new_incidents: list[str] = field(default_factory=list)
    highest_level: str | None = None
    detect_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp_ms": round(self.timestamp_ms, 1),
            "view": self.view.to_dict(),
            "tracks": self.tracks,
            "occupants": self.occupants,
            "machine": self.machine,
            "guard": self.guard,
            "postures": self.postures,
            "new_incidents": self.new_incidents,
            "highest_level": self.highest_level,
            "detect_ms": round(self.detect_ms, 2),
        }


class Keepout:
    """The watcher. Construct once per camera, feed frames in order."""

    def __init__(
        self,
        config: KeepoutConfig | None = None,
        detector: PersonDetector | None = None,
        *,
        on_evidence: Callable[[str, np.ndarray, dict[str, Any]], str] | None = None,
    ) -> None:
        self.config = config or KeepoutConfig()
        self.detector = detector
        self.on_evidence = on_evidence

        self.tracker = BoxTracker(self.config.tracker)
        self.view_guard = ViewGuard(self.config.view)
        self.machine = MotionEnergy(self.config.machine_zone, self.config.machine)
        self.guard = GuardChecker(self.config.guard)
        self.down = DownDetector(self.config.posture)
        self.blurrer = FaceBlurrer(self.config.privacy)
        self.log = IncidentLog()

        self.monitor: ZoneMonitor | None = None
        if self.config.danger_zone is not None:
            self.monitor = ZoneMonitor(self.config.danger_zone, self.config.dwell)

        self.frames_seen = 0
        self.frames_skipped = 0
        self.reference_frame: np.ndarray | None = None
        self.view_problems: dict[str, int] = {}
        self.timings: dict[str, float] = {}
        self.detect_calls = 0
        self._last_evidence_ms: dict[str, float] = {}
        # track_id -> (last seen ms, last box, depth inside the danger zone, first seen ms)
        self._last_known: dict[int, tuple[float, Box, float, float]] = {}
        self._vanished: set[int] = set()
        # Boxes of tracks that vanished inside the zone. We still believe somebody
        # is there, so the guard check must keep treating that area as occluded.
        self._shadow_boxes: dict[int, tuple[float, Box]] = {}

    # ---- setup ------------------------------------------------------------
    def set_reference(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> None:
        """Adopt the frame the zones were drawn against, and learn the guard from it."""
        self.reference_frame = frame.copy()
        self.view_guard.set_reference(frame)
        if self.config.guard_zone is not None:
            self.guard.learn(frame, self.config.guard_zone, timestamp_ms)
        if self.monitor is not None:
            self.monitor.frame_size = (frame.shape[1], frame.shape[0])

    # ---- detection --------------------------------------------------------
    def _detect(self, frame: np.ndarray) -> list[tuple[Box, float]]:
        if self.detector is None:
            self.detector = _default_detector(self.config.detection_score)
        people = self.detector(frame)
        floor = self.config.min_person_height_px
        return [(b, s) for b, s in people if (b[3] - b[1]) >= floor]

    # ---- the frame loop ---------------------------------------------------
    def process_frame(self, frame: np.ndarray, timestamp_ms: float, index: int | None = None
                      ) -> FrameResult:
        idx = self.frames_seen if index is None else index
        self.frames_seen += 1

        t0 = time.perf_counter()
        view = self.view_guard.check(frame)
        self._add_time("view_check", t0)
        for problem in view.problems:
            self.view_problems[problem.value] = self.view_problems.get(problem.value, 0) + 1

        result = FrameResult(index=idx, timestamp_ms=timestamp_ms, view=view)

        if not view.usable:
            self.frames_skipped += 1
            result.machine = self.machine.state.to_dict()
            result.machine["stale"] = True
            highest = self.log.highest_open_level()
            result.highest_level = highest.slug if highest else None
            self._maybe_view_evidence(frame, view, timestamp_ms)
            return result

        t0 = time.perf_counter()
        people = self._detect(frame)
        detect_ms = (time.perf_counter() - t0) * 1000.0
        self._add_time("detect", t0)
        self.detect_calls += 1
        result.detect_ms = detect_ms

        boxes = [b for b, _ in people]
        scores = [s for _, s in people]

        t0 = time.perf_counter()
        tracks = self.tracker.update(boxes, scores, timestamp_ms)
        self._add_time("track", t0)
        live_boxes = [t.box for t in tracks if t.misses == 0]

        t0 = time.perf_counter()
        machine_state = self.machine.update(frame, timestamp_ms, person_boxes=live_boxes)
        self._add_time("machine", t0)
        running = (self.config.assume_machine_running
                   if self.config.assume_machine_running is not None
                   else machine_state.running)

        t0 = time.perf_counter()
        # Include the last known box of anybody we believe is still there but can
        # no longer see. A person lying across the guard is exactly the case where
        # the detector has stopped producing a box, and calling that a removed
        # guard would bury the real emergency under a maintenance alert.
        guard_obs = self.guard.check(
            frame, timestamp_ms, person_boxes=live_boxes + self._shadow_list(timestamp_ms)
        )
        self._add_time("guard", t0)

        confirmed = [t for t in tracks if t.confirmed and t.misses == 0]
        t0 = time.perf_counter()
        postures = self.down.update(confirmed, timestamp_ms)
        self._add_time("posture", t0)

        t0 = time.perf_counter()
        occupants = self._update_zone(confirmed, timestamp_ms)
        new_ids = self._escalate(frame, confirmed, postures, occupants, guard_obs,
                                 running, timestamp_ms)
        self._add_time("escalate", t0)

        if self.tracker.removed_ids and self.monitor is not None:
            self.monitor.forget(set(self.tracker.removed_ids))

        result.tracks = [t.to_dict() for t in tracks if t.misses == 0]
        result.occupants = [o.to_dict(timestamp_ms) for o in occupants]
        result.machine = machine_state.to_dict()
        result.machine["effective_running"] = running
        result.guard = guard_obs.to_dict()
        result.postures = [p.to_dict() for p in postures]
        result.new_incidents = new_ids
        highest = self.log.highest_open_level()
        result.highest_level = highest.slug if highest else None
        return result

    # ---- steps ------------------------------------------------------------
    def _update_zone(self, tracks: list[Any], timestamp_ms: float) -> list[Any]:
        if self.monitor is None:
            return []
        for track in tracks:
            self.monitor.update(track.track_id, track.box, timestamp_ms)
        self.monitor.sweep(timestamp_ms)
        return self.monitor.qualified_occupants(timestamp_ms)

    def _escalate(self, frame, tracks, postures, occupants, guard_obs, running, ts
                  ) -> list[str]:
        new_ids: list[str] = []
        by_id = {t.track_id: t for t in tracks}
        posture_by_id = {p.track_id: p for p in postures}
        occupied_ids = {o.track_id for o in occupants}
        zone_name = self.config.danger_zone.name if self.config.danger_zone else "zone"

        # --- zone occupancy ------------------------------------------------
        for occ in occupants:
            level, reason = level_for_zone_entry(running)
            incident, changed = self.log.observe(
                kind="zone_entry", zone=zone_name, track_id=occ.track_id,
                level=level, reason=reason, timestamp_ms=ts, machine_running=running,
                dwell_ms=round(occ.dwell_ms(ts), 1),
                depth_px=round(occ.peak_depth_px, 1),
            )
            if changed:
                new_ids.append(incident.incident_id)
                track = by_id.get(occ.track_id)
                self._capture(frame, incident, ts,
                              boxes=[track.box] if track else [],
                              label=f"{incident.level.slug}-entry",
                              caption=reason)

        if self.monitor is not None:
            for state in self.monitor.states.values():
                if not state.inside and state.track_id not in occupied_ids:
                    self.log.clear("zone_entry", zone_name, state.track_id, ts)

        # --- person down ---------------------------------------------------
        for track in tracks:
            posture = posture_by_id.get(track.track_id)
            if posture is None or not posture.down:
                continue
            incident, changed = self.log.observe(
                kind="person_down", zone=zone_name, track_id=track.track_id,
                level=Level.CRITICAL, reason=REASONS["person_down"], timestamp_ms=ts,
                machine_running=running,
                aspect=round(posture.aspect, 3),
                still_for_ms=round(posture.duration_ms, 1),
                in_zone=track.track_id in occupied_ids,
            )
            if changed:
                new_ids.append(incident.incident_id)
                self._capture(frame, incident, ts, boxes=[track.box],
                              label="person-down", caption=posture.reason)

        # --- a person who stopped being visible inside the zone -------------
        for track in tracks:
            depth = 0.0
            if self.config.danger_zone is not None:
                from .zones import contact_point

                point = contact_point(track.box, self.config.danger_zone.contact)
                depth = self.config.danger_zone.signed_distance(
                    point, self.monitor.frame_size if self.monitor else None
                )
            self._last_known[track.track_id] = (ts, track.box, depth, track.first_ms)
            self._vanished.discard(track.track_id)
            self._shadow_boxes.pop(track.track_id, None)

        for incident_id in self._check_vanished(frame, ts, set(by_id)):
            new_ids.append(incident_id)

        # --- guard ---------------------------------------------------------
        if self.config.guard_zone is not None:
            guard_zone = self.config.guard_zone.name
            if guard_obs.status is GuardStatus.MISSING:
                incident, changed = self.log.observe(
                    kind="guard", zone=guard_zone, track_id=None,
                    level=Level.GUARD, reason=REASONS["guard_missing"], timestamp_ms=ts,
                    machine_running=running,
                    edge_score=round(guard_obs.edge_score, 3),
                    appearance_score=round(guard_obs.appearance_score, 3),
                )
                if changed:
                    new_ids.append(incident.incident_id)
                    self._capture(frame, incident, ts, boxes=[t.box for t in tracks],
                                  label="guard-missing", caption=guard_obs.note)
            elif guard_obs.status is GuardStatus.PRESENT:
                self.log.clear("guard", guard_zone, None, ts)

        return new_ids

    def _shadow_list(self, ts: float, hold_ms: float = 30_000.0) -> list[Box]:
        """Boxes where somebody vanished and has not been accounted for."""
        return [b for _, (seen, b) in self._shadow_boxes.items() if ts - seen <= hold_ms]

    def _check_vanished(self, frame: np.ndarray, ts: float, live: set[int]) -> list[str]:
        """Raise CRITICAL for anyone who stopped being detected deep inside the zone.

        This is the measured answer to a measured problem. YOLOX-tiny loses a
        person somewhere past forty degrees of body rotation (docs/evaluation.md),
        so a worker who collapses onto the floor frequently stops producing a
        detection at all. Posture cannot fire, because there is no box to measure.

        But the disappearance is itself the signal. A person standing well inside
        a zone cannot leave it without walking out across the boundary, which we
        would see. If they were there, and then there is nobody, and the boundary
        was never crossed, somebody has to go and look. This is the same
        conclusion a human watching the monitor would reach, and it is the one
        nobody reached in the HSE conveyor case.

        The rule is deliberately narrow: the last detection must have been at
        least `vanish_min_depth_px` inside the zone and at least
        `vanish_edge_margin_px` from the frame edge, so somebody walking out of
        shot does not trigger it.
        """
        if self.config.danger_zone is None:
            return []
        zone_name = self.config.danger_zone.name
        height, width = frame.shape[:2]
        raised: list[str] = []

        live_boxes = [b for tid, (_, b, _, _) in self._last_known.items() if tid in live]

        for track_id, (last_ms, box, depth, first_ms) in list(self._last_known.items()):
            if track_id in live or track_id in self._vanished:
                continue
            if ts - last_ms < self.config.vanish_grace_ms:
                continue

            # A track that only existed for a moment is a detector blip, not a person.
            if (last_ms - first_ms) < self.config.vanish_min_track_ms:
                self._vanished.add(track_id)
                continue

            # If somebody else was standing on top of them when they went, this is
            # mutual occlusion between two people, which happens constantly.
            from .track import iou as _iou

            if any(_iou(box, other) > self.config.vanish_max_neighbour_iou
                   for other in live_boxes):
                self._vanished.add(track_id)
                continue

            near_edge = (
                box[0] <= self.config.vanish_edge_margin_px
                or box[1] <= self.config.vanish_edge_margin_px
                or box[2] >= width - self.config.vanish_edge_margin_px
                or box[3] >= height - self.config.vanish_edge_margin_px
            )
            if depth < self.config.vanish_min_depth_px or near_edge:
                self._vanished.add(track_id)  # explained away; do not keep re-checking
                continue

            self._vanished.add(track_id)
            self._shadow_boxes[track_id] = (ts, box)
            incident, changed = self.log.observe(
                kind="person_unaccounted", zone=zone_name, track_id=track_id,
                level=Level.CRITICAL, reason=REASONS["person_unaccounted"],
                timestamp_ms=ts, machine_running=self.machine.state.running,
                last_seen_ms=round(last_ms, 1),
                missing_for_ms=round(ts - last_ms, 1),
                depth_inside_zone_px=round(depth, 1),
            )
            if changed:
                raised.append(incident.incident_id)
                self._capture(
                    frame, incident, ts, boxes=[box], label="person-unaccounted",
                    caption=(
                        f"last seen {round(depth)} px inside the zone "
                        f"{(ts - last_ms) / 1000:.1f} s ago, and has not been seen since"
                    ),
                )
        return raised

    # ---- evidence ---------------------------------------------------------
    def _capture(self, frame: np.ndarray, incident, ts: float, *, boxes: list[Box],
                 label: str, caption: str) -> None:
        if self.on_evidence is None:
            return
        key = f"{incident.incident_id}:{label}"
        last = self._last_evidence_ms.get(key)
        if last is not None and ts - last < EVIDENCE_COOLDOWN_MS:
            return
        self._last_evidence_ms[key] = ts

        annotated = self.annotate(frame, boxes=boxes, banner=f"{incident.level.label}: {caption}")
        redacted, report = self.blurrer.redact(annotated, boxes)
        name = f"{incident.incident_id}-{len(incident.evidence):02d}-{label}"
        uri = self.on_evidence(name, redacted, report)
        incident.add_evidence(EvidenceRef(
            label=label, uri=uri, timestamp_ms=ts, caption=caption,
            redacted=bool(report.get("enabled")),
        ))

    def _maybe_view_evidence(self, frame: np.ndarray, view: ViewStatus, ts: float) -> None:
        """One evidence frame the first time each view problem appears."""
        if self.on_evidence is None:
            return
        for problem in view.problems:
            key = f"view:{problem.value}"
            if key in self._last_evidence_ms:
                continue
            self._last_evidence_ms[key] = ts
            annotated = self.annotate(frame, boxes=[], banner=f"View unusable: {problem.value}",
                                      colour=(40, 140, 250))
            redacted, report = self.blurrer.redact(annotated, [])
            self.on_evidence(f"view-{problem.value}", redacted, report)

    def annotate(self, frame: np.ndarray, *, boxes: list[Box], banner: str = "",
                 colour: tuple[int, int, int] = (60, 90, 240)) -> np.ndarray:
        out = frame.copy()
        for zone, tint in (
            (self.config.machine_zone, (200, 160, 60)),
            (self.config.guard_zone, (90, 200, 140)),
            (self.config.danger_zone, (60, 90, 240)),
        ):
            if zone is not None:
                out = draw_zone(out, zone, colour=tint)
        for box in boxes:
            x1, y1, x2, y2 = (round(v) for v in box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        if banner:
            _banner(out, banner, colour)
        return out

    # ---- running a whole clip --------------------------------------------
    def run(self, frames: Iterable[tuple[np.ndarray, float]],
            progress: Callable[[int, float], None] | None = None) -> list[FrameResult]:
        results: list[FrameResult] = []
        for i, (frame, ts) in enumerate(frames):
            if self.reference_frame is None:
                self.set_reference(frame, ts)
            results.append(self.process_frame(frame, ts, index=i))
            if progress is not None:
                progress(i, ts)
        return results

    # ---- reporting --------------------------------------------------------
    def _add_time(self, name: str, t0: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    def summary(self) -> dict[str, Any]:
        incidents = self.log.summary()
        usable = self.frames_seen - self.frames_skipped
        return {
            "frames_seen": self.frames_seen,
            "frames_usable": usable,
            "frames_skipped_unusable": self.frames_skipped,
            "usable_fraction": round(usable / self.frames_seen, 4) if self.frames_seen else 0.0,
            "view_problems": dict(self.view_problems),
            "incidents": incidents,
            "machine": self.machine.state.to_dict(),
            "guard": {"status": self.guard.status.value, "since_ms": self.guard.since_ms},
            "tracks_created": self.tracker._next_id - 1,
            "detect_calls": self.detect_calls,
            "timings_ms": {k: round(v, 2) for k, v in self.timings.items()},
            "privacy": {
                "face_blur": self.config.privacy.enabled,
                "method": self.blurrer.method,
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(image: np.ndarray, text: str, colour: tuple[int, int, int]) -> None:
    h, w = image.shape[:2]
    bar = max(28, round(h * 0.062))
    strip = image[0:bar, 0:w]
    cv2.addWeighted(strip, 0.25, np.full_like(strip, 12), 0.75, 0.0, strip)
    cv2.rectangle(image, (0, 0), (w, bar), colour, 3)
    size = max(14, int(bar * 0.48))
    if hasattr(cv2, "FontFace"):
        cv2.putText(image, text[:96], (12, int(bar * 0.70)), (245, 245, 245),
                    cv2.FontFace("sans"), size)
    else:  # pragma: no cover
        cv2.putText(image, text[:96], (12, int(bar * 0.70)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (245, 245, 245), 2)


def _default_detector(score: float) -> PersonDetector:
    """YOLOX-tiny in `cv2.dnn`, person class only. Loaded lazily so tests stay fast."""
    from visioncore import YoloxDetector

    detector = YoloxDetector(score_threshold=score)

    def detect(frame: np.ndarray) -> list[tuple[Box, float]]:
        return [
            (d.bbox, d.score) for d in detector.detect(frame) if d.class_id == PERSON_CLASS_ID
        ]

    return detect


def default_zones(size: tuple[int, int]) -> dict[str, Zone]:
    """Sensible starting polygons for a frame of `size`, for the demo and for tests."""
    w, h = size
    return {
        "danger": Zone(
            name="conveyor danger zone",
            points=((w * 0.30, h * 0.42), (w * 0.86, h * 0.42),
                    (w * 0.92, h * 0.95), (w * 0.24, h * 0.95)),
            reference_size=(w, h),
            kind="danger",
            contact=ContactPoint.FEET,
        ),
        "machine": Zone(
            name="conveyor",
            points=((w * 0.34, h * 0.30), (w * 0.82, h * 0.30),
                    (w * 0.82, h * 0.52), (w * 0.34, h * 0.52)),
            reference_size=(w, h),
            kind="machine",
        ),
        "guard": Zone(
            name="fixed guard",
            points=((w * 0.36, h * 0.52), (w * 0.62, h * 0.52),
                    (w * 0.62, h * 0.72), (w * 0.36, h * 0.72)),
            reference_size=(w, h),
            kind="guard",
        ),
    }


__all__ = [
    "EVIDENCE_COOLDOWN_MS",
    "PERSON_CLASS_ID",
    "FrameResult",
    "Keepout",
    "KeepoutConfig",
    "PersonDetector",
    "ViewProblem",
    "default_zones",
]
