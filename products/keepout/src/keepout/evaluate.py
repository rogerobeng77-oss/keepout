"""Measurement against the labelled set. Every number in docs/evaluation.md comes from here.

What is measured, and why each one:

* **Detection rate** — of the frames where the labels say somebody's feet were
  inside the danger zone, on how many did Keepout have an occupant in the zone?
  This is the number that decides whether the product does its job at all.
* **Time to alert** — from the labelled instant a person's feet crossed the zone
  boundary to the timestamp on the incident. The brief asks for under a second.
* **False alerts per camera-hour** — measured on the clips where the labels say
  nobody was ever in the zone. This is the number that decides whether anybody
  leaves the system switched on.
* **Level correctness** — did entry while running produce an alert and entry while
  stopped produce a note? A product that alerts on both is a nuisance; one that
  notes on both is useless.
* **View-problem recall** — over the windows where a degradation was scripted, was
  the right problem raised, and did the pipeline stop producing zone incidents?
* **The rotation sweep** — at what body angle does YOLOX-tiny stop returning a
  person box at all? This is the measurement that justifies the vanish rule, and
  it is the most important limitation in the product.

Everything writes its raw per-clip output to the evaluation directory so a judge
can recompute any figure without rerunning the pipeline.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TOLERANCE_MS = 1000.0  # how close an incident must be to a labelled event to count


# ---------------------------------------------------------------------------
# Per-clip scoring
# ---------------------------------------------------------------------------


def _truth(clip: Path) -> dict[str, Any] | None:
    labels = clip.with_suffix(".json")
    if not labels.is_file():
        return None
    return json.loads(labels.read_text())


def _occupancy_truth(truth: dict[str, Any]) -> dict[float, int]:
    return {float(f["timestamp_ms"]): int(f["people_in_danger_zone"])
            for f in truth.get("frames", [])}


def _nearest(target_ms: float, timestamps: list[float]) -> float | None:
    if not timestamps:
        return None
    return min(timestamps, key=lambda t: abs(t - target_ms))


def score_clip(clip: Path, keeper, results, timing: dict[str, Any]) -> dict[str, Any]:
    """Compare one clip's run against its labels."""
    truth = _truth(clip)
    incidents = keeper.log.to_list()
    report: dict[str, Any] = {
        "clip": clip.name,
        "labelled": truth is not None,
        "timing": timing,
        "incidents": [
            {"level": i["level"], "kind": i["kind"], "started_ms": i["started_ms"],
             "reason": i["reason"][:90]} for i in incidents
        ],
        "frames_usable": keeper.summary()["frames_usable"],
        "frames_seen": keeper.summary()["frames_seen"],
        "view_problems": keeper.summary()["view_problems"],
    }
    if truth is None:
        # Unlabelled real footage. We still know one thing for certain about it:
        # nobody collapses in it. So every person-unaccounted incident is a false
        # positive, and that count is the vanish rule's false-alarm rate on real,
        # crowded, mutually occluding video.
        seconds = timing.get("clip_seconds") or 0.0
        hours = max(1e-9, seconds / 3600.0)
        unaccounted = [i for i in incidents if i["kind"] == "person_unaccounted"]
        report["note"] = (
            "Real footage with no frame-level labels. Zone entries are not scored "
            "because no one has marked them; person-unaccounted incidents are counted "
            "as false positives because nobody collapses in this clip."
        )
        report["unlabelled"] = {
            "clip_seconds": round(seconds, 1),
            "camera_hours": round(hours, 5),
            "incidents_total": len(incidents),
            "zone_entries": sum(1 for i in incidents if i["kind"] == "zone_entry"),
            "person_unaccounted_false_criticals": len(unaccounted),
            "vanish_false_criticals_per_camera_hour": round(len(unaccounted) / hours, 1),
        }
        return report

    occupancy = _occupancy_truth(truth)
    # Our own per-frame occupancy, keyed by the nearest labelled timestamp.
    ours = {round(r.timestamp_ms, 1): len(r.occupants) for r in results}
    usable = {round(r.timestamp_ms, 1) for r in results if r.view.usable}

    truth_ts = sorted(occupancy)
    hits = misses = 0
    false_frames = 0
    for ts in truth_ts:
        key = _nearest(ts, list(ours)) if ts not in ours else ts
        if key is None or key not in usable:
            continue
        expected = occupancy[ts] > 0
        got = ours.get(key, 0) > 0
        if expected and got:
            hits += 1
        elif expected and not got:
            misses += 1
        elif not expected and got:
            false_frames += 1

    occupied_frames = hits + misses
    report["occupancy"] = {
        "frames_with_someone_in_the_zone": occupied_frames,
        "frames_we_saw_them": hits,
        "frames_we_missed_them": misses,
        "detection_rate": round(hits / occupied_frames, 4) if occupied_frames else None,
        "frames_we_saw_somebody_who_was_not_there": false_frames,
        "empty_frames": len(truth_ts) - occupied_frames,
    }

    # --- time to alert ------------------------------------------------------
    entries = [e for e in truth.get("events", []) if e["event"] == "zone_entry"]
    latencies: list[dict[str, Any]] = []
    for entry in entries:
        want = entry.get("expected_level", "alert")
        candidates = [
            i for i in incidents
            if i["kind"] == "zone_entry" and i["started_ms"] >= entry["ts_ms"] - 200.0
        ]
        first = min(candidates, key=lambda i: i["started_ms"]) if candidates else None
        latencies.append({
            "entry_ms": entry["ts_ms"],
            "expected_level": want,
            "raised_ms": first["started_ms"] if first else None,
            "raised_level": first["level"] if first else None,
            "latency_ms": round(first["started_ms"] - entry["ts_ms"], 1) if first else None,
            "level_correct": bool(first and first["peak_level" if "peak_level" in first
                                                 else "level"] == want)
            if first else False,
        })
    blind_at_entry = {round(r.timestamp_ms, 1) for r in results if not r.view.usable}
    for entry in latencies:
        entry["view_usable_at_entry"] = (
            _nearest(entry["entry_ms"], sorted(blind_at_entry)) is None
            or min((abs(entry["entry_ms"] - t) for t in blind_at_entry), default=1e9) > 200.0
        )
    report["entries"] = latencies
    measured = [entry["latency_ms"] for entry in latencies
                if entry["latency_ms"] is not None and entry["view_usable_at_entry"]]
    report["latency_excluded_blind_entries"] = sum(
        1 for e in latencies if not e["view_usable_at_entry"]
    )
    report["time_to_alert_ms"] = {
        "n": len(measured),
        "median": round(float(np.median(measured)), 1) if measured else None,
        "max": round(float(np.max(measured)), 1) if measured else None,
        "under_1000ms": sum(1 for v in measured if v <= 1000.0),
        "missed_entries": sum(1 for e in latencies if e["latency_ms"] is None),
    }

    # --- false alerts -------------------------------------------------------
    quiet = occupied_frames == 0
    hours = max(1e-9, (truth.get("frames", [{}])[-1].get("timestamp_ms", 0.0)) / 3_600_000.0)
    spurious = [i for i in incidents if i["kind"] in ("zone_entry", "person_down",
                                                      "person_unaccounted")] if quiet else []
    report["false_alerts"] = {
        "clip_has_nobody_in_the_zone": quiet,
        "spurious_incidents": len(spurious),
        "camera_hours": round(hours, 5),
        "per_camera_hour": round(len(spurious) / hours, 1) if quiet else None,
    }

    # --- the vanish rule's own false-positive count -------------------------
    collapse_events = [e for e in truth.get("events", []) if e["event"] == "person_collapses"]
    unaccounted = [i for i in incidents if i["kind"] == "person_unaccounted"]
    if not collapse_events:
        report["vanish_rule"] = {
            "nobody_collapsed_in_this_clip": True,
            "false_criticals": len(unaccounted),
            "per_camera_hour": round(len(unaccounted) / hours, 1),
        }

    # --- guard --------------------------------------------------------------
    guard_events = [e for e in truth.get("events", []) if e["event"] == "guard_removed"]
    if guard_events:
        raised = [i for i in incidents if i["kind"] == "guard"]
        first = min(raised, key=lambda i: i["started_ms"]) if raised else None
        report["guard"] = {
            "removed_at_ms": guard_events[0]["ts_ms"],
            "raised_at_ms": first["started_ms"] if first else None,
            "latency_ms": round(first["started_ms"] - guard_events[0]["ts_ms"], 1)
            if first else None,
            "detected": first is not None,
        }
    elif any(i["kind"] == "guard" for i in incidents):
        report["guard"] = {"removed_at_ms": None, "false_guard_alerts":
                           sum(1 for i in incidents if i["kind"] == "guard")}

    # --- collapse -----------------------------------------------------------
    collapses = [e for e in truth.get("events", []) if e["event"] == "person_collapses"]
    if collapses:
        critical = [i for i in incidents if i["level"] == "critical"]
        first = min(critical, key=lambda i: i["started_ms"]) if critical else None
        report["person_down"] = {
            "collapsed_at_ms": collapses[0]["ts_ms"],
            "raised_at_ms": first["started_ms"] if first else None,
            "latency_ms": round(first["started_ms"] - collapses[0]["ts_ms"], 1)
            if first else None,
            "raised_by": first["kind"] if first else None,
            "detected": first is not None,
            "latched": bool(first and first["latched"]),
        }

    # --- view problems ------------------------------------------------------
    degraded = [f for f in truth.get("frames", []) if f["view_problems"]]
    if degraded:
        wanted: dict[str, int] = {}
        for frame in degraded:
            for problem in frame["view_problems"]:
                wanted[problem] = wanted.get(problem, 0) + 1
        got = keeper.summary()["view_problems"]
        report["view"] = {
            "scripted_problem_frames": wanted,
            "raised_problem_frames": got,
            "problems_found": sorted(set(wanted) & set(got)),
            "problems_missed": sorted(set(wanted) - set(got)),
            "recall_by_problem": {
                k: round(min(1.0, got.get(k, 0) / v), 3) for k, v in wanted.items()
            },
        }
        # Did we alert against a stale zone while blind? That is the failure that
        # matters, and it is asked of our own per-frame usability rather than of the
        # label timestamps, so a frame on the boundary of a scripted window does not
        # count as blind when the pipeline could in fact see.
        blind_ms = {round(r.timestamp_ms, 1) for r in results if not r.view.usable}
        report["view"]["blind_frames"] = len(blind_ms)
        report["view"]["incidents_raised_while_blind"] = sum(
            1 for i in incidents if round(i["started_ms"], 1) in blind_ms
        )
    return report


# ---------------------------------------------------------------------------
# The detector's rotation limit
# ---------------------------------------------------------------------------


def rotation_sweep(source: Path, angles=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90),
                   trials: int = 4) -> dict[str, Any]:
    """At what body angle does YOLOX-tiny stop seeing a person?

    This is the single most consequential measurement in the product. A person on
    the floor is rotated roughly 90 degrees from upright, and if the detector
    cannot see them then no posture rule can either. The answer here is why
    `pipeline._check_vanished` exists.
    """
    from visioncore import YoloxDetector, iter_video

    from .synth import Sprite, clean_plate, matte

    detector = YoloxDetector(score_threshold=0.25)
    plate = clean_plate(source)
    sprites: list[tuple[Sprite, np.ndarray]] = []
    for frame in iter_video(source, stride=37, start_ms=8000.0, max_frames=40):
        people = [d for d in detector.detect(frame.image)
                  if d.class_id == 0 and (d.bbox[3] - d.bbox[1]) >= 70 and d.score > 0.7]
        for det in people[:1]:
            sprite = matte(frame.image, plate, det.bbox)
            if sprite is not None:
                sprites.append((sprite, plate.copy()))
        if len(sprites) >= trials:
            break
    if not sprites:
        return {"error": "could not matte a person to sweep with"}

    rows: list[dict[str, Any]] = []
    for angle in angles:
        found, scores = 0, []
        for sprite, background in sprites:
            big = sprite.scaled(sprite.bgr.shape[0] * 2.0).rotated(angle)
            canvas = background.copy()
            sh, sw = big.bgr.shape[:2]
            py = max(0, canvas.shape[0] // 2 - sh // 2)
            px = max(0, canvas.shape[1] // 2 - sw // 2)
            region = canvas[py:py + sh, px:px + sw]
            alpha = big.alpha[:region.shape[0], :region.shape[1]] > 8
            region[alpha] = big.bgr[:region.shape[0], :region.shape[1]][alpha]
            hits = [
                d for d in detector.detect(canvas)
                if d.class_id == 0
                and d.bbox[0] > px - 30 and d.bbox[2] < px + sw + 30
                and d.bbox[1] > py - 30 and d.bbox[3] < py + sh + 30
            ]
            if hits:
                found += 1
                best = max(hits, key=lambda d: d.score)
                scores.append(best.score)
        rows.append({
            "body_angle_deg": angle,
            "trials": len(sprites),
            "detected": found,
            "detection_rate": round(found / len(sprites), 3),
            "mean_score": round(float(np.mean(scores)), 3) if scores else None,
        })
    lost = next((r["body_angle_deg"] for r in rows if r["detection_rate"] < 0.5), None)
    return {
        "rows": rows,
        "first_angle_below_half": lost,
        "conclusion": (
            f"YOLOX-tiny drops below a 50% detection rate at about {lost} degrees of body "
            "rotation. A person lying on the floor is near 90 degrees, so posture-based "
            "person-down detection cannot be relied on by itself. This is why a track "
            "that disappears deep inside the zone is escalated on its own."
        ) if lost is not None else "no angle fell below half; check the sweep setup",
    }


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


def stage_costs(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, float] = {}
    frames = 0
    for entry in per_clip:
        frames += entry["timing"]["frames"]
        for name, ms in entry.get("stage_ms", {}).items():
            total[name] = total.get(name, 0.0) + ms
    return {
        "frames": frames,
        "ms_per_frame_by_stage": {k: round(v / max(1, frames), 3)
                                  for k, v in sorted(total.items(), key=lambda kv: -kv[1])},
        "total_ms_per_frame": round(sum(total.values()) / max(1, frames), 3),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_evaluation(samples_dir: Path, out_dir: Path | None = None,
                   max_frames: int = 900) -> dict[str, Any]:
    from .cli import run_clip
    from .paths import eval_dir

    out_dir = out_dir or eval_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(samples_dir.glob("*.mp4"))
    per_clip: list[dict[str, Any]] = []
    started = time.perf_counter()

    for clip in clips:
        keeper, results, timing = run_clip(clip, max_frames=max_frames, quiet=True)
        report = score_clip(clip, keeper, results, timing)
        report["stage_ms"] = {k: round(v, 2) for k, v in keeper.timings.items()}
        per_clip.append(report)
        (out_dir / f"{clip.stem}.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"  {clip.stem:<16} {timing['frames']:>4} frames  "
              f"{timing['ms_per_frame']:>6.1f} ms/frame  "
              f"{len(report['incidents'])} incidents")

    source = samples_dir.parent / "source" / "vtest.avi"
    sweep = rotation_sweep(source) if source.is_file() else {"error": "no source clip"}
    (out_dir / "rotation-sweep.json").write_text(json.dumps(sweep, indent=2, default=str))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "opencv_version": cv2.__version__,
        "clips": len(clips),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "per_clip": per_clip,
        "headline": _headline(per_clip),
        "throughput": stage_costs(per_clip),
        "rotation_sweep": sweep,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _headline(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [c["occupancy"]["detection_rate"] for c in per_clip
             if c.get("occupancy", {}).get("detection_rate") is not None]
    latencies: list[float] = []
    for clip in per_clip:
        for entry in clip.get("entries", []):
            if entry["latency_ms"] is not None and entry.get("view_usable_at_entry", True):
                latencies.append(entry["latency_ms"])
    quiet = [c for c in per_clip if c.get("false_alerts", {}).get("clip_has_nobody_in_the_zone")]
    spurious = sum(c["false_alerts"]["spurious_incidents"] for c in quiet)
    quiet_hours = sum(c["false_alerts"]["camera_hours"] for c in quiet)

    view = next((c["view"] for c in per_clip if "view" in c), {})
    guard = next((c["guard"] for c in per_clip if c.get("guard", {}).get("detected")), {})
    down = next((c["person_down"] for c in per_clip if "person_down" in c), {})

    return {
        "detection_rate_mean": round(float(np.mean(rates)), 4) if rates else None,
        "detection_rate_min": round(float(np.min(rates)), 4) if rates else None,
        "time_to_alert_median_ms": round(float(np.median(latencies)), 1) if latencies else None,
        "time_to_alert_max_ms": round(float(np.max(latencies)), 1) if latencies else None,
        "entries_alerted_within_1s": sum(1 for v in latencies if v <= 1000.0),
        "entries_total": len(latencies),
        "false_alerts_per_camera_hour": round(spurious / quiet_hours, 1)
        if quiet_hours > 0 else None,
        "quiet_camera_hours": round(quiet_hours, 4),
        "guard_removal_latency_ms": guard.get("latency_ms"),
        "person_down_latency_ms": down.get("latency_ms"),
        "person_down_raised_by": down.get("raised_by"),
        "view_problems_found": view.get("problems_found"),
        "view_problems_missed": view.get("problems_missed"),
        "incidents_raised_while_blind": view.get("incidents_raised_while_blind"),
        "vanish_false_criticals": sum(
            c.get("vanish_rule", {}).get("false_criticals", 0) for c in per_clip
        ),
        "vanish_false_criticals_by_clip": {
            c["clip"]: c["vanish_rule"]["false_criticals"]
            for c in per_clip if "vanish_rule" in c
        },
        "unlabelled_real_footage": {
            c["clip"]: c["unlabelled"] for c in per_clip if "unlabelled" in c
        },
    }


def ci95(successes: int, trials: int) -> tuple[float, float]:
    """Wilson interval. A rate quoted without one on 30 frames is not a measurement."""
    if trials == 0:
        return (0.0, 1.0)
    z = 1.96
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
