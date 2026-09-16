"""The Keepout web service: the analyzer servicekit calls, plus the control-room routes.

servicekit gives us `/`, `/healthz`, `/version`, `POST /api/jobs`, the SSE progress
stream, the JSON result and the evidence endpoint. This module adds the four things
a control room needs that a generic upload page does not:

    GET  /api/demo                  the pre-computed demo run, served from disk
    GET  /api/samples               the bundled clips, with their ground truth
    GET  /api/samples/{name}/frame  the reference frame, for drawing a zone on
    POST /api/zones/propose         propose a zone from where the machine moves
    POST /api/jobs/{id}/incidents/{iid}/acknowledge

`/api/demo` exists because of one line in the brief: a judge must see an alert
within seconds of a cold start. Running the clip takes about ten seconds, so the
run is executed at **image build time** (`keepout demo --bake`) and the result and
its evidence frames are baked into the container. The page is therefore instant,
and the same clip can be re-run live from the UI to prove the result is real.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from servicekit import JobContext, ProductInfo, ServiceConfig, ServiceError, create_app
from servicekit.logs import get_logger
from visioncore import Evidence, RunRecord, iter_video, video_info

from . import HSE_QUOTE, HSE_SOURCE, __version__
from .escalate import Level
from .machine import MachineConfig
from .paths import demo_dir, describe, samples_dir, static_dir
from .pipeline import Keepout, KeepoutConfig, default_zones
from .privacy import PRIVACY_STATEMENT, FaceBlurrer, PrivacyConfig
from .storage import EvidenceSink
from .zones import ContactPoint, DwellConfig, Zone, propose_zone_from_motion

log = get_logger("keepout.service")

SAMPLES_DIR = samples_dir()
DEMO_DIR = demo_dir()
STATIC_DIR = static_dir()

ACCENT = "#F5C518"  # hazard yellow

MAX_ANALYSIS_FRAMES = int(os.environ.get("KEEPOUT_MAX_FRAMES", "900"))


# ---------------------------------------------------------------------------
# Config from request params
# ---------------------------------------------------------------------------


def _zone_from_param(value: Any, size: tuple[int, int], name: str, kind: str,
                     contact: ContactPoint = ContactPoint.FEET) -> Zone | None:
    """Accept either a full zone dict or a bare list of points."""
    if not value:
        return None
    if isinstance(value, dict) and "points" in value:
        data = dict(value)
        data.setdefault("name", name)
        data.setdefault("kind", kind)
        data.setdefault("reference_size", list(size))
        data.setdefault("contact", contact.value)
        return Zone.from_dict(data)
    if isinstance(value, list) and len(value) >= 3:
        return Zone(
            name=name,
            points=tuple((float(p[0]), float(p[1])) for p in value),
            reference_size=size,
            kind=kind,
            contact=contact,
        )
    raise ServiceError("BAD_REQUEST", f"{name} must be a polygon of at least 3 points")


def config_from_params(params: dict[str, Any], size: tuple[int, int]) -> KeepoutConfig:
    """Turn the UI's JSON into a `KeepoutConfig`, falling back to sensible defaults."""
    defaults = default_zones(size)
    danger = _zone_from_param(params.get("danger_zone"), size, "danger zone", "danger")
    machine = _zone_from_param(params.get("machine_zone"), size, "machine", "machine",
                               ContactPoint.CENTROID)
    guard = _zone_from_param(params.get("guard_zone"), size, "fixed guard", "guard",
                             ContactPoint.CENTROID)

    if danger is None and machine is None and guard is None:
        danger, machine, guard = defaults["danger"], defaults["machine"], defaults["guard"]

    machine_cfg = MachineConfig()
    if "running_threshold" in params:
        machine_cfg = MachineConfig(
            running_threshold=float(params["running_threshold"]),
            stopped_threshold=float(params.get("stopped_threshold",
                                               float(params["running_threshold"]) * 0.5)),
        )

    return KeepoutConfig(
        danger_zone=danger,
        machine_zone=machine,
        guard_zone=guard,
        dwell=DwellConfig(min_dwell_ms=float(params.get("min_dwell_ms", 0.0))),
        machine=machine_cfg,
        # Blurring is not a request parameter. Evidence frames are always redacted;
        # an old client that sends blur_faces=false is ignored, not obeyed.
        privacy=PrivacyConfig(enabled=True),
        detection_score=float(params.get("detection_score", 0.35)),
        tiled_detection=bool(params.get("tiled_detection", False)),
        stride=max(1, int(params.get("stride", 1))),
        max_side=int(params.get("max_side", 960)) or None,
        assume_machine_running=params.get("assume_machine_running"),
    )


# ---------------------------------------------------------------------------
# The analyzer
# ---------------------------------------------------------------------------


def analyze(ctx: JobContext) -> RunRecord:
    """Run the Keepout pipeline over an uploaded clip and fill in the RunRecord."""
    record = ctx.record
    sink: EvidenceSink = getattr(ctx, "_sink", None) or EvidenceSink.from_env()

    ctx.progress(2, "reading the clip")
    try:
        info = video_info(ctx.input_path)
    except Exception as exc:
        raise ServiceError("BAD_REQUEST", f"that file is not a video we can decode: {exc}") from exc
    if info.frame_count <= 0 and info.duration_ms <= 0:
        raise ServiceError("BAD_REQUEST", "the clip has no decodable frames")

    params = dict(ctx.params)
    size = (info.width, info.height)
    max_side = int(params.get("max_side", 960)) or None
    scale = 1.0
    if max_side and max(size) > max_side:
        scale = max_side / float(max(size))
    work_size = (round(size[0] * scale), round(size[1] * scale))

    config = config_from_params(params, work_size)
    record.params.update({"config": config.to_dict(), "source": info.to_dict(),
                          "work_size": list(work_size)})

    def on_evidence(name: str, image: np.ndarray, report: dict[str, Any]) -> str:
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        if not ok:
            return ""
        data = buffer.tobytes()
        uri = ctx.save_evidence(f"{name}.jpg", data)
        s3_uri = sink.mirror(f"{ctx.job_id}/{name}.jpg", data)
        record.add_evidence(Evidence(
            label=name, kind="frame", uri=uri, caption=report.get("note", ""),
            metrics={"redacted": report.get("enabled", False),
                     "blur_method": report.get("method"),
                     "face_detector": report.get("face_detector"),
                     "person_boxes_blurred": report.get("head_regions", 0),
                     "faces_detected": report.get("faces_detected", 0),
                     "person_sweep": report.get("person_sweep"),
                     "regions_blurred": report.get("regions", 0),
                     "s3": s3_uri},
        ))
        return uri

    keeper = Keepout(config, on_evidence=on_evidence)
    ctx.progress(6, "loading YOLOX-tiny through cv2.dnn")

    # The frame budget exists because App Runner runs YOLOX-tiny at about 440 ms a
    # frame, so 900 frames is already six or seven minutes. A clip longer than the
    # budget used to stop silently at frame 900: a 47 s clip was analysed for 36 s
    # and its real alert at 44.8 s never happened. Now, unless the caller chose a
    # stride, the stride is raised so the whole clip fits, and the record says so;
    # with `cover_whole_clip: false` the run is cut short, and the record says that.
    budget = MAX_ANALYSIS_FRAMES
    stride = config.stride
    coverage_note = None
    if (info.frame_count > budget * stride and "stride" not in params
            and bool(params.get("cover_whole_clip", True))):
        stride = math.ceil(info.frame_count / budget)
        coverage_note = (
            f"the clip has {info.frame_count} frames and this instance analyses at most "
            f"{budget}, so every {_ordinal(stride)} frame is analysed "
            f"({info.fps / stride:.1f} per second) to cover all "
            f"{info.duration_ms / 1000:.1f} s"
        )
        config.stride = stride
        record.params["config"] = config.to_dict()
        ctx.note(coverage_note)
    total_expected = min(budget, max(1, info.frame_count // stride) if info.frame_count else budget)

    started = time.perf_counter()
    last_report = 0.0
    frames_read = 0
    last_ms = 0.0
    live: list[dict[str, Any]] = []

    for frame in iter_video(ctx.input_path, stride=stride, max_frames=budget, max_side=max_side):
        result = keeper.process_frame(frame.image, frame.timestamp_ms, index=frame.index)
        for event in result.view_events:
            ctx.note(f"{event['timestamp_ms'] / 1000:.1f} s: {event['message']}")
        frames_read += 1
        last_ms = frame.timestamp_ms
        live.append(result.to_dict())

        now = time.perf_counter()
        if now - last_report > 0.35:
            last_report = now
            pct = 8.0 + 84.0 * min(1.0, frames_read / max(1, total_expected))
            ctx.progress(pct, f"frame {frames_read} of about {total_expected}",
                         highest_level=result.highest_level,
                         machine_running=result.machine.get("effective_running"),
                         incidents=len(keeper.log.incidents))
        for incident_id in result.new_incidents:
            incident = keeper.log.get(incident_id)
            if incident:
                ctx.note(f"{incident.level.label} at "
                         f"{incident.started_ms / 1000:.1f} s: {incident.reason}")

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ctx.progress(95, "writing the incident log")

    for name, ms in keeper.timings.items():
        record.add_stage(name, ms, calls=max(1, frames_read))

    summary = keeper.summary()
    incidents = keeper.log.to_list()
    record.results = incidents
    record.metrics.update(_metrics(keeper, summary, info, frames_read, elapsed_ms, sink))
    frame_gap_ms = 1000.0 * stride / info.fps if info.fps else 0.0
    analysed_ms = last_ms + frame_gap_ms
    whole = not info.duration_ms or analysed_ms >= info.duration_ms - 1.5 * frame_gap_ms
    record.metrics.update({
        "stride": stride,
        "frame_budget": budget,
        "analysed_seconds": round(min(analysed_ms, info.duration_ms or analysed_ms) / 1000, 2),
        "covered_whole_clip": whole,
        "coverage": coverage_note or (
            "the whole clip" if whole else
            f"the first {analysed_ms / 1000:.1f} s of {info.duration_ms / 1000:.1f} s "
            f"(the {budget}-frame budget)"),
        "person_size": summary["person_size"],
        "reference": summary["reference"],
        "incidents_rejoined": summary["incidents_rejoined"],
    })
    if not whole:
        record.warn(f"Analysed the first {analysed_ms / 1000:.1f} s of "
                    f"{info.duration_ms / 1000:.1f} s: this instance stops at {budget} frames. "
                    "Anything after that was not watched.")
    if summary["person_size"]["warning"]:
        record.warn(summary["person_size"]["warning"])
    for event in summary["reference"]["events"]:
        if event["event"] == "reference_replaced":
            record.warn(f"At {event['timestamp_ms'] / 1000:.1f} s: {event['message']}")
    record.params["live"] = live  # the live view replays this, all of it
    record.params["work_size"] = list(work_size)

    if summary["frames_usable"] == 0:
        record.refuse(
            "VIEW_UNUSABLE",
            "No frame in this clip was usable, so nothing was watched. "
            + "; ".join(sorted(summary["view_problems"])),
            view_problems=summary["view_problems"],
        )
    elif summary["usable_fraction"] < 0.5:
        record.warn(
            f"Only {summary['usable_fraction'] * 100:.0f}% of frames were usable. "
            f"Problems seen: {', '.join(sorted(summary['view_problems']))}."
        )
    if not keeper.guard.reference and config.guard_zone is None:
        record.warn("No guard region was set, so the fixed guard was not checked.")

    ctx.progress(100, "done")
    return record


def _ordinal(n: int) -> str:
    return {1: "", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def _metrics(keeper: Keepout, summary: dict[str, Any], info, frames: int,
             elapsed_ms: float, sink: EvidenceSink) -> dict[str, Any]:
    incidents = keeper.log.incidents
    first_alert = min(
        (i.started_ms for i in incidents if i.level >= Level.ALERT), default=None
    )
    watched_ms = (
        max((i.last_ms for i in incidents), default=0.0)
        if not info.duration_ms else info.duration_ms
    )
    hours = max(1e-9, watched_ms / 3_600_000.0)
    return {
        "frames_analysed": frames,
        "frames_usable": summary["frames_usable"],
        "frames_skipped_unusable": summary["frames_skipped_unusable"],
        "usable_fraction": summary["usable_fraction"],
        "clip_seconds": round(info.duration_ms / 1000.0, 2),
        "wall_seconds": round(elapsed_ms / 1000.0, 2),
        "ms_per_frame": round(elapsed_ms / max(1, frames), 2),
        "realtime_factor": round((info.duration_ms or 0.0) / max(1e-9, elapsed_ms), 2),
        "incidents_total": len(incidents),
        "incidents_open": summary["incidents"]["open"],
        "unacknowledged_critical": summary["incidents"]["unacknowledged_critical"],
        "highest_level": summary["incidents"]["highest_open_level"],
        "by_level": summary["incidents"]["by_peak_level"],
        "first_alert_ms": None if first_alert is None else round(first_alert, 1),
        "alerts_per_camera_hour": round(
            sum(1 for i in incidents if i.level >= Level.ALERT) / hours, 2
        ),
        "machine_running_at_end": summary["machine"]["running"],
        "guard_status": summary["guard"]["status"],
        "tracks_created": summary["tracks_created"],
        "view_problems": summary["view_problems"],
        "privacy": summary["privacy"],
        "evidence_storage": sink.to_dict(),
    }


# ---------------------------------------------------------------------------
# Extra routes
# ---------------------------------------------------------------------------


def _samples() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not SAMPLES_DIR.is_dir():
        return out
    for clip in sorted(SAMPLES_DIR.glob("*.mp4")):
        labels = clip.with_suffix(".json")
        meta: dict[str, Any] = {"name": clip.stem, "file": clip.name,
                                "bytes": clip.stat().st_size}
        if labels.is_file():
            try:
                truth = json.loads(labels.read_text())
            except json.JSONDecodeError:
                truth = {}
            meta["labelled"] = True
            meta["fps"] = truth.get("fps")
            meta["size"] = truth.get("size")
            meta["events"] = truth.get("events", [])[:24]
            meta["zones"] = truth.get("zones")
            meta["note"] = truth.get("note", "")
        else:
            meta["labelled"] = False
        readme = SAMPLES_DIR / f"{clip.stem}.txt"
        if readme.is_file():
            meta["description"] = readme.read_text().strip()
        out.append(meta)
    return out


def install_routes(app: FastAPI) -> None:
    @app.get("/api/keepout")
    async def keepout_info() -> dict[str, Any]:
        return {
            "version": __version__,
            "instance": os.environ.get("KEEPOUT_INSTANCE_LABEL", "local workstation"),
            "quote": HSE_QUOTE,
            "quote_source": HSE_SOURCE,
            "privacy": PRIVACY_STATEMENT,
            "levels": [
                {"slug": lvl.slug, "rank": int(lvl), "label": lvl.label,
                 "latching": lvl is Level.CRITICAL}
                for lvl in Level
            ],
            "demo_ready": (DEMO_DIR / "index.json").is_file()
            or (DEMO_DIR / "run.json").is_file(),
            "baked_clips": sorted(
                p.parent.name for p in DEMO_DIR.glob("*/run.json")
            ),
            "paths": describe(),
            "evidence_storage": EvidenceSink.from_env().to_dict(),
        }

    @app.get("/api/samples")
    async def samples() -> dict[str, Any]:
        return {"samples": _samples(), "dir": str(SAMPLES_DIR)}

    @app.get("/api/samples/{name}/clip")
    async def sample_clip(name: str) -> FileResponse:
        path = SAMPLES_DIR / f"{Path(name).name}.mp4"
        if not path.is_file():
            raise ServiceError("NOT_FOUND", f"no sample clip {name}")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/samples/{name}/frame")
    async def sample_frame(name: str, at_ms: float = 0.0) -> JSONResponse:
        """A reference frame as a data URI, so the zone editor can draw on it."""
        path = SAMPLES_DIR / f"{Path(name).name}.mp4"
        if not path.is_file():
            raise ServiceError("NOT_FOUND", f"no sample clip {name}")
        frames = list(iter_video(path, start_ms=at_ms, max_frames=1, max_side=960))
        if not frames:
            raise ServiceError("BAD_REQUEST", "could not decode a frame from that clip")
        image = frames[0].image
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise ServiceError("INTERNAL", "could not encode the reference frame")
        import base64

        return JSONResponse({
            "name": name,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "timestamp_ms": frames[0].timestamp_ms,
            "image": "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode(),
            "suggested": {k: v.to_dict() for k, v in
                          default_zones((image.shape[1], image.shape[0])).items()},
        })

    @app.post("/api/zones/propose")
    async def propose(request: Request) -> dict[str, Any]:
        """Propose a danger zone from where a sample clip actually moves."""
        body = await request.json()
        name = Path(str(body.get("sample", ""))).name
        path = SAMPLES_DIR / f"{name}.mp4"
        if not path.is_file():
            raise ServiceError("NOT_FOUND", f"no sample clip {name}")
        frames = [f.image for f in iter_video(path, stride=int(body.get("stride", 4)),
                                              max_frames=int(body.get("frames", 40)),
                                              max_side=960)]
        proposal = propose_zone_from_motion(
            frames, name="proposed danger zone",
            dilate_px=int(body.get("standoff_px", 45)),
        )
        return {
            "proposal": proposal.to_dict(),
            "accepted": proposal.zone is not None,
            "advice": (
                "This is a proposal from where the scene moved, not an answer. "
                "Check it covers everywhere a person could reach the machine, then save it."
            ),
        }

    @app.get("/api/demo")
    async def demo(clip: str | None = None) -> dict[str, Any]:
        """A pre-computed run, so a cold start shows a real alert immediately.

        Every bundled clip is baked at image build time. The hosted instance runs
        YOLOX-tiny at roughly 390 ms per frame, so analysing a twenty-four second
        clip live takes over a minute; that is fine as proof and useless as a
        first impression. The UI offers the live re-run beside the baked result.
        """
        name = Path(clip).name if clip else None
        if name:
            path = DEMO_DIR / name / "run.json"
        else:
            index = DEMO_DIR / "index.json"
            default = json.loads(index.read_text())["default"] if index.is_file() else None
            path = (DEMO_DIR / default / "run.json") if default else (DEMO_DIR / "run.json")
        if not path.is_file():
            raise ServiceError(
                "NOT_FOUND",
                f"no baked run for {name or 'the default clip'} in this image; "
                "run `keepout demo --bake` at build time",
            )
        payload = json.loads(path.read_text())
        payload["served_at"] = time.time()
        payload["baked"] = True
        return payload

    @app.get("/api/demo/index")
    async def demo_index() -> dict[str, Any]:
        path = DEMO_DIR / "index.json"
        if not path.is_file():
            return {"default": None, "clips": []}
        return json.loads(path.read_text())

    @app.get("/api/demo/{clip}/evidence/{name}")
    async def demo_clip_evidence(clip: str, name: str) -> FileResponse:
        path = DEMO_DIR / Path(clip).name / "evidence" / Path(name).name
        if not path.is_file():
            raise ServiceError("NOT_FOUND", f"no demo evidence {name} for {clip}")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/demo/evidence/{name}")
    async def demo_evidence(name: str) -> FileResponse:
        """Kept for the single-clip layout an earlier image used."""
        safe = Path(name).name
        for candidate in (DEMO_DIR / "evidence" / safe, *DEMO_DIR.glob(f"*/evidence/{safe}")):
            if candidate.is_file():
                return FileResponse(candidate, media_type="image/jpeg")
        raise ServiceError("NOT_FOUND", f"no demo evidence {name}")

    @app.post("/api/jobs/{job_id}/incidents/{incident_id}/acknowledge")
    async def acknowledge(request: Request, job_id: str, incident_id: str) -> dict[str, Any]:
        """A human clears a latched incident. The only state a person may change."""
        store = request.app.state.store
        job = store.get(job_id)
        if not job.result:
            raise ServiceError("BAD_REQUEST", "that job has not finished yet")
        body = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        who = str(body.get("by", "operator"))[:64]
        for incident in job.result.get("results", []):
            if incident["incident_id"] == incident_id:
                if incident.get("acknowledged_ms") is not None:
                    return {"ok": False, "reason": "already acknowledged",
                            "incident": incident}
                incident["acknowledged_ms"] = time.time() * 1000.0
                incident["acknowledged_by"] = who
                incident["open"] = False
                incident.setdefault("history", []).append(
                    {"ts_ms": incident["acknowledged_ms"], "event": "acknowledged", "by": who}
                )
                metrics = job.result.setdefault("metrics", {})
                metrics["unacknowledged_critical"] = sum(
                    1 for i in job.result.get("results", [])
                    if i.get("latched") and i.get("acknowledged_ms") is None
                )
                log.info("incident acknowledged",
                         extra={"job_id": job_id, "incident_id": incident_id, "by": who})
                return {"ok": True, "incident": incident}
        raise ServiceError("NOT_FOUND", f"no incident {incident_id} in job {job_id}")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

PARAMS_SCHEMA: list[dict[str, Any]] = [
    {"name": "tiled_detection", "label": "Tiled detection for small people", "type": "boolean",
     "default": False,
     "help": "Full frame plus four tiles: finds people down to about 8% of frame height "
             "instead of 15%, at about five times the time per frame."},
    {"name": "detection_score", "label": "Detection threshold", "type": "number",
     "default": 0.35, "min": 0.05, "max": 0.9, "step": 0.05,
     "help": "Lower finds more people and more furniture."},
    {"name": "min_dwell_ms", "label": "Ignore entries shorter than (ms)", "type": "number",
     "default": 0, "min": 0, "max": 5000, "step": 100},
    {"name": "stride", "label": "Analyse every Nth frame", "type": "number",
     "default": 1, "min": 1, "max": 5, "step": 1,
     "help": "Raise this to process a long clip faster, at the cost of time to alert."},
]


def build_app() -> FastAPI:
    product = ProductInfo(
        slug="keepout",
        title="Keepout",
        tagline="A fixed camera over machinery, watched by something that never blinks.",
        description=(
            "Keepout learns a danger zone, notices a person entering it while the machine "
            "is running, checks that the fixed guard is still in place, and escalates a "
            "person who is down and motionless until a human acknowledges it. "
            "OpenCV 5, YOLOX-tiny through cv2.dnn, no face recognition."
        ),
        accent=ACCENT,
        version=__version__,
        repo_url="https://github.com/rogerobeng77-oss/keepout",
    )
    config = ServiceConfig(
        product=product,
        models={"detector": {"name": "yolox-tiny", "licence": "Apache-2.0",
                             "runtime": "cv2.dnn (ONNX)", "classes": "COCO person only"}},
        static_dir=STATIC_DIR if STATIC_DIR.is_dir() else None,
        params_schema=PARAMS_SCHEMA,
        max_concurrent_jobs=int(os.environ.get("KEEPOUT_MAX_CONCURRENT", "2")),
    )
    # Fails here, at startup, if KEEPOUT_REQUIRE_FACE_DETECTOR is set and the YuNet
    # weights are missing. The container sets it, so a broken image never serves.
    privacy = FaceBlurrer(PrivacyConfig()).describe()
    app = create_app(config, analyze)
    install_routes(app)
    log.info("keepout ready", extra={"samples": len(_samples()),
                                     "demo_baked": (DEMO_DIR / "run.json").is_file(),
                                     "privacy": privacy})
    return app


app = build_app()
