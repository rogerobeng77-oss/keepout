"""Run Keepout over a real clip with a named config, streaming frames, and write a
compact result: frames usable, view events, incidents by kind/level with the
subject's box at the start, and the evidence frames (as the service would store them).

    python measure.py <run-name> <out-root>

Configs are the zones used on the first real-footage run, in source pixels, built through
`keepout.service.config_from_params` exactly as the live service does. Unlike the
earlier run_local.py this does NOT call set_reference on the first frame: it lets the
pipeline decide (the old pipeline has no such logic, so for "before" runs pass
--legacy-reference to reproduce what the service and the CLI did).
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
from visioncore import iter_video, video_info

from keepout.pipeline import Keepout
from keepout.service import config_from_params

HERE = Path(__file__).resolve().parent
PRODUCT = HERE.parents[1]
# The clips are not in this repository (identifiable faces, CC BY-SA); point this at
# a directory holding clips/ and source/ as listed in README.md next to this file.
REAL = Path(os.environ.get("KEEPOUT_REAL_FOOTAGE", HERE))

RUNS = {
    "courtyard": (REAL / "clips/courtyard-vtest-40s.mp4", "cli-courtyard"),
    "m4-default": (REAL / "clips/malta-mdina-shot4-74.35s-104.6s.mp4", {}),
    "m4-A": (REAL / "clips/malta-mdina-shot4-74.35s-104.6s.mp4", {
        "danger_zone": [[1000, 300], [1920, 240], [1920, 640], [1440, 640], [1040, 480]],
        "machine_zone": [[1100, 0], [1700, 0], [1700, 330], [1100, 330]]}),
    "m4-B": (REAL / "clips/malta-mdina-shot4-74.35s-104.6s.mp4", {
        "danger_zone": [[720, 380], [1100, 380], [1180, 1000], [740, 1000]],
        "machine_zone": [[0, 0], [300, 0], [300, 800], [0, 800]],
        "assume_machine_running": True}),
    "prefab1-default": (REAL / "clips/prefab-house-shot1-0.6s-47.5s.mp4", {}),
    "prefab1-crane": (REAL / "clips/prefab-house-shot1-0.6s-47.5s.mp4", {
        "danger_zone": [[150, 320], [600, 320], [610, 450], [140, 450]],
        "machine_zone": [[140, 0], [640, 0], [640, 215], [140, 215]]}),
    "amz2-default": (REAL / "source/Amazon_warehouse_BHX4_loading_docks_2.webm", {}),
    "amz2-tiled": (REAL / "source/Amazon_warehouse_BHX4_loading_docks_2.webm",
                   {"tiled_detection": True}),
    "prefab-full": (REAL / "source/Prefabricated_house_construction.ogv", {}),
}


def build(run: str):
    clip, params = RUNS[run]
    info = video_info(clip)
    max_side = 960
    scale = min(1.0, max_side / float(max(info.width, info.height)))
    size = (round(info.width * scale), round(info.height * scale))
    if params == "cli-courtyard":
        from keepout.cli import _config_for
        config = _config_for(PRODUCT / "data/samples/courtyard.mp4", size)
    else:
        params = dict(params)
        for k in ("danger_zone", "machine_zone", "guard_zone"):
            if isinstance(params.get(k), list):
                params[k] = [[x * scale, y * scale] for x, y in params[k]]
        config = config_from_params(params, size)
    return clip, info, config, max_side


def main() -> None:
    run, out_root = sys.argv[1], Path(sys.argv[2])
    legacy = "--legacy-reference" in sys.argv
    clip, _info, config, max_side = build(run)
    out = out_root / run
    out.mkdir(parents=True, exist_ok=True)
    reports = []

    def on_evidence(name, image, report):
        cv2.imwrite(str(out / f"{name}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        reports.append({"name": name, **{k: v for k, v in report.items() if k != "note"}})
        return name

    keeper = Keepout(config, on_evidence=on_evidence)
    start_boxes = {}
    view_events = []
    n = usable = 0
    t0 = time.perf_counter()
    for fr in iter_video(clip, max_frames=100_000, max_side=max_side):
        if legacy and keeper.reference_frame is None:
            keeper.set_reference(fr.image, fr.timestamp_ms)
        r = keeper.process_frame(fr.image, fr.timestamp_ms, fr.index)
        n += 1
        usable += bool(r.view.usable)
        for ev in getattr(r, "view_events", []) or []:
            view_events.append(ev)
        by_track = {t["track_id"]: t["bbox"] for t in r.tracks}
        for iid in r.new_incidents:
            inc = keeper.log.get(iid)
            if inc and iid not in start_boxes:
                start_boxes[iid] = by_track.get(inc.track_id)
    wall = time.perf_counter() - t0
    incidents = keeper.log.to_list()
    for inc in incidents:
        inc["start_box"] = start_boxes.get(inc["incident_id"])
    by_kind = {}
    for inc in incidents:
        key = f'{inc["kind"]}:{inc["peak_level"]}'
        by_kind[key] = by_kind.get(key, 0) + 1
    summary = keeper.summary()
    result = {
        "run": run, "clip": clip.name, "legacy_reference": legacy,
        "frames": n, "frames_usable": usable, "wall_s": round(wall, 1),
        "view_problems": summary["view_problems"], "view_events": view_events,
        "tracks_created": summary["tracks_created"],
        "incidents_by_kind": by_kind,
        "alerts": sum(1 for i in incidents if i["peak_level"] == "alert"),
        "criticals": sum(1 for i in incidents if i["peak_level"] == "critical"),
        "privacy": summary["privacy"], "evidence_reports": reports,
        "person_size": summary.get("person_size"),
        "reference": summary.get("reference"),
        "incidents_rejoined": summary.get("incidents_rejoined"),
        "ms_per_frame": round(wall * 1000 / max(1, n), 1),
        "incidents": incidents,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    hidden = ("incidents", "evidence_reports", "view_events", "reference")
    print(json.dumps({k: v for k, v in result.items() if k not in hidden}, default=str))


if __name__ == "__main__":
    main()
