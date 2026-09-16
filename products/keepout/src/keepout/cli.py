"""`keepout` command line: build the sample clips, bake the demo, run an analysis.

    keepout build-samples          # stage the labelled clips into data/samples
    keepout analyse CLIP           # run the pipeline and print the incident log
    keepout demo --bake            # pre-compute the demo the container serves
    keepout evaluate               # measure against the labelled set
    keepout version                # what is actually installed
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .paths import demo_dir, eval_dir, samples_dir, source_dir

PRODUCT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = samples_dir()
DEMO_DIR = demo_dir()
SOURCE_DIR = source_dir()

VTEST_URL = "https://raw.githubusercontent.com/opencv/opencv/5.x/samples/data/vtest.avi"
DEMO_CLIP = "cell-alert"


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def _fetch_source(verbose: bool = True) -> Path:
    """Download OpenCV's own pedestrian sample. Apache-2.0, same repo as the library."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    target = SOURCE_DIR / "vtest.avi"
    if target.is_file() and target.stat().st_size > 1_000_000:
        return target
    import urllib.request

    if verbose:
        print(f"fetching {VTEST_URL}")
    with urllib.request.urlopen(VTEST_URL, timeout=180) as response:
        target.write_bytes(response.read())
    return target


def _person_detector(score: float = 0.45):
    from visioncore import YoloxDetector

    detector = YoloxDetector(score_threshold=score)

    def detect(frame: np.ndarray):
        return [(d.bbox, d.score) for d in detector.detect(frame) if d.class_id == 0]

    return detect


def _scripts():
    """The staged clips, each one exercising a different part of the ladder."""
    from .synth import CELL_SIZE, Degradation, StageScript, Walk

    w, h = CELL_SIZE
    floor_y = h * 0.86
    near_y = h * 0.74

    return {
        # 1. The headline: someone walks into a running machine's zone.
        "cell-alert": StageScript(
            name="cell-alert",
            duration_s=24.0,
            running_spans=((0.0, 1e9),),
            walks=[
                Walk(enter_s=2.0, exit_s=23.0, height_px=190, sprite_index=0,
                     path=((2.0, -w * 0.06, floor_y), (9.0, w * 0.56, near_y),
                           (15.0, w * 0.56, near_y), (23.0, w * 1.12, floor_y))),
            ],
            note=("A running conveyor and one person who walks into the danger zone, "
                  "stands in it, and walks back out through the frame edge."),
        ),
        # 2. The same walk with the machine stopped: a note, not an alert.
        "cell-stopped": StageScript(
            name="cell-stopped",
            duration_s=20.0,
            running_spans=(),
            walks=[
                Walk(enter_s=2.0, exit_s=19.0, height_px=190, sprite_index=0,
                     path=((2.0, -w * 0.06, floor_y), (8.0, w * 0.56, near_y),
                           (12.0, w * 0.56, near_y), (19.0, w * 1.12, floor_y))),
            ],
            note="The identical approach with the conveyor stopped. Expect a note.",
        ),
        # 3. The guard comes off while the machine runs.
        "cell-guard": StageScript(
            name="cell-guard",
            duration_s=24.0,
            running_spans=((0.0, 1e9),),
            guard_removed_from_s=9.0,
            walks=[
                # Crosses the frame and leaves through the right edge. A walker who
                # stopped mid-floor would manufacture a person-unaccounted event and
                # the clip would be testing its own staging rather than the product.
                Walk(enter_s=1.0, exit_s=8.0, start=(-w * 0.08, floor_y),
                     end=(w * 1.10, floor_y), height_px=185, sprite_index=1),
            ],
            note="The fixed guard is removed nine seconds in, while the belt is running.",
        ),
        # 4. Somebody goes down inside the zone and stops moving.
        "cell-down": StageScript(
            name="cell-down",
            duration_s=30.0,
            running_spans=((0.0, 1e9),),
            walks=[
                Walk(enter_s=1.5, exit_s=29.0, start=(w * 0.10, floor_y),
                     end=(w * 0.52, near_y), height_px=190, sprite_index=0,
                     collapse_at_s=9.0, collapse_angle=80.0),
            ],
            note=("One person who collapses inside the zone at nine seconds and does not "
                  "move again. This is the clip the HSE conveyor case describes."),
        ),
        # 5. Four ways of being blind, one after another.
        "cell-unusable": StageScript(
            name="cell-unusable",
            duration_s=32.0,
            running_spans=((0.0, 1e9),),
            walks=[
                Walk(enter_s=1.0, exit_s=31.0, start=(-w * 0.06, floor_y),
                     end=(w * 1.10, near_y), height_px=190, sprite_index=0),
            ],
            degradations=[
                Degradation("camera_moved", 6.0, 12.0, magnitude=34.0),
                Degradation("lens_blocked", 14.0, 19.0, magnitude=41.0),
                Degradation("too_dark", 21.0, 25.0, magnitude=0.045),
                Degradation("frozen_feed", 27.0, 32.0),
            ],
            note=("The camera is nudged, then the lens is covered, then the lights go out, "
                  "then the feed freezes. Nothing here should produce a confident answer."),
        ),
        # 6. Nobody at all: the false-alert measurement.
        "cell-quiet": StageScript(
            name="cell-quiet",
            duration_s=40.0,
            running_spans=((0.0, 20.0), (28.0, 1e9)),
            walks=[],
            note=("A running machine with nobody in the room, stopping and starting. "
                  "Every incident from this clip is a false alert."),
        ),
    }


def build_samples(args: argparse.Namespace) -> int:
    from .synth import harvest_sprites, render_script

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    source = _fetch_source()
    print(f"source clip: {source} ({source.stat().st_size / 1e6:.1f} MB)")

    print("matting person sprites out of the source clip ...")
    sprites = harvest_sprites(source, _person_detector(), wanted=4)
    if not sprites:
        print("could not matte any people out of the source clip", file=sys.stderr)
        return 1
    print(f"  {len(sprites)} sprites, heights "
          f"{[s.bgr.shape[0] for s in sprites]}")

    # The courtyard clip is the source, unmodified: real footage, real people,
    # nothing composited. It is what the false-alert rate is measured on.
    courtyard = SAMPLES_DIR / "courtyard.mp4"
    if not courtyard.is_file() or args.force:
        _transcode(source, courtyard, seconds=40.0)
        _to_browser_codec(courtyard)
        (SAMPLES_DIR / "courtyard.txt").write_text(
            "Unmodified pedestrian footage: samples/data/vtest.avi from the OpenCV "
            "repository, Apache-2.0, transcoded to mp4 and trimmed to 40 seconds. "
            "Real people, real camera, nothing composited. There is no machine in it, "
            "so it is used to measure detection and tracking on real video and to count "
            "false alerts against a zone drawn on the walkway.\n"
        )
        print(f"  wrote {courtyard.name}")

    for name, script in _scripts().items():
        path = SAMPLES_DIR / f"{name}.mp4"
        if path.is_file() and not args.force:
            print(f"  {name}.mp4 exists, skipping")
            continue
        truth = render_script(script, sprites, path)
        _to_browser_codec(path)
        truth.save(path.with_suffix(".json"))
        (SAMPLES_DIR / f"{name}.txt").write_text(script.note + "\n")
        print(f"  wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB), "
              f"{len(truth.events)} labelled events")

    print(f"\n{len(list(SAMPLES_DIR.glob('*.mp4')))} clips in {SAMPLES_DIR}")
    return 0


def _to_browser_codec(path: Path) -> bool:
    """Re-encode to H.264 in place, so a browser can actually play the clip.

    `cv2.VideoWriter` writes MPEG-4 Part 2 (`mp4v`) here, which Chrome and Safari
    will not decode: the video element loads, reports a duration, and shows black.
    OpenCV's own build has no H.264 encoder (the wheel ships no x264), so the last
    step of building a sample is an ffmpeg re-encode. If ffmpeg is missing we say
    so and leave the clip alone rather than shipping a silently black demo.
    """
    import shutil as _shutil
    import subprocess

    if _shutil.which("ffmpeg") is None:
        print(f"  ffmpeg not found; {path.name} stays mp4v and will not play in a browser",
              file=sys.stderr)
        return False
    tmp = path.with_suffix(".h264.mp4")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not tmp.is_file():
        print(f"  ffmpeg failed on {path.name}: {result.stderr.strip()[:200]}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    return True


def _transcode(source: Path, target: Path, seconds: float = 40.0) -> None:
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for _ in range(int(seconds * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _config_for(clip: Path, size: tuple[int, int]):
    """Use a clip's own labelled zones when it has them, else the defaults."""
    from .pipeline import KeepoutConfig, default_zones
    from .zones import ContactPoint, Zone

    labels = clip.with_suffix(".json")
    if labels.is_file():
        truth = json.loads(labels.read_text())
        zones = truth.get("zones") or {}
        ref = tuple(truth.get("size", size))
        sx, sy = size[0] / ref[0], size[1] / ref[1]

        def make(key: str, name: str, kind: str, contact: ContactPoint) -> Zone | None:
            pts = zones.get(key)
            if not pts:
                return None
            return Zone(name=name, kind=kind, contact=contact, reference_size=size,
                        points=tuple((p[0] * sx, p[1] * sy) for p in pts))

        danger = make("danger", "conveyor danger zone", "danger", ContactPoint.FEET)
        if danger is not None:
            return KeepoutConfig(
                danger_zone=danger,
                machine_zone=make("machine", "conveyor", "machine", ContactPoint.CENTROID),
                guard_zone=make("guard", "fixed guard", "guard", ContactPoint.CENTROID),
            )
    zones = default_zones(size)
    if clip.stem == "courtyard":
        # Real pedestrian footage with no machine and no guard anywhere in frame.
        # Setting either would be measuring something that is not there, so the
        # clip carries a danger zone only and the machine is declared running.
        return KeepoutConfig(danger_zone=zones["danger"], machine_zone=None,
                             guard_zone=None, assume_machine_running=True)
    return KeepoutConfig(danger_zone=zones["danger"], machine_zone=zones["machine"],
                         guard_zone=zones["guard"])


def run_clip(clip: Path, *, evidence_dir: Path | None = None, max_frames: int = 900,
             max_side: int | None = 960, quiet: bool = False):
    """Run the pipeline over a clip. Returns (keeper, results, timing dict)."""
    from visioncore import iter_video, video_info

    from .pipeline import Keepout

    info = video_info(clip)
    scale = min(1.0, (max_side or info.width) / float(info.width))
    size = (round(info.width * scale), round(info.height * scale))
    config = _config_for(clip, size)

    on_evidence = None
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)

        def on_evidence(name: str, image: np.ndarray, report: dict[str, Any]) -> str:
            path = evidence_dir / f"{name}.jpg"
            cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            return f"/api/demo/evidence/{path.name}"

    keeper = Keepout(config, on_evidence=on_evidence)
    started = time.perf_counter()
    results = []
    for frame in iter_video(clip, max_frames=max_frames, max_side=max_side):
        results.append(keeper.process_frame(frame.image, frame.timestamp_ms, frame.index))
    elapsed = time.perf_counter() - started

    timing = {
        "clip": clip.name,
        "frames": len(results),
        "clip_seconds": round(info.duration_ms / 1000.0, 2),
        "wall_seconds": round(elapsed, 2),
        "ms_per_frame": round(elapsed * 1000.0 / max(1, len(results)), 2),
        "realtime_factor": round((info.duration_ms / 1000.0) / max(1e-9, elapsed), 2),
        "size": list(size),
    }
    if not quiet:
        print(json.dumps(timing, indent=2))
    return keeper, results, timing


def analyse(args: argparse.Namespace) -> int:
    clip = Path(args.clip)
    if not clip.is_file():
        print(f"no such clip: {clip}", file=sys.stderr)
        return 1
    evidence = Path(args.evidence) if args.evidence else None
    keeper, _results, timing = run_clip(clip, evidence_dir=evidence,
                                        max_frames=args.max_frames)
    summary = keeper.summary()
    print(json.dumps({"timing": timing, "summary": summary,
                      "incidents": keeper.log.to_list()}, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# Demo baking
# ---------------------------------------------------------------------------


def _bake_one(clip: Path, max_frames: int) -> dict[str, Any]:
    """Analyse one clip and write its result and evidence under data/demo/<stem>/."""
    target = DEMO_DIR / clip.stem
    evidence_dir = target / "evidence"
    keeper, results, timing = run_clip(clip, evidence_dir=evidence_dir,
                                       max_frames=max_frames, quiet=True)

    truth_path = clip.with_suffix(".json")
    truth = json.loads(truth_path.read_text()) if truth_path.is_file() else {}
    note = SAMPLES_DIR / f"{clip.stem}.txt"
    payload = {
        "clip": clip.name,
        "name": clip.stem,
        "description": note.read_text().strip() if note.is_file() else "",
        "timing": timing,
        "summary": keeper.summary(),
        "incidents": keeper.log.to_list(),
        "config": keeper.config.to_dict(),
        "live": [r.to_dict() for r in results],
        "ground_truth_events": truth.get("events", []),
        "baked_at": time.time(),
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "run.json").write_text(json.dumps(payload, default=str))
    return payload


def demo(args: argparse.Namespace) -> int:
    """Pre-compute the runs the container serves at /api/demo.

    A judge opening a cold container should see a real alert with its evidence
    frame immediately, not a spinner. The analysis still happens; it happens at
    image build time, on the same code, and the UI can re-run any clip live to
    prove the baked answer is real.

    Every bundled clip is baked, not just one, because App Runner's vCPUs run
    YOLOX-tiny at about 390 ms per frame. A live run of a twenty-four second clip
    therefore takes over a minute on the hosted instance, which is fine as proof
    and useless as a first impression.
    """
    if args.bake and DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    if args.clip == "all":
        clips = sorted(SAMPLES_DIR.glob("*.mp4"))
    else:
        one = SAMPLES_DIR / f"{args.clip}.mp4"
        if not one.is_file():
            print(f"no sample {args.clip} in {SAMPLES_DIR}; "
                  "run `keepout build-samples` first", file=sys.stderr)
            return 1
        clips = [one]
    if not clips:
        print(f"no sample clips in {SAMPLES_DIR}", file=sys.stderr)
        return 1

    index = []
    for clip in clips:
        payload = _bake_one(clip, args.max_frames)
        timing = payload["timing"]
        incidents = payload["incidents"]
        evidence = len(list((DEMO_DIR / clip.stem / "evidence").glob("*.jpg")))
        index.append({"name": clip.stem, "clip": clip.name,
                      "incidents": len(incidents), "frames": timing["frames"],
                      "ms_per_frame": timing["ms_per_frame"]})
        print(f"baked {clip.stem:<14} {timing['frames']:>4} frames  "
              f"{timing['ms_per_frame']:>6.1f} ms/frame  "
              f"{len(incidents)} incidents  {evidence} evidence frames")
        for incident in incidents:
            print(f"    {incident['level']:>8}  t={incident['started_ms'] / 1000:6.1f}s  "
                  f"{incident['reason'][:66]}")

    default = "cell-alert" if any(c["name"] == "cell-alert" for c in index) else index[0]["name"]
    (DEMO_DIR / "index.json").write_text(json.dumps(
        {"default": default, "clips": index, "baked_at": time.time()}, indent=2))
    # Kept so an older deployment that asks for /api/demo without a clip name
    # still gets something rather than a 404.
    shutil.copyfile(DEMO_DIR / default / "run.json", DEMO_DIR / "run.json")
    print(f"\n{len(index)} runs baked into {DEMO_DIR}, default {default}")
    return 0


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def version(_args: argparse.Namespace) -> int:
    import visioncore

    from . import __version__

    print(json.dumps({
        "keepout": __version__,
        "visioncore": visioncore.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "environment": visioncore.environment().to_dict(),
    }, indent=2, default=str))
    return 0


def evaluate_cmd(args: argparse.Namespace) -> int:
    from .evaluate import run_evaluation

    report = run_evaluation(SAMPLES_DIR, out_dir=Path(args.out) if args.out else eval_dir(),
                            max_frames=args.max_frames)
    print(json.dumps(report, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="keepout", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-samples", help="stage the labelled sample clips")
    p.add_argument("--force", action="store_true", help="rebuild clips that already exist")
    p.set_defaults(func=build_samples)

    p = sub.add_parser("analyse", help="run the pipeline over one clip")
    p.add_argument("clip")
    p.add_argument("--evidence", help="directory to write evidence frames into")
    p.add_argument("--max-frames", type=int, default=900)
    p.set_defaults(func=analyse)

    p = sub.add_parser("demo", help="pre-compute the demo the service serves")
    p.add_argument("--clip", default="all",
                   help="a sample name, or 'all' to bake every bundled clip")
    p.add_argument("--bake", action="store_true", help="clear and rewrite data/demo")
    p.add_argument("--max-frames", type=int, default=600)
    p.set_defaults(func=demo)

    p = sub.add_parser("evaluate", help="measure against the labelled set")
    p.add_argument("--out", help="directory for the evaluation artefacts")
    p.add_argument("--max-frames", type=int, default=900)
    p.set_defaults(func=evaluate_cmd)

    p = sub.add_parser("version", help="what is installed")
    p.set_defaults(func=version)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
