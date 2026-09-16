"""The live service's frame budget: cover the whole clip, or say plainly that it did not.

Found on real footage: a 47 s clip on the live service stopped at frame 900, 36 s
in, and its one real alert at 44.8 s never happened. Nothing in the result said so.
These tests run the real analyzer (so they need the YOLOX weights) on a small
generated clip with the budget turned down.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from .conftest import make_scene


def _weights_present() -> bool:
    try:
        from visioncore import YOLOX_TINY, fetch_model

        fetch_model(YOLOX_TINY)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.slow


def _clip(path: Path, frames: int = 60, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    base = make_scene((320, 180))
    for i in range(frames):
        frame = base.copy()
        cv2.circle(frame, (20 + i * 4, 90), 6, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def _run(tmp_path: Path, params: dict, monkeypatch) -> object:
    if not _weights_present():
        pytest.skip("YOLOX-tiny weights not available")
    from servicekit import JobContext
    from visioncore import RunRecord

    from keepout import service

    monkeypatch.setattr(service, "MAX_ANALYSIS_FRAMES", 20)
    clip = _clip(tmp_path / "clip.mp4")
    ctx = JobContext(job_id="t", input_path=clip, filename="clip.mp4", params=params,
                     record=RunRecord(product="keepout"), _emit=lambda _e: None,
                     evidence_dir=tmp_path / "evidence")
    return service.analyze(ctx)


def test_a_clip_longer_than_the_budget_is_covered_by_raising_the_stride(tmp_path, monkeypatch):
    record = _run(tmp_path, {}, monkeypatch)
    metrics = record.metrics
    assert metrics["stride"] == 3
    assert metrics["covered_whole_clip"] is True
    assert metrics["frames_analysed"] <= 20
    assert "every 3rd frame" in metrics["coverage"]
    assert not any("first" in w for w in record.warnings)


def test_a_run_cut_short_by_the_budget_says_so(tmp_path, monkeypatch):
    record = _run(tmp_path, {"cover_whole_clip": False}, monkeypatch)
    metrics = record.metrics
    assert metrics["stride"] == 1
    assert metrics["covered_whole_clip"] is False
    assert metrics["analysed_seconds"] == pytest.approx(2.0, abs=0.15)
    assert any("Analysed the first 2.0 s of 6.0 s" in w for w in record.warnings), record.warnings
