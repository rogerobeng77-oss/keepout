"""Fixtures for the Keepout tests.

Almost every test here builds its own input, so the expected answer is known by
construction rather than by eyeballing a real clip. The few tests that need the
real detector or real video decoding are marked `slow` and skipped when the
sample assets or the YOLOX weights are not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from keepout import ContactPoint, Zone

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = PRODUCT_ROOT / "data" / "samples"


@pytest.fixture
def frame_size() -> tuple[int, int]:
    return (640, 360)


@pytest.fixture
def danger_zone(frame_size) -> Zone:
    """A rectangle covering the right half of the lower frame."""
    w, h = frame_size
    return Zone(
        name="danger",
        points=((w * 0.5, h * 0.4), (w * 0.95, h * 0.4),
                (w * 0.95, h * 0.95), (w * 0.5, h * 0.95)),
        reference_size=(w, h),
        contact=ContactPoint.FEET,
    )


@pytest.fixture
def textured_frame(frame_size) -> np.ndarray:
    """A static scene with enough structure for ECC, Canny and the view checks."""
    return make_scene(frame_size)


def make_scene(size: tuple[int, int], seed: int = 3) -> np.ndarray:
    """A deterministic, detailed grey scene. Detail matters: a flat frame reads
    as a blocked lens, which would make half the view tests pass for the wrong
    reason."""
    w, h = size
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 190, (h // 8, w // 8), dtype=np.uint8)
    import cv2

    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(0, w, 40):
        cv2.line(img, (i, 0), (i, h), (30, 30, 30), 2)
    for j in range(0, h, 40):
        cv2.line(img, (0, j), (w, j), (220, 220, 220), 1)
    return img


def person_box(cx: float, feet_y: float, height: float = 120.0, aspect: float = 0.38
               ) -> tuple[float, float, float, float]:
    """A plausible upright person box with its feet at (cx, feet_y)."""
    width = height * aspect
    return (cx - width / 2.0, feet_y - height, cx + width / 2.0, feet_y)


def lying_box(cx: float, cy: float, length: float = 120.0, thickness: float = 46.0
              ) -> tuple[float, float, float, float]:
    """A person-on-the-floor box: wider than it is tall."""
    return (cx - length / 2.0, cy - thickness / 2.0, cx + length / 2.0, cy + thickness / 2.0)


def sample_clip(name: str) -> Path:
    path = SAMPLES / name
    if not path.is_file():
        pytest.skip(f"sample clip {name} not built; run `python -m keepout.cli build-samples`")
    return path
