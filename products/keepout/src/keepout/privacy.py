"""Privacy: blur faces in stored evidence, keep no identity, and prove it in the record.

The design constraint is stated once and enforced everywhere: **Keepout does not
know who anybody is.** Track ids are per-run integers that reset when the process
restarts and are never joined to a roster, a badge or a face. There is no face
recognition anywhere in this codebase, and there is no code path that could add
one without a new dependency.

What is here is the opposite: a face *detector* used only to destroy information.
OpenCV 5 ships `cv2.FaceDetectorYN` in the main wheel (research/FINDINGS.md §1.3),
which is a detector, not a recogniser: it returns boxes, and the model that would
turn a face into an embedding (`FaceRecognizerSF`) is never loaded.

The YuNet ONNX weights are an optional extra. When they are not present we still
blur, using a **fallback head region** derived from the person box: the top 28% of
the box, centred, widened slightly. That is deliberately generous, because the
failure mode of a face blur must be "blurred too much", never "missed a face".
The record says which method was used, and the UI shows it, so nobody assumes a
detector ran when it did not.

Blur, not a black box: a redacted evidence frame still has to let a supervisor
see what the person was doing, which is the whole point of the evidence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

Box = tuple[float, float, float, float]

YUNET_ENV = "KEEPOUT_YUNET_PATH"
YUNET_FILENAMES = ("face_detection_yunet_2023mar.onnx", "yunet.onnx")


@dataclass
class PrivacyConfig:
    enabled: bool = True
    head_fraction: float = 0.28  # top slice of a person box treated as the head
    head_widen: float = 1.12
    blur_strength: float = 0.11  # kernel as a fraction of the region's larger side
    min_kernel: int = 9
    detector_score: float = 0.62
    detector_nms: float = 0.3
    pixelate: bool = False  # pixelate instead of blur, if a site prefers it

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "head_fraction": self.head_fraction,
            "blur_strength": self.blur_strength,
            "pixelate": self.pixelate,
        }


def find_yunet(explicit: str | None = None) -> Path | None:
    """Locate YuNet weights, if this deployment shipped them. Never downloads."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get(YUNET_ENV)
    if env:
        candidates.append(Path(env))
    roots = [
        Path(__file__).resolve().parents[2] / "data" / "models",
        Path.home() / ".cache" / "opencv26" / "models",
    ]
    for root in roots:
        candidates.extend(root / name for name in YUNET_FILENAMES)
    return next((c for c in candidates if c.is_file()), None)


def _odd(value: float, minimum: int) -> int:
    value = max(minimum, round(value))
    return value if value % 2 == 1 else value + 1


def head_region(box: Box, shape: tuple[int, int], config: PrivacyConfig) -> Box:
    """The generous head slice of a person box, clipped to the frame."""
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    head_h = bh * config.head_fraction
    cx = (x1 + x2) / 2.0
    half = (bw * config.head_widen) / 2.0
    # Wide boxes mean the person is probably lying down and the head could be at
    # either end; blur a square at both ends rather than guess an orientation.
    return (
        max(0.0, cx - half),
        max(0.0, y1),
        min(float(w), cx + half),
        min(float(h), y1 + max(head_h, bw * 0.5 if bw > bh else head_h)),
    )


def _blur_region(image: np.ndarray, region: Box, config: PrivacyConfig) -> bool:
    x1, y1, x2, y2 = (round(v) for v in region)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False
    patch = image[y1:y2, x1:x2]
    if config.pixelate:
        blocks = max(2, int(min(patch.shape[:2]) * 0.18))
        small = cv2.resize(patch, (blocks, blocks), interpolation=cv2.INTER_AREA)
        image[y1:y2, x1:x2] = cv2.resize(
            small, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    else:
        k = _odd(max(patch.shape[:2]) * config.blur_strength, config.min_kernel)
        image[y1:y2, x1:x2] = cv2.GaussianBlur(patch, (k, k), 0)
    return True


class FaceBlurrer:
    """Blurs faces in evidence frames. Detector optional, blurring is not."""

    def __init__(self, config: PrivacyConfig | None = None, model_path: str | None = None) -> None:
        self.config = config or PrivacyConfig()
        self.model_path = find_yunet(model_path)
        self._detector: Any = None
        self._detector_size: tuple[int, int] | None = None
        self.method = "yunet+person-head" if self.model_path else "person-head"

    def _yunet(self, size: tuple[int, int]) -> Any:
        if self.model_path is None:
            return None
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(self.model_path), "", size,
                self.config.detector_score, self.config.detector_nms, 5000,
            )
            self._detector_size = size
        elif self._detector_size != size:
            self._detector.setInputSize(size)
            self._detector_size = size
        return self._detector

    def detect_faces(self, image: np.ndarray) -> list[Box]:
        detector = self._yunet((image.shape[1], image.shape[0]))
        if detector is None:
            return []
        try:
            _, faces = detector.detect(image)
        except cv2.error:
            return []
        if faces is None:
            return []
        return [
            (float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3])) for f in faces
        ]

    def redact(self, image: np.ndarray, person_boxes: list[Box] | None = None
               ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return a redacted copy and a report of exactly what was blurred and how."""
        out = image.copy()
        if not self.config.enabled:
            return out, {"enabled": False, "method": "none", "regions": 0,
                         "note": "face blurring is switched off for this run"}

        regions = 0
        faces = self.detect_faces(out)
        for face in faces:
            pad_x = (face[2] - face[0]) * 0.22
            pad_y = (face[3] - face[1]) * 0.28
            grown = (face[0] - pad_x, face[1] - pad_y, face[2] + pad_x, face[3] + pad_y)
            regions += int(_blur_region(out, grown, self.config))

        # Always also blur the head slice of every person box. A detector that
        # missed a face because it was turned away must not leave it sharp.
        for box in person_boxes or []:
            regions += int(_blur_region(out, head_region(box, out.shape[:2], self.config),
                                        self.config))

        return out, {
            "enabled": True,
            "method": self.method,
            "faces_detected": len(faces),
            "regions": regions,
            "style": "pixelate" if self.config.pixelate else "gaussian",
            "note": (
                "faces blurred before the frame was stored; no face embedding, no "
                "recognition, no identity is computed anywhere in this system"
            ),
        }


PRIVACY_STATEMENT = (
    "Keepout detects that a person is present, not who they are. It runs no face "
    "recognition and stores no biometric template. Track numbers are per-run counters "
    "that reset with the process. Faces are blurred in every stored evidence frame "
    "before it is written."
)
