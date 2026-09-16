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

**Every person box in a stored frame has its head region blurred**, whether or not
a face detector is available: the top 28% of the box, centred, widened slightly.
That is deliberately generous, because the failure mode of a face blur must be
"blurred too much", never "missed a face". YuNet, when its weights are present,
adds face boxes on top of that, which catches a face whose body the person
detector missed.

This was not always true, and real footage is how we found out. On a crowded
construction clip from Wikimedia Commons (Malta, 2013), a `person_down` evidence
frame blurred only the head of the incident's own subject, and every other
worker's face in the frame stayed sharp. The pipeline had passed the subject's
box and nothing else, and the YuNet weights were not in the image, so no detector
covered for it. `pipeline.Keepout._capture` now collects every person box it
knows about in that frame, plus a dedicated low-threshold, tiled person sweep of
the raw frame, and hands all of them here.

Missing YuNet weights are reported, never hidden: the report carries
`face_detector: "unavailable"` and `method: "person-head"`. A deployment that must
have the detector sets `KEEPOUT_REQUIRE_FACE_DETECTOR=1`, and then the process
refuses to start without the weights (the container image does this).

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
REQUIRE_ENV = "KEEPOUT_REQUIRE_FACE_DETECTOR"
YUNET_FILENAMES = ("face_detection_yunet_2023mar.onnx", "yunet.onnx")
# opencv_zoo, MIT licence. Verified at image build time against this digest.
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


class FaceDetectorMissing(RuntimeError):
    """Raised when a deployment requires YuNet and the weights are not there."""


@dataclass
class PrivacyConfig:
    enabled: bool = True
    head_fraction: float = 0.28  # top slice of a person box treated as the head
    head_widen: float = 1.12
    blur_strength: float = 0.11  # kernel as a fraction of the region's larger side
    min_kernel: int = 9
    detector_score: float = 0.5
    detector_nms: float = 0.3
    # Faces in a wide shot are 10 to 15 px across at the 960 px working size, below
    # what YuNet finds reliably. Upscale small frames before looking for faces.
    detector_min_side: int = 1600
    pixelate: bool = False  # pixelate instead of blur, if a site prefers it
    use_face_detector: bool = True  # False forces the head-region-only path (tests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "head_fraction": self.head_fraction,
            "blur_strength": self.blur_strength,
            "pixelate": self.pixelate,
            "use_face_detector": self.use_face_detector,
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
    if os.environ.get("OPENCV26_MODEL_DIR"):
        roots.insert(0, Path(os.environ["OPENCV26_MODEL_DIR"]))
    for root in roots:
        candidates.extend(root / name for name in YUNET_FILENAMES)
    return next((c for c in candidates if c.is_file()), None)


def face_detector_required() -> bool:
    return os.environ.get(REQUIRE_ENV, "").strip().lower() in {"1", "true", "yes"}


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
    """Blurs faces in evidence frames. Detector optional, blurring is not.

    `use_detector=False` forces the head-region-only path, which is what the tests
    use so that a YuNet file in somebody's model cache cannot change their answer.
    """

    def __init__(self, config: PrivacyConfig | None = None, model_path: str | None = None,
                 *, use_detector: bool = True) -> None:
        self.config = config or PrivacyConfig()
        self.model_path = find_yunet(model_path) if use_detector else None
        if self.model_path is None and use_detector and face_detector_required():
            raise FaceDetectorMissing(
                f"{REQUIRE_ENV} is set but no YuNet weights were found (looked for "
                f"{', '.join(YUNET_FILENAMES)} via {YUNET_ENV}, OPENCV26_MODEL_DIR and the "
                "model cache). Refusing to run rather than store evidence with a weaker blur."
            )
        self._detector: Any = None
        self._detector_size: tuple[int, int] | None = None
        self.method = "yunet+person-head" if self.model_path else "person-head"
        self.face_detector = "yunet" if self.model_path else "unavailable"

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
        if self.model_path is None:
            return []
        h, w = image.shape[:2]
        scale = max(1.0, self.config.detector_min_side / float(max(h, w)))
        work = image if scale == 1.0 else cv2.resize(
            image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
        detector = self._yunet((work.shape[1], work.shape[0]))
        try:
            _, faces = detector.detect(work)
        except cv2.error:
            return []
        if faces is None:
            return []
        return [
            (float(f[0]) / scale, float(f[1]) / scale,
             float(f[0] + f[2]) / scale, float(f[1] + f[3]) / scale) for f in faces
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "face_blur": self.config.enabled,
            "method": self.method,
            "face_detector": self.face_detector,
            "head_region_of_every_person_box": True,
        }

    def redact(self, image: np.ndarray, person_boxes: list[Box] | None = None,
               *, source: np.ndarray | None = None, sweep: str = "none"
               ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return a redacted copy and a report of exactly what was blurred and how.

        `image` is what gets stored (usually annotated); `source` is the raw frame
        faces are looked for in, so a banner or a drawn box cannot hide a face from
        the detector. `person_boxes` must be every person the caller knows about in
        the frame, not just the one the incident is about.
        """
        out = image.copy()
        boxes = list(person_boxes or [])
        if not self.config.enabled:
            return out, {"enabled": False, "method": "none", "regions": 0,
                         "person_boxes": len(boxes), "face_detector": self.face_detector,
                         "note": "face blurring is switched off for this run"}

        regions = 0
        faces = self.detect_faces(source if source is not None else image)
        for face in faces:
            pad_x = (face[2] - face[0]) * 0.22
            pad_y = (face[3] - face[1]) * 0.28
            grown = (face[0] - pad_x, face[1] - pad_y, face[2] + pad_x, face[3] + pad_y)
            regions += int(_blur_region(out, grown, self.config))

        # Always blur the head slice of every person box. A detector that missed a
        # face because it was turned away, small or absent must not leave it sharp.
        head_regions = 0
        for box in boxes:
            blurred = _blur_region(out, head_region(box, out.shape[:2], self.config),
                                   self.config)
            head_regions += int(blurred)
        regions += head_regions

        return out, {
            "enabled": True,
            "method": self.method,
            "face_detector": self.face_detector,
            "faces_detected": len(faces),
            "person_boxes": len(boxes),
            "head_regions": head_regions,
            "person_sweep": sweep,
            "regions": regions,
            "style": "pixelate" if self.config.pixelate else "gaussian",
            "note": (
                "the head region of every detected person, and every face YuNet found, "
                "blurred before the frame was stored; no face embedding, no recognition, "
                "no identity is computed anywhere in this system"
                if self.model_path else
                "the head region of every detected person blurred before the frame was "
                "stored (YuNet weights not present, so no separate face detector ran); no "
                "face embedding, no recognition, no identity is computed anywhere"
            ),
        }


PRIVACY_STATEMENT = (
    "Keepout detects that a person is present, not who they are. It runs no face "
    "recognition and stores no biometric template. Track numbers are per-run counters "
    "that reset with the process. Before any evidence frame is written, the head "
    "region of every person detected in it is blurred, and every face the YuNet face "
    "detector finds. A person no detector finds at all is not blurred; the record says "
    "which detectors ran."
)
