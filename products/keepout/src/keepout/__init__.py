"""Keepout - a fixed camera over machinery, watched by something that never blinks.

    "In an effort to raise the alarm, he repeatedly waved at a CCTV camera in the
    hope that someone monitoring the system would spot him and come to his aid,
    nobody did."
        -- HSE, Factory Services UK Limited, Knowsley, 9 September 2026

Everything in this package is built on OpenCV 5 (`opencv-python-headless==5.0.0.93`)
and `visioncore`. Importing `visioncore` already asserts the OpenCV major version,
so a 4.x wheel fails at import rather than silently producing a 4.x entry.
"""

from __future__ import annotations

from .escalate import EvidenceRef, Incident, IncidentLog, Level
from .guard import GuardChecker, GuardConfig, GuardObservation, GuardStatus
from .machine import MachineConfig, MachineState, MotionEnergy, suggest_thresholds
from .pipeline import FrameResult, Keepout, KeepoutConfig, default_zones
from .posture import DownDetector, PostureConfig, PostureState, box_aspect, motion_ratio
from .privacy import PRIVACY_STATEMENT, FaceBlurrer, PrivacyConfig
from .track import BoxTracker, Track, TrackerConfig, hungarian, iou
from .viewcheck import ViewConfig, ViewGuard, ViewProblem, ViewStatus
from .zones import (
    ContactPoint,
    DwellConfig,
    Occupancy,
    Zone,
    ZoneMonitor,
    ZoneProposal,
    draw_zone,
    propose_zone_from_motion,
)

__version__ = "1.0.0"

HSE_QUOTE = (
    "In an effort to raise the alarm, he repeatedly waved at a CCTV camera in the hope "
    "that someone monitoring the system would spot him and come to his aid, nobody did."
)
HSE_SOURCE = (
    "HSE press release, 9 September 2026, Factory Services UK Limited, Knowsley. "
    "https://press.hse.gov.uk/2026/09/09/manufacturer-fined-after-worker-suffers-"
    "life-changing-injuries-in-conveyor-incident/"
)

__all__ = [
    "HSE_QUOTE",
    "HSE_SOURCE",
    "PRIVACY_STATEMENT",
    "BoxTracker",
    "ContactPoint",
    "DownDetector",
    "DwellConfig",
    "EvidenceRef",
    "FaceBlurrer",
    "FrameResult",
    "GuardChecker",
    "GuardConfig",
    "GuardObservation",
    "GuardStatus",
    "Incident",
    "IncidentLog",
    "Keepout",
    "KeepoutConfig",
    "Level",
    "MachineConfig",
    "MachineState",
    "MotionEnergy",
    "Occupancy",
    "PostureConfig",
    "PostureState",
    "PrivacyConfig",
    "Track",
    "TrackerConfig",
    "ViewConfig",
    "ViewGuard",
    "ViewProblem",
    "ViewStatus",
    "Zone",
    "ZoneMonitor",
    "ZoneProposal",
    "__version__",
    "box_aspect",
    "default_zones",
    "draw_zone",
    "hungarian",
    "iou",
    "motion_ratio",
    "propose_zone_from_motion",
    "suggest_thresholds",
]
