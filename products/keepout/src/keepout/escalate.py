"""The escalation ladder, and the incident objects that climb it.

Four levels, in the order the brief sets out:

| level | what causes it | behaviour |
|---|---|---|
| `NOTE` | someone in the zone while the machine is stopped | closes by itself when they leave |
| `ALERT` | someone in the zone while the machine is running | closes when they leave |
| `GUARD` | a fixed guard is open, removed or covered | closes when the guard comes back |
| `CRITICAL` | a person down and motionless | **latches** until acknowledged |

The latch is the whole point. A note or an alert describes a condition, and when
the condition ends the incident ends. A person on the floor is not a condition to
be cleared by the person getting up, because by then either somebody came and
helped, in which case the record should show that, or the person moved and the
state can still be wrong. `CRITICAL` stays raised until somebody presses
acknowledge, and the acknowledgement is recorded with its timestamp. This is the
one piece of state in the system that a human, not the pipeline, is allowed to
change.

An incident **upgrades in place**. A worker who steps into a zone while the
machine is stopped gets a note; if the machine then starts, the same incident
becomes an alert rather than a second incident, so the list stays readable and
the evidence frames stay attached to one story. Incidents never downgrade: once
something was worth an alert, the record says so even if the machine later stops.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Level(IntEnum):
    """Ordered so `max()` and `>` mean what they look like."""

    NOTE = 1
    GUARD = 2
    ALERT = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return {
            Level.NOTE: "Note",
            Level.GUARD: "Guard open",
            Level.ALERT: "Alert",
            Level.CRITICAL: "Person down",
        }[self]

    @property
    def slug(self) -> str:
        return self.name.lower()


LATCHING: frozenset[Level] = frozenset({Level.CRITICAL})

REASONS: dict[str, str] = {
    "zone_entry_stopped":
        "A person entered the danger zone while the machine was stopped. Logged as a note.",
    "zone_entry_running":
        "A person entered the danger zone while the machine was running.",
    "guard_missing":
        "The fixed guard is open, removed, or no longer matches its reference.",
    "person_down":
        "A person is not upright and has not moved. This stays raised until somebody "
        "acknowledges it.",
    "person_unaccounted":
        "Somebody was standing inside the danger zone and then stopped being visible "
        "without walking out of it. Send somebody to look. This stays raised until "
        "acknowledged.",
}


@dataclass
class EvidenceRef:
    """A pointer to a stored frame. The bytes live wherever the service put them."""

    label: str
    uri: str
    timestamp_ms: float
    caption: str = ""
    redacted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "uri": self.uri,
            "timestamp_ms": round(self.timestamp_ms, 1),
            "caption": self.caption,
            "redacted": self.redacted,
        }


@dataclass
class Incident:
    """One continuous safety condition, from first evidence to acknowledgement."""

    incident_id: str
    kind: str  # zone_entry | guard | person_down
    level: Level
    zone: str
    track_id: int | None
    started_ms: float
    last_ms: float
    reason: str
    peak_level: Level = Level.NOTE
    ended_ms: float | None = None
    acknowledged_ms: float | None = None
    acknowledged_by: str | None = None
    latched: bool = False
    machine_running: bool = False
    detail: dict[str, Any] = field(default_factory=dict)
    evidence: list[EvidenceRef] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    MAX_EVIDENCE = 6

    def __post_init__(self) -> None:
        self.peak_level = max(self.peak_level, self.level)
        self.latched = self.latched or self.level in LATCHING

    # ---- state ------------------------------------------------------------
    @property
    def open(self) -> bool:
        """Open means it still needs somebody's attention."""
        if self.latched:
            return self.acknowledged_ms is None
        return self.ended_ms is None

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.ended_ms or self.last_ms) - self.started_ms)

    def raise_to(self, level: Level, reason: str, timestamp_ms: float,
                 **detail: Any) -> bool:
        """Upgrade in place. Returns True if the level actually changed."""
        if level <= self.level:
            self.last_ms = timestamp_ms
            self.detail.update(detail)
            return False
        self.history.append({
            "ts_ms": round(timestamp_ms, 1),
            "from": self.level.slug,
            "to": level.slug,
            "reason": reason,
        })
        self.level = level
        self.peak_level = max(self.peak_level, level)
        self.reason = reason
        self.last_ms = timestamp_ms
        self.detail.update(detail)
        if level in LATCHING:
            self.latched = True
            self.ended_ms = None
        return True

    def touch(self, timestamp_ms: float, **detail: Any) -> None:
        self.last_ms = timestamp_ms
        self.ended_ms = None
        if detail:
            self.detail.update(detail)

    def end(self, timestamp_ms: float) -> None:
        """The condition stopped. A latched incident stays open regardless."""
        if self.ended_ms is None:
            self.ended_ms = timestamp_ms
            self.history.append({"ts_ms": round(timestamp_ms, 1), "event": "condition cleared"})

    def acknowledge(self, timestamp_ms: float, by: str = "operator") -> bool:
        if self.acknowledged_ms is not None:
            return False
        self.acknowledged_ms = timestamp_ms
        self.acknowledged_by = by
        self.history.append({
            "ts_ms": round(timestamp_ms, 1), "event": "acknowledged", "by": by,
        })
        return True

    def add_evidence(self, ref: EvidenceRef) -> EvidenceRef:
        """Keep the first frames and the most recent one; the middle is repetition."""
        self.evidence.append(ref)
        if len(self.evidence) > self.MAX_EVIDENCE:
            self.evidence = [*self.evidence[: self.MAX_EVIDENCE - 1], self.evidence[-1]]
        return ref

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "kind": self.kind,
            "level": self.level.slug,
            "level_rank": int(self.level),
            "level_label": self.level.label,
            "peak_level": self.peak_level.slug,
            "zone": self.zone,
            "track_id": self.track_id,
            "started_ms": round(self.started_ms, 1),
            "last_ms": round(self.last_ms, 1),
            "ended_ms": None if self.ended_ms is None else round(self.ended_ms, 1),
            "duration_ms": round(self.duration_ms, 1),
            "open": self.open,
            "latched": self.latched,
            "acknowledged_ms": self.acknowledged_ms,
            "acknowledged_by": self.acknowledged_by,
            "machine_running": self.machine_running,
            "reason": self.reason,
            "detail": dict(self.detail),
            "evidence": [e.to_dict() for e in self.evidence],
            "history": list(self.history),
        }


class IncidentLog:
    """Creates, upgrades and closes incidents. Holds no frames, only references."""

    def __init__(self) -> None:
        self.incidents: list[Incident] = []
        self._by_key: dict[tuple[str, str, int | None], Incident] = {}

    # ---- lookup -----------------------------------------------------------
    def _key(self, kind: str, zone: str, track_id: int | None
             ) -> tuple[str, str, int | None]:
        return (kind, zone, track_id)

    def active(self, kind: str, zone: str, track_id: int | None) -> Incident | None:
        incident = self._by_key.get(self._key(kind, zone, track_id))
        if incident is None:
            return None
        if incident.ended_ms is not None and not incident.latched:
            return None
        return incident

    def get(self, incident_id: str) -> Incident | None:
        return next((i for i in self.incidents if i.incident_id == incident_id), None)

    # ---- mutation ---------------------------------------------------------
    def observe(
        self,
        *,
        kind: str,
        zone: str,
        track_id: int | None,
        level: Level,
        reason: str,
        timestamp_ms: float,
        machine_running: bool = False,
        **detail: Any,
    ) -> tuple[Incident, bool]:
        """Report a live condition. Returns (incident, is_new_or_upgraded)."""
        existing = self.active(kind, zone, track_id)
        if existing is not None:
            existing.machine_running = machine_running
            changed = existing.raise_to(level, reason, timestamp_ms, **detail)
            existing.touch(timestamp_ms)
            return existing, changed

        incident = Incident(
            incident_id=uuid.uuid4().hex[:10],
            kind=kind,
            level=level,
            zone=zone,
            track_id=track_id,
            started_ms=timestamp_ms,
            last_ms=timestamp_ms,
            reason=reason,
            machine_running=machine_running,
            detail=dict(detail),
        )
        self.incidents.append(incident)
        self._by_key[self._key(kind, zone, track_id)] = incident
        return incident, True

    def clear(self, kind: str, zone: str, track_id: int | None, timestamp_ms: float
              ) -> Incident | None:
        incident = self.active(kind, zone, track_id)
        if incident is None:
            return None
        incident.end(timestamp_ms)
        if not incident.latched:
            self._by_key.pop(self._key(kind, zone, track_id), None)
        return incident

    def acknowledge(self, incident_id: str, timestamp_ms: float, by: str = "operator") -> bool:
        incident = self.get(incident_id)
        if incident is None:
            return False
        done = incident.acknowledge(timestamp_ms, by)
        if done:
            self._by_key.pop(self._key(incident.kind, incident.zone, incident.track_id), None)
        return done

    # ---- views ------------------------------------------------------------
    def open_incidents(self) -> list[Incident]:
        return [i for i in self.incidents if i.open]

    def highest_open_level(self) -> Level | None:
        levels = [i.level for i in self.open_incidents()]
        return max(levels) if levels else None

    def summary(self) -> dict[str, Any]:
        counts = {level.slug: 0 for level in Level}
        for incident in self.incidents:
            counts[incident.peak_level.slug] += 1
        highest = self.highest_open_level()
        return {
            "total": len(self.incidents),
            "open": len(self.open_incidents()),
            "unacknowledged_critical": sum(
                1 for i in self.incidents if i.latched and i.acknowledged_ms is None
            ),
            "by_peak_level": counts,
            "highest_open_level": highest.slug if highest else None,
        }

    def to_list(self) -> list[dict[str, Any]]:
        return [i.to_dict() for i in sorted(
            self.incidents, key=lambda i: (-int(i.peak_level), i.started_ms)
        )]


def level_for_zone_entry(machine_running: bool) -> tuple[Level, str]:
    if machine_running:
        return Level.ALERT, REASONS["zone_entry_running"]
    return Level.NOTE, REASONS["zone_entry_stopped"]
