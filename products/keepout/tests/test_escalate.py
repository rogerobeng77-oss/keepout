"""The escalation ladder: note, guard, alert, critical, and the latch."""

from __future__ import annotations

import pytest

from keepout import IncidentLog, Level
from keepout.escalate import REASONS, EvidenceRef, Incident, level_for_zone_entry


def test_levels_are_ordered_so_max_means_worst():
    assert Level.NOTE < Level.GUARD < Level.ALERT < Level.CRITICAL
    assert max([Level.NOTE, Level.CRITICAL, Level.ALERT]) is Level.CRITICAL
    assert Level.CRITICAL.label == "Person down"


def test_the_machine_state_decides_note_versus_alert():
    level, reason = level_for_zone_entry(machine_running=False)
    assert level is Level.NOTE
    assert "stopped" in reason

    level, reason = level_for_zone_entry(machine_running=True)
    assert level is Level.ALERT
    assert "running" in reason


def test_a_note_becomes_an_alert_in_place_when_the_machine_starts():
    """One person, one continuous presence, one incident that gets worse."""
    log = IncidentLog()
    first, created = log.observe(kind="zone_entry", zone="z", track_id=1, level=Level.NOTE,
                                reason=REASONS["zone_entry_stopped"], timestamp_ms=0.0)
    assert created and first.level is Level.NOTE

    second, changed = log.observe(kind="zone_entry", zone="z", track_id=1, level=Level.ALERT,
                                  reason=REASONS["zone_entry_running"], timestamp_ms=3000.0,
                                  machine_running=True)
    assert changed is True
    assert second is first, "it must be the same incident, not a second one"
    assert len(log.incidents) == 1
    assert first.level is Level.ALERT
    assert first.history[-1]["from"] == "note" and first.history[-1]["to"] == "alert"


def test_an_incident_never_downgrades():
    log = IncidentLog()
    incident, _ = log.observe(kind="zone_entry", zone="z", track_id=1, level=Level.ALERT,
                              reason="r", timestamp_ms=0.0, machine_running=True)
    _, changed = log.observe(kind="zone_entry", zone="z", track_id=1, level=Level.NOTE,
                             reason="r", timestamp_ms=1000.0)
    assert changed is False
    assert incident.level is Level.ALERT
    assert incident.peak_level is Level.ALERT


def test_an_unlatched_incident_closes_when_the_condition_ends():
    log = IncidentLog()
    log.observe(kind="zone_entry", zone="z", track_id=1, level=Level.ALERT,
                reason="r", timestamp_ms=0.0)
    assert log.open_incidents()
    closed = log.clear("zone_entry", "z", 1, timestamp_ms=5000.0)
    assert closed is not None
    assert closed.open is False
    assert closed.ended_ms == 5000.0
    assert closed.duration_ms == 5000.0
    assert log.open_incidents() == []


def test_the_same_person_re_entering_later_opens_a_new_incident():
    log = IncidentLog()
    log.observe(kind="zone_entry", zone="z", track_id=1, level=Level.NOTE,
                reason="r", timestamp_ms=0.0)
    log.clear("zone_entry", "z", 1, timestamp_ms=2000.0)
    log.observe(kind="zone_entry", zone="z", track_id=1, level=Level.NOTE,
                reason="r", timestamp_ms=9000.0)
    assert len(log.incidents) == 2


def test_critical_latches_and_only_a_human_clears_it():
    """The core rule: a person on the floor does not stop mattering by itself."""
    log = IncidentLog()
    incident, _ = log.observe(kind="person_down", zone="z", track_id=4, level=Level.CRITICAL,
                              reason=REASONS["person_down"], timestamp_ms=1000.0)
    assert incident.latched is True

    log.clear("person_down", "z", 4, timestamp_ms=4000.0)
    assert incident.open is True, "a latched incident must not close on its own"
    assert log.summary()["unacknowledged_critical"] == 1

    assert log.acknowledge(incident.incident_id, timestamp_ms=9000.0, by="shift supervisor")
    assert incident.open is False
    assert incident.acknowledged_by == "shift supervisor"
    assert log.summary()["unacknowledged_critical"] == 0


def test_acknowledging_twice_is_a_no_op():
    log = IncidentLog()
    incident, _ = log.observe(kind="person_down", zone="z", track_id=1, level=Level.CRITICAL,
                              reason="r", timestamp_ms=0.0)
    assert log.acknowledge(incident.incident_id, 100.0) is True
    assert log.acknowledge(incident.incident_id, 200.0) is False
    assert incident.acknowledged_ms == 100.0


def test_acknowledging_an_unknown_incident_is_false_not_an_exception():
    assert IncidentLog().acknowledge("nope", 0.0) is False


def test_a_zone_entry_that_becomes_critical_stays_raised():
    """A person who walks in and then collapses: one story, ending latched."""
    log = IncidentLog()
    incident, _ = log.observe(kind="zone_entry", zone="z", track_id=2, level=Level.ALERT,
                              reason="entered while running", timestamp_ms=0.0,
                              machine_running=True)
    log.observe(kind="zone_entry", zone="z", track_id=2, level=Level.CRITICAL,
                reason=REASONS["person_down"], timestamp_ms=8000.0)
    assert incident.level is Level.CRITICAL
    assert incident.latched is True
    log.clear("zone_entry", "z", 2, timestamp_ms=12000.0)
    assert incident.open is True


def test_highest_open_level_and_summary():
    log = IncidentLog()
    log.observe(kind="zone_entry", zone="z", track_id=1, level=Level.NOTE,
                reason="r", timestamp_ms=0.0)
    log.observe(kind="guard", zone="g", track_id=None, level=Level.GUARD,
                reason="r", timestamp_ms=0.0)
    assert log.highest_open_level() is Level.GUARD

    log.observe(kind="person_down", zone="z", track_id=3, level=Level.CRITICAL,
                reason="r", timestamp_ms=0.0)
    assert log.highest_open_level() is Level.CRITICAL

    summary = log.summary()
    assert summary["total"] == 3
    assert summary["open"] == 3
    assert summary["by_peak_level"]["critical"] == 1
    assert summary["highest_open_level"] == "critical"


def test_evidence_keeps_the_first_frames_and_the_latest_one():
    incident = Incident(incident_id="x", kind="k", level=Level.NOTE, zone="z", track_id=1,
                        started_ms=0.0, last_ms=0.0, reason="r")
    for i in range(12):
        incident.add_evidence(EvidenceRef(label=f"e{i}", uri=f"/u/{i}", timestamp_ms=i * 100.0))
    assert len(incident.evidence) == Incident.MAX_EVIDENCE
    assert incident.evidence[0].label == "e0"
    assert incident.evidence[-1].label == "e11", "the most recent frame must be kept"


def test_serialisation_carries_everything_the_ui_needs():
    log = IncidentLog()
    incident, _ = log.observe(kind="zone_entry", zone="conveyor", track_id=5, level=Level.ALERT,
                              reason="r", timestamp_ms=1234.0, machine_running=True,
                              dwell_ms=900.0)
    incident.add_evidence(EvidenceRef("frame", "/e/1", 1234.0, caption="c"))
    payload = incident.to_dict()
    for key in ("incident_id", "level", "level_label", "zone", "track_id", "open",
                "latched", "reason", "detail", "evidence", "history", "machine_running"):
        assert key in payload
    assert payload["level"] == "alert"
    assert payload["detail"]["dwell_ms"] == 900.0
    assert payload["evidence"][0]["redacted"] is True

    listed = log.to_list()
    assert listed[0]["incident_id"] == incident.incident_id


def test_the_list_is_ordered_worst_first():
    log = IncidentLog()
    log.observe(kind="zone_entry", zone="z", track_id=1, level=Level.NOTE,
                reason="r", timestamp_ms=0.0)
    log.observe(kind="person_down", zone="z", track_id=2, level=Level.CRITICAL,
                reason="r", timestamp_ms=100.0)
    log.observe(kind="zone_entry", zone="z", track_id=3, level=Level.ALERT,
                reason="r", timestamp_ms=200.0)
    levels = [i["level"] for i in log.to_list()]
    assert levels == ["critical", "alert", "note"]


@pytest.mark.parametrize("key", ["zone_entry_stopped", "zone_entry_running",
                                 "guard_missing", "person_down", "person_unaccounted"])
def test_every_reason_is_a_sentence_an_operator_can_read(key):
    assert key in REASONS
    assert REASONS[key][0].isupper() and REASONS[key].endswith(".")
