"""Подбор аналогов из базы событий (этап B2, требование 7)."""

from atomic.diagnostics.analogues import AnalogueFinder

EVENTS = [
    {"event_id": "ethz:1", "site": "A", "country": "X", "date": "1988-01-01", "origin": "Steam Generator",
     "description": "A primary-to-secondary leak was detected: steam generator tube rupture.", "cause": None},
    {"event_id": "ethz:2", "site": "A", "country": "X", "date": "1988-01-01", "origin": "Steam Generator",
     "description": "A primary-to-secondary leak was detected: steam generator tube rupture.", "cause": None},  # дубль (второй энергоблок)
    {"event_id": "ethz:3", "site": "B", "country": "Y", "date": "1990-05-05", "origin": "Main Feedwater",
     "description": "Feedwater pipe rupture during start-up of a feedwater pump.", "cause": "erosion"},
    {"event_id": "ethz:4", "site": "C", "country": "Z", "date": "2001-02-02", "origin": "Offsite Power Systems",
     "description": "Loss of offsite power due to grid disturbance.", "cause": None},
]
MAPPING = {
    "version": "test", "origin_boost": 2.0, "min_score": 0.5,
    "scenario": {
        "SGATR": {"query": "steam generator tube rupture", "origin": ["Steam Generator"]},
        "FLB": {"query": "feedwater pipe rupture", "origin": ["Main Feedwater"]},
        "Normal": {"query": "", "origin": []},
    },
}


def finder():
    return AnalogueFinder(EVENTS, MAPPING, source_version="t")


def test_relevant_event_first_and_duplicates_removed():
    res = finder().find("SGATR", k=3)
    assert res[0].event_id == "ethz:1"
    assert [a.event_id for a in res].count("ethz:2") == 0
    assert "steam_generator" in res[0].matched_terms


def test_feedwater_query_finds_feedwater_event():
    assert finder().find("FLB", k=1)[0].event_id == "ethz:3"


def test_empty_or_unknown_scenario_gives_no_analogues():
    assert finder().find("Normal") == [] and finder().find("NOPE") == []


def test_min_score_filters_weak_matches():
    strict = AnalogueFinder(EVENTS, {**MAPPING, "min_score": 1e6})
    assert strict.find("SGATR") == []


def test_analogue_wording_and_resolvable_ids():
    f = finder()
    a = f.find("SGATR", k=1)[0]
    assert a.describe().startswith("В аналогичных случаях наблюдалось")
    assert f.has_event(a.event_id) and not f.has_event("ethz:999")
    ref = a.to_evidence_ref(0.6)
    assert ref.kind.value == "historical_event" and ref.ref_id == "ethz:1"
