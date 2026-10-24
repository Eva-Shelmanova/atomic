"""Семантическая модель: схема связей, происхождение, сериализация (этап A)."""

import numpy as np
import pytest

from atomic.contracts import HypothesisSet, ObservationWindow, ScenarioHypothesis
from atomic.diagnostics.channels import default_registry
from atomic.knowledge import EntityType, KnowledgeGraph, Provenance, RelationType, schema_mermaid
from atomic.knowledge.builders import add_hypotheses, add_registry, add_scenarios, add_window


def test_every_entity_type_from_roadmap_present():
    required = {
        "channel", "window", "scenario", "historical_event", "document", "fragment",
        "hypothesis", "evidence", "claim", "rule", "policy_decision",
    }
    assert required <= {e.value for e in EntityType}


def test_invalid_relation_rejected():
    g = KnowledgeGraph()
    p = Provenance("test")
    g.add("rule:1", EntityType.RULE, p)
    g.add("channel:P", EntityType.CHANNEL, p)
    with pytest.raises(ValueError):
        g.link("rule:1", RelationType.OBSERVES, "channel:P")
    with pytest.raises(KeyError):
        g.link("rule:1", RelationType.PART_OF, "missing")


def test_node_type_conflict_rejected():
    g = KnowledgeGraph()
    g.add("x", EntityType.RULE, Provenance("t"))
    with pytest.raises(ValueError):
        g.add("x", EntityType.CHANNEL, Provenance("t"))


def test_graph_roundtrip_and_path_window_to_scenario():
    reg = default_registry()
    g = KnowledgeGraph()
    add_registry(g, reg, "nppad@test")
    add_scenarios(g, {"LOCA": "течь"}, "config")
    w = ObservationWindow("w1", "r1", 0.0, 110.0, reg.names, np.zeros((12, 96)), np.zeros((12, 96)))
    add_window(g, w, "nppad@test", channels=("P", "TAVG"))
    hids = add_hypotheses(g, HypothesisSet("w1", (ScenarioHypothesis("LOCA", 0.7, 1),), "gen-test"))
    assert [n.id for n in g.neighbors(hids[0], RelationType.PROPOSES)] == ["scenario:LOCA"]
    assert {n.id for n in g.neighbors("window:w1", RelationType.OBSERVES)} == {"channel:P", "channel:TAVG"}
    again = KnowledgeGraph.from_dict(g.to_dict())
    assert again.stats() == g.stats()
    assert all(n.provenance.source for n in again.nodes())


def test_schema_mermaid_lists_relations():
    text = schema_mermaid()
    assert text.startswith("flowchart LR") and "proposes" in text
