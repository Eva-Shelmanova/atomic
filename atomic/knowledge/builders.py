"""
Наполнение графа знаний из контрактов: каналы, сценарии, окна, гипотезы,
вердикты, утверждения и решения политики.

По графу восстанавливается путь «данные → гипотеза → проверка → ответ → решение»
(основа трассируемости, требование 10).
"""

from __future__ import annotations

from atomic.contracts.hypothesis import (
    EvidenceDirection,
    HypothesisSet,
    HypothesisVerdict,
    ObservationWindow,
)
from atomic.contracts.verification import EvidenceKind, VerificationBundle
from atomic.diagnostics.channels import ChannelRegistry
from atomic.knowledge.schema import EntityType as E
from atomic.knowledge.schema import KnowledgeGraph, Provenance
from atomic.knowledge.schema import RelationType as R


def channel_node_id(name: str) -> str:
    return f"channel:{name}"


def scenario_node_id(code: str) -> str:
    return f"scenario:{code}"


def add_registry(g: KnowledgeGraph, registry: ChannelRegistry, source: str) -> None:
    prov = Provenance(source=source, version=registry.schema_version)
    for c in registry.channels:
        g.add(
            channel_node_id(c.name), E.CHANNEL, prov,
            name=c.name, label=c.label, unit=c.unit, subsystem=c.subsystem, label_proxy=c.label_proxy,
        )


def add_scenarios(g: KnowledgeGraph, titles: dict[str, str], source: str) -> None:
    prov = Provenance(source=source)
    for code, title in titles.items():
        g.add(scenario_node_id(code), E.SCENARIO, prov, code=code, title=title)


def add_window(g: KnowledgeGraph, w: ObservationWindow, source: str, channels: tuple[str, ...] | None = None) -> None:
    """Окно и его прогон. channels — какие связи OBSERVES проводить (по умолчанию все)."""
    prov = Provenance(source=source)
    run_node = f"run:{w.run_id}"
    if not g.has(run_node):
        g.add(run_node, E.RUN, prov, run_id=w.run_id)
    g.add(f"window:{w.window_id}", E.WINDOW, prov, t_start=w.t_start, t_end=w.t_end, n=len(w.values))
    g.link(f"window:{w.window_id}", R.PART_OF, run_node)
    for c in channels or w.channels:
        if g.has(channel_node_id(c)):
            g.link(f"window:{w.window_id}", R.OBSERVES, channel_node_id(c))


def add_hypotheses(g: KnowledgeGraph, hs: HypothesisSet) -> list[str]:
    prov = Provenance(source="generator", version=hs.generator_version)
    ids = []
    for h in hs.hypotheses:
        hid = f"hypothesis:{hs.window_id}:{h.rank}"
        g.add(hid, E.HYPOTHESIS, prov, scenario=h.scenario, confidence=h.confidence, rank=h.rank)
        g.link(hid, R.ABOUT, f"window:{hs.window_id}")
        if g.has(scenario_node_id(h.scenario)):
            g.link(hid, R.PROPOSES, scenario_node_id(h.scenario))
        ids.append(hid)
    return ids


def add_verdict(g: KnowledgeGraph, hypothesis_id: str, v: HypothesisVerdict) -> None:
    prov = Provenance(source="hypothesis_verifier", version=v.verifier_version)
    g.add(hypothesis_id, E.HYPOTHESIS, g.node(hypothesis_id).provenance,
          **{**g.node(hypothesis_id).attrs, "status": v.status.value, "support_ratio": v.support_ratio})
    for ev in (*v.evidence_for, *v.evidence_against):
        eid = f"evidence:{ev.ref_id}:{hypothesis_id}"
        g.add(eid, E.EVIDENCE, prov, **ev.to_dict())
        rel = R.SUPPORTS if ev.direction == EvidenceDirection.SUPPORTS else R.CONTRADICTS
        g.link(eid, rel, hypothesis_id)
        if g.has(channel_node_id(ev.channel)):
            g.link(eid, R.REFERS_TO, channel_node_id(ev.channel))
        if g.has(f"window:{ev.window_id}"):
            g.link(eid, R.REFERS_TO, f"window:{ev.window_id}")
    for a in v.analogues:
        event_node = f"event:{a.event_id}"
        if not g.has(event_node):
            g.add(event_node, E.HISTORICAL_EVENT, Provenance(source="ethz"), title=a.title)
        eid = f"evidence:{a.event_id}:{hypothesis_id}"
        g.add(eid, E.EVIDENCE, prov, kind="historical_event", score=a.score)
        g.link(eid, R.SUPPORTS, hypothesis_id)
        g.link(eid, R.REFERS_TO, event_node)


def add_bundle(g: KnowledgeGraph, bundle: VerificationBundle, decision_id: str) -> None:
    """Утверждения, правила, пакет и решение политики одного ответа."""
    d = bundle.decision
    prov = Provenance(source="answer_verifier", version=d.applied_pack_id)
    pack_node = f"rulepack:{d.applied_pack_id}"
    if not g.has(pack_node):
        g.add(pack_node, E.RULE_PACK, prov, pack_id=d.applied_pack_id)
    for rule_id in bundle.rule_results:
        rnode = f"rule:{d.applied_pack_id}:{rule_id}"
        if not g.has(rnode):
            g.add(rnode, E.RULE, prov, rule_id=rule_id)
            g.link(rnode, R.PART_OF, pack_node)
    g.add(decision_id, E.POLICY_DECISION, prov, **d.to_dict())
    g.link(decision_id, R.APPLIES, pack_node)
    for c in bundle.claims:
        cnode = f"claim:{decision_id}:{c.claim_id}"
        g.add(cnode, E.CLAIM, prov, text=c.text, strength=c.strength())
        g.link(decision_id, R.DECIDES, cnode)
        for rule_id in bundle.rule_results:
            g.link(cnode, R.CHECKED_BY, f"rule:{d.applied_pack_id}:{rule_id}")
        for e in c.evidence:
            enode = f"evidence:{e.ref_id}:{cnode}"
            g.add(enode, E.EVIDENCE, prov, **e.to_dict())
            g.link(enode, R.SUPPORTS, cnode)
            target = {
                EvidenceKind.RAG_CHUNK: f"fragment:{e.ref_id}",
                EvidenceKind.HISTORICAL_EVENT: f"event:{e.ref_id}",
                EvidenceKind.TELEMETRY: f"window:{e.ref_id.split('#')[0]}",
            }.get(e.kind)
            if target and g.has(target):
                g.link(enode, R.REFERS_TO, target)
