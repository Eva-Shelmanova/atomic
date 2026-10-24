"""
Сквозной сценарий этапа C: окно NPPAD → гипотезы → проверка → ответ AnalystAgent →
PolicyDecision (требование 11). LLM подменяется; модели и данные — реальные (val, не test).
"""

import numpy as np
import pytest

from atomic.contracts.hypothesis import DiagnosisOutcome
from atomic.contracts.verification import PolicyEffect
from atomic.diagnostics.config import default_config

MODELS = default_config().resolve("models")
READY = all((MODELS / p).exists() for p in ("generator/generator.joblib", "verifier/signatures.npz",
                                              "verifier/onset/signatures.npz", "novelty/novelty.npz")) and \
    default_config().resolve("ethz_events").exists()
pytestmark = pytest.mark.skipif(not READY, reason="модели или база ETHZ не подготовлены")


class FakeLLM:
    def __init__(self, reply="ok"):
        self.reply = reply

    def chat(self, messages, temperature=0.3, max_tokens=2048):
        return self.reply


@pytest.fixture(scope="module")
def env():
    from atomic.diagnostics.labels import LabelStore
    from atomic.diagnostics.nppad import NPPADDataset
    from atomic.memory.vector import VectorMemory
    from atomic.orchestration.workflow import AtomicWorkflow

    ds = NPPADDataset()
    labels = LabelStore.load(ds.directory)
    wf = AtomicWorkflow(vector_memory=VectorMemory(), verification_mode="shadow")
    wf.analyst.llm = FakeLLM()
    return ds, labels, wf


def run_of(ds, labels, scenario, split="val"):
    return next(r for r in ds.split(split) if labels.scenario(r) == scenario)


def test_hot_leg_loca_identified_and_allowed(env):
    from atomic.diagnostics.telemetry import TelemetryReplay

    ds, labels, wf = env
    r = wf.diagnose(TelemetryReplay(ds, run_of(ds, labels, "LOCA")), t=600)
    assert wf.last_diagnosis.outcome is DiagnosisOutcome.IDENTIFIED
    assert wf.last_diagnosis.selected.scenario == "LOCA"
    assert r.bundle.decision.effect is PolicyEffect.ALLOW
    assert r.mode.value == "enforce"  # диагностика всегда в ENFORCE, даже если чат в shadow
    assert wf.chat_memory.get_history(limit=1)[-1]["content"] == r.answer


def test_no_event_run_gives_no_anomaly(env):
    from atomic.diagnostics.telemetry import TelemetryReplay

    ds, labels, wf = env
    wf.diagnose(TelemetryReplay(ds, run_of(ds, labels, "MD", "probe")), t=900)
    assert wf.last_diagnosis.outcome is DiagnosisOutcome.NO_ANOMALY


def test_out_of_scope_requires_human(env):
    from atomic.diagnostics.telemetry import TelemetryReplay

    ds, labels, wf = env
    r = wf.diagnose(TelemetryReplay(ds, run_of(ds, labels, "TT", "probe")), t=600)
    assert wf.last_diagnosis.outcome is DiagnosisOutcome.NOVEL
    assert r.bundle.decision.effect is PolicyEffect.REQUIRE_HUMAN
    assert r.answer.startswith("[Требуется оценка оператора")


def test_llm_directive_is_blocked_and_dropped(env):
    from atomic.diagnostics.telemetry import TelemetryReplay

    ds, labels, wf = env
    wf.analyst.llm = FakeLLM("Давление падает [E1]. Необходимо снизить мощность [E1].")
    try:
        r = wf.diagnose(TelemetryReplay(ds, run_of(ds, labels, "SGBTR")), t=420, use_llm=True)
    finally:
        wf.analyst.llm = FakeLLM()
    # пояснение заблокировано (D-041): выпускается только детерминированная часть с пометкой
    rejected = r.response.metadata["narrative_rejected"]
    assert rejected["decision"]["effect"] == "block" and "NO_DIRECTIVES" in rejected["decision"]["violated_rule_ids"]
    assert r.bundle.decision.effect is PolicyEffect.ALLOW
    assert "снизить мощность" not in r.answer and "Пояснение языковой модели не выпущено" in r.answer
    assert not any(c["claim_id"].startswith("narrative") for c in r.response.metadata["verification"]["claims"])


def test_telemetry_replay_resolves_only_own_windows(env):
    from atomic.diagnostics.telemetry import TelemetryReplay

    ds, labels, _ = env
    src = TelemetryReplay(ds, run_of(ds, labels, "LOCA"))
    w = src.window_at(300)
    assert w.t_end <= 300
    assert src.resolves(w.window_id) and src.resolves(f"{w.window_id}#P")
    assert not src.resolves(f"{w.window_id}#NOPE") and not src.resolves("deadbeef#P")


def test_variable_power_loader():
    from atomic.diagnostics.variable_power import VP_DIR, load_vp_runs, segment_start, vp_episode, vp_windows
    from atomic.diagnostics.nppad import NPPADDataset

    if not (VP_DIR / "manifest.json").exists():
        pytest.skip("Variable_Power не сконвертирован")
    ds = NPPADDataset()
    runs = {r.scenario: r for r in load_vp_runs()}
    assert len(runs) == 12
    loca = runs["LOCA_100_to_80"]
    ws = vp_windows(loca, ds)
    assert all(abs((w.t_end - w.t_start) - 110) < 1e-6 for w in ws)  # окна не пересекают разрыв во времени
    ep = vp_episode(loca, ds, upto_t=loca.injection_time_s + 60)
    assert ep.horizon_s <= 60 + 10

    # причинное начало события (D-039, D-042): во время снижения мощности начало — последняя
    # стационарная точка до него; станция не возвращается к 100 % — событие не сбрасывается
    from atomic.diagnostics.pipeline import DiagnosticPipeline

    pipe = DiagnosticPipeline.load(with_analogues=False)
    norm = runs["NORM_100_to_80"]
    on = pipe.onset_track(norm.raw, segment_start(norm))
    i = len(on) - 1
    assert on[i] < i - 30  # к концу записи событие (снижение мощности) всё ещё длится: защёлка
    assert norm.time[on[i]] < 600  # и началось до/в начале снижения (≈ 420–520 с), а не в начале записи
    ep = vp_episode(norm, ds, float(norm.time[i]), onsets=on)
    assert np.allclose(ep.baseline_raw, norm.raw[on[i]])


def test_analogues_prefer_initiating_event():
    from atomic.diagnostics.analogues import AnalogueFinder

    f = AnalogueFinder.load()
    res = f.find("SGATR", k=3)
    assert res and all(a.match_basis == "initiating_event" for a in res)
    ies = {e["event_id"]: e.get("initiating_event") for e in f.events}
    assert all(ies[a.event_id] == "Steam Generator Tube Rupture" for a in res)


def test_window_before_first_end_is_not_served(env):
    from atomic.diagnostics.telemetry import NotEnoughData, TelemetryReplay

    ds, labels, _ = env
    src = TelemetryReplay(ds, run_of(ds, labels, "LOCA"))
    with pytest.raises(NotEnoughData):
        src.window_at(src.windows[0].t_end - 1)  # иначе в ответ попали бы данные из будущего
    assert src.window_at(src.windows[0].t_end).window_id == src.windows[0].window_id


def test_identified_answer_cites_scenario_reference(env):
    """Справка по сценарию (D-043) попадает в ответ как [D1] и разрешается верификатором (RAG_CHUNK)."""
    from atomic.diagnostics.telemetry import TelemetryReplay
    from atomic.memory.scenario_kb import SCENARIO_KB_DIR, seed_scenario_kb

    if not (SCENARIO_KB_DIR / "LOCA.txt").exists():
        pytest.skip("справка по сценариям не построена")
    ds, labels, wf = env
    seed_scenario_kb(wf.vector_memory)
    assert seed_scenario_kb(wf.vector_memory) == 0  # повторный вызов не дублирует
    r = wf.diagnose(TelemetryReplay(ds, run_of(ds, labels, "LOCA")), t=600)
    docs = [c for c in r.response.metadata["verification"]["claims"] if c["attributes"]["type"] == "document"]
    assert docs and r.bundle.decision.effect is PolicyEffect.ALLOW
    assert "[D1]" in r.answer and "dataset/scenarios/LOCA.txt" in r.answer and "LOCAC.txt" not in r.answer
