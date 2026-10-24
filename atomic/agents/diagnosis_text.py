"""
Текст и утверждения диагностического ответа (этап C, требование 8).

Детерминированная часть строится только из DiagnosticReport: каждое утверждение
ссылается на окно/канал телеметрии, на событие ETHZ или на фрагмент базы знаний
(справочно, только для установленного сценария). Необязательное пояснение LLM
разбирается по предложениям: предложение со ссылками [E…]/[A…]/[D…] опирается на них,
предложение без ссылок — только на текст модели (правило NO_MODEL_ONLY).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from atomic.contracts.hypothesis import DiagnosisOutcome, DiagnosticReport
from atomic.verification.rules import DIRECTIVE_PATTERNS

STATUS_RU = {"confirmed": "подтверждена", "rejected": "отклонена", "uncertain": "не определено"}
CITE = re.compile(r"\[([EAD]\d+)\]")


@dataclass
class DiagnosisText:
    text: str
    claims: list[dict]
    citations: dict[str, dict] = field(default_factory=dict)  # E1/A1/D1 → {"ref_id", "kind", "text"}


def _clip(text: str, limit: int) -> str:
    """Обрезка по границе пункта («; ») или слова, с многоточием."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    cut = cut[: cut.rfind("; ")] if "; " in cut else cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ;,") + "…"


def _ref(ref_id: str, kind: str) -> dict:
    return {"ref_id": ref_id, "kind": kind}


def build_diagnosis_text(report: DiagnosticReport, titles: dict[str, str], t_range: tuple[float, float] | None = None,
                         max_evidence: int = 5, docs: list[dict] | None = None, max_docs: int = 2,
                         excerpt_chars: int = 320) -> DiagnosisText:
    """docs — фрагменты базы знаний ({"id", "content", "metadata"}), найденные по установленному сценарию.
    Фрагменты с указаниями к действию пропускаются (правило NO_DIRECTIVES, инвариант «без рекомендаций»).
    Цитата — пункты списка документа (у справки по сценарию — последовательность событий), иначе начало текста."""
    title = lambda s: f"{s} ({titles.get(s, s)})"  # noqa: E731
    out = report.outcome
    lines: list[str] = []
    claims: list[dict] = []
    cites: dict[str, dict] = {}
    window_ref = _ref(report.window_id, "telemetry")

    def cite(ev) -> str:
        key = f"E{len([k for k in cites if k.startswith('E')]) + 1}"
        cites[key] = {"ref_id": ev.ref_id, "kind": "telemetry", "text": ev.detail}
        return key

    when = f" (окно {t_range[0]:.0f}–{t_range[1]:.0f} с)" if t_range else ""
    sel_v = report.verdict(report.selected.scenario) if report.selected else None
    if out is DiagnosisOutcome.IDENTIFIED:
        head = (f"Вероятна ситуация {title(report.selected.scenario)}: подтверждена независимой проверкой, "
                f"{sel_v.support_ratio:.0%} взвешенных свидетельств «за».")
        diag_ev = [_ref(e.ref_id, "telemetry") for e in sel_v.evidence_for[:8]] or [window_ref]
    elif out is DiagnosisOutcome.AMBIGUOUS:
        head = ("Подтверждено несколько гипотез с близкой поддержкой: " + ", ".join(title(s) for s in report.candidates)
                + ". Различить их по имеющимся данным не удаётся.")
        diag_ev = [_ref(e.ref_id, "telemetry") for s in report.candidates for e in report.verdict(s).evidence_for[:4]] or [window_ref]
    elif out is DiagnosisOutcome.UNCONFIRMED:
        rest = ", ".join(title(s) for s in report.candidates) or "нет"
        head = f"Ни одна гипотеза не подтверждена независимой проверкой. Остаются в рассмотрении: {rest}."
        diag_ev = [_ref(e.ref_id, "telemetry") for v in report.verdicts for e in (*v.evidence_for, *v.evidence_against)[:3]] or [window_ref]
    elif out is DiagnosisOutcome.NOVEL:
        near = report.novelty.nearest_scenario
        head = ("Ситуация не похожа ни на один известный сценарий"
                + (f" (ближайший известный — {title(near)})" if near else "")
                + ". Гипотезы генератора приведены справочно.")
        diag_ev = [_ref(f"{report.window_id}#{c}", "telemetry") for c in report.novelty.deviating_channels[:8]] or [window_ref]
    else:
        head = "Отклонений от исходного состояния не обнаружено."
        diag_ev = [window_ref]
    lines.append(f"Оценка ситуации{when}: {head}")
    claims.append({"claim_id": "diagnosis", "text": head, "evidence": diag_ev,
                   "attributes": {"type": "diagnosis", "outcome": out.value}})

    if out is not DiagnosisOutcome.NO_ANOMALY:
        lines.append("\nГипотезы (генератор по 6 каналам → проверка по всем наблюдаемым каналам):")
        for v in report.verdicts:
            h = v.hypothesis
            keys = [cite(e) for e in v.evidence_for[:max_evidence]] + [cite(e) for e in v.evidence_against[:max_evidence]]
            refs = "".join(f"[{k}]" for k in keys)
            text = (f"{h.rank}. {title(h.scenario)} — уверенность генератора {h.confidence:.0%}; проверка: "
                    f"{STATUS_RU[v.status.value]}, доля подтверждений {v.support_ratio:.0%} "
                    f"(за {len(v.evidence_for)}, против {len(v.evidence_against)})")
            lines.append(f"  {text} {refs}".rstrip())
            ev = [_ref(cites[k]["ref_id"], "telemetry") for k in keys] or [window_ref]
            claims.append({"claim_id": f"hypothesis-{h.rank}", "text": text, "evidence": ev,
                           "attributes": {"type": "hypothesis", "scenario": h.scenario, "status": v.status.value}})
        if cites:
            lines.append("\nСвидетельства:")
            for k, c in cites.items():
                lines.append(f"  [{k}] {c['text']} (ссылка {c['ref_id']})")
        anchor = sel_v or (report.verdicts[0] if report.verdicts else None)
        if anchor and anchor.analogues and out is not DiagnosisOutcome.NOVEL:
            lines.append(f"\nАналогичные события из базы ETHZ для {anchor.hypothesis.scenario} (не предписания):")
            for j, a in enumerate(anchor.analogues, start=1):
                key = f"A{j}"
                cites[key] = {"ref_id": a.event_id, "kind": "historical_event", "text": a.describe()}
                lines.append(f"  [{key}] {a.describe()} ({a.event_id})")
                claims.append({"claim_id": f"analogue-{j}", "text": a.describe(),
                               "evidence": [_ref(a.event_id, "historical_event")], "attributes": {"type": "analogue"}})
        if docs and out is DiagnosisOutcome.IDENTIFIED:
            usable = [d for d in docs if d.get("id") and not DIRECTIVE_PATTERNS.search(d.get("content", ""))][:max_docs]
            if usable:
                lines.append(f"\nСправочные материалы базы знаний по {report.selected.scenario}:")
            for j, d in enumerate(usable, start=1):
                key = f"D{j}"
                bullets = [ln[2:].strip() for ln in d["content"].splitlines() if ln.startswith("- ")]
                excerpt = _clip("; ".join(bullets) if bullets else " ".join(d["content"].split()), excerpt_chars)
                source = (d.get("metadata") or {}).get("source", "база знаний")
                cites[key] = {"ref_id": d["id"], "kind": "rag_chunk", "text": excerpt}
                lines.append(f"  [{key}] «{excerpt}» ({source}, {d['id']})")
                claims.append({"claim_id": f"document-{j}", "text": excerpt,
                               "evidence": [_ref(d["id"], "rag_chunk")], "attributes": {"type": "document"}})
    lines.append("\nСистема только интерпретирует данные; решение о действиях принимает оператор.")
    return DiagnosisText(text="\n".join(lines), claims=claims, citations=cites)


SENTENCE = re.compile(r"(?<=[.!?])\s+")
THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def narrative_claims(narrative: str, citations: dict[str, dict]) -> tuple[str, list[dict], dict]:
    """Разбор пояснения LLM: утверждение на предложение; ссылки [E…]/[A…]/[D…] → свидетельства."""
    text = THINK.sub("", narrative or "").strip()
    claims, stats = [], {"sentences": 0, "cited": 0, "unknown_citations": 0}
    for i, sent in enumerate(s for s in SENTENCE.split(text) if s.strip()):
        stats["sentences"] += 1
        keys = CITE.findall(sent)
        ev = []
        for k in keys:
            if k in citations:
                ev.append(_ref(citations[k]["ref_id"], citations[k]["kind"]))
            else:
                stats["unknown_citations"] += 1
                ev.append(_ref(f"citation:{k}", "telemetry"))  # несуществующая ссылка → REF_RESOLVABLE
        if ev:
            stats["cited"] += 1
        else:
            ev = [_ref("analyst:narrative", "model")]
        claims.append({"claim_id": f"narrative-{i + 1}", "text": sent.strip(), "evidence": ev,
                       "attributes": {"type": "narrative"}})
    return text, claims, stats
