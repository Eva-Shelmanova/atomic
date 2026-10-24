"""
Разрешение ссылок на артефакты (для правила REF_RESOLVABLE).

Для каждого вида свидетельства регистрируется функция «существует ли артефакт
с таким id». Вид без зарегистрированной функции считается неразрешимым:
непроверяемая ссылка не может обосновать ответ.
"""

from __future__ import annotations

from collections.abc import Callable

from atomic.contracts.verification import EvidenceKind, EvidenceRef

Checker = Callable[[str], bool]


class EvidenceResolver:
    def __init__(self) -> None:
        self._checkers: dict[EvidenceKind, Checker] = {}

    def register(self, kind: EvidenceKind, checker: Checker) -> EvidenceResolver:
        self._checkers[kind] = checker
        return self

    def register_ids(self, kind: EvidenceKind, ids: set[str] | frozenset[str]) -> EvidenceResolver:
        known = frozenset(ids)
        return self.register(kind, lambda ref_id: ref_id in known)

    def supports(self, kind: EvidenceKind) -> bool:
        return kind in self._checkers

    def resolves(self, ref: EvidenceRef) -> bool:
        if ref.kind is EvidenceKind.MODEL_GENERATED:
            return True  # не ссылка на артефакт; ограничивается правилом NO_MODEL_ONLY
        checker = self._checkers.get(ref.kind)
        if checker is None:
            return False
        try:
            return bool(checker(ref.ref_id))
        except Exception:
            return False
