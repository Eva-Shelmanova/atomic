"""
Причинное определение начала события по потоку телеметрии (D-039, D-042).

Точка стационарна, если ни один наблюдаемый недрейфующий канал не изменился больше порога
проверки отклонения за предыдущие lookback шагов. Начало события — последняя стационарная
точка перед текущим отклонением. Событие «защёлкивается»: оно заканчивается, только когда станция
простояла в стационаре не меньше reset_steps шагов подряд И вернулась к состоянию до события
(проверка отклонения относительно него молчит). Новое стационарное состояние после аварии
(например, после срабатывания защиты) событие не сбрасывает. Все функции используют только прошлое.
"""

from __future__ import annotations

import numpy as np

from atomic.diagnostics.novelty import AnomalyGate


def stationary_points(raw: np.ndarray, gate: AnomalyGate, start: int = 0, lookback: int = 6) -> np.ndarray:
    """(T,) bool. Точка start (начало записи или непрерывного участка) считается стационарной."""
    st = np.zeros(len(raw), dtype=bool)
    st[start] = True
    for i in range(start + 1, len(raw)):
        st[i] = all(not gate.check(raw[i], raw[j])[0] for j in range(max(start, i - lookback), i))
    return st


def onset_track(
    stationary: np.ndarray, start: int = 0, reset_steps: int = 0,
    raw: np.ndarray | None = None, gate: AnomalyGate | None = None,
) -> np.ndarray:
    """(T,) int: начало события для каждой точки (причинно).

    reset_steps = 0 — без защёлки: начало равно последней стационарной точке (D-039).
    reset_steps > 0 — защёлка (D-042): событие сбрасывается после reset_steps стационарных точек подряд,
    а если переданы raw и gate — ещё и только при возврате к состоянию до события.
    """
    out = np.full(len(stationary), start, dtype=int)
    onset, in_event, quiet = start, False, reset_steps  # начало записи считается установившимся стационаром
    for i in range(start, len(stationary)):
        quiet = quiet + 1 if stationary[i] else 0
        if in_event and quiet >= max(reset_steps, 1):
            back = raw is None or gate is None or reset_steps == 0 or not gate.check(raw[i], raw[onset])[0]
            in_event = not back
        if not in_event:
            if stationary[i]:
                onset = i
            else:
                in_event = True
        out[i] = onset
    return out
