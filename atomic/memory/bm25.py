"""
BM25 (Okapi) — разреженный поиск по ключевым словам без внешних зависимостей.

Используется:
- для подбора исторических аналогов в базе ETHZ (этап B2);
- как поиск в in-memory VectorMemory, когда PostgreSQL + pgvector недоступен
  (запасной путь, заявленный в отчёте 6-го семестра).

Токены — слова в нижнем регистре (Unicode \\w+, длина ≥ 2) и биграммы соседних
слов: фразовый запрос «steam generator tube» сильнее совпадает с фразой, чем
с разрозненными словами.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str, bigrams: bool = True) -> list[str]:
    words = [w for w in _WORD.findall((text or "").lower()) if len(w) >= 2]
    if not bigrams:
        return words
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75, bigrams: bool = True):
        self.k1, self.b, self.bigrams = k1, b, bigrams
        self._tf: list[Counter[str]] = [Counter(tokenize(d, bigrams)) for d in documents]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._tf)
        self._idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __len__(self) -> int:
        return len(self._tf)

    def scores(self, query: str) -> list[float]:
        q = tokenize(query, self.bigrams)
        out = []
        for tf, length in zip(self._tf, self._len, strict=True):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * (length / self._avg if self._avg else 0.0))
            for t in q:
                f = tf.get(t)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    def top(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Индексы и оценки k лучших документов с ненулевой оценкой."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]

    def matched_terms(self, query: str, index: int) -> list[str]:
        tf = self._tf[index]
        return sorted({t for t in tokenize(query, self.bigrams) if t in tf})
