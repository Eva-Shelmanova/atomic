#!/usr/bin/env python3
"""
Загрузка документов из dataset/ в векторную БД atomic_documents.

Чанкинг: topic/entity-aware
  - структурные границы: нумерованные разделы, markdown-заголовки, маркеры списков;
  - семантика: соседние предложения сравниваются по косинусу эмбеддингов;
    при падении сходства — новый чанк (смена темы).

Требования: Ollama (nomic-embed-text), PostgreSQL (docker compose up -d).

Запуск: uv run python scripts/load_dataset.py

Переменные окружения (опционально):
  ATOMIC_CHUNK_SIM_THRESHOLD — порог косинуса соседних предложений (0–1), ниже = новый чанк (по умолчанию 0.72)
  ATOMIC_CHUNK_MAX_CHARS — макс. размер чанка (по умолчанию 2500)
  ATOMIC_CHUNK_MIN_CHARS — мин. размер; меньшие сливаются с соседом (по умолчанию 80)
  ATOMIC_CHUNK_DISABLE_SEMANTIC — если true, только структурные границы + абзацы
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Включаем PgVector
os.environ.setdefault("ATOMIC_USE_PGVECTOR", "true")

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = _ROOT / "dataset"

# --- topic / entity-aware chunking -------------------------------------------------

# Заголовки, нумерация, маркеры разделов (строка целиком или начало)
_SECTION_LINE = re.compile(
    r"^\s*(?:"
    r"#+\s+.+"  # markdown
    r"|\d+(?:\.\d+)*[\.)]\s+.+"  # 1. 1.1) ...
    r"|[•\-*]\s+[А-ЯЁA-Z].{3,}"  # маркированный пункт с заглавной
    r"|[А-ЯЁ][А-ЯЁA-ZА-ЯЁ0-9IVXLC]{1,48}\s*$"  # ВСЕ ЗАГЛАВНЫЕ короткая строка (рубрика)
    r")",
    re.MULTILINE,
)

# Явные «сущности»: даты, URL, email (граница перед ними — часто новый блок)
_ENTITY_PREFIX = re.compile(
    r"(?:^|\n)\s*(?:https?://|www\.|[\w.-]+@[\w.-]+\.\w{2,}|\d{1,2}[./]\d{1,2}[./]\d{2,4})",
    re.IGNORECASE,
)


def _split_structural_blocks(text: str) -> list[str]:
    """
    Режем по строкам-заголовкам и крупным entity-префиксам, затем склеиваем пустое.
    """
    text = text.strip()
    if not text:
        return []

    # Точки вставки: начало строк, похожей на секцию
    lines = text.split("\n")
    blocks: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            chunk = "\n".join(buf).strip()
            if chunk:
                blocks.append(chunk)
            buf = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_section = bool(_SECTION_LINE.match(line)) if stripped else False
        # Новый блок перед этой строкой (кроме самого начала)
        if is_section and buf:
            flush()
        buf.append(line)

    flush()

    # Если не нашли секций — один блок
    if not blocks:
        return [text]

    # Дополнительно: разрез по крупным entity-вставкам внутри длинных блоков
    refined: list[str] = []
    for b in blocks:
        parts = _ENTITY_PREFIX.split(b)
        if len(parts) == 1:
            refined.append(b)
            continue
        cur = parts[0].strip()
        for p in parts[1:]:
            p = p.strip()
            if not p:
                continue
            if cur:
                refined.append(cur)
            cur = p
        if cur:
            refined.append(cur)

    return refined if refined else [text]


def _split_sentences(text: str) -> list[str]:
    """Простое разбиение на предложения (RU/EN)."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    # Не режем после одной буквы (т. д., т.п.)
    parts = re.split(r"(?<=[.!?…])\s+(?=[«\"А-ЯA-ZЁ0-9(])", text)
    out = [p.strip() for p in parts if p.strip()]
    return out if out else [text]


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _merge_small_chunks(chunks: list[str], min_chars: int) -> list[str]:
    if not chunks:
        return []
    merged: list[str] = [chunks[0]]
    for c in chunks[1:]:
        if len(merged[-1]) < min_chars:
            merged[-1] = (merged[-1] + " " + c).strip()
        elif len(c) < min_chars:
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)
    return merged


def _force_max_length(chunk: str, max_chars: int) -> list[str]:
    if len(chunk) <= max_chars:
        return [chunk]
    sents = _split_sentences(chunk)
    if len(sents) <= 1:
        # одно длинное предложение — режем по словам
        out: list[str] = []
        rest = chunk
        while len(rest) > max_chars:
            cut = rest[:max_chars].rsplit(" ", 1)[0] or rest[:max_chars]
            out.append(cut.strip())
            rest = rest[len(cut) :].strip()
        if rest:
            out.append(rest)
        return out
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for s in sents:
        add = len(s) + (1 if buf else 0)
        if size + add > max_chars and buf:
            out.append(" ".join(buf))
            buf = [s]
            size = len(s)
        else:
            buf.append(s)
            size += add
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_topic_entity_aware(
    text: str,
    embedder,
    *,
    sim_threshold: float | None = None,
    max_chars: int | None = None,
    min_chars: int | None = None,
    semantic: bool = True,
) -> list[tuple[str, dict]]:
    """
    Возвращает список (текст_чанка, metadata).
    """
    sim_threshold = sim_threshold if sim_threshold is not None else float(
        os.getenv("ATOMIC_CHUNK_SIM_THRESHOLD", "0.72")
    )
    max_chars = max_chars if max_chars is not None else int(os.getenv("ATOMIC_CHUNK_MAX_CHARS", "2500"))
    min_chars = min_chars if min_chars is not None else int(os.getenv("ATOMIC_CHUNK_MIN_CHARS", "80"))

    disable_sem = os.getenv("ATOMIC_CHUNK_DISABLE_SEMANTIC", "").lower() in ("1", "true", "yes")
    semantic = semantic and not disable_sem

    structural = _split_structural_blocks(text)
    all_chunks: list[str] = []

    for block in structural:
        if not semantic:
            # только абзацы внутри блока
            for para in re.split(r"\n\s*\n", block):
                p = para.strip()
                if p:
                    all_chunks.extend(_force_max_length(p, max_chars))
            continue

        sentences = _split_sentences(block)
        if not sentences:
            continue
        if len(sentences) == 1:
            all_chunks.extend(_force_max_length(sentences[0], max_chars))
            continue

        # Батч эмбеддингов
        embs: list[list[float]] = []
        batch_size = 48
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            embs.extend(embedder.embed_batch(batch))

        current: list[str] = [sentences[0]]
        for i in range(len(sentences) - 1):
            sim = _cosine(embs[i], embs[i + 1])
            merged_len = len(" ".join(current)) + 1 + len(sentences[i + 1])
            if sim < sim_threshold and merged_len > min_chars:
                piece = " ".join(current)
                all_chunks.extend(_force_max_length(piece, max_chars))
                current = [sentences[i + 1]]
            else:
                current.append(sentences[i + 1])
                # жёсткий потолок по символам
                if len(" ".join(current)) > max_chars:
                    piece = " ".join(current[:-1])
                    if piece.strip():
                        all_chunks.extend(_force_max_length(piece.strip(), max_chars))
                    current = [current[-1]]

        if current:
            piece = " ".join(current)
            all_chunks.extend(_force_max_length(piece.strip(), max_chars))

    all_chunks = [c for c in all_chunks if c.strip()]
    all_chunks = _merge_small_chunks(all_chunks, min_chars)

    result: list[tuple[str, dict]] = []
    for idx, ch in enumerate(all_chunks):
        result.append(
            (
                ch,
                {
                    "chunking": "topic_entity_aware",
                    "chunk_index": idx,
                },
            )
        )
    return result


def main() -> None:
    from atomic.embeddings.client import EmbeddingClient
    from atomic.memory.pgvector_memory import PgVectorMemory

    memory = PgVectorMemory()
    embedder = EmbeddingClient()
    loaded_files = 0
    loaded_chunks = 0

    for path in sorted(DATASET_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        loaded_files += 1
        pairs = chunk_topic_entity_aware(content, embedder)
        if not pairs:
            continue
        for chunk_text, extra_meta in pairs:
            meta = {"source": path.name, **extra_meta}
            memory.add(chunk_text, metadata=meta)
            loaded_chunks += 1
        print(f"  + {path.name} → {len(pairs)} чанков")

    print(f"\nФайлов: {loaded_files}, чанков в atomic_documents: {loaded_chunks}.")


if __name__ == "__main__":
    main()
