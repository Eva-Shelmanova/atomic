#!/usr/bin/env python3
"""
Загрузка документов из dataset/ в векторную БД atomic_documents.

Требования: Ollama (nomic-embed-text), PostgreSQL (docker compose up -d).

Запуск: uv run python scripts/load_dataset.py
"""

import os
from pathlib import Path

# Включаем PgVector
os.environ.setdefault("ATOMIC_USE_PGVECTOR", "true")

# Загружаем .env
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


def main():
    from atomic.memory.pgvector_memory import PgVectorMemory

    memory = PgVectorMemory()
    loaded = 0

    for path in sorted(DATASET_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        memory.add(content, metadata={"source": path.name})
        loaded += 1
        print(f"  + {path.name}")

    print(f"\nЗагружено {loaded} документов в atomic_documents.")


if __name__ == "__main__":
    main()
