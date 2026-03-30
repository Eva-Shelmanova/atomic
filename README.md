# atomic — AI Agent System

Архитектура JARVIS-подобного AI-агента: **Router → специализированные агенты → память → ответ**.

📖 **[Инструкция по использованию (USAGE.md)](USAGE.md)** — установка, настройка, API, Langfuse.  
🧩 **[Модули и связи (MODULES.md)](MODULES.md)** — структура пакета `atomic/`, зависимости между слоями.

## Структура проекта

```
atomic/
├── api/              # Interface Layer — точка входа
├── router/           # RouterAgent — оркестратор, intent detection
├── agents/           # Chain of Responsibility
│   ├── retriever     # RAG, поиск в знаниях
│   ├── command       # CodeGen, Execute, Debug, Review
│   └── analyst       # рассуждения, планирование
├── tools/            # code_executor, api_tools, search
├── memory/           # Chat + Vector memory
└── orchestration/    # Workflow engine
```

## Требования

- Python 3.10+
- [Ollama](https://ollama.com) с моделями: deepseek-r1, vikhr-llama3.1-8b-instruct-r-21-09-24.Q4_K_M, nomic-embed-text
- PostgreSQL с расширением [pgvector](https://github.com/pgvector/pgvector)

## Установка

[uv](https://docs.astral.sh/uv/) — менеджер пакетов и запуска:

```bash
# Установка uv (если ещё нет)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Установка зависимостей
uv sync
```

Экспорт в requirements.txt (для pip):
```bash
uv export --no-dev -o requirements.txt
```

Или с pip напрямую:
```bash
pip install -r requirements.txt
```

**PostgreSQL (Docker):**
```bash
docker compose up -d
export ATOMIC_PG_VECTOR_DSN="postgresql://atomic:atomic@localhost:5432/atomic"
```

**Загрузка dataset в RAG:**
```bash
uv run python scripts/load_dataset.py
```

Или локальный PostgreSQL:
```bash
createdb atomic
export ATOMIC_PG_VECTOR_DSN="postgresql://user:pass@localhost/atomic"
```

Переменные окружения (опционально):
| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| ATOMIC_MODEL_ROUTER | Модель для роутера | vikhr-llama3.1-8b-instruct-r-21-09-24.Q4_K_M:latest |
| ATOMIC_MODEL_RETRIEVER | Модель для RAG | deepseek-r1:latest |
| ATOMIC_MODEL_COMMAND | Модель для Command | deepseek-r1:latest |
| ATOMIC_MODEL_ANALYST | Модель для Analyst | vikhr-llama3.1-8b-instruct-r-21-09-24.Q4_K_M:latest |
| ATOMIC_MODEL_EMBEDDING | Модель эмбеддингов | nomic-embed-text:latest |
| OLLAMA_HOST | URL Ollama | http://localhost:11434 |
| ATOMIC_USE_PGVECTOR | Использовать PostgreSQL | true |
| ATOMIC_SEED_DEMO | Вставить 3 демо-строки в RAG при старте (без дубликатов) | false |
| LANGFUSE_SECRET_KEY | Langfuse (трейсинг) | — |
| LANGFUSE_PUBLIC_KEY | Langfuse public key | — |
| LANGFUSE_BASE_URL | Langfuse host | https://cloud.langfuse.com |

## Запуск

**CLI (интерактивный режим):**
```bash
uv run python main.py
```

Без PostgreSQL (in-memory, для быстрого старта):
```bash
ATOMIC_USE_PGVECTOR=false uv run python main.py
```

**API сервер:**
```bash
uv run python main.py --api
# или
uv run uvicorn atomic.api.server:app --reload --port 8000
```

**Управление пакетами:**
```bash
uv add <package>      # добавить зависимость
uv remove <package>   # удалить
uv sync              # установить по uv.lock
uv lock              # обновить lock-файл
```

Эндпоинт: `POST /chat` с телом `{"message": "найди информацию об atomic"}`.

## Примеры запросов

| Intent    | Пример                          | Агент      |
|-----------|----------------------------------|------------|
| Retrieval | "найди информацию об atomic"    | Retriever  |
| Command   | "выполни код print(1+1)"         | Command    |
| Analysis  | "проанализируй архитектуру"      | Analyst    |

## Тесты (DeepEval)

```bash
uv sync
uv run deepeval set-ollama --model=deepseek-r1:latest  # модель для evals
uv run pytest tests/test_atomic.py -v
```

**Быстрые тесты** (без LLM-as-a-Judge, ~1 мин):
```bash
uv run pytest tests/test_atomic.py -m "not slow" -v
```

**Полные тесты** (с AnswerRelevancyMetric): требуют Ollama, могут занять 5–10 мин на Mac.

## Технологический стек (расширение)

- **Orchestration:** LangGraph, Semantic Kernel
- **Memory:** Qdrant, Weaviate, FAISS
- **Routing:** LLM function calling, Semantic Router
- **Tools:** Docker sandbox, API connectors

## Лицензия

MIT
