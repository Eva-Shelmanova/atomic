# Модули проекта atomic и их связи

Документ описывает пакет `atomic/`, точки входа, скрипты и поток зависимостей.

---

## Схема связей (упрощённо)

```
                    main.py
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
   run_cli()                    run_api()
         │                           │
         └─────────────┬─────────────┘
                       ▼
              AtomicWorkflow  ◄────────── config (модели, DSN, флаги)
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   RouterAgent   ChatMemory    vector_memory ◄── factory → PgVectorMemory | VectorMemory
         │                           │
         │                    EmbeddingClient + PgVectorStore
         ▼
   RetrieverAgent ──► LLMClient, vector_memory.search
   CommandAgent   ──► LLMClient, CodeExecutor
   AnalystAgent   ──► LLMClient

LLMClient / EmbeddingClient ──► Ollama (внешний)

AtomicWorkflow.ask() ──► observability/langfuse (trace_ask, опционально)
LLMClient / EmbeddingClient ──► observe_llm / observe_embedding (опционально)
```

---

## Корень репозитория (не пакет `atomic`)

| Файл / каталог | Назначение |
|----------------|------------|
| `main.py` | Точка входа: CLI-чат и запуск API через uvicorn. Создаёт `AtomicWorkflow`, вызывает `seed_demo_vector_memory`. |
| `pyproject.toml` / `uv.lock` | Зависимости и сборка пакета `atomic`. |
| `docker-compose.yml` | PostgreSQL + pgvector для RAG. |
| `dataset/` | Исходные `.txt` для `scripts/load_dataset.py`. |
| `scripts/load_dataset.py` | Загрузка датасета в БД с topic/entity-aware чанкингом. |
| `tests/` | Pytest + DeepEval; фикстура `workflow` в `conftest.py`. |
| `langfuse/` | Отдельный docker-compose для self-hosted Langfuse. |
| `README.md`, `USAGE.md` | Документация пользователя. |

---

## `atomic/config.py`

**Роль:** единое чтение переменных окружения.

**Содержит:** имена моделей Ollama (router, retriever, command, analyst, embedding), `OLLAMA_HOST`, Langfuse-ключи, `PG_VECTOR_DSN`, `USE_PGVECTOR`, `SEED_DEMO`.

**Кто импортирует:** почти все слои (`llm`, `embeddings`, `memory`, `router`, `agents`, `observability`).

---

## `atomic/api/` — интерфейс HTTP

| Модуль | Роль | Зависимости |
|--------|------|-------------|
| `server.py` | FastAPI: `lifespan` создаёт глобальный `AtomicWorkflow`, `seed_demo_vector_memory`, эндпоинты `/chat`, `/health`. | `workflow`, `ChatMemory`, `create_vector_memory`, `demo_seed` |

**Связь:** клиенты → HTTP → `workflow.ask()`.

---

## `atomic/orchestration/` — оркестрация

| Модуль | Роль | Зависимости |
|--------|------|-------------|
| `workflow.py` | `AtomicWorkflow`: роутинг, цепочка Retriever → Command → Analyst, `ask()` с Langfuse `trace_ask`. | `RouterAgent`, три агента, `ChatMemory`, `vector_memory`, `CodeExecutor`; опционально `langfuse` |

**Связь:** центральный узел между входом (CLI/API) и агентами.

---

## `atomic/router/`

| Модуль | Роль | Зависимости |
|--------|------|-------------|
| `agent.py` | `RouterAgent`: LLM-классификация intent (`retrieval` / `command` / `analysis` / `general`), `prepare_context` с историей чата. | `LLMClient`, `config.MODEL_ROUTER`, `AgentType` / `Intent` |

**Связь:** выдаёт `context["intent"]` для `can_handle()` у агентов.

---

## `atomic/agents/`

| Модуль | Роль | Зависимости |
|--------|------|-------------|
| `base.py` | `BaseAgent`, `AgentResponse`, `AgentType` — контракт цепочки. | — |
| `retriever.py` | RAG: `vector_memory.search`, промпт, ответ LLM. | `Intent`, `LLMClient`, `config.MODEL_RETRIEVER` |
| `command.py` | Генерация Python-кода LLM, выполнение через `CodeExecutor`. | `Intent`, `LLMClient`, `CodeExecutor` |
| `analyst.py` | Общие вопросы и анализ, fallback. | `LLMClient`, история из `context` |

**Связь:** все наследуют идею `can_handle` / `process`; используются только из `AtomicWorkflow`.

---

## `atomic/memory/` — память

| Модуль | Роль | Зависимости |
|--------|------|-------------|
| `chat.py` | `ChatMemory`: короткая история диалога (deque). | — |
| `vector.py` | `VectorMemory`: in-memory «документы», демо-поиск по подстроке. | — |
| `pgvector_store.py` | SQL: таблица `atomic_documents`, HNSW, `add` / `search` / `content_exists`. | `psycopg2`, `pgvector`, `config` |
| `pgvector_memory.py` | `PgVectorMemory`: эмбеддинг → `PgVectorStore`. | `EmbeddingClient`, `PgVectorStore` |
| `factory.py` | `create_vector_memory()`: PG или in-memory по `USE_PGVECTOR`. | `PgVectorMemory`, `VectorMemory` |
| `demo_seed.py` | Идемпотентная подстановка демо-текстов при `ATOMIC_SEED_DEMO`. | `config.SEED_DEMO`, `has_content` на памяти |

**Связь:** `AtomicWorkflow` держит один экземпляр векторной памяти + `ChatMemory`; Retriever читает только векторную память.

---

## `atomic/llm/` и `atomic/embeddings/`

| Модуль | Роль | Зависимости |
|--------|------|-------------|
| `llm/client.py` | `LLMClient.chat()` → Ollama; опционально `observe_llm`. | `ollama`, `config`, `langfuse` |
| `embeddings/client.py` | `EmbeddingClient.embed` / `embed_batch` → Ollama. | `ollama`, `config`, `langfuse` |

**Связь:** все агенты и роутер используют `LLMClient`; эмбеддинги — `PgVectorMemory` и (косвенно) `load_dataset.py`.

---

## `atomic/tools/`

| Модуль | Роль | Зависимости |
|--------|------|-------------|
| `code_executor.py` | `CodeExecutor.run()`: временный `.py` + `subprocess` (Python 3). | stdlib |

**Связь:** только `CommandAgent` (и `AtomicWorkflow` при передаче в конструктор).

---

## `atomic/observability/`

| Модуль | Роль | Зависимости |
|--------|------|-------------|
| `langfuse.py` | `trace_ask`, `observe_llm`, `observe_embedding` при наличии ключей. | `langfuse` SDK, `config` |

**Связь:** вызывается из `workflow.ask`, `LLMClient`, `EmbeddingClient` (мягко, без падения при отключении).

---

## `atomic/__init__.py`

Обычно версия/пакет; публичный API минимальный — основной сценарий через `main.py` и `atomic.api.server:app`.

---

## Внешние системы

| Система | Для чего |
|---------|----------|
| **Ollama** | LLM и эмбеддинги. |
| **PostgreSQL + pgvector** | Персистентная векторная БД (если не `ATOMIC_USE_PGVECTOR=false`). |
| **Langfuse** | Трейсинг (опционально). |

---

## Краткая таблица «кто кого импортирует»

| Модуль | Основные импорты из проекта |
|--------|-----------------------------|
| `workflow.py` | `agents.*`, `router`, `memory.*`, `tools`, `langfuse` (lazy) |
| `server.py` | `orchestration.workflow`, `memory.*` |
| `retriever.py` | `base`, `router.Intent`, `llm`, `config` |
| `command.py` | `base`, `router.Intent`, `llm`, `tools` |
| `analyst.py` | `base`, `llm`, `config` |
| `router/agent.py` | `llm`, `config`, `agents.base.AgentType` |
| `pgvector_memory.py` | `embeddings`, `pgvector_store` |
| `load_dataset.py` (скрипт) | `embeddings`, `memory.pgvector_memory` |

---

Подробнее про запуск и конфигурацию см. **[USAGE.md](USAGE.md)** и **[README.md](README.md)**.
