# Инструкция по использованию atomic

Пошаговое руководство по установке, настройке и запуску AI-агентной системы atomic.

---

## 1. Требования

- **Python 3.10+**
- **[Ollama](https://ollama.com)** — для запуска LLM локально
- **Docker** (опционально) — для PostgreSQL и Langfuse

---

## 2. Установка

### 2.1. uv (менеджер пакетов)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.2. Зависимости проекта

```bash
cd atomic
uv sync
```

### 2.3. Модели Ollama

Скачайте и запустите Ollama, затем загрузите модели:

```bash
ollama pull deepseek-r1
ollama pull qcwind/qwen2.5-7B-instruct-Q4_K_M
ollama pull nomic-embed-text
```

---

## 3. База данных (PostgreSQL + pgvector)

### Вариант A: Docker (рекомендуется)

```bash
docker compose up -d
```

PostgreSQL будет доступен на `localhost:5432` с учётными данными `atomic:atomic`, БД `atomic`.

### Вариант B: Без PostgreSQL (in-memory)

Для быстрого старта без Docker задайте переменную окружения:

```bash
export ATOMIC_USE_PGVECTOR=false
```

### Загрузка документов в RAG

Если используете PostgreSQL:

```bash
uv run python scripts/load_dataset.py
```

Документы из `dataset/*.txt` загружаются **чанками** (topic/entity-aware): сначала границы по разделам/нумерации, затем разбиение по сходству эмбеддингов соседних предложений.

Опционально: `ATOMIC_CHUNK_SIM_THRESHOLD` (по умолчанию 0.72), `ATOMIC_CHUNK_MAX_CHARS`, `ATOMIC_CHUNK_MIN_CHARS`, `ATOMIC_CHUNK_DISABLE_SEMANTIC=true` — только структурные границы.

---

## 4. Конфигурация (.env)

Создайте файл `.env` в корне проекта:

```env
# Langfuse (self-hosted или Cloud) — опционально
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_BASE_URL="http://localhost:3000"   # self-hosted
# LANGFUSE_BASE_URL="https://cloud.langfuse.com"  # Cloud

# PostgreSQL (если не стандартный)
# ATOMIC_PG_VECTOR_DSN="postgresql://atomic:atomic@localhost:5432/atomic"

# Ollama (если не localhost:11434)
# OLLAMA_HOST="http://localhost:11434"

# Три демо-фразы в векторную БД при старте CLI/API (идемпотентно, без дублей по тексту)
# ATOMIC_SEED_DEMO=true
```

---

## 5. Langfuse (трейсинг, опционально)

### Self-hosted Langfuse

```bash
cd langfuse
docker compose up -d
```

Интерфейс: http://localhost:3000

В `.env` укажите:
```
LANGFUSE_BASE_URL="http://localhost:3000"
```

### Langfuse Cloud

Зарегистрируйтесь на [cloud.langfuse.com](https://cloud.langfuse.com), создайте проект и скопируйте ключи в `.env`:

```
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
```

> **Примечание:** atomic и Langfuse используют разные docker-compose. PostgreSQL atomic — порт 5432, Langfuse Postgres — порт 5433. Конфликтов нет.

---

## 6. Запуск

### CLI (интерактивный режим)

```bash
uv run python main.py
```

Примеры запросов:
- `найди информацию об atomic` — поиск в базе знаний (Retriever)
- `выполни код print(1+1)` — выполнение кода (Command)
- `проанализируй архитектуру` — анализ и рассуждения (Analyst)

Выход: `exit`, `quit` или `q`.

### API-сервер

```bash
uv run python main.py --api
```

Или через uvicorn:

```bash
uv run uvicorn atomic.api.server:app --reload --port 8000
```

API доступен на http://localhost:8000

---

## 7. API

### POST /chat

**Тело запроса:**
```json
{
  "message": "найди информацию об atomic",
  "session_id": "user-123"
}
```

| Поле        | Тип    | Обязательно | Описание                          |
|-------------|--------|-------------|-----------------------------------|
| message     | string | да          | Текст запроса                     |
| session_id  | string | нет         | ID сессии для Langfuse (до 200 символов) |

**Заголовок (альтернатива session_id в body):**
```
X-Session-ID: user-123
```

**Примеры:**

```bash
# Простой запрос
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "привет"}'

# С session_id в body
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "привет", "session_id": "chat-abc"}'

# С session_id в заголовке
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: chat-abc" \
  -d '{"message": "привет"}'
```

**Ответ:**
```json
{
  "response": "Текст ответа агента"
}
```

### GET /health

Проверка доступности сервиса:
```bash
curl http://localhost:8000/health
```

---

## 8. Переменные окружения

| Переменная              | Описание                    | По умолчанию                          |
|-------------------------|-----------------------------|---------------------------------------|
| ATOMIC_MODEL_ROUTER     | Модель роутера              | qcwind/qwen2.5-7B-instruct-Q4_K_M     |
| ATOMIC_MODEL_RETRIEVER  | Модель RAG                  | deepseek-r1                           |
| ATOMIC_MODEL_COMMAND    | Модель Command-агента       | deepseek-r1                           |
| ATOMIC_MODEL_ANALYST    | Модель Analyst               | qcwind/qwen2.5-7B-instruct-Q4_K_M     |
| ATOMIC_MODEL_EMBEDDING  | Модель эмбеддингов          | nomic-embed-text                      |
| OLLAMA_HOST             | URL Ollama                  | http://localhost:11434                |
| ATOMIC_PG_VECTOR_DSN    | DSN PostgreSQL              | postgresql://atomic:atomic@localhost:5432/atomic |
| ATOMIC_USE_PGVECTOR      | Использовать pgvector       | true                                  |
| ATOMIC_SEED_DEMO         | Демо-строки в RAG при старте (без дублей) | false              |
| LANGFUSE_SECRET_KEY      | Langfuse secret key         | —                                     |
| LANGFUSE_PUBLIC_KEY      | Langfuse public key         | —                                     |
| LANGFUSE_BASE_URL        | Langfuse URL                | https://cloud.langfuse.com             |

---

## 9. Типы запросов и агенты

| Intent    | Примеры запросов                    | Агент     |
|-----------|--------------------------------------|-----------|
| Retrieval | "найди информацию", "как запустить"  | Retriever |
| Command   | "выполни код print(1+1)"             | Command   |
| Analysis  | "проанализируй", "объясни"           | Analyst   |
| General   | Любой другой вопрос                  | Analyst   |

---

## 10. Тесты

```bash
# Быстрые тесты (без LLM-as-a-Judge)
uv run pytest tests/test_atomic.py -m "not slow" -v

# Полные тесты (требуют Ollama, 5–10 мин)
uv run pytest tests/test_atomic.py -v
```

---

## 11. Устранение неполадок

| Проблема                    | Решение                                              |
|-----------------------------|------------------------------------------------------|
| Ошибка подключения к Ollama | Проверьте `ollama serve` и `OLLAMA_HOST`             |
| Ошибка PostgreSQL           | `docker compose up -d` в корне проекта               |
| Langfuse не показывает трассы | Проверьте ключи в `.env`, Langfuse должен быть запущен |
| Конфликт портов 5432        | atomic и Langfuse используют разные порты (5432/5433) |
