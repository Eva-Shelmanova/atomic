"""
API — точка входа (FastAPI).

Функции: приём запроса, контекст, rate limit (заглушка).
"""

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header
from pydantic import BaseModel

from atomic.memory.chat import ChatMemory
from atomic.memory.demo_seed import seed_demo_vector_memory
from atomic.memory.factory import create_vector_memory
from atomic.orchestration.workflow import AtomicWorkflow


# Глобальный workflow (в production — per-user/session)
workflow: AtomicWorkflow | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global workflow
    workflow = AtomicWorkflow(
        chat_memory=ChatMemory(),
        vector_memory=create_vector_memory(),
    )
    seed_demo_vector_memory(workflow.vector_memory)
    yield
    workflow = None


app = FastAPI(title="atomic API", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # для Langfuse Sessions (группировка трасс)


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
) -> ChatResponse:
    """Эндпоинт чата. session_id — из body или заголовка X-Session-ID."""
    if not workflow:
        return ChatResponse(response="Сервис не инициализирован.")
    session_id = req.session_id or x_session_id
    return ChatResponse(response=workflow.ask(req.message, session_id=session_id))


@app.get("/health")
def health():
    return {"status": "ok"}
