"""
API — точка входа (FastAPI).

Функции: приём запроса, контекст, rate limit (заглушка).
"""

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from atomic.memory.chat import ChatMemory
from atomic.memory.demo_seed import seed_demo_vector_memory
from atomic.memory.scenario_kb import seed_scenario_kb
from atomic.memory.factory import create_vector_memory
from atomic.orchestration.workflow import AtomicWorkflow


# Глобальный workflow (в production — per-user/session)
workflow: AtomicWorkflow | None = None
_dataset = None  # NPPADDataset, загружается при первом запросе /diagnose


def nppad_dataset():
    global _dataset
    if _dataset is None:
        from atomic.diagnostics.nppad import NPPADDataset

        _dataset = NPPADDataset()
    return _dataset


@asynccontextmanager
async def lifespan(app: FastAPI):
    global workflow
    workflow = AtomicWorkflow(
        chat_memory=ChatMemory(),
        vector_memory=create_vector_memory(),
    )
    seed_demo_vector_memory(workflow.vector_memory)
    seed_scenario_kb(workflow.vector_memory)  # справка по сценариям NPPAD (D-043)
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


class DiagnoseRequest(BaseModel):
    run_id: str  # непрозрачный id прогона NPPAD (воспроизведение телеметрии, только чтение)
    t: float  # момент времени от начала записи, с
    explain: bool = False  # добавить пояснение LLM (каждое предложение — со ссылками)
    session_id: str | None = None


class DiagnoseResponse(BaseModel):
    response: str
    decision: str
    outcome: str
    verification: dict


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest) -> DiagnoseResponse:
    """Диагностика по окну телеметрии: гипотезы → проверка → ответ → решение политики (этап C)."""
    from atomic.diagnostics.telemetry import NotEnoughData, TelemetryReplay

    if not workflow:
        return DiagnoseResponse(response="Сервис не инициализирован.", decision="block", outcome="", verification={})
    try:
        source = TelemetryReplay(nppad_dataset(), req.run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="прогон не найден") from None
    try:
        r = workflow.diagnose(source, req.t, use_llm=req.explain, session_id=req.session_id)
    except NotEnoughData as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    return DiagnoseResponse(
        response=r.answer, decision=r.bundle.decision.effect.value,
        # исход берётся из ответа, а не из workflow.last_diagnosis: запросы обрабатываются параллельно
        outcome=r.response.metadata["diagnosis"]["outcome"], verification=r.bundle.to_dict(),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
