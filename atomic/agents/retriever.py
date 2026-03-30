"""
RetrieverAgent — RAG слой.

Pipeline: Query → Embedding → Vector Search → Top-K docs → Context Builder → LLM
"""

from atomic.agents.base import AgentResponse, AgentType, BaseAgent
from atomic.config import MODEL_RETRIEVER
from atomic.llm.client import LLMClient
from atomic.router.agent import Intent


class RetrieverAgent(BaseAgent):
    """
    Агент поиска по знаниям.
    Vector DB: PostgreSQL + pgvector.
    LLM: синтез ответа на основе найденных документов.
    """

    agent_type = AgentType.RETRIEVER

    RETRIEVER_PROMPT = """На основе найденных документов ответь на вопрос пользователя.
Если в документах нет релевантной информации — честно скажи об этом.

Документы:
{context}

Вопрос: {query}

Ответ:"""

    def __init__(self, vector_memory=None, llm_client: LLMClient | None = None):
        self.vector_memory = vector_memory
        self.llm = llm_client or LLMClient(model=MODEL_RETRIEVER)

    def can_handle(self, query: str, context: dict) -> bool:
        return context.get("intent") == Intent.RETRIEVAL

    def process(self, query: str, context: dict) -> AgentResponse:
        if not self.can_handle(query, context):
            return AgentResponse(
                content="",
                agent_type=self.agent_type,
                handled=False,
            )

        docs = []
        if self.vector_memory:
            docs = self.vector_memory.search(query, top_k=5)

        retrieved = "\n\n".join(d.get("content", str(d)) for d in docs) if docs else "Нет релевантных документов в базе знаний."

        prompt = self.RETRIEVER_PROMPT.format(context=retrieved, query=query)
        content = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )

        return AgentResponse(
            content=content or "Не удалось сформировать ответ.",
            agent_type=self.agent_type,
            handled=True,
            metadata={"docs": docs},
        )
