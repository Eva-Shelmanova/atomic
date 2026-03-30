"""
LLM клиент — вызовы Ollama для генерации текста.
"""

from typing import Optional

from atomic.config import OLLAMA_BASE_URL


def _trace_llm(model: str, messages: list, output: str) -> None:
    try:
        from atomic.observability.langfuse import observe_llm

        observe_llm(model=model, messages=messages, output=output)
    except Exception:
        pass


class LLMClient:
    """Клиент для chat completion через Ollama API."""

    def __init__(self, model: str, base_url: Optional[str] = None):
        self.model = model.replace("ollama/", "")  # убираем префикс если есть
        self.base_url = base_url or OLLAMA_BASE_URL

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Генерация ответа по списку сообщений.
        messages: [{"role": "user"|"system"|"assistant", "content": "..."}]
        """
        try:
            from ollama import Client

            client = Client(host=self.base_url)
            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            content = response.get("message", {}).get("content", "").strip()
            _trace_llm(self.model, messages, content)
            return content
        except Exception as e:
            return f"[Ошибка LLM: {e}]"
