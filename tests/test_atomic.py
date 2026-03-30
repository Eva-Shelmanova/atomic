"""
Тесты atomic с DeepEval.

Запуск: deepeval test run tests/test_atomic.py
Или: uv run pytest tests/test_atomic.py -v

Быстрые тесты (без LLM-as-a-Judge): uv run pytest tests/test_atomic.py -m "not slow" -v
"""

import os

import pytest

# In-memory для тестов
os.environ.setdefault("ATOMIC_USE_PGVECTOR", "false")

# Увеличенный timeout для Ollama на Mac (deepseek-r1 очень медленный)
# Для AnswerRelevancyMetric лучше использовать быструю модель (см. EVAL_MODEL ниже)
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")

# Быстрая модель для LLM-as-a-Judge (qwen быстрее deepseek-r1 на Mac)
EVAL_MODEL = os.getenv("DEEPEVAL_EVAL_MODEL", "qcwind/qwen2.5-7B-instruct-Q4_K_M:latest")


@pytest.mark.slow
def test_retrieval_answer_relevancy(workflow):
    """Retriever: ответ должен быть релевантен запросу."""
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.models import OllamaModel
    from deepeval.test_case import LLMTestCase

    model = OllamaModel(model=EVAL_MODEL, base_url="http://localhost:11434")
    metric = AnswerRelevancyMetric(model=model, threshold=0.5)

    query = "найди информацию об atomic"
    actual_output = workflow.ask(query)

    test_case = LLMTestCase(input=query, actual_output=actual_output)
    assert_test(test_case, [metric])


def test_command_code_execution(workflow):
    """Command: выполнение кода должно вернуть корректный результат (детерминированная проверка)."""
    query = "выполни код print(1+1)"
    actual_output = workflow.ask(query)

    assert "2" in actual_output, f"Ожидается результат 2, получено: {actual_output[:200]}"
    assert "ошибк" not in actual_output.lower() and "error" not in actual_output.lower()


@pytest.mark.slow
def test_analyst_answer_relevancy(workflow):
    """Analyst: ответ должен быть релевантен запросу."""
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.models import OllamaModel
    from deepeval.test_case import LLMTestCase

    model = OllamaModel(model=EVAL_MODEL, base_url="http://localhost:11434")
    metric = AnswerRelevancyMetric(model=model, threshold=0.5)

    query = "проанализируй архитектуру AI-агентов"
    actual_output = workflow.ask(query)

    test_case = LLMTestCase(input=query, actual_output=actual_output)
    assert_test(test_case, [metric])


def test_retrieval_returns_context(workflow):
    """Retriever: ответ должен содержать информацию из базы знаний."""
    query = "найди информацию об atomic"
    actual_output = workflow.ask(query)

    assert "atomic" in actual_output.lower()
    assert len(actual_output) > 20
