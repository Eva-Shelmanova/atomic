#!/usr/bin/env python3
"""
Точка входа atomic — CLI и запуск API.

Использование:
    python main.py              # интерактивный режим
    python main.py --api        # запуск FastAPI сервера
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import sys
import threading
import uuid

from atomic.orchestration.workflow import AtomicWorkflow
from atomic.memory.chat import ChatMemory
from atomic.memory.demo_seed import seed_demo_vector_memory
from atomic.memory.factory import create_vector_memory
from atomic.tools.code_executor import CodeExecutor


def _thinking_spinner(stop_event: threading.Event) -> None:
    """Спиннер в отдельном потоке: показывает, что агент думает."""
    chars = "|/-\\"
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r  {chars[i % len(chars)]} Думаю... ")
        sys.stdout.flush()
        i += 1
        stop_event.wait(0.08)


def run_cli():
    """Интерактивный чат в консоли."""
    workflow = AtomicWorkflow(
        chat_memory=ChatMemory(),
        vector_memory=create_vector_memory(),
        code_executor=CodeExecutor(),
    )
    # Демо в БД только при ATOMIC_SEED_DEMO=true и без дубликатов по тексту
    seed_demo_vector_memory(workflow.vector_memory)

    # Один session_id на всю CLI-сессию для Langfuse
    session_id = f"cli-{uuid.uuid4().hex[:12]}"

    print("atomic CLI. Введите запрос (или 'exit' для выхода).\n")
    print("Примеры: 'найди информацию об atomic', 'выполни код print(1+1)'\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break

            stop = threading.Event()
            spinner_thread = threading.Thread(target=_thinking_spinner, args=(stop,), daemon=True)
            spinner_thread.start()
            try:
                answer = workflow.ask(query, session_id=session_id)
            finally:
                stop.set()
                spinner_thread.join(timeout=0.2)
                sys.stdout.write("\r" + " " * 25 + "\r")
                sys.stdout.flush()

            print(f"atomic: {answer}\n")
        except KeyboardInterrupt:
            print("\nВыход.")
            break


def run_api():
    """Запуск FastAPI сервера."""
    import uvicorn
    from atomic.api.server import app
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    parser = argparse.ArgumentParser(description="atomic AI Agent")
    parser.add_argument("--api", action="store_true", help="Запустить API сервер")
    args = parser.parse_args()

    if args.api:
        run_api()
    else:
        run_cli()


if __name__ == "__main__":
    main()
