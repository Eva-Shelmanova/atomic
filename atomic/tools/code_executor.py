"""
Code Executor — выполнение кода в sandbox.

Production: Docker, E2B, Lambda.
Демо: subprocess с ограничениями (опционально).
"""

import hashlib
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple


class CodeExecutor:
    """
    Выполняет Python-код в изолированном процессе.
    Внимание: для production нужен полноценный sandbox (Docker и т.п.).

    Каждый запуск журналируется с id (exec:<хэш>), чтобы ответ CommandAgent мог
    сослаться на конкретный запуск, а верификатор ответа — проверить ссылку.
    """

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.last_run_id: str | None = None

    def has_run(self, run_id: str) -> bool:
        return run_id in self.runs

    def _record(self, code: str, output: str, success: bool) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        run_id = "exec:" + hashlib.sha256(f"{len(self.runs)}|{stamp}|{code}".encode()).hexdigest()[:12]
        self.runs[run_id] = {"code": code, "output": output, "success": success, "at": stamp}
        self.last_run_id = run_id

    def run(self, code: str, timeout: int = 10) -> Tuple[str, bool]:
        """Выполняет код и журналирует запуск. Возвращает (stdout+stderr, success)."""
        output, success = self._run(code, timeout)
        self._record(code, output, success)
        return output, success

    def _run(self, code: str, timeout: int = 10) -> Tuple[str, bool]:
        """
        Выполняет код. Возвращает (stdout+stderr, success).
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                f.write(code)
                path = Path(f.name)

            result = subprocess.run(
                ["python3", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            path.unlink(missing_ok=True)

            out = result.stdout or ""
            err = result.stderr or ""
            combined = (out + "\n" + err).strip() or "(пусто)"
            return combined, result.returncode == 0
        except subprocess.TimeoutExpired:
            return "Timeout: выполнение превысило лимит.", False
        except Exception as e:
            return f"Ошибка: {e}", False
