"""
Code Executor — выполнение кода в sandbox.

Production: Docker, E2B, Lambda.
Демо: subprocess с ограничениями (опционально).
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Tuple


class CodeExecutor:
    """
    Выполняет Python-код в изолированном процессе.
    Внимание: для production нужен полноценный sandbox (Docker и т.п.).
    """

    def run(self, code: str, timeout: int = 10) -> Tuple[str, bool]:
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
