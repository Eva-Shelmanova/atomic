#!/usr/bin/env python3
"""
Сравнение двух прогонов benchmark (воспроизводимость, требование 13, этап E).

    uv run python scripts/compare_benchmarks.py A.json B.json [--out docs/test-results/<файл>.json]

Метрики и паспорт версий должны совпасть полностью; задержка, время работы и дата прогона
зависят от машины и в сравнение не входят.
"""

import argparse
import json
import sys
from pathlib import Path

VOLATILE = {"created_at", "latency", "runtime_s", "reproducibility"}


def diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: есть только в {'B' if k not in a else 'A'}")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
        return out
    return [] if a == b else [f"{path}: {a!r} ≠ {b!r}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.a, args.b))
    d = diff(a, b)
    for line in d[:50]:
        print(line)
    print("совпадают" if not d else f"расхождений: {len(d)}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "a": Path(args.a).name, "b": Path(args.b).name, "ignored": sorted(VOLATILE),
            "identical": not d, "differences": d,
            "versions_a": a.get("versions"), "latency_a": a.get("latency"), "latency_b": b.get("latency"),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.exit(0 if not d else 1)


if __name__ == "__main__":
    main()
