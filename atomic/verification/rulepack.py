"""Загрузка версионируемого пакета правил из config/rulepacks/<pack_id>.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

from atomic.contracts.verification import EvidenceKind, Rule, RulePack
from atomic.diagnostics.config import PROJECT_ROOT

RULEPACK_DIR = PROJECT_ROOT / "config" / "rulepacks"
DEFAULT_PACK_ID = "aes-core-1.1.0"


def load_rulepack(pack_id: str = DEFAULT_PACK_ID, directory: Path | None = None) -> RulePack:
    path = (directory or RULEPACK_DIR) / f"{pack_id}.toml"
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    if raw["pack_id"] != pack_id:
        raise ValueError(f"{path}: pack_id {raw['pack_id']!r} не совпадает с именем файла")
    rules = tuple(Rule(r["rule_id"], r["severity"], r.get("description", "")) for r in raw["rule"])
    trust = {EvidenceKind(k): float(v) for k, v in raw.get("trust_base", {}).items()}
    missing = set(EvidenceKind) - set(trust)
    if missing:
        raise ValueError(f"{path}: не задан trust_base для {sorted(m.value for m in missing)}")
    return RulePack(
        pack_id=pack_id,
        rules=rules,
        min_confidence=float(raw.get("min_confidence", 0.5)),
        params={"trust_base": trust},
    )
