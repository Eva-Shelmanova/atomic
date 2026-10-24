"""
Контрактные тесты информационного барьера (требование 5, критерий готовности этапа A).

Генератор не должен видеть скрытых каналов, меток сценария, исходных путей
и времени от начала прогона.
"""

import ast
import copy
import dataclasses
from pathlib import Path

import pytest

from atomic.contracts.hypothesis import RestrictedWindow
from atomic.diagnostics.barrier import InformationBarrier
from atomic.diagnostics.channels import default_registry
from atomic.diagnostics.config import PROJECT_ROOT, DiagnosticsConfig, default_config
from atomic.diagnostics.nppad import NPPADDataset, build_processed
from tests.diag_fixtures import SYN_SCENARIOS, synthetic_config, write_synthetic_nppad

FORBIDDEN_FIELDS = {"run_id", "t_start", "t_end", "label", "scenario", "severity", "source_file", "raw_values"}

# Модули рабочего пути генератора: им запрещено импортировать хранилище меток.
GENERATOR_PATH_MODULES = ["atomic/diagnostics/barrier.py", "atomic/diagnostics/features.py", "atomic/diagnostics/generator.py"]


@pytest.fixture
def syn_dataset(tmp_path):
    write_synthetic_nppad(tmp_path)
    cfg = synthetic_config(tmp_path)
    build_processed(config=cfg)
    return cfg, NPPADDataset(config=cfg)


def test_restricted_window_type_has_no_label_fields():
    names = {f.name for f in dataclasses.fields(RestrictedWindow)}
    assert names == {"window_id", "channels", "values"}
    assert not names & FORBIDDEN_FIELDS


def test_barrier_output_contains_only_allowed_channels(syn_dataset):
    cfg, ds = syn_dataset
    barrier = InformationBarrier(config=cfg)
    for run in ds.split("train"):
        for w in ds.windows(run):
            r = barrier.restrict(w)
            assert r.channels == cfg.allowed_channels
            assert r.values.shape == (w.values.shape[0], len(cfg.allowed_channels))
            hidden = set(w.channels) - set(cfg.allowed_channels)
            assert not hidden & set(r.channels)
            # значения совпадают с исходными столбцами разрешённых каналов
            for j, c in enumerate(r.channels):
                assert (r.values[:, j] == w.channel(c)).all()


def test_window_id_does_not_leak_scenario(syn_dataset):
    cfg, ds = syn_dataset
    barrier = InformationBarrier(config=cfg)
    for w in ds.iter_windows("train"):
        r = barrier.restrict(w)
        assert all(s.lower() not in r.window_id.lower() for s in SYN_SCENARIOS)
        assert w.run_id not in r.window_id


def test_config_rejects_label_proxy_channel():
    base = default_config()
    raw = copy.deepcopy(base.raw)
    raw["generator"]["allowed_channels"] = ["P", "WLR"]  # WLR — расход течи (label_proxy)
    with pytest.raises(ValueError, match="label_proxy"):
        InformationBarrier(config=DiagnosticsConfig(raw=raw, path=base.path))


def test_default_config_allowed_channels_valid():
    cfg = default_config()
    default_registry().validate_allowed(cfg.allowed_channels)
    assert len(cfg.allowed_channels) == 6


def test_generator_path_does_not_import_labels():
    for rel in GENERATOR_PATH_MODULES:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue  # модуль ещё не создан (этап B)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert "atomic.diagnostics.labels" not in imported, f"{rel} импортирует метки"


@pytest.mark.skipif(
    not (default_config().resolve("processed") / "manifest.json").exists(),
    reason="реальный NPPAD не подготовлен (scripts/load_nppad.py)",
)
def test_barrier_on_real_nppad_sample():
    ds = NPPADDataset()
    barrier = InformationBarrier()
    from atomic.diagnostics.labels import LabelStore

    labels = LabelStore.load(ds.directory)
    scen = {labels.get(r).scenario for r in labels.run_ids()}
    for run in ds.split("test")[:20]:
        for w in ds.windows(run):
            r = barrier.restrict(w)
            assert r.channels == default_config().allowed_channels
            assert all(s.lower() not in r.window_id.lower() for s in scen)
