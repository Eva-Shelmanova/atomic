"""Загрузчик и нормализатор NPPAD на синтетическом наборе (требование 2)."""


import numpy as np
import pytest

from atomic.diagnostics.labels import LabelStore
from atomic.diagnostics.nppad import NPPADDataset, build_processed, make_run_id, read_raw_run
from atomic.diagnostics.channels import default_registry
from atomic.diagnostics.splits import stratified_run_split
from tests.diag_fixtures import SYN_SCENARIOS, synthetic_config, write_synthetic_nppad


@pytest.fixture
def processed(tmp_path):
    write_synthetic_nppad(tmp_path)
    cfg = synthetic_config(tmp_path)
    manifest = build_processed(config=cfg)
    return cfg, manifest


def test_columns_reordered_to_registry(tmp_path):
    write_synthetic_nppad(tmp_path)
    run = read_raw_run(tmp_path / "raw" / "AAA" / "2.csv", "AAA", default_registry())
    p = default_registry().names.index("P")
    assert run.data.shape[1] == 96
    assert run.data[0, p] == pytest.approx(155.0)
    assert run.data[1, p] == pytest.approx(155.0 - 0.2)


def test_unexpected_column_rejected(tmp_path):
    write_synthetic_nppad(tmp_path)
    f = tmp_path / "raw" / "AAA" / "1.csv"
    lines = f.read_text().splitlines()
    f.write_text("\n".join([lines[0] + ",SURPRISE"] + [l + ",1" for l in lines[1:]]))
    with pytest.raises(ValueError, match="неожиданные столбцы"):
        read_raw_run(f, "AAA", default_registry())


def test_manifest_and_labels_separate(processed):
    cfg, m = processed
    out = cfg.resolve("processed")
    assert m["n_runs"] == 11 and len(m["channels"]) == 96
    labels = LabelStore.load(out)
    assert len(labels) == 11
    # метки не попадают в runs.npz и манифест (кроме имён сценариев в статистике)
    npz = np.load(out / "runs.npz")
    assert set(npz.files) == {"data", "time", "offsets", "run_ids"}
    assert all(s not in r for r in npz["run_ids"] for s in SYN_SCENARIOS)
    lab = labels.get(make_run_id("AAA", "3"))
    assert lab.scenario == "AAA" and lab.severity == pytest.approx(3.0) and lab.malfunction == 7


def test_split_by_runs_disjoint_and_probe_separate(processed):
    _, m = processed
    s = m["split"]
    sets = [set(s[k]) for k in ("train", "val", "test", "probe")]
    assert sum(len(x) for x in sets) == len(set().union(*sets)) == 11
    assert len(s["probe"]) == 1


def test_normalization_fitted_on_train_only(processed):
    cfg, m = processed
    ds = NPPADDataset(config=cfg)
    train = np.concatenate([ds.run_raw(r)[1] for r in ds.split("train")]).astype(np.float64)
    assert np.allclose(m["normalization"]["mean"], train.mean(axis=0), atol=1e-3)
    z = ds.normalize(train.astype(np.float32))
    assert np.allclose(z.mean(axis=0), 0, atol=1e-3)


def test_windows_shape_ids_and_stride(processed):
    cfg, _ = processed
    ds = NPPADDataset(config=cfg)
    run = ds.split("train")[0]
    wins = list(ds.windows(run))
    assert len(wins) == (40 - 8) // 4 + 1
    assert wins[0].values.shape == (8, 96)
    assert wins[1].t_start - wins[0].t_start == pytest.approx(40.0)
    assert len({w.window_id for w in wins}) == len(wins)
    assert ds.resolve_window(wins[2].window_id).t_start == wins[2].t_start
    assert list(ds.windows(run, length=100)) == []  # прогон короче окна → нет окон


def test_build_is_deterministic(tmp_path):
    write_synthetic_nppad(tmp_path)
    cfg = synthetic_config(tmp_path)
    a = build_processed(config=cfg)
    b = build_processed(config=cfg)
    assert a["split"] == b["split"] and a["normalization"] == b["normalization"]


def test_stratified_split_fractions():
    runs = {f"r{i}": ("X" if i < 10 else "Y") for i in range(20)}
    s = stratified_run_split(runs, (0.6, 0.2, 0.2), seed=1)
    assert len(s["train"]) == 12 and len(s["val"]) == 4 and len(s["test"]) == 4
    assert stratified_run_split(runs, (0.6, 0.2, 0.2), seed=1) == s
    with pytest.raises(ValueError):
        stratified_run_split(runs, (0.5, 0.2, 0.2), seed=1)
