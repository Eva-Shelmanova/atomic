"""HTTP API диагностики: POST /diagnose (этап C). LLM не вызывается (explain=false)."""

import pytest

from atomic.diagnostics.config import default_config

MODELS = default_config().resolve("models")
READY = (MODELS / "novelty" / "novelty.npz").exists() and default_config().resolve("ethz_events").exists()
pytestmark = pytest.mark.skipif(not READY, reason="модели или база ETHZ не подготовлены")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from atomic.api.server import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def loca_run():
    from atomic.api.server import nppad_dataset
    from atomic.diagnostics.labels import LabelStore

    ds = nppad_dataset()
    labels = LabelStore.load(ds.directory)
    return next(r for r in ds.split("val") if labels.scenario(r) == "LOCA")


def test_diagnose_returns_verified_answer(client, loca_run):
    r = client.post("/diagnose", json={"run_id": loca_run, "t": 600})
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "identified" and body["decision"] == "allow"
    assert body["verification"]["decision"]["applied_pack_id"] == "aes-core-1.1.0"
    assert "LOCA" in body["response"]


def test_unknown_run_is_404(client):
    assert client.post("/diagnose", json={"run_id": "deadbeef", "t": 600}).status_code == 404


def test_moment_before_first_window_is_422(client, loca_run):
    r = client.post("/diagnose", json={"run_id": loca_run, "t": 50})
    assert r.status_code == 422 and "первое окно" in r.json()["detail"]
