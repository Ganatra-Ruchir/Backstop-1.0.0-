"""
The scoring service: contract, failure modes, and agreement with offline scoring.

These need a trained model, so they skip on a fresh clone until
`scripts/train.py` has been run.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow

V_KEYS = [f"V{i}" for i in range(1, 29)]


def base_payload(**overrides) -> dict:
    body = {"time": 3600.0, "amount": 42.50}
    body.update(dict.fromkeys(V_KEYS, 0.0))
    body.update(overrides)
    return body


@pytest.fixture(scope="module")
def client(trained_artifact):
    from fastapi.testclient import TestClient

    from backstop.serving.app import app

    with TestClient(app) as c:
        yield c


def test_health_reports_the_loaded_model(client, trained_artifact):
    _, manifest = trained_artifact
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == manifest.version
    assert body["threshold"] == pytest.approx(manifest.threshold)
    assert body["features"] == manifest.n_features


def test_scoring_returns_a_decision_and_a_threshold(client):
    body = client.post("/score", json=base_payload()).json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in {"review", "approve"}
    assert body["score"] >= body["threshold"] or body["decision"] == "approve"


def test_transaction_id_is_echoed_for_tracing(client):
    body = client.post("/score", json=base_payload(transaction_id="abc-123")).json()
    assert body["transaction_id"] == "abc-123"


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("missing components", {"time": 1.0, "amount": 5.0}),
        ("negative amount", base_payload(amount=-1.0)),
        ("absurd amount", base_payload(amount=1e12)),
        ("negative time", base_payload(time=-5.0)),
        ("component out of range", base_payload(V1=5000.0)),
        ("unknown field", base_payload(shipping_country="IN")),
        ("string where a number belongs", base_payload(amount="forty")),
        ("null component", base_payload(V3=None)),
    ],
)
def test_bad_input_is_rejected_not_imputed(client, label, payload):
    """
    A service that quietly imputes a missing feature returns a confident number
    for an input it never saw, and nothing downstream can tell.
    """
    assert client.post("/score", json=payload).status_code == 422, label


def test_empty_batch_is_rejected(client):
    assert client.post("/score/batch", json={"transactions": []}).status_code == 422


def test_oversized_batch_is_rejected(client):
    body = {"transactions": [base_payload() for _ in range(1001)]}
    assert client.post("/score/batch", json=body).status_code == 422


def test_batch_matches_single_scoring(client):
    payloads = [base_payload(amount=a) for a in (1.0, 50.0, 500.0)]
    singles = [client.post("/score?explain=false", json=p).json()["score"] for p in payloads]
    batch = client.post(
        "/score/batch", json={"transactions": payloads, "explain": False}
    ).json()
    assert batch["count"] == 3
    for one, many in zip(singles, batch["decisions"], strict=True):
        assert one == pytest.approx(many["score"], abs=1e-9)


def test_flagged_transactions_come_back_explained(client, real_data):
    """An alert with no reason costs an analyst as much time as no alert."""
    from backstop.data.splits import temporal_split

    ds = temporal_split(real_data)
    scores = np.load(
        __import__("backstop.config", fromlist=["CONFIG"]).CONFIG.artifact_dir
        / "test_predictions.npz"
    )["score"]
    row = ds.X_test.iloc[int(np.argmax(scores))]
    payload = {"time": float(row["Time"]), "amount": float(row["Amount"])}
    payload.update({k: float(row[k]) for k in V_KEYS})

    body = client.post("/score", json=payload).json()
    assert body["decision"] == "review"
    assert body["reasons"], "a flagged transaction must say why"
    assert all(r["direction"] in {"raises", "lowers"} for r in body["reasons"])


def test_service_reproduces_offline_scores(client, real_data):
    """
    The number the service returns must be the number the model was evaluated on.

    A service that reproduces the model to three decimal places is a service
    running a different model.
    """
    from backstop.config import CONFIG
    from backstop.data.splits import temporal_split

    ds = temporal_split(real_data)
    offline = np.load(CONFIG.artifact_dir / "test_predictions.npz")["score"]
    rng = np.random.default_rng(0)
    for i in rng.choice(len(ds.X_test), size=25, replace=False):
        row = ds.X_test.iloc[int(i)]
        payload = {"time": float(row["Time"]), "amount": float(row["Amount"])}
        payload.update({k: float(row[k]) for k in V_KEYS})
        served = client.post("/score?explain=false", json=payload).json()["score"]
        assert served == pytest.approx(round(float(offline[i]), 6), abs=1e-6)


def test_metrics_track_what_was_scored(client):
    before = client.get("/metrics").json()["scored_total"]
    client.post("/score?explain=false", json=base_payload())
    after = client.get("/metrics").json()
    assert after["scored_total"] == before + 1
    assert "latency_ms" in after and "score" in after


def test_single_transaction_latency_stays_inside_the_budget(client):
    import time

    from backstop.config import CONFIG

    payload = base_payload()
    for _ in range(20):
        client.post("/score", json=payload)
    timings = []
    for _ in range(120):
        started = time.perf_counter()
        client.post("/score", json=payload)
        timings.append((time.perf_counter() - started) * 1000)

    p99 = float(np.percentile(timings, 99))
    assert p99 < CONFIG.serving.latency_budget_ms, (
        f"p99 {p99:.1f} ms exceeds the {CONFIG.serving.latency_budget_ms:.0f} ms budget"
    )
