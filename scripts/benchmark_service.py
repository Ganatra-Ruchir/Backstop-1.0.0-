"""Latency and correctness of the scoring service, measured against real rows."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from fastapi.testclient import TestClient

from backstop.config import CONFIG
from backstop.data.loader import deduplicate, load_raw
from backstop.data.splits import temporal_split
from backstop.serving.app import app


def payload(row) -> dict:
    body = {"time": float(row["Time"]), "amount": float(row["Amount"])}
    body.update({f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)})
    return body


def percentiles(values: list[float]) -> dict:
    a = np.asarray(values)
    return {
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
        "max": round(float(a.max()), 2),
    }


def main() -> None:
    df, _ = deduplicate(load_raw())
    ds = temporal_split(df)
    rng = np.random.default_rng(CONFIG.seed)

    with TestClient(app) as client:
        health = client.get("/health").json()
        print(f"model {health['model_version']} ({health['model_name']}), "
              f"threshold {health['threshold']:.6f}, "
              f"explainer {'ready' if health['explainer_ready'] else 'unavailable'}\n")

        idx = rng.choice(len(ds.X_test), size=300, replace=False)
        rows = [payload(ds.X_test.iloc[i]) for i in idx]

        for label, explain in (("without explanations", False), ("with explanations", True)):
            for r in rows[:20]:
                client.post(f"/score?explain={str(explain).lower()}", json=r)
            timings = []
            for r in rows:
                start = time.perf_counter()
                resp = client.post(f"/score?explain={str(explain).lower()}", json=r)
                timings.append((time.perf_counter() - start) * 1000)
                assert resp.status_code == 200, resp.text
            p = percentiles(timings)
            budget = CONFIG.serving.latency_budget_ms
            verdict = "within" if p["p99"] <= budget else "OVER"
            print(f"single transaction, {label:<20} "
                  f"p50 {p['p50']:>6.2f} ms  p95 {p['p95']:>6.2f}  p99 {p['p99']:>6.2f}  "
                  f"({verdict} the {budget:.0f} ms budget)")

        for size in (10, 100, 1000):
            batch = [payload(ds.X_test.iloc[i]) for i in rng.choice(len(ds.X_test), size)]
            client.post("/score/batch", json={"transactions": batch})
            start = time.perf_counter()
            resp = client.post("/score/batch", json={"transactions": batch})
            elapsed = (time.perf_counter() - start) * 1000
            assert resp.status_code == 200, resp.text
            print(f"batch of {size:>4}: {elapsed:>8.1f} ms total, "
                  f"{elapsed / size:>6.3f} ms per transaction")

        # Correctness: the service must agree with offline scoring exactly.
        offline = np.load(CONFIG.artifact_dir / "test_predictions.npz")["score"]
        check = rng.choice(len(ds.X_test), size=200, replace=False)
        served = np.array([
            client.post("/score?explain=false", json=payload(ds.X_test.iloc[i])).json()["score"]
            for i in check
        ])
        diff = np.abs(served - offline[check].round(6)).max()
        print(f"\nservice vs offline scores: max absolute difference {diff:.2e}")
        assert diff < 1e-6, "the service is not reproducing the offline model"

        # A known fraud should come back explained.
        fraud_idx = int(np.flatnonzero(ds.y_test.to_numpy() == 1)[0])
        r = client.post("/score", json=payload(ds.X_test.iloc[fraud_idx])).json()
        print(f"\nexample decision: {r['decision']}, score {r['score']:.4f}")
        for reason in r["reasons"][:3]:
            print(f"  {reason['direction']} risk {reason['contribution']:+.3f}  "
                  f"{reason['explanation']}")

        print("\n/metrics:", json.dumps(client.get("/metrics").json(), indent=2)[:420])


if __name__ == "__main__":
    main()
