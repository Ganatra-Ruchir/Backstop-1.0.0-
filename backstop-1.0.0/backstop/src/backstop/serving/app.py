"""
The scoring service.

Three things this does that a `predict()` endpoint does not.

*It loads a verified artifact.* The model, its threshold and the metrics it was
accepted on all come from one manifest whose checksum is checked at startup.
The service cannot be running a different model from the one that was
evaluated, and it will refuse to start rather than find out later.

*It answers with a reason.* An alert without an explanation costs an analyst
the same time as no alert at all, so every flagged transaction comes back with
the features that pushed it over the line.

*It measures itself.* Latency percentiles and score distribution are tracked in
process and exposed on `/metrics`, because the first sign of a broken feature
pipeline is usually that the scores all move, not that anything raises.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from backstop.config import RAW_COLUMNS
from backstop.explain.reasons import Explainer
from backstop.models import registry
from backstop.serving.schemas import (
    BatchRequest,
    BatchResponse,
    Decision,
    Health,
    ReasonOut,
    Transaction,
)

log = logging.getLogger("backstop")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INPUT_COLUMNS = [c for c in RAW_COLUMNS if c != "Class"]
V_INPUTS = [c for c in INPUT_COLUMNS if c.startswith("V")]


class Service:
    """Holds the loaded model and the counters, so nothing is a global."""

    def __init__(self) -> None:
        self.pipeline = None
        self.manifest = None
        self.explainer: Explainer | None = None
        self.latencies: deque[float] = deque(maxlen=5000)
        self.scores: deque[float] = deque(maxlen=5000)
        self.scored = 0
        self.flagged = 0

    def load(self, version: str | None = None) -> None:
        self.pipeline, self.manifest = registry.load(version)
        log.info(
            "loaded %s (%s), threshold %.6f, %d features",
            self.manifest.version, self.manifest.model_name,
            self.manifest.threshold, self.manifest.n_features,
        )
        try:
            self.explainer = Explainer(self.pipeline)
        except Exception as exc:
            # Explanations are valuable but not load-bearing: a service that
            # scores without them still protects money, so this degrades
            # rather than refusing to start.
            log.warning("explainer unavailable, scoring without reasons: %s", exc)
            self.explainer = None

    def to_frame(self, transactions: list[Transaction]) -> pd.DataFrame:
        rows = [
            [getattr(t, "time" if c == "Time" else "amount" if c == "Amount" else c)
             for c in INPUT_COLUMNS]
            for t in transactions
        ]
        return pd.DataFrame(rows, columns=INPUT_COLUMNS, dtype=float)

    def raw_matrix(self, transactions: list[Transaction]) -> np.ndarray:
        """
        Model inputs without going through pandas.

        Profiling the single-transaction path found 3.6 of its 6.3 ms inside
        the pandas-based feature transformer, against 0.3 ms in the booster
        itself. This route builds the same feature matrix with numpy and hands
        it to the model directly.
        """
        features = self.pipeline.named_steps["features"]
        time_s = np.fromiter((t.time for t in transactions), dtype=float, count=len(transactions))
        amount = np.fromiter((t.amount for t in transactions), dtype=float, count=len(transactions))
        v = np.array([[getattr(t, c) for c in V_INPUTS] for t in transactions], dtype=float)

        matrix = features.transform_array(time_s, amount, v)
        # Whatever follows the feature step (a scaler, or a no-op) still has to
        # run: skipping it would score through a different pipeline than the
        # one the model was fitted on.
        for _, step in self.pipeline.steps[1:-1]:
            matrix = step.transform(matrix)
        return np.asarray(matrix, dtype=float)

    def record(self, latency_ms: float, scores: np.ndarray, flagged: int) -> None:
        self.latencies.append(latency_ms)
        self.scores.extend(float(s) for s in scores[:100])
        self.scored += len(scores)
        self.flagged += flagged


service = Service()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pinning a version is how a rollback happens: point the environment at the
    # previous artifact and restart, rather than retraining under pressure.
    service.load(os.environ.get("BACKSTOP_MODEL_VERSION") or None)
    yield


app = FastAPI(
    title="Backstop",
    version="1.0.0",
    description=(
        "Card-fraud decisioning. Scores a transaction, decides whether it goes "
        "to review, and says why."
    ),
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "internal error"},
    )


def _decide(
    transactions: list[Transaction], *, explain: bool
) -> tuple[list[Decision], float]:
    if service.pipeline is None or service.manifest is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")

    started = time.perf_counter()
    matrix = service.raw_matrix(transactions)
    scores = service.pipeline[-1].predict_proba(matrix)[:, 1]
    threshold = service.manifest.threshold
    flagged_mask = scores >= threshold

    explanations = None
    if explain and service.explainer is not None:
        # Only explain what was flagged: SHAP is the expensive part, and an
        # approved transaction nobody will look at does not need a reason.
        idx = np.flatnonzero(flagged_mask)
        if len(idx):
            explanations = dict(
                zip(
                    idx.tolist(),
                    service.explainer.explain_matrix(matrix[idx]),
                    strict=True,
                )
            )

    elapsed_ms = (time.perf_counter() - started) * 1000
    per_row = elapsed_ms / max(1, len(transactions))

    decisions = []
    for i, (t, s) in enumerate(zip(transactions, scores, strict=True)):
        reasons: list[ReasonOut] = []
        if explanations and i in explanations:
            reasons = [
                ReasonOut(
                    feature=r.feature,
                    contribution=round(r.contribution, 4),
                    direction=r.direction,
                    explanation=r.text,
                )
                for r in explanations[i].reasons
            ]
        decisions.append(
            Decision(
                transaction_id=t.transaction_id,
                score=round(float(s), 6),
                decision="review" if s >= threshold else "approve",
                threshold=threshold,
                model_version=service.manifest.version,
                reasons=reasons,
                latency_ms=round(per_row, 3),
            )
        )

    service.record(elapsed_ms, scores, int(flagged_mask.sum()))
    return decisions, elapsed_ms


@app.post("/score", response_model=Decision, tags=["scoring"])
def score(transaction: Transaction, explain: bool = True) -> Decision:
    """Score one transaction."""
    decisions, _ = _decide([transaction], explain=explain)
    return decisions[0]


@app.post("/score/batch", response_model=BatchResponse, tags=["scoring"])
def score_batch(body: BatchRequest) -> BatchResponse:
    """Score up to a thousand transactions in one call."""
    decisions, elapsed = _decide(body.transactions, explain=body.explain)
    return BatchResponse(
        decisions=decisions,
        count=len(decisions),
        flagged=sum(d.decision == "review" for d in decisions),
        latency_ms=round(elapsed, 3),
    )


@app.get("/health", response_model=Health, tags=["system"])
def health() -> Health:
    if service.manifest is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
    m = service.manifest
    return Health(
        status="ok",
        model_version=m.version,
        model_name=m.model_name,
        threshold=m.threshold,
        features=m.n_features,
        trained_rows=m.trained_rows,
        explainer_ready=service.explainer is not None,
    )


@app.get("/metrics", tags=["system"])
def metrics() -> dict:
    """
    In-process operational metrics.

    The score percentiles are here for a specific reason: if a feature pipeline
    breaks upstream — a column arrives null, a unit changes — the service keeps
    returning 200s and the only visible symptom is that the score distribution
    moves. Watching p50 and p99 of the score catches that within minutes;
    waiting for labels catches it in weeks.
    """
    lat = np.array(service.latencies, dtype=float)
    sc = np.array(service.scores, dtype=float)
    out: dict = {
        "scored_total": service.scored,
        "flagged_total": service.flagged,
        "flag_rate": service.flagged / service.scored if service.scored else None,
        "model_version": service.manifest.version if service.manifest else None,
    }
    if len(lat):
        out["latency_ms"] = {
            "p50": round(float(np.percentile(lat, 50)), 3),
            "p95": round(float(np.percentile(lat, 95)), 3),
            "p99": round(float(np.percentile(lat, 99)), 3),
            "max": round(float(lat.max()), 3),
        }
    if len(sc):
        out["score"] = {
            "p50": round(float(np.percentile(sc, 50)), 6),
            "p95": round(float(np.percentile(sc, 95)), 6),
            "p99": round(float(np.percentile(sc, 99)), 6),
            "mean": round(float(sc.mean()), 6),
        }
    return out
