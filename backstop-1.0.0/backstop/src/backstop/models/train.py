"""
Cross-validation over time, and what it produces.

Each candidate is fitted once per expanding fold and scored on the block that
follows it. Two things come out:

* Per-fold metrics, whose *spread* matters as much as their mean. A model whose
  average precision swings from 0.55 to 0.85 across folds is not a 0.70 model;
  it is a model whose performance depends on which two days you caught.
* Pooled out-of-fold predictions, used to place the decision threshold. Pooling
  is what makes the threshold estimable: any single fold has too few frauds.

Latency is measured here rather than asserted, because a model that scores in
40 ms cannot sit in a card authorisation path whatever its recall is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backstop.config import CONFIG
from backstop.data.splits import Fold
from backstop.evaluation.metrics import ScoreReport, score_report
from backstop.models.zoo import Candidate


@dataclass
class FoldResult:
    fold: int
    report: ScoreReport
    fit_seconds: float
    valid_idx: np.ndarray
    scores: np.ndarray


@dataclass
class CVResult:
    """Everything one candidate produced across the whole development window."""

    name: str
    note: str
    tags: tuple[str, ...]
    folds: list[FoldResult] = field(default_factory=list)
    oof_scores: np.ndarray | None = None
    oof_mask: np.ndarray | None = None
    fit_seconds_total: float = 0.0
    predict_micros_per_row: float = float("nan")
    model_bytes: int = 0

    # -- aggregates ------------------------------------------------------
    def _stat(self, attr: str) -> tuple[float, float]:
        vals = [getattr(f.report, attr) for f in self.folds]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    @property
    def ap_mean(self) -> float:
        return self._stat("average_precision")[0]

    @property
    def ap_std(self) -> float:
        return self._stat("average_precision")[1]

    @property
    def recall_budget_mean(self) -> float:
        return self._stat("recall_at_budget")[0]

    @property
    def brier_mean(self) -> float:
        return self._stat("brier")[0]

    def row(self) -> dict:
        return {
            "model": self.name,
            "ap_mean": self.ap_mean,
            "ap_std": self.ap_std,
            "ap_min": min((f.report.average_precision for f in self.folds), default=float("nan")),
            "recall_at_budget": self.recall_budget_mean,
            "roc_auc": self._stat("roc_auc")[0],
            "brier": self.brier_mean,
            "fit_seconds": self.fit_seconds_total,
            "predict_us_per_row": self.predict_micros_per_row,
            "model_kb": self.model_bytes / 1024,
            "note": self.note,
        }


def _measure_latency(pipeline, X: pd.DataFrame, *, n_rows: int = 1, repeats: int = 60) -> float:
    """
    Microseconds to score one transaction, single row at a time.

    Batch throughput is the wrong measurement for this system: an authorisation
    decision arrives alone and has to be answered before the terminal times out.
    A model that scores 50,000 rows per second in a batch can still take 30 ms
    on a single row once per-call overhead is counted.
    """
    sample = X.iloc[:n_rows]
    pipeline.predict_proba(sample)  # warm caches and any lazy imports
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        pipeline.predict_proba(sample)
        timings.append((time.perf_counter() - start) * 1e6)
    return float(np.median(timings) / n_rows)


def _model_size(pipeline) -> int:
    import io
    import pickle

    buf = io.BytesIO()
    pickle.dump(pipeline, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.tell()


def cross_validate(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[Fold],
    *,
    budget: float | None = None,
    measure_latency: bool = True,
) -> CVResult:
    """Fit and score one candidate across every fold."""
    budget = CONFIG.serving.alert_budget if budget is None else budget
    result = CVResult(name=candidate.name, note=candidate.note, tags=candidate.tags)

    oof = np.full(len(y), np.nan, dtype=float)
    last_pipeline = None

    for fold in folds:
        X_tr, y_tr = X.iloc[fold.train_idx], y.iloc[fold.train_idx]
        X_va, y_va = X.iloc[fold.valid_idx], y.iloc[fold.valid_idx]

        pipeline = candidate.build()
        started = time.perf_counter()
        pipeline.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - started

        scores = pipeline.predict_proba(X_va)[:, 1]
        # Later folds overwrite earlier ones where validation blocks overlap;
        # they were fitted on strictly more data, so their prediction is the
        # one closer to what a deployed model would have said.
        oof[fold.valid_idx] = scores

        result.folds.append(
            FoldResult(
                fold=fold.index,
                report=score_report(y_va, scores, budget=budget),
                fit_seconds=elapsed,
                valid_idx=fold.valid_idx,
                scores=scores,
            )
        )
        result.fit_seconds_total += elapsed
        last_pipeline = pipeline

    result.oof_scores = oof
    result.oof_mask = ~np.isnan(oof)
    if last_pipeline is not None:
        result.model_bytes = _model_size(last_pipeline)
        if measure_latency:
            result.predict_micros_per_row = _measure_latency(last_pipeline, X)
    return result


def comparison_table(results: list[CVResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.row() for r in results])
    return df.sort_values("ap_mean", ascending=False).reset_index(drop=True)


def render_table(df: pd.DataFrame) -> str:
    show = df.copy()
    show["AP"] = show.apply(lambda r: f"{r.ap_mean:.4f} ±{r.ap_std:.3f}", axis=1)
    show["recall@0.1%"] = show.recall_at_budget.map("{:.3f}".format)
    show["Brier"] = show.brier.map("{:.5f}".format)
    show["fit s"] = show.fit_seconds.map("{:.1f}".format)
    show["µs/row"] = show.predict_us_per_row.map("{:.0f}".format)
    show["KB"] = show.model_kb.map("{:,.0f}".format)
    cols = ["model", "AP", "recall@0.1%", "Brier", "fit s", "µs/row", "KB"]
    widths = {c: max(len(c), *(len(str(v)) for v in show[c])) for c in cols}
    head = "  ".join(c.ljust(widths[c]) for c in cols)
    line = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join(
        "  ".join(str(row[c]).ljust(widths[c]) for c in cols)
        for _, row in show.iterrows()
    )
    return f"{head}\n{line}\n{body}"
