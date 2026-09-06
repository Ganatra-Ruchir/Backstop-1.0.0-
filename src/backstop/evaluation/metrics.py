"""
Metrics for a problem where 99.83% of the answers are "no".

Accuracy is not on this list. A model that predicts "legitimate" for every
transaction is 99.83% accurate on this dataset and worth nothing, which is why
the primary metric here is average precision — the area under the
precision-recall curve — and why every headline number is accompanied by what
it costs at the operating point actually used.

ROC-AUC is reported too, but as a secondary. With 0.17% positives, the false
positive rate moves so little across the useful range of thresholds that
ROC-AUC compresses large practical differences into the third decimal place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass(frozen=True)
class ScoreReport:
    """Everything worth knowing about one set of predicted probabilities."""

    n: int
    positives: int
    average_precision: float
    roc_auc: float
    brier: float
    #: Recall achievable when only `alert_budget` of traffic can be reviewed.
    recall_at_budget: float
    precision_at_budget: float
    alerts_at_budget: int
    #: Recall at the fixed, very low false-positive rates fraud teams operate at.
    recall_at_fpr_001: float
    recall_at_fpr_0001: float

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self, label: str = "") -> str:
        head = f"{label}  " if label else ""
        return (
            f"{head}AP={self.average_precision:.4f}  ROC-AUC={self.roc_auc:.4f}  "
            f"Brier={self.brier:.5f}  "
            f"recall@budget={self.recall_at_budget:.3f} "
            f"(precision {self.precision_at_budget:.3f}, {self.alerts_at_budget:,} alerts)  "
            f"recall@FPR0.1%={self.recall_at_fpr_001:.3f}"
        )


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_score {s.shape}")
    if not np.isfinite(s).all():
        raise ValueError("y_score contains NaN or infinity")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("y_true must be binary")
    return y, s


def recall_at_budget(y_true, y_score, budget: float) -> tuple[float, float, int]:
    """
    What the model catches when a review team can look at `budget` of traffic.

    This is the metric a fraud lead actually asks for: "we can review a
    thousand transactions a day — how many frauds does that buy?" It is the
    operating constraint, and a model with a better PR curve that only reaches
    it at a 5% alert rate loses to a worse one that fits inside the budget.

    Returns (recall, precision, number of alerts).
    """
    y, s = _as_arrays(y_true, y_score)
    positives = int(y.sum())
    if positives == 0:
        return float("nan"), float("nan"), 0

    k = max(1, round(len(y) * budget))
    # argpartition is O(n) where a full sort is O(n log n); at 280k rows scored
    # on every evaluation that difference is worth the extra line.
    top = np.argpartition(-s, k - 1)[:k]
    caught = int(y[top].sum())
    return caught / positives, caught / k, k


def recall_at_fpr(y_true, y_score, max_fpr: float) -> float:
    """Recall at the highest threshold whose false-positive rate stays under `max_fpr`."""
    y, s = _as_arrays(y_true, y_score)
    negatives = int((y == 0).sum())
    positives = int(y.sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    false_positives = np.cumsum(y_sorted == 0)
    true_positives = np.cumsum(y_sorted == 1)
    allowed = false_positives <= max_fpr * negatives
    if not allowed.any():
        return 0.0
    return float(true_positives[allowed][-1] / positives)


def score_report(y_true, y_score, *, budget: float) -> ScoreReport:
    y, s = _as_arrays(y_true, y_score)
    recall_b, precision_b, alerts = recall_at_budget(y, s, budget)
    return ScoreReport(
        n=len(y),
        positives=int(y.sum()),
        average_precision=float(average_precision_score(y, s)),
        roc_auc=float(roc_auc_score(y, s)),
        brier=float(brier_score_loss(y, np.clip(s, 0, 1))),
        recall_at_budget=float(recall_b),
        precision_at_budget=float(precision_b),
        alerts_at_budget=int(alerts),
        recall_at_fpr_001=recall_at_fpr(y, s, 0.001),
        recall_at_fpr_0001=recall_at_fpr(y, s, 0.0001),
    )


def pr_curve(y_true, y_score) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, s = _as_arrays(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y, s)
    return precision, recall, thresholds


def bootstrap_ci(
    y_true,
    y_score,
    statistic,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap interval for a metric.

    With 97 positives in the test window, a point estimate of average precision
    is a number with a lot of sampling noise behind it. Reporting the interval
    is the difference between "0.82" and "0.82, and a rerun on different traffic
    could plausibly give 0.72".

    Resampling is stratified by class, so every replicate keeps the same number
    of positives — otherwise a replicate can draw zero frauds and the metric is
    undefined.
    """
    y, s = _as_arrays(y_true, y_score)
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        values[i] = statistic(y[idx], s[idx])

    point = float(statistic(y, s))
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
