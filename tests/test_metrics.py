"""Metrics, including the cases where an ordinary implementation goes wrong."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from backstop.evaluation.metrics import (
    bootstrap_ci,
    recall_at_budget,
    recall_at_fpr,
    score_report,
)


def test_a_perfect_ranker_hits_the_ceiling_the_budget_allows():
    """
    With more positives than the budget can review, perfect recall is
    impossible and the metric must say so rather than report 1.0.
    """
    n, positives = 10_000, 50
    y = np.zeros(n, dtype=int)
    y[:positives] = 1
    scores = y.astype(float) + np.linspace(0, 1e-9, n)

    recall, precision, alerts = recall_at_budget(y, scores, budget=0.002)  # 20 alerts
    assert alerts == 20
    assert recall == pytest.approx(20 / positives)
    assert precision == pytest.approx(1.0)


def test_random_scores_catch_nothing_at_a_strict_fpr():
    rng = np.random.default_rng(0)
    y = (rng.random(50_000) < 0.002).astype(int)
    assert recall_at_fpr(y, rng.random(50_000), 0.0001) < 0.05


def test_recall_at_fpr_is_monotone_in_the_allowance():
    rng = np.random.default_rng(1)
    y = (rng.random(20_000) < 0.01).astype(int)
    s = np.clip(rng.normal(0.4 * y, 0.2), 0, 1)
    strict = recall_at_fpr(y, s, 0.0001)
    loose = recall_at_fpr(y, s, 0.01)
    assert loose >= strict


def test_no_positives_is_nan_not_zero():
    """Undefined and zero are different answers, and conflating them hides bugs."""
    y = np.zeros(100, dtype=int)
    recall, precision, _ = recall_at_budget(y, np.random.random(100), 0.1)
    assert np.isnan(recall) and np.isnan(precision)


def test_shape_and_value_errors_are_caught():
    with pytest.raises(ValueError, match="shape mismatch"):
        score_report([0, 1], [0.1, 0.2, 0.3], budget=0.1)
    with pytest.raises(ValueError, match="NaN"):
        score_report([0, 1], [0.1, np.nan], budget=0.1)
    with pytest.raises(ValueError, match="binary"):
        score_report([0, 2], [0.1, 0.2], budget=0.1)


def test_bootstrap_keeps_the_positive_count_fixed():
    """
    Unstratified resampling can draw zero positives, and average precision on a
    replicate with no positives is undefined — the interval then quietly
    contains NaNs.
    """
    rng = np.random.default_rng(3)
    n = 4000
    y = np.zeros(n, dtype=int)
    y[:12] = 1
    s = np.clip(rng.normal(0.5 * y + 0.05, 0.1), 0, 1)

    point, lo, hi = bootstrap_ci(y, s, average_precision_score, n_boot=200, seed=5)
    assert np.isfinite([point, lo, hi]).all()
    assert lo <= point <= hi


def test_report_is_serialisable(frame):
    rng = np.random.default_rng(2)
    y = frame["Class"].to_numpy()
    s = np.clip(rng.normal(0.5 * y + 0.02, 0.1), 0, 1)
    report = score_report(y, s, budget=0.01)
    payload = report.to_dict()
    assert set(payload) >= {"average_precision", "roc_auc", "brier", "recall_at_budget"}
    assert "AP=" in report.render("x")
