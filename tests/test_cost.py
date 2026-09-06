"""The cost model and threshold selection."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.config import CostConfig
from backstop.evaluation.cost import (
    candidate_thresholds,
    choose_threshold,
    evaluate_threshold,
    sensitivity,
)


@pytest.fixture()
def scored():
    rng = np.random.default_rng(4)
    n = 20_000
    y = (rng.random(n) < 0.004).astype(int)
    amounts = np.round(np.exp(rng.normal(3.2, 1.3, n)), 2)
    scores = np.clip(rng.beta(1, 60, n) + 0.65 * y, 0, 1)
    return y, scores, amounts


def test_flagging_nothing_loses_every_fraud(scored):
    y, s, amt = scored
    b = evaluate_threshold(y, s, amt, threshold=1.1)
    assert b.alerts == 0
    assert b.true_positives == 0
    assert b.total_cost == pytest.approx(b.do_nothing_cost)


def test_flagging_everything_costs_more_than_the_fraud(scored):
    y, s, amt = scored
    b = evaluate_threshold(y, s, amt, threshold=0.0)
    assert b.alerts == len(y)
    assert b.false_negatives == 0
    assert b.total_cost > b.do_nothing_cost, "reviewing everything cannot be optimal"


def test_the_chosen_threshold_respects_the_alert_budget(scored):
    y, s, amt = scored
    for budget in (0.0005, 0.001, 0.005):
        b = choose_threshold(y, s, amt, max_alert_rate=budget)
        assert b.alert_rate <= budget + 1e-9, f"budget {budget} exceeded"


def test_candidate_thresholds_resolve_the_tail():
    """
    A quantile grid alone cannot see an 0.1% budget.

    With 300 evenly spaced quantile points on 20,000 rows the finest step still
    moves the alert count by ~67, so the search cannot place a 20-alert
    threshold and returns whichever single point happens to be feasible. This
    was a real bug; the enumerated tail is the fix.
    """
    rng = np.random.default_rng(9)
    scores = rng.beta(1, 40, 20_000)
    coarse = np.quantile(scores, np.linspace(0, 1, 300))
    fine = candidate_thresholds(scores, max_alert_rate=0.001)

    def alert_counts(ts):
        return {int((scores >= t).sum()) for t in ts}

    budget = 20
    assert not any(c <= budget for c in alert_counts(coarse) if c > 1), (
        "the coarse grid should not be able to hit the budget"
    )
    assert sum(1 for c in alert_counts(fine) if 0 < c <= budget) >= 10


def test_savings_rate_is_bounded_by_the_oracle(scored):
    y, s, amt = scored
    b = choose_threshold(y, s, amt, max_alert_rate=0.01)
    assert b.savings_rate <= 1.0 + 1e-9


def test_a_higher_review_cost_never_widens_the_net(scored):
    y, s, amt = scored
    cheap = choose_threshold(
        y, s, amt, cost=CostConfig(review_cost=1.0), max_alert_rate=0.05
    )
    dear = choose_threshold(
        y, s, amt, cost=CostConfig(review_cost=100.0), max_alert_rate=0.05
    )
    assert dear.alerts <= cheap.alerts


def test_sensitivity_covers_the_grid(scored):
    y, s, amt = scored
    rows = sensitivity(y, s, amt, review_costs=(1.0, 10.0), frictions=(0.0, 0.1))
    assert len(rows) == 4
    assert all({"threshold", "alert_rate", "recall", "savings"} <= set(r) for r in rows)


def test_mismatched_lengths_raise(scored):
    y, s, amt = scored
    with pytest.raises(ValueError, match="same length"):
        evaluate_threshold(y, s, amt[:-1], 0.5)
