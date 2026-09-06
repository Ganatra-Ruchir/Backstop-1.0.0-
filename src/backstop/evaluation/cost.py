"""
Turning a probability into a decision, and pricing the decision.

A fraud model does not output a decision; it outputs a number between 0 and 1.
Choosing where to cut that number is a business decision, not a modelling one,
and picking the cut that maximises F1 — the usual default — quietly asserts
that a missed €2,000 fraud and a missed €5 fraud cost the same, and that a
false alarm costs exactly as much as a missed fraud. Neither is true.

What is true on this data:

* Missing a fraud costs its amount. The distribution of those amounts is
  heavily skewed: median €15, maximum €25,691. An amount-blind threshold is
  therefore leaving money on the table.
* A false alarm costs an analyst's time, plus some share of a good customer's
  transaction if it is declined outright.
* Catching a fraud still costs the review.

The numbers below are assumptions, and `sensitivity()` exists because they are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backstop.config import CONFIG, CostConfig


@dataclass(frozen=True)
class CostBreakdown:
    threshold: float
    alerts: int
    alert_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    review_cost: float
    fraud_loss: float
    friction_cost: float
    total_cost: float
    #: Cost if nothing were flagged at all — every fraud paid out in full.
    do_nothing_cost: float
    #: Cost if a perfect oracle flagged exactly the frauds and nothing else.
    oracle_cost: float

    @property
    def savings(self) -> float:
        return self.do_nothing_cost - self.total_cost

    @property
    def savings_rate(self) -> float:
        """Share of the achievable saving that this threshold actually captures."""
        head_room = self.do_nothing_cost - self.oracle_cost
        return self.savings / head_room if head_room > 0 else float("nan")

    def render(self) -> str:
        return (
            f"threshold {self.threshold:.6f}: "
            f"{self.alerts:,} alerts ({self.alert_rate * 100:.3f}% of traffic), "
            f"caught {self.true_positives}, missed {self.false_negatives}, "
            f"cost EUR {self.total_cost:,.0f} vs EUR {self.do_nothing_cost:,.0f} "
            f"unmanaged — saves EUR {self.savings:,.0f} "
            f"({self.savings_rate * 100:.1f}% of what an oracle would save)"
        )


def evaluate_threshold(
    y_true,
    y_score,
    amounts,
    threshold: float,
    cost: CostConfig | None = None,
) -> CostBreakdown:
    """Price one operating point."""
    cost = cost or CONFIG.cost
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    amt = np.asarray(amounts, dtype=float).ravel()
    if not (y.shape == s.shape == amt.shape):
        raise ValueError("y_true, y_score and amounts must be the same length")

    flagged = s >= threshold
    tp = flagged & (y == 1)
    fp = flagged & (y == 0)
    fn = ~flagged & (y == 1)

    review = float(flagged.sum()) * cost.review_cost
    # A caught fraud still leaks whatever the recovery rate does not recover.
    caught_leak = float(amt[tp].sum()) * (1.0 - cost.recovery_rate)
    missed = float(amt[fn].sum())
    friction = float(amt[fp].sum()) * cost.false_decline_friction

    return CostBreakdown(
        threshold=float(threshold),
        alerts=int(flagged.sum()),
        alert_rate=float(flagged.mean()),
        true_positives=int(tp.sum()),
        false_positives=int(fp.sum()),
        false_negatives=int(fn.sum()),
        review_cost=review,
        fraud_loss=missed + caught_leak,
        friction_cost=friction,
        total_cost=review + missed + caught_leak + friction,
        do_nothing_cost=float(amt[y == 1].sum()),
        oracle_cost=float(y.sum()) * cost.review_cost
        + float(amt[y == 1].sum()) * (1.0 - cost.recovery_rate),
    )


def candidate_thresholds(
    y_score, *, max_alert_rate: float | None = None, n_coarse: int = 300
) -> np.ndarray:
    """
    Where it is worth evaluating a threshold.

    The decision lives in the extreme upper tail of the score distribution — an
    alert budget of 0.1% means the interesting thresholds are the top ~70 scores
    out of 73,000. A uniform grid, or even an evenly spaced quantile grid, has
    almost no resolution there: with 300 quantile points the finest step still
    moves the alert count by hundreds, so the search cannot see the budget at
    all and returns whatever single point happens to be feasible.

    So the tail is enumerated exactly — every distinct score down to several
    times the budget, each of which moves the alert count by one — and the rest
    of the range gets a coarse grid, which is all it needs.
    """
    s = np.asarray(y_score, dtype=float).ravel()
    n = len(s)
    budget = CONFIG.serving.alert_budget if max_alert_rate is None else max_alert_rate

    # Enumerate generously past the budget so the optimiser can see that the
    # cost is still falling when it hits the constraint.
    tail_size = int(np.clip(np.ceil(budget * n) * 5, 200, max(200, n)))
    tail = np.sort(s)[-min(tail_size, n):]

    coarse = np.quantile(s, np.linspace(0.0, 1.0, n_coarse))
    return np.unique(np.concatenate([coarse, tail]))


def cost_curve(
    y_true,
    y_score,
    amounts,
    *,
    cost: CostConfig | None = None,
    max_alert_rate: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Total cost across candidate thresholds."""
    s = np.asarray(y_score, dtype=float).ravel()
    thresholds = candidate_thresholds(s, max_alert_rate=max_alert_rate)

    costs = np.empty(len(thresholds))
    alert_rates = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        b = evaluate_threshold(y_true, s, amounts, t, cost)
        costs[i] = b.total_cost
        alert_rates[i] = b.alert_rate
    return thresholds, costs, alert_rates


def choose_threshold(
    y_true,
    y_score,
    amounts,
    *,
    cost: CostConfig | None = None,
    max_alert_rate: float | None = None,
) -> CostBreakdown:
    """
    The cheapest threshold that a review team can actually staff.

    Without the alert-rate constraint the optimiser will happily propose
    reviewing 4% of all traffic, because analyst time is cheap relative to the
    fraud it prevents. That is arithmetically correct and operationally
    fictional: there is no queue that absorbs 4% of a card network's volume.
    """
    cost = cost or CONFIG.cost
    max_alert_rate = (
        CONFIG.serving.alert_budget if max_alert_rate is None else max_alert_rate
    )
    thresholds, costs, alert_rates = cost_curve(
        y_true, y_score, amounts, cost=cost, max_alert_rate=max_alert_rate
    )

    feasible = alert_rates <= max_alert_rate
    if not feasible.any():
        # Every candidate exceeds the budget: fall back to the most selective.
        best = int(np.argmin(alert_rates))
    else:
        masked = np.where(feasible, costs, np.inf)
        best = int(np.argmin(masked))
    return evaluate_threshold(y_true, y_score, amounts, thresholds[best], cost)


def sensitivity(
    y_true,
    y_score,
    amounts,
    *,
    review_costs=(1.0, 3.0, 10.0, 30.0),
    frictions=(0.0, 0.05, 0.20),
    max_alert_rate: float | None = None,
) -> list[dict]:
    """
    How much the chosen threshold depends on the cost assumptions.

    If the answer is "hardly at all", the assumptions were not load-bearing and
    the operating point is robust. If it swings wildly, the honest thing is to
    say the model needs real cost data before it can be deployed, and this
    table is how you find out which of the two you are in.
    """
    rows = []
    for rc in review_costs:
        for fr in frictions:
            cfg = CostConfig(review_cost=rc, false_decline_friction=fr)
            b = choose_threshold(
                y_true, y_score, amounts, cost=cfg, max_alert_rate=max_alert_rate
            )
            rows.append(
                {
                    "review_cost": rc,
                    "friction": fr,
                    "threshold": b.threshold,
                    "alert_rate": b.alert_rate,
                    "recall": b.true_positives / max(1, b.true_positives + b.false_negatives),
                    "total_cost": b.total_cost,
                    "savings": b.savings,
                }
            )
    return rows
