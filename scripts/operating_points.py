"""
What each review capacity buys.

The threshold turned out to be insensitive to the cost assumptions — every
combination of review cost and friction in `sensitivity()` picks the same
number, because the alert-rate constraint binds long before the cost trade-off
does. That is worth knowing: it means the operating point is set by how many
transactions a team can review, not by how the costs are priced, and arguing
about the price of an analyst-hour will not move it.

So this is the table that actually informs the decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from backstop.config import CONFIG
from backstop.evaluation.cost import choose_threshold

BUDGETS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]


def main() -> None:
    dev = np.load(CONFIG.artifact_dir / "oof_scores.npz")
    test = np.load(CONFIG.artifact_dir / "test_predictions.npz")

    s = dev["xgboost"]
    mask = ~np.isnan(s)
    y_dev, s_dev, amt_dev = dev["y"][mask], s[mask], dev["amount"][mask]
    y_te, s_te, amt_te = test["y"], test["score"], test["amount"]

    rows = []
    print(f"{'budget':>8} {'threshold':>10} {'alerts':>7} {'recall':>7} {'precision':>10} "
          f"{'saved EUR':>11} {'% of oracle':>12}   (test window)")
    print("-" * 82)
    for b in BUDGETS:
        # The threshold is always chosen on development data; the test window
        # is only ever measured at it.
        picked = choose_threshold(y_dev, s_dev, amt_dev, max_alert_rate=b)
        from backstop.evaluation.cost import evaluate_threshold

        t = evaluate_threshold(y_te, s_te, amt_te, picked.threshold)
        recall = t.true_positives / max(1, t.true_positives + t.false_negatives)
        precision = t.true_positives / max(1, t.alerts)
        rows.append(
            {
                "budget": b,
                "threshold": picked.threshold,
                "test_alerts": t.alerts,
                "test_alert_rate": t.alert_rate,
                "test_recall": recall,
                "test_precision": precision,
                "test_savings": t.savings,
                "test_savings_rate": t.savings_rate,
            }
        )
        print(f"{b:>8.2%} {picked.threshold:>10.6f} {t.alerts:>7,} {recall:>7.3f} "
              f"{precision:>10.3f} {t.savings:>11,.0f} {t.savings_rate:>11.1%}")

    out = CONFIG.artifact_dir / "operating_points.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
