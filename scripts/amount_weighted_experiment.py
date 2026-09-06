"""
Does training on money instead of counts catch more money?

The error analysis found value-weighted recall (0.627) below count-weighted
recall (0.670): the model misses a larger share of the euros than of the
transactions. The obvious response is to tell the loss function that a
EUR 1,000 fraud matters more than a EUR 1 fraud, by weighting each fraud
example by its amount.

The obvious response is not always the right one, so it is measured here rather
than assumed, on out-of-fold development predictions only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import average_precision_score

from backstop.config import CONFIG
from backstop.data.loader import deduplicate, load_raw
from backstop.data.splits import expanding_folds, temporal_split
from backstop.evaluation.cost import choose_threshold, evaluate_threshold
from backstop.evaluation.metrics import recall_at_budget
from backstop.models.zoo import by_name

SCHEMES = {
    "none": lambda amt, y: np.ones(len(y)),
    "log-amount": lambda amt, y: np.where(y == 1, 1.0 + np.log1p(amt), 1.0),
    "sqrt-amount": lambda amt, y: np.where(y == 1, 1.0 + np.sqrt(amt), 1.0),
    "amount": lambda amt, y: np.where(y == 1, 1.0 + amt, 1.0),
}


def main() -> None:
    budget = CONFIG.serving.alert_budget
    df, _ = deduplicate(load_raw())
    ds = temporal_split(df)
    folds = expanding_folds(ds.dev)
    pos_weight = float((ds.y_dev == 0).sum() / max(1, (ds.y_dev == 1).sum()))

    amounts = ds.X_dev["Amount"].to_numpy()
    y = ds.y_dev.to_numpy()

    print(f"{'weighting':<14} {'AP':>7} {'recall':>8} {'value-recall':>13} "
          f"{'EUR saved':>11}   (pooled out-of-fold, development only)")
    print("-" * 74)

    for name, scheme in SCHEMES.items():
        oof = np.full(len(y), np.nan)
        for fold in folds:
            cand = by_name("xgboost", pos_weight=pos_weight)
            pipe = cand.build()
            w = scheme(amounts[fold.train_idx], y[fold.train_idx])
            pipe.fit(
                ds.X_dev.iloc[fold.train_idx],
                ds.y_dev.iloc[fold.train_idx],
                model__sample_weight=w,
            )
            oof[fold.valid_idx] = pipe.predict_proba(ds.X_dev.iloc[fold.valid_idx])[:, 1]

        mask = ~np.isnan(oof)
        yy, ss, aa = y[mask], oof[mask], amounts[mask]
        picked = choose_threshold(yy, ss, aa, max_alert_rate=budget)
        b = evaluate_threshold(yy, ss, aa, picked.threshold)

        flagged = ss >= picked.threshold
        fraud_value = aa[yy == 1].sum()
        caught_value = aa[(yy == 1) & flagged].sum()

        print(f"{name:<14} {average_precision_score(yy, ss):>7.4f} "
              f"{recall_at_budget(yy, ss, budget)[0]:>8.3f} "
              f"{caught_value / fraud_value:>13.3f} {b.savings:>11,.0f}")


if __name__ == "__main__":
    main()
