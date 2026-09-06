"""
What a random split is worth on this dataset.

Almost every public notebook on this data reports a random 80/20 split. The
number that comes out is higher than the one this project reports, and the gap
is not a modelling difference — it is the difference between grading a model on
the future and grading it on a shuffled version of the past.

Three configurations are trained, identical apart from how the rows were split:

  random            shuffle everything, hold out 26%
  random-with-dupes the same, but leaving the 1,081 duplicate rows in
  temporal          train on the first 74% of the window, test on the rest

Nothing here is used to train the shipped model. It exists so the README can
state the size of the gap instead of asserting that one exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import average_precision_score, roc_auc_score

from backstop.config import CONFIG
from backstop.data.loader import deduplicate, load_raw
from backstop.data.splits import (
    random_split_for_comparison,
    temporal_split,
)
from backstop.evaluation.metrics import recall_at_budget
from backstop.models.zoo import by_name

TARGET = "Class"


def fit_and_score(train, test, *, pos_weight: float) -> dict:
    cand = by_name("xgboost", pos_weight=pos_weight)
    pipe = cand.build()
    pipe.fit(train.drop(columns=[TARGET]), train[TARGET])
    scores = pipe.predict_proba(test.drop(columns=[TARGET]))[:, 1]
    y = test[TARGET].to_numpy()
    return {
        "n_train": len(train),
        "n_test": len(test),
        "test_frauds": int(y.sum()),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "recall_at_budget": float(
            recall_at_budget(y, scores, CONFIG.serving.alert_budget)[0]
        ),
    }


def main() -> None:
    raw = load_raw()
    clean, dropped = deduplicate(raw)
    pos_weight = float((clean[TARGET] == 0).sum() / (clean[TARGET] == 1).sum())

    # The temporal split holds out the final 20% of the window; match the test
    # fraction in the random splits so the comparison is like for like.
    ds = temporal_split(clean)
    test_fraction = len(ds.test) / len(clean)

    results = {}

    tr, te = random_split_for_comparison(clean, test_size=test_fraction, seed=CONFIG.seed)
    results["random"] = fit_and_score(tr, te, pos_weight=pos_weight)

    tr, te = random_split_for_comparison(raw, test_size=test_fraction, seed=CONFIG.seed)
    results["random-with-dupes"] = fit_and_score(tr, te, pos_weight=pos_weight)

    results["temporal"] = fit_and_score(ds.dev, ds.test, pos_weight=pos_weight)

    print(f"{'split':<20} {'AP':>8} {'ROC-AUC':>9} {'recall@0.1%':>12} {'test frauds':>12}")
    print("-" * 66)
    for name in ("random-with-dupes", "random", "temporal"):
        r = results[name]
        print(f"{name:<20} {r['average_precision']:>8.4f} {r['roc_auc']:>9.4f} "
              f"{r['recall_at_budget']:>12.3f} {r['test_frauds']:>12,}")

    gap = results["random"]["average_precision"] - results["temporal"]["average_precision"]
    dupe_gap = (
        results["random-with-dupes"]["average_precision"]
        - results["random"]["average_precision"]
    )
    print(f"\nshuffling time is worth {gap:+.4f} average precision")
    print(f"leaving duplicates in is worth a further {dupe_gap:+.4f}")
    print(f"({dropped:,} duplicate rows exist in the raw file)")

    (CONFIG.artifact_dir / "leakage_demo.json").write_text(
        json.dumps(
            {"results": results, "temporal_gap": gap, "duplicate_gap": dupe_gap},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
