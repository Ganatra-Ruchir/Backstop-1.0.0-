"""
Which frauds the model misses, and whether the misses have a shape.

A single recall number says a third of fraud got through. It does not say
whether that third is random — in which case the model is simply as good as
this feature set allows — or concentrated, in which case there is a specific
gap and a specific thing to do about it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from backstop.config import CONFIG
from backstop.models import registry


def band(amount: float) -> str:
    for edge, label in [(1, "0-1"), (10, "1-10"), (50, "10-50"),
                        (200, "50-200"), (1000, "200-1k")]:
        if amount < edge:
            return label
    return "1k+"


def main() -> None:
    _, manifest = registry.load()
    t = manifest.threshold
    d = np.load(CONFIG.artifact_dir / "test_predictions.npz")

    df = pd.DataFrame(
        {"y": d["y"], "score": d["score"], "amount": d["amount"], "time": d["time"]}
    )
    df["hour"] = (df.time / 3600) % 24
    df["flagged"] = df.score >= t
    df["band"] = df.amount.map(band)

    fraud = df[df.y == 1]
    caught, missed = fraud[fraud.flagged], fraud[~fraud.flagged]

    print(f"threshold {t:.6f} — {len(caught)} caught, {len(missed)} missed "
          f"of {len(fraud)} frauds in the test window\n")

    print("recall by amount band")
    order = ["0-1", "1-10", "10-50", "50-200", "200-1k", "1k+"]
    rows = []
    for b in order:
        sub = fraud[fraud.band == b]
        if not len(sub):
            continue
        rec = sub.flagged.mean()
        exposure = sub.amount.sum()
        recovered = sub[sub.flagged].amount.sum()
        rows.append({"band": b, "frauds": len(sub), "recall": rec,
                     "exposure": exposure, "recovered": recovered})
        print(f"  {b:>7}  n={len(sub):>3}  recall={rec:>5.2f}  "
              f"exposure EUR {exposure:>9,.0f}  recovered EUR {recovered:>9,.0f}")

    print("\nvalue-weighted recall "
          f"{caught.amount.sum() / fraud.amount.sum():.3f} "
          f"vs count-weighted {len(caught) / len(fraud):.3f}")

    print("\nthe misses, by score")
    print(f"  median score of a miss   {missed.score.median():.5f}")
    print(f"  median score of a catch  {caught.score.median():.5f}")
    near = int((missed.score > t / 10).sum())
    print(f"  {near} of {len(missed)} misses scored within a factor of ten of the "
          f"threshold — the rest were not close")

    print("\nrecall by time of day")
    for lo, hi, label in [(0, 8, "night 00-08"), (8, 16, "day 08-16"),
                          (16, 24, "evening 16-24")]:
        sub = fraud[(fraud.hour >= lo) & (fraud.hour < hi)]
        if len(sub):
            print(f"  {label:<14} n={len(sub):>3}  recall={sub.flagged.mean():.2f}")

    print("\nthe five most expensive misses")
    for _, r in missed.nlargest(5, "amount").iterrows():
        print(f"  EUR {r.amount:>9,.2f}  score {r.score:.6f}  at {r.hour:>5.1f} h")

    (CONFIG.artifact_dir / "error_analysis.json").write_text(
        json.dumps(
            {
                "threshold": t,
                "caught": len(caught),
                "missed": len(missed),
                "value_weighted_recall": float(caught.amount.sum() / fraud.amount.sum()),
                "count_weighted_recall": float(len(caught) / len(fraud)),
                "by_band": rows,
            },
            indent=2, default=float,
        )
    )


if __name__ == "__main__":
    main()
