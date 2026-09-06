"""Fit every candidate across the expanding folds and print the comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from backstop.config import CONFIG
from backstop.data.loader import deduplicate, load_raw, profile
from backstop.data.splits import expanding_folds, temporal_split
from backstop.models.train import comparison_table, cross_validate, render_table
from backstop.models.zoo import candidates


def main() -> None:
    raw = load_raw()
    print(profile(raw).render(), "\n")

    df, dropped = deduplicate(raw)
    print(f"dropped {dropped:,} duplicate rows before splitting\n")

    ds = temporal_split(df)
    print(ds.summary(), "\n")

    folds = expanding_folds(ds.dev)
    for f in folds:
        print(" ", f.describe(ds.y_dev))
    print()

    y = ds.y_dev
    pos_weight = float((y == 0).sum() / max(1, (y == 1).sum()))
    print(f"class ratio in development: {pos_weight:.0f} negatives per positive\n")

    results = []
    for cand in candidates(pos_weight=pos_weight):
        res = cross_validate(cand, ds.X_dev, y, folds)
        results.append(res)
        print(f"  {cand.name:<24} AP {res.ap_mean:.4f} ±{res.ap_std:.3f}   "
              f"recall@budget {res.recall_budget_mean:.3f}   "
              f"{res.fit_seconds_total:.1f}s")

    table = comparison_table(results)
    print("\n" + render_table(table))

    out = CONFIG.artifact_dir / "model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    per_fold = {
        r.name: [
            {"fold": f.fold, **f.report.to_dict(), "fit_seconds": f.fit_seconds}
            for f in r.folds
        ]
        for r in results
    }
    (CONFIG.artifact_dir / "per_fold_metrics.json").write_text(json.dumps(per_fold, indent=2))

    np.savez_compressed(
        CONFIG.artifact_dir / "oof_scores.npz",
        y=y.to_numpy(),
        amount=ds.X_dev["Amount"].to_numpy(),
        time=ds.X_dev["Time"].to_numpy(),
        **{r.name: r.oof_scores for r in results},
    )
    print(f"\nwrote {out} and oof_scores.npz")


if __name__ == "__main__":
    main()
