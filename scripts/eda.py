"""
Exploratory analysis, run on development data only.

Every hypothesis below was tested here before anything was added to the feature
pipeline, and one of them was rejected. Doing this on the whole dataset would
mean choosing features using information from the test window, which is a
quieter form of the same leakage the split is designed to prevent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from scipy import stats

from backstop.config import CONFIG
from backstop.data.loader import deduplicate, load_raw, profile
from backstop.data.splits import temporal_split

TARGET = "Class"


def binary_lift(df: pd.DataFrame, column: str) -> dict:
    """Fraud rate with and without a binary flag, plus a chi-square test."""
    base = df[TARGET].mean()
    table = pd.crosstab(df[column], df[TARGET])
    chi2, p, _, _ = stats.chi2_contingency(table)
    groups = df.groupby(column)[TARGET].agg(["size", "sum", "mean"])
    on = groups.loc[1] if 1 in groups.index else None
    return {
        "feature": column,
        "n_flagged": int(on["size"]) if on is not None else 0,
        "frauds_flagged": int(on["sum"]) if on is not None else 0,
        "rate_flagged": float(on["mean"]) if on is not None else float("nan"),
        "base_rate": float(base),
        "lift": float(on["mean"] / base) if on is not None else float("nan"),
        "chi2": float(chi2),
        "p_value": float(p),
        "verdict": "kept" if p < 1e-4 and on is not None and on["mean"] / base > 1.2
                   else "rejected",
    }


def main() -> None:
    raw = load_raw()
    report = profile(raw)
    print(report.render(), "\n")

    df, dropped = deduplicate(raw)
    dev = temporal_split(df).dev.copy()
    print(f"analysis window: {len(dev):,} transactions, "
          f"{int(dev[TARGET].sum())} frauds, development only\n")

    hour = (dev["Time"] / 3600) % 24
    dev["is_night"] = ((hour >= 0) & (hour < 8)).astype(int)
    dev["is_whole_euro"] = (dev["Amount"] % 1 == 0).astype(int)
    dev["is_multiple_of_ten"] = (dev["Amount"] % 10 == 0).astype(int)
    dev["is_zero_amount"] = (dev["Amount"] == 0).astype(int)

    tests = [binary_lift(dev, c) for c in
             ("is_night", "is_zero_amount", "is_whole_euro", "is_multiple_of_ten")]

    print(f"{'hypothesis':<20} {'n':>8} {'frauds':>7} {'rate':>8} {'lift':>6} "
          f"{'p':>10}  verdict")
    print("-" * 72)
    for t in tests:
        print(f"{t['feature']:<20} {t['n_flagged']:>8,} {t['frauds_flagged']:>7} "
              f"{t['rate_flagged']:>7.3%} {t['lift']:>5.2f}x {t['p_value']:>10.2e}  "
              f"{t['verdict']}")

    fraud_amt = dev.loc[dev[TARGET] == 1, "Amount"]
    legit_amt = dev.loc[dev[TARGET] == 0, "Amount"]
    _u_stat, u_p = stats.mannwhitneyu(fraud_amt, legit_amt, alternative="two-sided")
    print(f"\namount: fraud median EUR {fraud_amt.median():.2f} vs legit EUR "
          f"{legit_amt.median():.2f}, but fraud mean EUR {fraud_amt.mean():.2f} vs "
          f"EUR {legit_amt.mean():.2f}")
    print(f"        Mann-Whitney p = {u_p:.3g} — fraud is *smaller* typically and "
          "*larger* in the tail")

    hourly = dev.assign(hour=hour.astype(int)).groupby("hour")[TARGET].agg(
        ["size", "sum", "mean"]
    )

    v_cols = [f"V{i}" for i in range(1, 29)]
    smd = {
        v: float(
            abs(dev.loc[dev[TARGET] == 1, v].mean() - dev.loc[dev[TARGET] == 0, v].mean())
            / dev[v].std()
        )
        for v in v_cols
    }
    top = sorted(smd.items(), key=lambda kv: -kv[1])[:8]
    print("\nstrongest separating components (standardised mean difference):")
    print("  " + ", ".join(f"{k} {v:.1f}" for k, v in top))

    out = {
        "quality": report.to_dict(),
        "duplicates_dropped": dropped,
        "development_rows": len(dev),
        "development_frauds": int(dev[TARGET].sum()),
        "base_rate": float(dev[TARGET].mean()),
        "hypothesis_tests": tests,
        "amount": {
            "fraud_median": float(fraud_amt.median()),
            "legit_median": float(legit_amt.median()),
            "fraud_mean": float(fraud_amt.mean()),
            "legit_mean": float(legit_amt.mean()),
            "mannwhitney_p": float(u_p),
        },
        "hourly": {
            "hour": hourly.index.tolist(),
            "n": hourly["size"].tolist(),
            "frauds": hourly["sum"].astype(int).tolist(),
            "fraud_rate": hourly["mean"].tolist(),
        },
        "component_separation": dict(top),
    }
    CONFIG.artifact_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.artifact_dir / "eda_findings.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CONFIG.artifact_dir / 'eda_findings.json'}")


if __name__ == "__main__":
    main()
