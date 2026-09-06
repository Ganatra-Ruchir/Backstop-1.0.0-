"""Generate every figure the documentation uses, from saved run artifacts."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backstop.config import CONFIG
from backstop.evaluation.figures import (
    BLUES,
    DIM,
    INK,
    INK_2,
    MUTED,
    SERIES,
    STATUS,
    caption,
    strip,
    use_style,
)
from backstop.evaluation.metrics import pr_curve

FIG = CONFIG.figure_dir
ART = CONFIG.artifact_dir
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.relative_to(CONFIG.artifact_dir.parent)}")


# ── 1. precision-recall, out-of-fold ────────────────────────────────────
def pr_curves() -> None:
    d = np.load(ART / "oof_scores.npz")
    y = d["y"]
    shown = [
        ("xgboost", SERIES[0], 2.4, "XGBoost"),
        ("lightgbm", SERIES[1], 1.6, "LightGBM"),
        ("hist-gradient-boosting", SERIES[2], 1.6, "HistGradientBoosting"),
        ("logistic", SERIES[6], 1.6, "Logistic regression"),
        ("amount-rule", DIM, 1.6, "Amount rule (no learning)"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, colour, width, label in shown:
        s = d[key]
        mask = ~np.isnan(s)
        precision, recall, _ = pr_curve(y[mask], s[mask])
        ax.plot(recall, precision, color=colour, linewidth=width, label=label,
                zorder=3 if key == "xgboost" else 2)

    base = float(y[~np.isnan(d["xgboost"])].mean())
    ax.axhline(base, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"no signal at all — {base:.3%} of transactions are fraud",
                xy=(0.30, base), xytext=(0.30, 0.115),
                color=MUTED, fontsize=8.5,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.9})

    ax.set_xlabel("Recall — share of fraud caught")
    ax.set_ylabel("Precision — share of alerts that are fraud")
    ax.set_title("Precision and recall, out-of-fold on the development window")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", bbox_to_anchor=(0.015, 0.02))
    strip(ax)
    caption(fig, "Pooled across four expanding-window folds (190,568 transactions, "
                 "295 frauds). The rule a team writes before it has a model sits on "
                 "the floor.")
    save(fig, "pr_curves")


# ── 2. model comparison, emphasis on the one that shipped ───────────────
def model_comparison() -> None:
    df = pd.read_csv(ART / "model_comparison.csv")
    df = df[~df.model.isin(["prior-baseline"])].sort_values("ap_mean")

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    colours = [SERIES[0] if m == "xgboost" else DIM for m in df.model]
    ax.barh(df.model, df.ap_mean, xerr=df.ap_std, color=colours, height=0.62,
            error_kw={"ecolor": MUTED, "elinewidth": 1.2, "capsize": 3})
    for name, value, sd, colour in zip(df.model, df.ap_mean, df.ap_std, colours,
                                       strict=True):
        ax.text(value + sd + 0.028, name, f"{value:.3f}", va="center", fontsize=9,
                color=INK if colour != DIM else INK_2,
                fontweight="600" if colour != DIM else "normal")

    ax.set_xlabel("Average precision, mean across folds (bars show ±1 SD)")
    ax.set_title("Eleven candidates; the boosted trees are within noise of each other")
    ax.set_xlim(0, 1.0)
    strip(ax, y=False)
    ax.tick_params(axis="y", length=0)
    caption(fig, "The spread matters as much as the mean: random forest reaches a "
                 "similar average with three times the variance, and takes 240 s to "
                 "fit against XGBoost's 10 s.")
    save(fig, "model_comparison")


def _percent_ticks(ax, values) -> None:
    """Label a log axis with the percentages it actually plots, not 10^-1."""
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v:g}%" for v in values])
    ax.minorticks_off()


# ── 3. what each review capacity buys ───────────────────────────────────
def operating_points() -> None:
    rows = json.loads((ART / "operating_points.json").read_text())
    budget = np.array([r["budget"] for r in rows]) * 100
    recall = np.array([r["test_recall"] for r in rows])
    precision = np.array([r["test_precision"] for r in rows])
    savings = np.array([r["test_savings"] for r in rows])
    chosen = CONFIG.serving.alert_budget * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    ax1.plot(budget, recall, color=SERIES[0], marker="o", markersize=6, label="Recall")
    ax1.plot(budget, precision, color=SERIES[1], marker="s", markersize=6,
             label="Precision")
    ax1.axvline(chosen, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)))
    ax1.set_xscale("log")
    _percent_ticks(ax1, budget)
    ax1.set_xlabel("Share of traffic sent to review (log scale)")
    ax1.set_ylabel("Rate")
    ax1.set_title("More review capacity stops paying at 0.2%")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower left")
    strip(ax1)

    ax2.plot(budget, savings, color=SERIES[0], marker="o", markersize=6)
    ax2.axvline(chosen, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)))
    idx = int(np.argmin(np.abs(budget - chosen)))
    ax2.annotate(f"shipped\nEUR {savings[idx]:,.0f}",
                 xy=(budget[idx], savings[idx]), xytext=(budget[idx] * 1.35, savings[idx] * 0.62),
                 color=INK, fontsize=9,
                 arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1})
    ax2.set_xscale("log")
    _percent_ticks(ax2, budget)
    ax2.set_xlabel("Share of traffic sent to review (log scale)")
    ax2.set_ylabel("Fraud loss avoided, EUR")
    ax2.set_title("Value recovered on the held-out test window")
    strip(ax2)

    caption(fig, "Two measures, two panels — never two y-axes on one plot. The "
                 "threshold at each capacity was chosen on development data; only "
                 "the outcome is measured on test.")
    fig.tight_layout()
    save(fig, "operating_points")


# ── 4. does the probability mean anything ───────────────────────────────
def calibration() -> None:
    d = np.load(ART / "test_predictions.npz")
    y, s = d["y"], d["score"]
    order = np.argsort(-s)
    y, s = y[order], s[order]

    edges = [0, 50, 150, 400, 1200, 5000, 20000, len(y)]
    predicted, observed, sizes = [], [], []
    for lo, hi in itertools.pairwise(edges):
        chunk_y, chunk_s = y[lo:hi], s[lo:hi]
        if len(chunk_y) == 0:
            continue
        predicted.append(chunk_s.mean())
        observed.append(chunk_y.mean())
        sizes.append(len(chunk_y))

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    lims = [3e-6, 1.4]
    ax.plot(lims, lims, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
            label="perfect calibration", zorder=1)
    ax.scatter(predicted, observed, s=[30 + 90 * np.log10(n) for n in sizes],
               color=SERIES[0], zorder=3, edgecolor="#fcfcfb", linewidth=1.5,
               label="score band (area = transactions)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Predicted probabilities land close to observed rates")
    ax.legend(loc="upper left")
    strip(ax)
    caption(fig, "Bands by score rank, not equal-width bins: at a 0.13% base rate a "
                 "uniform binning puts 99.9% of the data in one bucket. Isotonic and "
                 "Platt recalibration were both tried and both made the Brier score "
                 "worse.")
    save(fig, "calibration")


# ── 5. what the model looks at ──────────────────────────────────────────
def feature_importance() -> None:
    gi = json.loads((ART / "global_importance.json").read_text())
    items = list(gi.items())[:15][::-1]
    names = [k for k, _ in items]
    values = [v for _, v in items]

    engineered = {"log_amount", "amount_pct_rank", "hour_sin", "hour_cos",
                  "is_night", "is_zero_amount", "is_whole_euro"}
    colours = [SERIES[1] if n in engineered else SERIES[0] for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.barh(names, values, color=colours, height=0.66)
    for n, v in zip(names, values, strict=True):
        ax.text(v + max(values) * 0.015, n, f"{v:.3f}", va="center",
                fontsize=8.5, color=INK_2)

    handles = [
        plt.Line2D([], [], color=SERIES[0], linewidth=7, label="anonymised component"),
        plt.Line2D([], [], color=SERIES[1], linewidth=7, label="engineered here"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_xlabel("Mean |SHAP value| over 4,000 development transactions")
    ax.set_title("Time of day and amount earn their place among the components")
    strip(ax, y=False)
    ax.tick_params(axis="y", length=0)
    caption(fig, "Twenty-eight of the thirty-five inputs are published as principal "
                 "components, so 'V14 is low' is a reproducible signal but not a "
                 "business reason. The features this project built are named.")
    save(fig, "feature_importance")


# ── 6. drift between training and production windows ────────────────────
def drift() -> None:
    df = pd.read_csv(ART / "drift_report.csv").sort_values("psi", ascending=False)
    df = df.head(14)[::-1]

    palette = {"shifted": STATUS["critical"], "watch": STATUS["warning"],
               "stable": DIM, "seasonal": SERIES[6]}
    colours = [palette[v] for v in df.verdict]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.barh(df.feature, df.psi, color=colours, height=0.66)
    top_y = len(df) - 0.35
    for x, label in ((0.10, "watch"), (0.25, "shifted")):
        ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=0)
        ax.text(x, top_y, f" {label} ≥ {x:.2f}", fontsize=8, color=MUTED, va="bottom")

    present = list(dict.fromkeys(df.verdict))
    labels = {"seasonal": "seasonal — excluded from the alarm",
              "shifted": "shifted", "watch": "watch", "stable": "stable"}
    order = [k for k in ("seasonal", "shifted", "watch", "stable") if k in present]
    handles = [
        plt.Line2D([], [], color=palette[k], linewidth=7, label=labels[k]) for k in order
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, -0.015))
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xticks([0, 0.1, 0.25, 1, 5])
    ax.set_xticklabels(["0", "0.10", "0.25", "1", "5"])
    ax.minorticks_off()
    ax.set_ylim(-0.8, len(df) - 0.2)
    ax.set_xlabel("Population stability index (symlog scale)")
    ax.set_title("Inputs moved between the two windows; the decisions did not")
    strip(ax, y=False)
    ax.tick_params(axis="y", length=0)
    caption(fig, "Score-distribution PSI is 0.036 — the model reads different inputs "
                 "and reaches the same distribution of conclusions. The two hour "
                 "features are supposed to differ between two time windows and would "
                 "otherwise drown out every real signal.")
    save(fig, "drift")


# ── 7. the cost of shuffling time ───────────────────────────────────────
def leakage() -> None:
    data = json.loads((ART / "leakage_demo.json").read_text())["results"]
    labels = ["Random split,\nduplicates left in", "Random split,\nduplicates removed",
              "Temporal split\n(what this project reports)"]
    keys = ["random-with-dupes", "random", "temporal"]
    values = [data[k]["average_precision"] for k in keys]
    colours = [STATUS["critical"], STATUS["serious"], SERIES[0]]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars = ax.bar(labels, values, color=colours, width=0.56)
    for bar, v in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.018, f"{v:.3f}",
                ha="center", fontsize=10.5, color=INK, fontweight="600")

    ax.set_ylabel("Average precision on the held-out data")
    ax.set_title("The same model, graded three ways")
    ax.set_ylim(0, 1.0)
    strip(ax, x=False)
    ax.tick_params(axis="x", length=0)
    caption(fig, "Identical model and hyperparameters; only the split differs. "
                 "Shuffling time is worth +0.049 average precision and leaving the "
                 "1,081 duplicate rows in is worth a further +0.023 — 0.072 of "
                 "headline performance that would not survive contact with "
                 "tomorrow's traffic.")
    save(fig, "leakage")


# ── 8. the finding that produced a feature ──────────────────────────────
def fraud_by_hour() -> None:
    stats = json.loads((ART / "eda_findings.json").read_text())
    hours = np.array(stats["hourly"]["hour"])
    rates = np.array(stats["hourly"]["fraud_rate"]) * 100
    base = stats["base_rate"] * 100

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colours = [BLUES[5] if h < 8 else BLUES[2] for h in hours]
    ax.bar(hours, rates, color=colours, width=0.72)
    ax.axhline(base, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(23.4, base * 1.12, f"overall {base:.3f}%", ha="right", fontsize=8.5,
            color=MUTED)
    ax.axvspan(-0.6, 7.6, color=BLUES[0], alpha=0.35, zorder=0)
    ax.text(3.5, max(rates) * 0.92, "00:00 – 08:00\n2.3x the fraud rate",
            ha="center", fontsize=9, color=INK)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraud rate, %")
    ax.set_title("Fraud is concentrated overnight — measured, then used as a feature")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.7, 23.7)
    strip(ax)
    caption(fig, "Development window only; chi-square p = 5.8e-31. The raw Time "
                 "column is discarded and the hour is kept, because 'seconds since "
                 "the file began' does not generalise past these two days.")
    save(fig, "fraud_by_hour")


def main() -> None:
    use_style()
    print("figures:")
    pr_curves()
    model_comparison()
    operating_points()
    calibration()
    feature_importance()
    drift()
    leakage()
    fraud_by_hour()


if __name__ == "__main__":
    main()
