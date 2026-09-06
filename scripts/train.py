"""
Train the shipped model, choose its threshold, and score the test window once.

The order here is the whole point. The threshold comes from pooled out-of-fold
predictions on the development window; the test window is scored exactly once,
at the end, with everything already fixed. Nothing after the test evaluation
feeds back into a choice made before it.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import average_precision_score

from backstop.config import CONFIG
from backstop.data.loader import deduplicate, file_digest, load_raw
from backstop.data.splits import expanding_folds, temporal_split
from backstop.evaluation.cost import choose_threshold, evaluate_threshold, sensitivity
from backstop.evaluation.metrics import bootstrap_ci, score_report
from backstop.features.pipeline import FEATURE_NAMES
from backstop.models import registry
from backstop.models.train import cross_validate
from backstop.models.zoo import by_name

CHOSEN = "xgboost"


def main() -> None:
    budget = CONFIG.serving.alert_budget

    df, dropped = deduplicate(load_raw())
    ds = temporal_split(df)
    folds = expanding_folds(ds.dev)
    pos_weight = float((ds.y_dev == 0).sum() / max(1, (ds.y_dev == 1).sum()))
    candidate = by_name(CHOSEN, pos_weight=pos_weight)

    print(f"chosen model: {CHOSEN}")
    print(ds.summary())
    print(f"dropped {dropped:,} duplicates before splitting\n")

    # ── 1. out-of-fold predictions, for the threshold ────────────────────
    print("cross-validating for out-of-fold predictions...")
    cv = cross_validate(candidate, ds.X_dev, ds.y_dev, folds, budget=budget)
    mask = cv.oof_mask
    y_oof = ds.y_dev.to_numpy()[mask]
    s_oof = cv.oof_scores[mask]
    amt_oof = ds.X_dev["Amount"].to_numpy()[mask]

    oof_report = score_report(y_oof, s_oof, budget=budget)
    print("  out-of-fold ", oof_report.render())

    # ── 2. threshold, chosen on development data only ────────────────────
    chosen = choose_threshold(y_oof, s_oof, amt_oof, max_alert_rate=budget)
    print("\nthreshold selection (development, pooled out-of-fold)")
    print(" ", chosen.render())

    sens = sensitivity(y_oof, s_oof, amt_oof, max_alert_rate=budget)
    spread = [r["threshold"] for r in sens]
    print(f"  sensitivity: threshold ranges {min(spread):.6f} - {max(spread):.6f} "
          f"across {len(sens)} cost assumptions")

    # ── 3. refit on the whole development window ─────────────────────────
    print("\nrefitting on all development data...")
    pipeline = candidate.build().fit(ds.X_dev, ds.y_dev)

    # ── 4. the single test evaluation ────────────────────────────────────
    test_scores = pipeline.predict_proba(ds.X_test)[:, 1]
    y_test = ds.y_test.to_numpy()
    amt_test = ds.X_test["Amount"].to_numpy()

    test_report = score_report(y_test, test_scores, budget=budget)
    ap, lo, hi = bootstrap_ci(y_test, test_scores, average_precision_score,
                              n_boot=1000, seed=CONFIG.seed)
    test_cost = evaluate_threshold(y_test, test_scores, amt_test, chosen.threshold)

    print("\n" + "=" * 74)
    print("TEST WINDOW — scored once, with the threshold already fixed")
    print("=" * 74)
    print(" ", test_report.render())
    print(f"  average precision {ap:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(" ", test_cost.render())
    print(f"  caught {test_cost.true_positives}/{test_cost.true_positives + test_cost.false_negatives} "
          f"frauds, {test_cost.false_positives} false alarms")

    # ── 5. save the artifact ─────────────────────────────────────────────
    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    metrics = {
        "development_oof": oof_report.to_dict(),
        "test": test_report.to_dict(),
        "test_average_precision_ci": [ap, lo, hi],
        "test_cost": {
            k: v for k, v in test_cost.__dict__.items() if isinstance(v, int | float)
        },
        "test_savings": test_cost.savings,
        "test_savings_rate": test_cost.savings_rate,
        "cv_ap_mean": cv.ap_mean,
        "cv_ap_std": cv.ap_std,
        "predict_us_per_row": cv.predict_micros_per_row,
    }
    path = registry.save(
        pipeline,
        version=version,
        model_name=CHOSEN,
        threshold=chosen.threshold,
        threshold_basis=(
            f"minimum expected cost on {len(y_oof):,} pooled out-of-fold development "
            f"predictions ({int(y_oof.sum())} frauds), constrained to an alert rate "
            f"at or below {budget:.3%}"
        ),
        feature_names=FEATURE_NAMES,
        trained_rows=len(ds.dev),
        trained_frauds=int(ds.y_dev.sum()),
        train_window_hours=float(ds.dev_end_time / 3600),
        data_sha256=file_digest(CONFIG.raw_csv),
        metrics=metrics,
        notes=[
            "Probabilities are the model's raw output. Isotonic and Platt "
            "calibration were both fitted on held-out development predictions "
            "and both made the Brier score worse (0.00054 raw vs 0.00074 "
            "isotonic, 0.00064 Platt); isotonic also cost 6 points of average "
            "precision by collapsing distinct scores into ties.",
            "Class reweighting was tested and rejected: scale_pos_weight cost "
            "2 points of AP, and LightGBM's is_unbalance cost 35.",
        ],
    )
    print(f"\nsaved artifact → {path}")

    np.savez_compressed(
        CONFIG.artifact_dir / "test_predictions.npz",
        y=y_test, score=test_scores, amount=amt_test,
        time=ds.X_test["Time"].to_numpy(),
    )
    (CONFIG.artifact_dir / "threshold_sensitivity.json").write_text(json.dumps(sens, indent=2))
    print("saved test_predictions.npz and threshold_sensitivity.json")


if __name__ == "__main__":
    main()
