"""End-to-end training behaviour, drift, and the registry contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backstop.data.splits import expanding_folds, temporal_split
from backstop.models import registry
from backstop.models.train import comparison_table, cross_validate
from backstop.models.zoo import AmountRule, PriorBaseline, by_name, candidates
from backstop.monitoring.drift import compare, population_stability_index


def test_every_candidate_builds_and_fits(tiny_frame):
    ds = temporal_split(tiny_frame)
    for candidate in candidates(pos_weight=20.0):
        pipe = candidate.build().fit(ds.X_dev, ds.y_dev)
        proba = pipe.predict_proba(ds.X_test)
        assert proba.shape == (len(ds.X_test), 2)
        assert np.isfinite(proba).all()
        assert ((proba >= 0) & (proba <= 1)).all()
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_baseline_predicts_the_training_prior(tiny_frame):
    model = PriorBaseline().fit(tiny_frame, tiny_frame["Class"])
    proba = model.predict_proba(tiny_frame)[:, 1]
    assert np.allclose(proba, tiny_frame["Class"].mean())


def test_a_real_model_beats_both_baselines(tiny_frame):
    """If it cannot beat 'flag the biggest amounts', it is not worth operating."""
    from sklearn.metrics import average_precision_score

    ds = temporal_split(tiny_frame)
    scores = {}
    for name in ("prior-baseline", "amount-rule", "xgboost"):
        pipe = by_name(name, pos_weight=20.0).build().fit(ds.X_dev, ds.y_dev)
        scores[name] = average_precision_score(
            ds.y_test, pipe.predict_proba(ds.X_test)[:, 1]
        )
    assert scores["xgboost"] > scores["amount-rule"]
    assert scores["xgboost"] > scores["prior-baseline"]


def test_amount_rule_ranks_by_amount(tiny_frame):
    from backstop.features.pipeline import build_feature_pipeline

    features = build_feature_pipeline(scale=False).fit(tiny_frame)
    matrix = np.asarray(features.transform(tiny_frame))
    rule = AmountRule().fit(matrix, tiny_frame["Class"])
    scores = rule.predict_proba(matrix)[:, 1]

    # Spearman, not Pearson: the rule is a monotone ranking of a log-normal
    # quantity, so rank agreement is what "ranks by amount" means. Pearson
    # against the raw euro value would be about 0.6 for a perfect ranking,
    # which says something about the amount distribution, not the rule.
    from scipy.stats import spearmanr

    rho = spearmanr(scores, tiny_frame["Amount"].to_numpy()).statistic
    assert rho > 0.99


def test_cross_validation_leaves_out_of_fold_predictions_only(tiny_frame):
    ds = temporal_split(tiny_frame)
    folds = expanding_folds(ds.dev)
    result = cross_validate(
        by_name("logistic", pos_weight=20.0), ds.X_dev, ds.y_dev, folds,
        budget=0.05, measure_latency=False,
    )
    assert result.oof_scores is not None
    covered = np.flatnonzero(result.oof_mask)
    validated = set()
    for fold in folds:
        validated |= set(fold.valid_idx.tolist())
    assert set(covered.tolist()) <= validated, "a row was scored by a model that trained on it"
    assert len(result.folds) == len(folds)


def test_comparison_table_is_sorted_and_complete(tiny_frame):
    ds = temporal_split(tiny_frame)
    folds = expanding_folds(ds.dev)
    results = [
        cross_validate(by_name(n, pos_weight=20.0), ds.X_dev, ds.y_dev, folds,
                       budget=0.05, measure_latency=False)
        for n in ("prior-baseline", "logistic")
    ]
    table = comparison_table(results)
    assert list(table.model) == sorted(
        table.model, key=lambda m: -table.set_index("model").ap_mean[m]
    )
    assert {"ap_mean", "ap_std", "brier", "fit_seconds"} <= set(table.columns)


def test_unknown_model_name_raises():
    with pytest.raises(KeyError, match="unknown model"):
        by_name("does-not-exist", pos_weight=1.0)


# ── drift ───────────────────────────────────────────────────────────────
def test_psi_is_zero_for_identical_samples():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 20_000)
    small = population_stability_index(reference, rng.normal(0.2, 1, 20_000))
    large = population_stability_index(reference, rng.normal(2.0, 1, 20_000))
    assert large > small > 0


def test_an_emptied_bin_does_not_vanish():
    """A bin populated in training and empty now is the strongest signal there is."""
    reference = np.concatenate([np.zeros(5000), np.ones(5000) * 10])
    current = np.zeros(5000)
    assert population_stability_index(reference, current) > 1.0


def test_seasonal_features_do_not_raise_the_alarm():
    rng = np.random.default_rng(2)
    reference = pd.DataFrame({
        "hour_sin": rng.normal(0, 1, 5000),
        "V1": rng.normal(0, 1, 5000),
    })
    current = pd.DataFrame({
        "hour_sin": rng.normal(3, 1, 5000),   # a different time of day
        "V1": rng.normal(0, 1, 5000),          # genuinely unchanged
    })
    report = compare(reference, current)
    assert [f.feature for f in report.seasonal] == ["hour_sin"]
    assert not report.shifted
    assert not report.alarm


def test_prediction_drift_alone_raises_the_alarm():
    rng = np.random.default_rng(3)
    stable = pd.DataFrame({"V1": rng.normal(0, 1, 5000)})
    report = compare(
        stable, stable,
        reference_scores=rng.beta(1, 50, 5000),
        current_scores=rng.beta(4, 6, 5000),
    )
    assert not report.shifted
    assert report.score_psi > 0.25
    assert report.alarm, "the decisions moved even though no input did"


# ── registry ────────────────────────────────────────────────────────────
def test_saved_model_round_trips(tiny_frame, tmp_path):
    ds = temporal_split(tiny_frame)
    pipe = by_name("logistic", pos_weight=20.0).build().fit(ds.X_dev, ds.y_dev)

    from backstop.features.pipeline import FEATURE_NAMES

    registry.save(
        pipe, version="test-1", model_name="logistic", threshold=0.5,
        threshold_basis="fixture", feature_names=FEATURE_NAMES,
        trained_rows=len(ds.dev), trained_frauds=int(ds.y_dev.sum()),
        train_window_hours=1.0, data_sha256="deadbeef", metrics={"ap": 0.5},
        root=tmp_path,
    )
    loaded, manifest = registry.load("test-1", root=tmp_path)
    assert manifest.threshold == 0.5
    assert manifest.n_features == len(FEATURE_NAMES)
    np.testing.assert_allclose(
        loaded.predict_proba(ds.X_test), pipe.predict_proba(ds.X_test)
    )


def test_a_tampered_artifact_is_refused(tiny_frame, tmp_path):
    """A pickle whose bytes changed since the manifest was written is not the model."""
    ds = temporal_split(tiny_frame)
    pipe = by_name("logistic", pos_weight=20.0).build().fit(ds.X_dev, ds.y_dev)
    path = registry.save(
        pipe, version="test-2", model_name="logistic", threshold=0.5,
        threshold_basis="fixture", feature_names=["a"], trained_rows=1,
        trained_frauds=1, train_window_hours=1.0, data_sha256="x", metrics={},
        root=tmp_path,
    )
    (path / "model.joblib").write_bytes(b"not a model")
    with pytest.raises(registry.ModelError, match="checksum"):
        registry.load("test-2", root=tmp_path)


def test_loading_without_any_model_says_so(tmp_path):
    with pytest.raises(registry.ModelError, match="no model has been trained"):
        registry.load(root=tmp_path)


def test_latest_pointer_follows_the_newest_save(tiny_frame, tmp_path):
    ds = temporal_split(tiny_frame)
    pipe = by_name("logistic", pos_weight=20.0).build().fit(ds.X_dev, ds.y_dev)
    for version in ("v1", "v2"):
        registry.save(
            pipe, version=version, model_name="logistic", threshold=0.5,
            threshold_basis="f", feature_names=["a"], trained_rows=1, trained_frauds=1,
            train_window_hours=1.0, data_sha256="x", metrics={}, root=tmp_path,
        )
    _, manifest = registry.load(root=tmp_path)
    assert manifest.version == "v2"
    assert registry.list_versions(tmp_path) == ["v1", "v2"]
