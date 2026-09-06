"""SHAP explanations and the reason codes built from them."""

from __future__ import annotations

import numpy as np
import pytest

from backstop.data.splits import temporal_split
from backstop.explain.reasons import Explainer, global_importance
from backstop.features.pipeline import FEATURE_NAMES
from backstop.models.zoo import by_name


@pytest.fixture(scope="module")
def fitted(tiny_frame_module):
    ds = temporal_split(tiny_frame_module)
    pipe = by_name("xgboost", pos_weight=20.0).build().fit(ds.X_dev, ds.y_dev)
    return pipe, ds


@pytest.fixture(scope="module")
def tiny_frame_module():
    from tests.conftest import make_frame

    return make_frame(n=800, fraud_rate=0.06, seed=13)


def test_explanations_reconstruct_the_score(fitted):
    """
    SHAP values plus the base value must add back up to the model's output.

    An explanation that does not reconstruct the prediction is decoration, and
    an analyst acting on it is acting on nothing.
    """
    pipe, ds = fitted
    rows = ds.X_test.iloc[:20]
    predicted = pipe.predict_proba(rows)[:, 1]
    explained = [e.score for e in Explainer(pipe).explain(rows)]
    np.testing.assert_allclose(explained, predicted, atol=1e-5)


def test_contributions_sum_back_to_the_prediction(fitted):
    """The additive guarantee, checked directly against the model's margin."""
    pipe, ds = fitted
    matrix = np.asarray(pipe[:-1].transform(ds.X_test.iloc[:50]), dtype=float)
    assert Explainer(pipe).reconstruction_error(matrix) < 1e-4


def test_reasons_are_ordered_by_influence(fitted):
    pipe, ds = fitted
    for explanation in Explainer(pipe).explain(ds.X_test.iloc[:10]):
        magnitudes = [abs(r.contribution) for r in explanation.reasons]
        assert magnitudes == sorted(magnitudes, reverse=True)


def test_reason_features_are_real_features(fitted):
    pipe, ds = fitted
    for explanation in Explainer(pipe).explain(ds.X_test.iloc[:10]):
        for reason in explanation.reasons:
            assert reason.feature in FEATURE_NAMES


def test_named_features_get_plain_language(fitted):
    """An anonymised component says so; a real one is described."""
    from backstop.explain.reasons import _phrase

    assert "amount" in _phrase("log_amount", 3.0, 0.5).lower()
    assert "overnight" in _phrase("is_night", 1.0, 0.5)
    assert "card testing" in _phrase("is_zero_amount", 1.0, 0.5)
    assert "anonymised" in _phrase("V14", -3.0, 0.5)


def test_matrix_and_frame_paths_agree(fitted):
    """The serving path explains a prepared matrix; both must give one answer."""
    pipe, ds = fitted
    rows = ds.X_test.iloc[:8]
    explainer = Explainer(pipe)
    matrix = np.asarray(pipe[:-1].transform(rows), dtype=float)

    from_frame = explainer.explain(rows)
    from_matrix = explainer.explain_matrix(matrix)
    for a, b in zip(from_frame, from_matrix, strict=True):
        assert a.score == pytest.approx(b.score)
        assert [r.feature for r in a.reasons] == [r.feature for r in b.reasons]


def test_reason_codes_only_describe_risk_raising_features(fitted):
    pipe, ds = fitted
    for explanation in Explainer(pipe).explain(ds.X_test.iloc[:15]):
        raising = {r.feature.upper() for r in explanation.reasons if r.contribution > 0}
        assert all(code.split("_")[0] in raising for code in explanation.codes())


def test_global_importance_covers_every_feature(fitted):
    pipe, ds = fitted
    importance = global_importance(pipe, ds.X_dev, sample=200, seed=0)
    assert set(importance) == set(FEATURE_NAMES)
    assert all(v >= 0 for v in importance.values())
    values = list(importance.values())
    assert values == sorted(values, reverse=True)
