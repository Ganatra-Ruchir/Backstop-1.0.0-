"""Feature construction, and the two properties it must never lose."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backstop.data.splits import temporal_split
from backstop.features.pipeline import (
    FEATURE_NAMES,
    V_COLUMNS,
    TransactionFeatures,
    build_feature_pipeline,
)


def test_output_shape_and_names(frame):
    out = TransactionFeatures().fit(frame).transform(frame)
    assert list(out.columns) == FEATURE_NAMES
    assert len(out) == len(frame)


def test_raw_time_is_not_a_feature():
    """Seconds-since-the-file-began does not generalise past this file."""
    assert "Time" not in FEATURE_NAMES


def test_hour_is_cyclical(frame):
    """23:59 and 00:01 must be adjacent, not 24 hours apart."""
    late = frame.iloc[[0]].copy()
    late["Time"] = 23.99 * 3600
    early = frame.iloc[[0]].copy()
    early["Time"] = 0.01 * 3600

    fitted = TransactionFeatures().fit(frame)
    a = fitted.transform(late)[["hour_sin", "hour_cos"]].to_numpy()
    b = fitted.transform(early)[["hour_sin", "hour_cos"]].to_numpy()
    assert np.linalg.norm(a - b) < 0.01


def test_night_flag_matches_the_definition(frame):
    fitted = TransactionFeatures().fit(frame)
    out = fitted.transform(frame)
    hour = (frame["Time"].to_numpy() / 3600) % 24
    expected = ((hour >= 0) & (hour < 8)).astype(float)
    np.testing.assert_array_equal(out["is_night"].to_numpy(), expected)


def test_amount_rank_is_learned_from_training_only(frame):
    """
    The percentile a transaction sits at is a *fitted* statistic.

    Fitting it on the whole dataset would put test-window information into a
    training feature, which is the quiet version of the leak the split
    prevents. Fitting on a subset must therefore give a different answer.
    """
    small = frame.iloc[:500]
    on_all = TransactionFeatures().fit(frame).transform(frame)["amount_pct_rank"]
    on_part = TransactionFeatures().fit(small).transform(frame)["amount_pct_rank"]
    assert not np.allclose(on_all, on_part)


def test_amount_rank_saturates_outside_the_training_range(frame):
    fitted = TransactionFeatures().fit(frame)
    extreme = frame.iloc[[0]].copy()
    extreme["Amount"] = 1e9
    assert fitted.transform(extreme)["amount_pct_rank"].iloc[0] <= 1.0

    zero = frame.iloc[[0]].copy()
    zero["Amount"] = 0.0
    assert 0.0 <= fitted.transform(zero)["amount_pct_rank"].iloc[0] <= 1.0


def test_transform_before_fit_raises(frame):
    with pytest.raises(RuntimeError, match="before fit"):
        TransactionFeatures().transform(frame)


def test_fast_path_agrees_with_the_pandas_path_exactly(frame):
    """
    The serving fast path must not be a second implementation that drifts.

    It is four times faster and therefore worth having; a fast path that
    quietly disagrees with the path the model was trained through would be a
    far worse bug than a slow one.
    """
    fitted = TransactionFeatures().fit(frame)
    slow = fitted.transform(frame).to_numpy(dtype=float)
    fast = fitted.transform_array(
        frame["Time"].to_numpy(), frame["Amount"].to_numpy(),
        frame[V_COLUMNS].to_numpy(),
    )
    np.testing.assert_array_equal(slow, fast)


def test_pipeline_produces_no_nan_or_inf(frame):
    for scale in (True, False):
        out = np.asarray(build_feature_pipeline(scale=scale).fit_transform(frame))
        assert np.isfinite(out).all()


def test_scaler_statistics_come_from_training_only(frame):
    """A scaler fitted on everything leaks the test window's mean and variance."""
    ds = temporal_split(frame)
    pipe = build_feature_pipeline(scale=True).fit(ds.X_dev)
    dev_means = np.asarray(pipe.transform(ds.X_dev)).mean(axis=0)
    test_means = np.asarray(pipe.transform(ds.X_test)).mean(axis=0)
    assert np.allclose(dev_means, 0, atol=1e-6), "training data should standardise to 0"
    assert not np.allclose(test_means, 0, atol=1e-3), (
        "test data standardising to exactly 0 means the scaler saw it"
    )


def test_zero_and_whole_euro_flags():
    rows = pd.DataFrame({
        "Time": [0.0, 0.0, 0.0],
        "Amount": [0.0, 25.00, 25.37],
        **{c: [0.0, 0.0, 0.0] for c in V_COLUMNS},
    })
    out = TransactionFeatures().fit(rows).transform(rows)
    np.testing.assert_array_equal(out["is_zero_amount"].to_numpy(), [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(out["is_whole_euro"].to_numpy(), [1.0, 1.0, 0.0])
