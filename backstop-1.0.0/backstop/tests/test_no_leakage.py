"""
The claim this project is built on: nothing the model is graded on was
available to it at training time.

Three separate mechanisms can break that, and each gets its own assertion. This
file is the reason the headline number is lower than the one most notebooks on
this dataset report — and the reason it is the one worth reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from backstop.data.loader import deduplicate
from backstop.data.splits import (
    expanding_folds,
    random_split_for_comparison,
    temporal_split,
)
from backstop.models.zoo import by_name


def test_no_training_row_occurs_after_any_test_row(frame):
    ds = temporal_split(frame)
    assert ds.dev["Time"].max() <= ds.test["Time"].min()


def test_no_duplicate_straddles_the_split(frame):
    """
    The same transaction on both sides is memorisation graded as generalisation.

    Deduplication runs before the split for exactly this reason, so after it no
    row in test may be byte-identical to a row in development.
    """
    doubled = pd.concat([frame, frame.iloc[:80]], ignore_index=True)
    clean, _ = deduplicate(doubled)
    ds = temporal_split(clean)

    feature_cols = [c for c in clean.columns if c != "Class"]
    dev_rows = {tuple(r) for r in ds.dev[feature_cols].to_numpy().tolist()}
    test_rows = {tuple(r) for r in ds.test[feature_cols].to_numpy().tolist()}
    assert not (dev_rows & test_rows), "a transaction appears on both sides of the split"


def test_the_test_window_is_untouched_by_fitting(frame):
    """
    Every fitted statistic must come from training rows.

    Fitting the pipeline on development data and then transforming the test
    window must give a different answer from fitting on the test window
    directly — if the two agree, something saw data it should not have.
    """
    from backstop.features.pipeline import build_feature_pipeline

    ds = temporal_split(frame)
    honest = build_feature_pipeline(scale=True).fit(ds.X_dev)
    cheating = build_feature_pipeline(scale=True).fit(ds.X_test)

    a = np.asarray(honest.transform(ds.X_test))
    b = np.asarray(cheating.transform(ds.X_test))
    assert not np.allclose(a, b)


def test_folds_are_ordered_in_time(frame):
    ds = temporal_split(frame)
    times = ds.dev["Time"].to_numpy()
    for fold in expanding_folds(ds.dev):
        assert times[fold.train_idx].max() < times[fold.valid_idx].min()


@pytest.mark.slow
def test_shuffling_time_inflates_the_score(real_data):
    """
    The measurement behind the headline claim.

    Identical model, identical hyperparameters; only the split differs. If the
    random split ever stops looking better, either the leak has been fixed
    upstream or this test has stopped testing anything — both worth knowing.
    """
    pos_weight = float(
        (real_data["Class"] == 0).sum() / (real_data["Class"] == 1).sum()
    )
    ds = temporal_split(real_data)
    test_fraction = len(ds.test) / len(real_data)

    def fit_and_score(train, test) -> float:
        pipe = by_name("xgboost", pos_weight=pos_weight).build()
        pipe.fit(train.drop(columns=["Class"]), train["Class"])
        scores = pipe.predict_proba(test.drop(columns=["Class"]))[:, 1]
        return float(average_precision_score(test["Class"], scores))

    temporal_ap = fit_and_score(ds.dev, ds.test)
    shuffled_train, shuffled_test = random_split_for_comparison(
        real_data, test_size=test_fraction, seed=20260907
    )
    random_ap = fit_and_score(shuffled_train, shuffled_test)

    assert random_ap > temporal_ap, (
        "the random split no longer looks better than the temporal one — "
        "check whether the leak was fixed or this test broke"
    )
    assert random_ap - temporal_ap > 0.02, (
        f"expected a gap of at least 0.02, measured {random_ap - temporal_ap:.4f}"
    )
