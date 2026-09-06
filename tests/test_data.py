"""Loading, profiling, deduplication and the split contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backstop.data.loader import DataError, deduplicate, load_raw, profile
from backstop.data.splits import (
    Dataset,
    expanding_folds,
    random_split_for_comparison,
    temporal_split,
)


def test_profile_counts_duplicates(frame):
    doubled = pd.concat([frame, frame.iloc[:50]], ignore_index=True)
    report = profile(doubled)
    assert report.exact_duplicates == 50
    assert "byte-identical" in " ".join(report.notes)


def test_deduplicate_keeps_first_occurrence(frame):
    doubled = pd.concat([frame, frame.iloc[:20]], ignore_index=True)
    out, dropped = deduplicate(doubled)
    assert dropped == 20
    assert len(out) == len(frame)
    assert not out.duplicated().any()


def test_temporal_split_puts_nothing_from_the_future_in_training(frame):
    ds = temporal_split(frame)
    assert ds.dev["Time"].max() <= ds.test["Time"].min()
    assert len(ds.dev) + len(ds.test) == len(frame)


def test_temporal_split_is_deterministic(frame):
    a, b = temporal_split(frame), temporal_split(frame)
    pd.testing.assert_frame_equal(a.dev, b.dev)
    pd.testing.assert_frame_equal(a.test, b.test)


def test_split_refuses_a_test_window_without_fraud(frame):
    late = frame["Time"] > frame["Time"].quantile(0.7)
    clean = frame.copy()
    clean.loc[late, "Class"] = 0
    with pytest.raises(ValueError, match="no fraud"):
        temporal_split(clean)


def test_folds_never_validate_on_data_they_trained_on(frame):
    ds = temporal_split(frame)
    for fold in expanding_folds(ds.dev):
        assert not set(fold.train_idx) & set(fold.valid_idx)
        train_max = ds.dev["Time"].to_numpy()[fold.train_idx].max()
        valid_min = ds.dev["Time"].to_numpy()[fold.valid_idx].min()
        assert valid_min > train_max, "a fold validated on a row it trained on"


def test_folds_expand(frame):
    ds = temporal_split(frame)
    sizes = [len(f.train_idx) for f in expanding_folds(ds.dev)]
    assert sizes == sorted(sizes), "training windows must grow, not slide"


def test_purge_gap_is_respected(frame):
    from backstop.config import SplitConfig

    cfg = SplitConfig(purge_seconds=3600.0)
    ds = temporal_split(frame, cfg)
    times = ds.dev["Time"].to_numpy()
    for fold in expanding_folds(ds.dev, cfg):
        assert times[fold.valid_idx].min() > fold.train_end_time + cfg.purge_seconds - 1e-9


def test_random_split_is_labelled_as_a_comparison_only(frame):
    train, test = random_split_for_comparison(frame, test_size=0.25, seed=1)
    assert len(train) + len(test) == len(frame)
    # The point of the helper: it does *not* respect time.
    assert train["Time"].max() > test["Time"].min()


def test_missing_file_says_what_to_do(tmp_path):
    with pytest.raises(DataError, match="download_data"):
        load_raw(tmp_path / "absent.csv")


@pytest.mark.slow
def test_the_real_file_is_the_one_the_metrics_describe(real_data):
    assert len(real_data) == 283_726
    assert int(real_data["Class"].sum()) == 473
    assert real_data.isna().sum().sum() == 0


@pytest.mark.slow
def test_real_split_sizes_match_the_documentation(real_data):
    ds = temporal_split(real_data)
    assert isinstance(ds, Dataset)
    assert len(ds.dev) == 210_292
    assert len(ds.test) == 73_434
    assert int(ds.y_test.sum()) == 97
    assert np.isclose(ds.dev_end_time / 3600, 38.4, atol=0.1)
