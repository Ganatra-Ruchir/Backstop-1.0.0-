"""
Time-respecting splits.

Everything here exists to stop the model being graded on information it could
not have had. There are three separate ways that happens on this dataset, and
each has a countermeasure:

1. *Random splitting* puts later transactions in training and earlier ones in
   test. Countermeasure: order by time and cut.
2. *Duplicate rows* straddling a boundary put the same transaction on both
   sides. Countermeasure: `loader.deduplicate` runs before splitting.
3. *Fitting a scaler or an imputer on everything* leaks test-set statistics
   into training. Countermeasure: every transformer is fitted inside the
   pipeline, on training rows only (`features/pipeline.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backstop.config import CONFIG, SplitConfig

TIME = "Time"
TARGET = "Class"


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold: everything before `train_end`, then a block."""

    index: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    train_end_time: float
    valid_end_time: float

    def describe(self, y: pd.Series) -> str:
        tr, va = y.iloc[self.train_idx], y.iloc[self.valid_idx]
        return (
            f"fold {self.index}: train n={len(tr):,} f={int(tr.sum()):>3} | "
            f"valid n={len(va):,} f={int(va.sum()):>3} "
            f"(t ≤ {self.train_end_time / 3600:.1f}h → {self.valid_end_time / 3600:.1f}h)"
        )


@dataclass(frozen=True)
class Dataset:
    """Development and test frames, already ordered by time."""

    dev: pd.DataFrame
    test: pd.DataFrame
    dev_end_time: float

    @property
    def X_dev(self) -> pd.DataFrame:
        return self.dev.drop(columns=[TARGET])

    @property
    def y_dev(self) -> pd.Series:
        return self.dev[TARGET]

    @property
    def X_test(self) -> pd.DataFrame:
        return self.test.drop(columns=[TARGET])

    @property
    def y_test(self) -> pd.Series:
        return self.test[TARGET]

    def summary(self) -> str:
        return (
            f"development  n={len(self.dev):,}  frauds={int(self.dev[TARGET].sum()):,} "
            f"({self.dev[TARGET].mean() * 100:.4f}%)\n"
            f"test         n={len(self.test):,}  frauds={int(self.test[TARGET].sum()):,} "
            f"({self.test[TARGET].mean() * 100:.4f}%)\n"
            f"boundary at t = {self.dev_end_time / 3600:.1f} h"
        )


def temporal_split(df: pd.DataFrame, cfg: SplitConfig | None = None) -> Dataset:
    """Cut the observation window into development and test by time."""
    cfg = cfg or CONFIG.split
    ordered = df.sort_values(TIME, kind="mergesort").reset_index(drop=True)
    t0, t1 = ordered[TIME].min(), ordered[TIME].max()
    boundary = t0 + (t1 - t0) * cfg.dev_end

    dev = ordered[ordered[TIME] <= boundary].reset_index(drop=True)
    test = ordered[ordered[TIME] > boundary].reset_index(drop=True)

    if int(test[TARGET].sum()) == 0:
        raise ValueError("the test window contains no fraud; the split is unusable")
    return Dataset(dev=dev, test=test, dev_end_time=float(boundary))


def expanding_folds(
    dev: pd.DataFrame, cfg: SplitConfig | None = None
) -> list[Fold]:
    """
    Expanding-window folds over the development data.

    Fold *k* trains on everything up to a boundary and validates on the block
    that follows it, which is how the model will actually be used: fit on the
    past, score the future. A purge gap after each training edge keeps rows
    adjacent to the boundary out of validation.
    """
    cfg = cfg or CONFIG.split
    times = dev[TIME].to_numpy()
    t0, t1 = times.min(), times.max()

    # Blocks 1..n_folds partition the window; fold k trains on blocks < k.
    edges = np.linspace(t0, t1, cfg.n_folds + 2)[1:]

    folds: list[Fold] = []
    for k in range(cfg.n_folds):
        train_end, valid_end = edges[k], edges[k + 1]
        train_idx = np.flatnonzero(times <= train_end)
        valid_idx = np.flatnonzero(
            (times > train_end + cfg.purge_seconds) & (times <= valid_end)
        )
        if len(valid_idx) == 0 or dev[TARGET].to_numpy()[valid_idx].sum() == 0:
            # A fold with no positives cannot inform a threshold; skip it and
            # say so, rather than silently averaging a meaningless number in.
            continue
        folds.append(
            Fold(
                index=k + 1,
                train_idx=train_idx,
                valid_idx=valid_idx,
                train_end_time=float(train_end),
                valid_end_time=float(valid_end),
            )
        )
    if not folds:
        raise ValueError("no usable folds; widen the development window")
    return folds


def random_split_for_comparison(
    df: pd.DataFrame, *, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A deliberately wrong split, used only to quantify how wrong it is.

    This is not used to train the shipped model. It exists so the README can
    state the size of the gap on this data instead of asserting that one exists.
    """
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
