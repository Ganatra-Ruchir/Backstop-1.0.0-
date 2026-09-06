"""
Feature construction.

Two rules govern everything here.

*Nothing may be fitted outside a fold.* Every statistic a feature depends on —
the amount distribution used for percentile ranking, the scaler's mean and
variance — is learned from training rows only, inside a scikit-learn pipeline.
A `.fit()` on the full frame before splitting is the most common way a fraud
notebook reports a number it cannot reproduce in production.

*Nothing may encode position in this particular window.* The raw `Time` column
is seconds since the first transaction in the file. A tree given that column
will happily learn "fraud is likely around t = 4,000" — true of these two days
and useless on any other. What generalises is the time of day, so `Time` is
turned into cyclical hour features and then dropped.

Which features exist at all was decided by testing hypotheses on development
data (`scripts/eda.py`), not by adding whatever came to mind:

    is_night        2.34x lift, chi2 p = 5.8e-31   → kept
    zero amount     6.96x lift, chi2 p = 1.3e-20   → kept
    whole-euro amt  1.42x lift, chi2 p = 2.1e-06   → kept
    multiple of 10  1.00x lift, chi2 p = 1.00      → rejected
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
NIGHT_START, NIGHT_END = 0, 8

ENGINEERED = [
    "log_amount",
    "amount_pct_rank",
    "hour_sin",
    "hour_cos",
    "is_night",
    "is_zero_amount",
    "is_whole_euro",
]
FEATURE_NAMES = V_COLUMNS + ENGINEERED


class TransactionFeatures(BaseEstimator, TransformerMixin):
    """
    Derive model inputs from a raw transaction frame.

    The only fitted state is the training amount distribution, used to express
    an amount as a percentile of what the model saw during training. That is
    deliberately a *learned* feature: a €500 transaction is unremarkable in one
    portfolio and a red flag in another, and the percentile carries that,
    whereas the raw number does not.
    """

    def __init__(self, n_quantiles: int = 1000) -> None:
        self.n_quantiles = n_quantiles

    def fit(self, X: pd.DataFrame, y=None) -> TransactionFeatures:
        amounts = np.asarray(X["Amount"], dtype=float)
        qs = np.linspace(0.0, 1.0, self.n_quantiles + 1)
        self.amount_quantiles_ = np.quantile(amounts, qs)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "amount_quantiles_"):
            raise RuntimeError("TransactionFeatures.transform called before fit")

        amount = np.asarray(X["Amount"], dtype=float)
        time_s = np.asarray(X["Time"], dtype=float)
        hour = (time_s / 3600.0) % 24.0

        out = X[V_COLUMNS].copy()
        out["log_amount"] = np.log1p(np.clip(amount, 0, None))
        # searchsorted against training quantiles: values beyond anything seen
        # in training saturate at 0 or 1 rather than extrapolating.
        out["amount_pct_rank"] = (
            np.searchsorted(self.amount_quantiles_, amount, side="right")
            / len(self.amount_quantiles_)
        ).clip(0.0, 1.0)
        # Hour is circular: 23:59 and 00:01 are two minutes apart, and a raw
        # 0-24 feature tells a linear model they are 24 hours apart.
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["is_night"] = ((hour >= NIGHT_START) & (hour < NIGHT_END)).astype(float)
        out["is_zero_amount"] = (amount == 0).astype(float)
        out["is_whole_euro"] = ((amount % 1.0) == 0).astype(float)
        return out[FEATURE_NAMES]

    def transform_array(self, time_s: np.ndarray, amount: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        The same transform, straight into a float array.

        `transform` builds and slices pandas frames, which costs about 3.6 ms
        for a single row — an order of magnitude more than the model itself
        takes to score it. On a batch that overhead disappears into the noise;
        on the one-row calls an authorisation path actually makes, it is most
        of the response time.

        So single-row scoring takes this path instead. It is the same
        arithmetic, and `tests/test_features.py` asserts the two agree
        bit-for-bit on real rows — a fast path that quietly disagrees with the
        path the model was trained through is a far worse bug than a slow one.
        """
        if not hasattr(self, "amount_quantiles_"):
            raise RuntimeError("transform_array called before fit")

        n = len(amount)
        out = np.empty((n, len(FEATURE_NAMES)), dtype=float)
        out[:, : len(V_COLUMNS)] = v

        hour = (time_s / 3600.0) % 24.0
        base = len(V_COLUMNS)
        out[:, base + 0] = np.log1p(np.clip(amount, 0, None))
        out[:, base + 1] = np.clip(
            np.searchsorted(self.amount_quantiles_, amount, side="right")
            / len(self.amount_quantiles_),
            0.0,
            1.0,
        )
        out[:, base + 2] = np.sin(2 * np.pi * hour / 24.0)
        out[:, base + 3] = np.cos(2 * np.pi * hour / 24.0)
        out[:, base + 4] = (hour >= NIGHT_START) & (hour < NIGHT_END)
        out[:, base + 5] = amount == 0
        out[:, base + 6] = (amount % 1.0) == 0
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(FEATURE_NAMES, dtype=object)


def build_feature_pipeline(*, scale: bool) -> Pipeline:
    """
    Feature steps for one model family.

    `scale=True` for anything that measures distance or fits coefficients —
    logistic regression, k-NN, neural networks. `scale=False` for trees, which
    are invariant to monotone rescaling and only pay for it in fit time.
    """
    steps: list[tuple[str, object]] = [("features", TransactionFeatures())]
    if scale:
        steps.append(("scale", StandardScaler()))
    else:
        # Keep the shape of the pipeline identical across families so that
        # serving code, explainers and drift monitors need no special cases.
        steps.append(("passthrough", FunctionTransformer(feature_names_out="one-to-one")))
    return Pipeline(steps)


def as_frame(array: np.ndarray) -> pd.DataFrame:
    """Re-attach feature names after a step that returns a bare array."""
    return pd.DataFrame(array, columns=FEATURE_NAMES)


__all__ = [
    "ENGINEERED",
    "FEATURE_NAMES",
    "V_COLUMNS",
    "ColumnTransformer",
    "TransactionFeatures",
    "as_frame",
    "build_feature_pipeline",
]
