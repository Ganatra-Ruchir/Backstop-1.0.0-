"""
The candidate models, and the two baselines they have to beat.

A model comparison without a floor is a ranking of numbers with no meaning.
Two floors are defined here:

* `PriorBaseline` scores every transaction identically. Its average precision
  is the fraud rate, which is what "no signal at all" looks like on this data.
* `AmountRule` is the fraud rule a team would write on day one without any
  machine learning: flag the largest transactions. If a gradient-boosted
  ensemble cannot beat that, the ensemble is not worth operating.

Class imbalance is handled by *not* resampling. SMOTE and random undersampling
both change the base rate the model sees, which shifts every predicted
probability away from the real world — and this system needs calibrated
probabilities, because the threshold is chosen by expected cost. Where a model
family offers a class weight, both settings are trained and compared, and
`scripts/train.py` reports what the reweighting actually bought.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backstop.config import RANDOM_SEED
from backstop.features.pipeline import FEATURE_NAMES, build_feature_pipeline


class PriorBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the training base rate for everything. The floor."""

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.prior_ = float(y.mean())
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.prior_, dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class AmountRule(BaseEstimator, ClassifierMixin):
    """
    The rule a team writes before it has a model: bigger amount, more suspicious.

    Scored as a percentile of the training amount distribution so that it
    produces a ranking comparable with a model's probabilities. It is not
    calibrated and does not pretend to be.
    """

    def __init__(self, column: str = "log_amount") -> None:
        self.column = column

    def fit(self, X, y):
        values = self._column(X)
        self.classes_ = np.array([0, 1])
        self.quantiles_ = np.quantile(values, np.linspace(0, 1, 1001))
        return self

    def _column(self, X) -> np.ndarray:
        idx = FEATURE_NAMES.index(self.column)
        arr = np.asarray(X)
        return arr[:, idx].astype(float)

    def predict_proba(self, X):
        v = self._column(X)
        p = np.searchsorted(self.quantiles_, v, side="right") / len(self.quantiles_)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.999).astype(int)


@dataclass(frozen=True)
class Candidate:
    """One model family plus how its inputs must be prepared."""

    name: str
    estimator: object
    scale: bool
    #: Free-text note that ends up in the comparison table.
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def build(self) -> Pipeline:
        pipe = build_feature_pipeline(scale=self.scale)
        return Pipeline([*pipe.steps, ("model", self.estimator)])


def _xgboost(scale_pos_weight: float | None):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_pos_weight or 1.0,
    )


def _lightgbm(is_unbalance: bool):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=30,
        reg_lambda=2.0,
        objective="binary",
        n_jobs=4,
        random_state=RANDOM_SEED,
        is_unbalance=is_unbalance,
        verbose=-1,
    )


def candidates(*, pos_weight: float) -> list[Candidate]:
    """
    Every model in the comparison.

    `pos_weight` is negatives/positives in the training fold — roughly 560:1
    here — and is passed to the families that accept it so the "reweighted"
    variants are meaningfully different from the plain ones.
    """
    return [
        Candidate("prior-baseline", PriorBaseline(), scale=False,
                  note="predicts the base rate", tags=("baseline",)),
        Candidate("amount-rule", AmountRule(), scale=False,
                  note="flag the largest amounts; no learning", tags=("baseline",)),
        Candidate(
            "logistic", 
            LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs"),
            scale=True, note="linear, interpretable, fast", tags=("linear",),
        ),
        Candidate(
            "logistic-balanced",
            LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs",
                               class_weight="balanced"),
            scale=True, note="same, with class weights", tags=("linear", "reweighted"),
        ),
        Candidate(
            "random-forest",
            RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                n_jobs=4, random_state=RANDOM_SEED,
            ),
            scale=False, note="bagged trees", tags=("tree",),
        ),
        Candidate(
            "random-forest-balanced",
            RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=4,
                random_state=RANDOM_SEED,
            ),
            scale=False, note="bagged trees, class weighted",
            tags=("tree", "reweighted"),
        ),
        Candidate(
            "hist-gradient-boosting",
            HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                min_samples_leaf=30, l2_regularization=1.0,
                random_state=RANDOM_SEED,
            ),
            scale=False, note="sklearn's boosted trees, no extra dependency",
            tags=("tree", "boosting"),
        ),
        Candidate("xgboost", _xgboost(None), scale=False,
                  note="gradient boosting", tags=("tree", "boosting")),
        Candidate("xgboost-weighted", _xgboost(pos_weight), scale=False,
                  note="gradient boosting, scale_pos_weight",
                  tags=("tree", "boosting", "reweighted")),
        Candidate("lightgbm", _lightgbm(False), scale=False,
                  note="gradient boosting, leaf-wise", tags=("tree", "boosting")),
        Candidate("lightgbm-weighted", _lightgbm(True), scale=False,
                  note="gradient boosting, is_unbalance",
                  tags=("tree", "boosting", "reweighted")),
    ]


def by_name(name: str, *, pos_weight: float) -> Candidate:
    for c in candidates(pos_weight=pos_weight):
        if c.name == name:
            return c
    raise KeyError(f"unknown model {name!r}")
