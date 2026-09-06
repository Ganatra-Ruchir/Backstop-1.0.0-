"""
Why this transaction, in words a human can act on.

An analyst opening an alert needs to know what to check first, and in a
regulated setting a declined customer is entitled to be told why. "The model
said 0.94" satisfies neither. So every alert carries the handful of features
that pushed it over the line, with the direction and size of each push.

SHAP is used rather than global feature importance because the question is
about *this* transaction. Global importance says V14 matters across the
portfolio; it cannot say that V14 is the reason this particular card was
blocked, and the two are routinely different.

The honest limitation, stated here and in the model card: twenty-eight of the
thirty-five features are anonymised principal components. "V14 is unusually
low" is a true and useful signal — it tells an analyst the transaction is
strange in a specific, reproducible way and can be compared against other
alerts — but it is not a business reason, and this project does not dress it up
as one. Features that *do* carry meaning — the amount, the time of day, the
percentile rank — are named in plain language wherever they appear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backstop.features.pipeline import FEATURE_NAMES

#: Plain-language templates for the features that are not anonymised.
PHRASES: dict[str, tuple[str, str]] = {
    "log_amount": ("unusually large amount", "unusually small amount"),
    "amount_pct_rank": (
        "amount is high for this portfolio",
        "amount is low for this portfolio",
    ),
    "is_night": ("occurred overnight", "occurred during the day"),
    "hour_sin": ("time of day", "time of day"),
    "hour_cos": ("time of day", "time of day"),
    "is_zero_amount": ("zero-amount authorisation (card testing pattern)", ""),
    "is_whole_euro": ("whole-euro amount", "amount has cents"),
}


@dataclass(frozen=True)
class Reason:
    feature: str
    contribution: float
    value: float
    text: str

    @property
    def direction(self) -> str:
        return "raises" if self.contribution > 0 else "lowers"


@dataclass(frozen=True)
class Explanation:
    score: float
    baseline: float
    reasons: list[Reason]

    def render(self) -> str:
        head = f"score {self.score:.4f} (portfolio baseline {self.baseline:.5f})"
        body = "\n".join(
            f"  {r.direction} risk  {r.contribution:+.3f}  {r.text}" for r in self.reasons
        )
        return f"{head}\n{body}"

    def codes(self) -> list[str]:
        """Short reason codes, the form an adverse-action notice takes."""
        return [
            f"{r.feature.upper()}_{'HI' if r.value > 0 else 'LO'}"
            for r in self.reasons
            if r.contribution > 0
        ]


def _phrase(feature: str, value: float, contribution: float) -> str:
    if feature in PHRASES:
        high, low = PHRASES[feature]
        if feature in {"is_zero_amount", "is_whole_euro"}:
            return high if value > 0.5 else (low or f"{feature} absent")
        return high if value > 0 else low
    # Anonymised component: say exactly what is known and no more.
    side = "high" if value > 0 else "low"
    return f"{feature} is {side} ({value:+.2f}); anonymised component"


class Explainer:
    """
    Wraps a SHAP TreeExplainer over a fitted pipeline.

    The explainer runs on the *transformed* matrix, so it must be given the
    pipeline's feature steps as well as its model. Building it is expensive;
    scoring one row with it is not, which is why it is constructed once at
    service start rather than per request.
    """

    def __init__(self, pipeline, background: np.ndarray | None = None) -> None:
        import shap

        self.feature_steps = pipeline[:-1]
        self.model = pipeline[-1]
        self._explainer = shap.TreeExplainer(self.model, data=background)

        # `expected_value` is not settled until the explainer has run once:
        # read before the first `shap_values` call it can be a different number
        # from the one the values are actually referenced to, and the
        # reconstruction then misses by a constant. It cost an afternoon to
        # find, so the warm-up is explicit and the constant is verified below
        # rather than trusted.
        probe = np.zeros((1, len(FEATURE_NAMES)), dtype=float)
        self._raw_values(probe)
        self.expected_value = float(np.ravel(self._explainer.expected_value)[0])

    def _raw_values(self, matrix: np.ndarray) -> np.ndarray:
        values = self._explainer.shap_values(matrix)
        if isinstance(values, list):  # older shap returns one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        return values

    def reconstruction_error(self, matrix: np.ndarray) -> float:
        """
        How far the parts are from the whole.

        An explanation whose contributions do not sum back to the prediction is
        decoration, and an analyst acting on it is acting on nothing. This is
        the check, and `tests/test_explain.py` asserts it stays near zero.
        """
        matrix = np.asarray(matrix, dtype=float)
        rebuilt = self.expected_value + self._raw_values(matrix).sum(axis=1)
        actual = np.asarray(self.model.predict(matrix, output_margin=True), dtype=float)
        return float(np.abs(rebuilt - actual).max())

    def explain(self, X_raw, *, top_k: int = 4) -> list[Explanation]:
        """Explain a batch of raw transactions."""
        return self.explain_matrix(
            np.asarray(self.feature_steps.transform(X_raw), dtype=float), top_k=top_k
        )

    def explain_matrix(self, matrix: np.ndarray, *, top_k: int = 4) -> list[Explanation]:
        """Explain rows that have already been through the feature pipeline."""
        matrix = np.asarray(matrix, dtype=float)
        values = self._raw_values(matrix)

        # SHAP values live in log-odds space for a binary booster; converting
        # the base value back through the logistic gives a probability an
        # analyst can compare with the score they were shown.
        baseline = 1.0 / (1.0 + np.exp(-self.expected_value))

        out: list[Explanation] = []
        for row_values, row_features in zip(values, matrix, strict=True):
            order = np.argsort(-np.abs(row_values))[:top_k]
            reasons = [
                Reason(
                    feature=FEATURE_NAMES[i],
                    contribution=float(row_values[i]),
                    value=float(row_features[i]),
                    text=_phrase(FEATURE_NAMES[i], row_features[i], row_values[i]),
                )
                for i in order
            ]
            score = 1.0 / (1.0 + np.exp(-(self.expected_value + row_values.sum())))
            out.append(Explanation(score=float(score), baseline=float(baseline), reasons=reasons))
        return out


def global_importance(pipeline, X_raw, *, sample: int = 5000, seed: int = 0) -> dict:
    """Mean absolute SHAP value per feature, over a sample. For the model card."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_raw), size=min(sample, len(X_raw)), replace=False)
    matrix = np.asarray(pipeline[:-1].transform(X_raw.iloc[idx]), dtype=float)

    mean_abs = np.abs(Explainer(pipeline)._raw_values(matrix)).mean(axis=0)
    return dict(
        sorted(
            zip(FEATURE_NAMES, mean_abs.tolist(), strict=True),
            key=lambda kv: -kv[1],
        )
    )
