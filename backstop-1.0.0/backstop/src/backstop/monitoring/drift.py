"""
Noticing when the world has moved away from the training data.

A fraud model degrades in two distinguishable ways and the difference decides
what you do about it.

*Data drift* — the inputs change. New merchant categories, a new acquirer, a
seasonal shift. Detectable immediately, from unlabelled traffic, which matters
because labels arrive weeks late: a chargeback is not confirmed the day the
transaction clears.

*Concept drift* — the inputs look the same but their meaning has changed,
because the fraudsters adapted. Only detectable once labels arrive, so the data
drift signal is what you actually operate on day to day.

Two complementary statistics are computed per feature. PSI is the industry
convention in credit risk and has thresholds people already trust. The KS
statistic is distribution-free and has a p-value, which PSI does not — and with
enough rows, a KS test flags differences far too small to act on, so both are
reported and neither is read alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

#: Conventional PSI bands from credit-risk practice.
PSI_STABLE = 0.10
PSI_SHIFTED = 0.25

#: Features whose distribution is *supposed* to move between two windows.
#:
#: Comparing 09:00-17:00 against 22:00-06:00 makes an hour feature look
#: catastrophically drifted, and it is — but the model has not degraded and
#: nobody should be paged. On this dataset those two features produce PSI of
#: 5.23 and 3.74, an order of magnitude above the loudest genuine signal, and
#: they would drown it out entirely. They are still measured and reported;
#: they just do not raise the alarm.
SEASONAL_FEATURES = frozenset({"hour_sin", "hour_cos", "is_night"})


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float
    seasonal: bool = False

    @property
    def verdict(self) -> str:
        if self.seasonal:
            return "seasonal"
        if self.psi >= PSI_SHIFTED:
            return "shifted"
        if self.psi >= PSI_STABLE:
            return "watch"
        return "stable"


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    features: list[FeatureDrift] = field(default_factory=list)
    #: Prediction drift: the score distribution itself, which moves even when
    #: no single input looks alarming.
    score_psi: float | None = None

    @property
    def shifted(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.verdict == "shifted"]

    @property
    def watch(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.verdict == "watch"]

    @property
    def seasonal(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.verdict == "seasonal"]

    @property
    def alarm(self) -> bool:
        """
        Whether a human should look.

        Prediction drift is weighted more heavily than input drift, and that is
        deliberate. Inputs move constantly; what matters is whether the moving
        inputs changed what the model *decides*. On the test window here, six
        features cross the shifted threshold while the score distribution
        barely moves (PSI 0.036) — the model is reading different inputs and
        reaching the same distribution of conclusions, which is a model
        generalising, not a model breaking.
        """
        if self.score_psi is not None and self.score_psi >= PSI_SHIFTED:
            return True
        return len(self.shifted) >= 3

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ | {"verdict": f.verdict} for f in self.features])

    def render(self, top: int = 10) -> str:
        rows = sorted(self.features, key=lambda f: -f.psi)[:top]
        width = max((len(r.feature) for r in rows), default=8)
        lines = [
            f"reference n={self.n_reference:,}  current n={self.n_current:,}",
            f"{'feature'.ljust(width)}  {'PSI':>7}  {'KS':>6}  {'p':>9}  verdict",
            f"{'-' * width}  {'-' * 7}  {'-' * 6}  {'-' * 9}  -------",
        ]
        for r in rows:
            lines.append(
                f"{r.feature.ljust(width)}  {r.psi:>7.4f}  {r.ks_statistic:>6.3f}  "
                f"{r.ks_pvalue:>9.2e}  {r.verdict}"
            )
        if self.seasonal:
            names = ", ".join(f"{f.feature} (PSI {f.psi:.2f})" for f in self.seasonal)
            lines.append(f"\nseasonal, excluded from the alarm: {names}")
        if self.score_psi is not None:
            lines.append(f"score distribution PSI {self.score_psi:.4f}")
        lines.append(
            f"\nALARM — {len(self.shifted)} features shifted"
            + (
                f" and the score distribution moved (PSI {self.score_psi:.3f})"
                if self.score_psi is not None and self.score_psi >= PSI_SHIFTED
                else ""
            )
            if self.alarm
            else f"\nno alarm — {len(self.shifted)} shifted, "
            f"{len(self.watch)} on watch, predictions stable"
        )
        return "\n".join(lines)


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> float:
    """
    PSI between two samples of one feature.

    Bin edges come from the *reference* quantiles, never from the pooled data:
    binning on the combination hides exactly the movement the statistic is meant
    to detect. Empty bins are floored rather than dropped, because a bin that
    was populated in training and is empty now is the strongest possible signal
    and must not silently become a zero term.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # A near-constant feature has no distribution to compare.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    floor = 1e-6
    ref_pct = np.maximum(ref_counts / len(reference), floor)
    cur_pct = np.maximum(cur_counts / len(current), floor)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compare(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
    bins: int = 10,
) -> DriftReport:
    """Per-feature drift between a reference window and a current one."""
    shared = [c for c in reference.columns if c in current.columns]
    report = DriftReport(n_reference=len(reference), n_current=len(current))

    for column in shared:
        ref = reference[column].to_numpy(dtype=float)
        cur = current[column].to_numpy(dtype=float)
        ks = stats.ks_2samp(ref, cur)
        report.features.append(
            FeatureDrift(
                feature=column,
                psi=population_stability_index(ref, cur, bins=bins),
                ks_statistic=float(ks.statistic),
                ks_pvalue=float(ks.pvalue),
                reference_mean=float(ref.mean()),
                current_mean=float(cur.mean()),
                seasonal=column in SEASONAL_FEATURES,
            )
        )

    if reference_scores is not None and current_scores is not None:
        report.score_psi = population_stability_index(
            reference_scores, current_scores, bins=bins
        )
    return report
