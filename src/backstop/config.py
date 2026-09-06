"""
Configuration.

Everything that changes an experiment's outcome lives here, not scattered
through scripts. A run is reproducible from this file plus a random seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
FIGURE_DIR = ROOT / "docs" / "figures"

RANDOM_SEED = 20260907

#: Columns as they appear in the source file, which ships without a header.
RAW_COLUMNS: list[str] = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/creditcard.csv.zip"


@dataclass(frozen=True)
class SplitConfig:
    """
    Temporal split, as a fraction of the observation window.

    Fraud models are deployed to score transactions that have not happened yet,
    so the only split that estimates deployed performance is one that respects
    time. A random split lets a model learn from transactions occurring *after*
    the ones it is graded on; `scripts/leakage_demo.py` measures how much that
    inflates the result on this exact data.

    Model choice, hyperparameters and the decision threshold are all settled
    inside the development window using expanding-window cross-validation. The
    test window is scored once, at the end.

    Why not a single validation slice: 20% of the window holds roughly 45
    frauds. A threshold estimated from 45 positives has a confidence interval
    wide enough to be meaningless. Pooling out-of-fold predictions across the
    development folds gives ~370 positives to place it with.
    """

    dev_end: float = 0.80          # remainder is the held-out test window
    n_folds: int = 4               # expanding-window folds inside development
    #: Rows within this many seconds after a fold's training edge are dropped
    #: from that fold's validation side. Nothing in this dataset carries state
    #: across rows, but the gap makes the guarantee structural rather than
    #: dependent on that continuing to be true.
    purge_seconds: float = 600.0


@dataclass(frozen=True)
class CostConfig:
    """
    What a decision is worth, in the currency of the dataset (EUR).

    These are assumptions, not measurements, and the README says so. Every
    number below is varied in the sensitivity analysis, because a threshold
    chosen under one cost ratio is not the right threshold under another.
    """

    #: What an analyst review costs, whether or not the transaction is fraud.
    review_cost: float = 3.00
    #: Share of a blocked legitimate transaction's value lost to abandonment.
    #: A declined customer does not always come back.
    false_decline_friction: float = 0.05
    #: Share of a caught fraud that is actually recovered by blocking it.
    recovery_rate: float = 1.00


@dataclass(frozen=True)
class ServingConfig:
    #: Fraction of traffic a human review team can absorb. Threshold selection
    #: is constrained by this: a model that flags 5% of traffic is unusable
    #: however good its recall looks.
    alert_budget: float = 0.001
    latency_budget_ms: float = 25.0


@dataclass(frozen=True)
class Config:
    seed: int = RANDOM_SEED
    split: SplitConfig = field(default_factory=SplitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    raw_csv: Path = DATA_DIR / "raw" / "creditcard.csv"
    artifact_dir: Path = ARTIFACT_DIR
    figure_dir: Path = FIGURE_DIR


CONFIG = Config()
