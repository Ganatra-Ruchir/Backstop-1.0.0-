"""
Fixtures.

Most tests run against a small synthetic frame rather than the real 284,807-row
file: they are testing that the code does what it says, and a suite that takes
four minutes stops being run. The tests that genuinely need the real data are
marked `slow` and skip cleanly when the file is absent, so a fresh clone can run
the suite before downloading anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backstop.config import CONFIG, RAW_COLUMNS

V_COLUMNS = [f"V{i}" for i in range(1, 29)]


def make_frame(
    n: int = 4000, *, fraud_rate: float = 0.02, seed: int = 7, hours: float = 48.0
) -> pd.DataFrame:
    """
    A frame shaped like the real data, with signal a model can actually find.

    The fraud rate is deliberately higher than reality: at 0.17% a 4,000-row
    fixture holds seven positives, and a test that asserts anything about
    ranking would be measuring noise.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(np.int8)

    frame = pd.DataFrame(
        rng.normal(0, 1, size=(n, len(V_COLUMNS))).astype(float), columns=V_COLUMNS
    )
    # Two components carry the signal, as V14 and V17 do in the real data.
    frame.loc[y == 1, "V14"] -= rng.normal(4.0, 1.0, size=int(y.sum()))
    frame.loc[y == 1, "V17"] -= rng.normal(3.0, 1.0, size=int(y.sum()))

    frame["Time"] = np.sort(rng.uniform(0, hours * 3600, size=n))
    frame["Amount"] = np.round(np.exp(rng.normal(3.0, 1.2, size=n)), 2)
    frame["Class"] = y
    return frame[RAW_COLUMNS]


@pytest.fixture()
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture()
def tiny_frame() -> pd.DataFrame:
    return make_frame(n=600, fraud_rate=0.05, seed=11)


@pytest.fixture(scope="session")
def real_data():
    """The actual dataset, or a skip if it has not been downloaded."""
    from backstop.data.loader import DataError, deduplicate, load_raw

    if not CONFIG.raw_csv.exists():
        pytest.skip("run `python scripts/download_data.py` for the slow tests")
    try:
        df, _ = deduplicate(load_raw())
    except DataError as exc:
        pytest.skip(f"dataset unusable: {exc}")
    return df


@pytest.fixture(scope="session")
def trained_artifact():
    """The saved model, or a skip if `scripts/train.py` has not been run."""
    from backstop.models.registry import ModelError, load

    try:
        return load()
    except ModelError as exc:
        pytest.skip(f"no trained model: {exc}")
