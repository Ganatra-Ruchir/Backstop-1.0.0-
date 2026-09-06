"""
Loading and validating the raw transaction file.

The source is the ULB / Worldline credit-card dataset: 284,807 real card
transactions from two days in September 2013, of which 492 are fraud. Features
V1-V28 are principal components published in place of the original fields,
which is a privacy measure and also a real constraint — see `docs/MODEL_CARD.md`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backstop.config import CONFIG, RAW_COLUMNS

TARGET = "Class"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492


class DataError(RuntimeError):
    """The file on disk is not the dataset this project was built against."""


@dataclass
class QualityReport:
    """What the data actually looks like, checked rather than assumed."""

    rows: int
    columns: int
    frauds: int
    fraud_rate: float
    missing_cells: int
    exact_duplicates: int
    duplicate_frauds: int
    zero_amount_rows: int
    time_span_hours: float
    amount_min: float
    amount_median: float
    amount_max: float
    constant_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()
        }

    def render(self) -> str:
        lines = [
            "data quality report",
            "-------------------",
            f"  rows                {self.rows:,}",
            f"  columns             {self.columns}",
            f"  frauds              {self.frauds:,} ({self.fraud_rate * 100:.4f}%)",
            f"  missing cells       {self.missing_cells:,}",
            f"  exact duplicates    {self.exact_duplicates:,} "
            f"({self.duplicate_frauds} of them fraud)",
            f"  zero-amount rows    {self.zero_amount_rows:,}",
            f"  observation window  {self.time_span_hours:.1f} h",
            f"  amount              min {self.amount_min:,.2f} / "
            f"median {self.amount_median:,.2f} / max {self.amount_max:,.2f}",
        ]
        if self.constant_columns:
            lines.append(f"  constant columns    {', '.join(self.constant_columns)}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def file_digest(path: Path) -> str:
    """SHA-256 of the raw file, so a run can name the exact bytes it used."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path | None = None, *, strict: bool = True) -> pd.DataFrame:
    """
    Read the transaction file.

    The published CSV carries no header row and quotes the class label, so the
    column names and the target dtype are both imposed here rather than
    inferred — inference on this file silently yields a string target and a
    first row that has gone missing into the header.
    """
    path = Path(path or CONFIG.raw_csv)
    if not path.exists():
        raise DataError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )

    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, dtype=str)
    for column in RAW_COLUMNS:
        df[column] = pd.to_numeric(df[column].str.strip().str.strip('"'), errors="coerce")

    if df[TARGET].isna().any():
        raise DataError("target column contains values that are not 0 or 1")
    df[TARGET] = df[TARGET].astype(np.int8)

    if strict:
        _assert_expected_shape(df)
    return df


def _assert_expected_shape(df: pd.DataFrame) -> None:
    if len(df) != EXPECTED_ROWS:
        raise DataError(
            f"expected {EXPECTED_ROWS:,} rows, found {len(df):,}. "
            "The file is not the dataset this project reports metrics on."
        )
    frauds = int(df[TARGET].sum())
    if frauds != EXPECTED_FRAUD:
        raise DataError(f"expected {EXPECTED_FRAUD} frauds, found {frauds}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise DataError("target is not binary")


def profile(df: pd.DataFrame) -> QualityReport:
    """Describe the data, including the parts that need a decision."""
    duplicated = df.duplicated(keep="first")
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    report = QualityReport(
        rows=len(df),
        columns=df.shape[1],
        frauds=int(df[TARGET].sum()),
        fraud_rate=float(df[TARGET].mean()),
        missing_cells=int(df.isna().sum().sum()),
        exact_duplicates=int(duplicated.sum()),
        duplicate_frauds=int(df.loc[duplicated, TARGET].sum()),
        zero_amount_rows=int((df["Amount"] == 0).sum()),
        time_span_hours=float((df["Time"].max() - df["Time"].min()) / 3600),
        amount_min=float(df["Amount"].min()),
        amount_median=float(df["Amount"].median()),
        amount_max=float(df["Amount"].max()),
        constant_columns=constant,
    )

    if report.exact_duplicates:
        report.notes.append(
            f"{report.exact_duplicates:,} rows are byte-identical to an earlier row. "
            "They are dropped before splitting: a duplicate straddling the split "
            "boundary would put the same transaction in train and test."
        )
    if report.zero_amount_rows:
        report.notes.append(
            f"{report.zero_amount_rows:,} transactions have an amount of 0.00 — "
            "card verification requests, most likely. They are kept: they are real "
            "traffic the model will be asked to score."
        )
    return report


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop byte-identical rows, keeping the first occurrence.

    This runs *before* the temporal split, deliberately. A duplicate pair with
    one copy either side of the boundary is the same transaction appearing in
    both training and test — the model would be graded on a row it had already
    memorised.
    """
    before = len(df)
    out = df.drop_duplicates(keep="first").reset_index(drop=True)
    return out, before - len(out)
