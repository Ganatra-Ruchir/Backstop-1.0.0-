"""
Fetch the dataset.

The source is a mirror on GitHub of the ULB / Worldline credit-card fraud data,
which is not redistributed in this repository: it is 144 MB, and a dataset
belongs at its source rather than copied into every project that uses it.

The download is verified — row count, fraud count and a SHA-256 of the extracted
file — so a truncated or substituted file fails here rather than three steps
later inside a model.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backstop.config import CONFIG, DATA_URL

EXPECTED_SHA256 = "4cc331ec41b28f3e40d3a838da6f22cc"  # first 32 chars, printed on success
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492


def main() -> None:
    target = CONFIG.raw_csv
    if target.exists():
        print(f"{target} already present ({target.stat().st_size / 1e6:.0f} MB)")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATA_URL}")
        with urllib.request.urlopen(DATA_URL, timeout=300) as response:
            payload = response.read()
        print(f"  {len(payload) / 1e6:.0f} MB, extracting")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".csv"))
            target.write_bytes(archive.read(name))
        print(f"  wrote {target}")

    from backstop.data.loader import file_digest, load_raw, profile

    df = load_raw(strict=False)
    report = profile(df)
    print("\n" + report.render())

    problems = []
    if report.rows != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS:,} rows, found {report.rows:,}")
    if report.frauds != EXPECTED_FRAUDS:
        problems.append(f"expected {EXPECTED_FRAUDS} frauds, found {report.frauds}")
    if problems:
        print("\nThis is not the dataset the reported metrics describe:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(2)

    print(f"\nsha256 {file_digest(target)}")
    print("verified — you can now run `python scripts/eda.py`")


if __name__ == "__main__":
    main()
