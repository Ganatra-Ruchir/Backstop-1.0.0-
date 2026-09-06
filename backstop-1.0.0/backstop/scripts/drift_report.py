"""Drift between the training window and the production window."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from backstop.config import CONFIG
from backstop.data.loader import deduplicate, load_raw
from backstop.data.splits import temporal_split
from backstop.models import registry
from backstop.monitoring.drift import compare


def main() -> None:
    pipeline, manifest = registry.load()
    df, _ = deduplicate(load_raw())
    ds = temporal_split(df)

    # Compare the model's own inputs, not the raw columns: what the monitor
    # must watch is what the model actually consumes.
    features = pipeline[:-1]
    ref = features.transform(ds.X_dev)
    cur = features.transform(ds.X_test)
    if not hasattr(ref, "columns"):
        import pandas as pd

        from backstop.features.pipeline import FEATURE_NAMES

        ref = pd.DataFrame(ref, columns=FEATURE_NAMES)
        cur = pd.DataFrame(cur, columns=FEATURE_NAMES)

    report = compare(
        ref,
        cur,
        reference_scores=pipeline.predict_proba(ds.X_dev)[:, 1],
        current_scores=pipeline.predict_proba(ds.X_test)[:, 1],
    )
    print(f"model {manifest.version} ({manifest.model_name})\n")
    print(report.render(top=12))

    frame = report.to_frame().sort_values("psi", ascending=False)
    frame.to_csv(CONFIG.artifact_dir / "drift_report.csv", index=False)
    (CONFIG.artifact_dir / "drift_summary.json").write_text(
        json.dumps(
            {
                "n_reference": report.n_reference,
                "n_current": report.n_current,
                "score_psi": report.score_psi,
                "shifted": [f.feature for f in report.shifted],
                "watch": [f.feature for f in report.watch],
                "alarm": report.alarm,
            },
            indent=2,
        )
    )
    print(f"\nwrote {CONFIG.artifact_dir / 'drift_report.csv'}")


if __name__ == "__main__":
    main()
