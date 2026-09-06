"""Global SHAP importance for the shipped model, for the model card."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backstop.config import CONFIG
from backstop.data.loader import deduplicate, load_raw
from backstop.data.splits import temporal_split
from backstop.explain.reasons import global_importance
from backstop.models import registry


def main() -> None:
    pipeline, manifest = registry.load()
    df, _ = deduplicate(load_raw())
    ds = temporal_split(df)

    importance = global_importance(pipeline, ds.X_dev, sample=4000, seed=CONFIG.seed)
    (CONFIG.artifact_dir / "global_importance.json").write_text(
        json.dumps(importance, indent=2)
    )
    print(f"model {manifest.version} — mean |SHAP| over 4,000 development rows\n")
    for name, value in list(importance.items())[:12]:
        print(f"  {name:<18} {value:.4f}")
    print(f"\nwrote {CONFIG.artifact_dir / 'global_importance.json'}")


if __name__ == "__main__":
    main()
