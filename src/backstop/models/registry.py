"""
Saving a model so that the thing you load is the thing you evaluated.

A pickle on its own is not a model artifact. Loading one tells you nothing
about what data it saw, which library versions produced it, what threshold it
is supposed to run at, or what it scored — and every one of those is needed
before anyone should let it decline a customer's card.

So an artifact here is the fitted pipeline plus a manifest, and loading
verifies the manifest: same feature list, same library major versions, and a
checksum over the pickle. A mismatch raises instead of quietly scoring.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib

from backstop.config import CONFIG

MANIFEST_NAME = "manifest.json"
MODEL_NAME = "model.joblib"


class ModelError(RuntimeError):
    """The artifact on disk cannot be trusted to score traffic."""


@dataclass
class Manifest:
    """Everything needed to reproduce, audit or reject a saved model."""

    version: str
    model_name: str
    created_at: str
    #: Decision threshold this model is meant to run at, and how it was chosen.
    threshold: float
    threshold_basis: str
    feature_names: list[str]
    n_features: int
    trained_rows: int
    trained_frauds: int
    train_window_hours: float
    data_sha256: str
    model_sha256: str
    metrics: dict = field(default_factory=dict)
    library_versions: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions() -> dict:
    import numpy
    import pandas
    import sklearn

    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for name in ("xgboost", "lightgbm"):
        # An absent optional dependency is not an error: the core install
        # ships without LightGBM, and an artifact trained without it must
        # still load and score.
        with contextlib.suppress(Exception):
            versions[name] = __import__(name).__version__
    return versions


def save(
    pipeline,
    *,
    version: str,
    model_name: str,
    threshold: float,
    threshold_basis: str,
    feature_names: list[str],
    trained_rows: int,
    trained_frauds: int,
    train_window_hours: float,
    data_sha256: str,
    metrics: dict,
    notes: list[str] | None = None,
    root: Path | None = None,
) -> Path:
    """Write a versioned artifact directory and return its path."""
    root = Path(root or CONFIG.artifact_dir) / "models" / version
    root.mkdir(parents=True, exist_ok=True)

    model_path = root / MODEL_NAME
    joblib.dump(pipeline, model_path, compress=3)

    manifest = Manifest(
        version=version,
        model_name=model_name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        threshold=float(threshold),
        threshold_basis=threshold_basis,
        feature_names=list(feature_names),
        n_features=len(feature_names),
        trained_rows=int(trained_rows),
        trained_frauds=int(trained_frauds),
        train_window_hours=float(train_window_hours),
        data_sha256=data_sha256,
        model_sha256=_sha256_file(model_path),
        metrics=metrics,
        library_versions=library_versions(),
        notes=notes or [],
    )
    (root / MANIFEST_NAME).write_text(manifest.to_json())
    (Path(root).parent / "latest.txt").write_text(version)
    return root


def load(version: str | None = None, *, root: Path | None = None, strict: bool = True):
    """
    Load an artifact, refusing to return one that does not verify.

    `strict=False` downgrades the library-version check to a warning, which is
    what you want when deliberately testing an old artifact on a new stack and
    never what you want in production.
    """
    models_root = Path(root or CONFIG.artifact_dir) / "models"
    if version is None:
        pointer = models_root / "latest.txt"
        if not pointer.exists():
            raise ModelError(f"no model has been trained yet (looked in {models_root})")
        version = pointer.read_text().strip()

    directory = models_root / version
    manifest_path = directory / MANIFEST_NAME
    model_path = directory / MODEL_NAME
    if not manifest_path.exists() or not model_path.exists():
        raise ModelError(f"artifact {version} is incomplete at {directory}")

    manifest = Manifest(**json.loads(manifest_path.read_text()))

    digest = _sha256_file(model_path)
    if digest != manifest.model_sha256:
        raise ModelError(
            f"{model_path} does not match its manifest checksum — the artifact "
            "has been modified or truncated since it was written"
        )

    current, recorded = library_versions(), manifest.library_versions
    drifted = [
        f"{k}: trained on {v}, running {current.get(k, 'absent')}"
        for k, v in recorded.items()
        if k in current and current[k].split(".")[0] != v.split(".")[0]
    ]
    if drifted:
        message = "major library version change since training — " + "; ".join(drifted)
        if strict:
            raise ModelError(message)
        import warnings

        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return joblib.load(model_path), manifest


def list_versions(root: Path | None = None) -> list[str]:
    models_root = Path(root or CONFIG.artifact_dir) / "models"
    if not models_root.exists():
        return []
    return sorted(p.name for p in models_root.iterdir() if p.is_dir())
