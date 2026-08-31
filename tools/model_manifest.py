from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class HuggingFaceArtifact:
    model: str
    repo_id: str
    revision: str | None


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError(f"Unsupported model manifest: {path}")
    return manifest


def iter_huggingface_artifacts(
    manifest: dict[str, Any], selected_models: Iterable[str] | None = None
) -> list[HuggingFaceArtifact]:
    models = manifest.get("huggingface_models", {})
    if not isinstance(models, dict):
        raise ValueError("huggingface_models must be a mapping")

    selected = set(selected_models or models)
    unknown = selected.difference(models)
    if unknown:
        raise ValueError(f"Unknown model selection: {', '.join(sorted(unknown))}")

    artifacts: list[HuggingFaceArtifact] = []
    for model, entry in models.items():
        if model not in selected:
            continue
        for artifact in entry.get("artifacts", []):
            artifacts.append(
                HuggingFaceArtifact(
                    model=model,
                    repo_id=artifact["repo_id"],
                    revision=artifact.get("revision"),
                )
            )
    return artifacts


def require_pinned(artifacts: Iterable[HuggingFaceArtifact]) -> None:
    unpinned = [f"{item.model}: {item.repo_id}" for item in artifacts if not item.revision]
    if unpinned:
        details = "\n  - ".join(unpinned)
        raise ValueError(
            "The following Hugging Face artifacts do not have pinned revisions:\n"
            f"  - {details}\n"
            "Recover and record their exact commit IDs in models/manifest.yaml."
        )

