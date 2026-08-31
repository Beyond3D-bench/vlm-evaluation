from pathlib import Path

import pytest

from tools.model_manifest import (
    iter_huggingface_artifacts,
    iter_source_repositories,
    load_manifest,
    require_pinned,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "models" / "manifest.yaml"


def test_source_revisions_and_patches_are_pinned() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    sources = iter_source_repositories(manifest, MANIFEST_PATH)

    for source in sources:
        revision = source.revision
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)
        for patch in source.patches:
            assert patch.file.is_file(), patch.file


def test_known_vlm3r_assets_are_pinned() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    artifacts = iter_huggingface_artifacts(manifest, ["vlm3r"])

    require_pinned(artifacts)
    assert {artifact.repo_id for artifact in artifacts} == {
        "Journey9ni/vlm-3r-llava-qwen2-lora",
        "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2",
    }


def test_unresolved_revision_is_rejected() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    artifacts = iter_huggingface_artifacts(manifest, ["qwen3_6"])

    with pytest.raises(ValueError, match="do not have pinned revisions"):
        require_pinned(artifacts)
