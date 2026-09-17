import json
from pathlib import Path

import pytest

from tools.environment_profiles import get_profile, load_environment_manifest, profile_for_preset


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "environments" / "profiles.json"


def test_all_profiles_use_the_single_cuda_stack_and_have_locks() -> None:
    manifest = load_environment_manifest(MANIFEST_PATH)

    assert manifest["python"] == "3.10"
    assert manifest["cuda"] == "12.8"
    assert manifest["torch"] == "2.7.1"
    assert manifest["torchvision"] == "0.22.1"

    for name in manifest["profiles"]:
        profile = get_profile(name, MANIFEST_PATH)
        assert profile.requirements.is_file()
        assert profile.lock.is_file()
        lock = profile.lock.read_text(encoding="utf-8")
        assert "torch==2.7.1+cu128" in lock
        assert "torchvision==0.22.1+cu128" in lock


def test_every_launcher_preset_has_a_profile() -> None:
    common = (REPO_ROOT / "launchers" / "common.sh").read_text(encoding="utf-8")
    presets_line = next(line for line in common.splitlines() if line.startswith("OOS_MODEL_PRESETS="))
    presets = presets_line.split('"', 2)[1].split()

    for preset in presets:
        if preset == "custom":
            continue
        assert profile_for_preset(preset, MANIFEST_PATH)


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="No environment profile"):
        profile_for_preset("not-a-model", MANIFEST_PATH)
