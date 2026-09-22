from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "environments" / "profiles.json"


@dataclass(frozen=True)
class EnvironmentProfile:
    name: str
    directory: str
    requirements: Path
    lock: Path


def load_environment_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported environment manifest: {path}")
    return manifest


def get_profile(name: str, path: Path = DEFAULT_MANIFEST) -> EnvironmentProfile:
    manifest = load_environment_manifest(path)
    try:
        entry = manifest["profiles"][name]
    except KeyError as exc:
        known = ", ".join(sorted(manifest.get("profiles", {})))
        raise ValueError(f"Unknown environment profile {name!r}; choose one of: {known}") from exc
    root = path.resolve().parents[1]
    return EnvironmentProfile(
        name=name,
        directory=entry["directory"],
        requirements=root / entry["requirements"],
        lock=root / entry["lock"],
    )


def profile_for_preset(preset: str, path: Path = DEFAULT_MANIFEST) -> str:
    manifest = load_environment_manifest(path)
    try:
        return manifest["preset_profiles"][preset]
    except KeyError as exc:
        raise ValueError(f"No environment profile is declared for model preset {preset!r}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve model environment profile metadata.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile")
    selection.add_argument("--preset")
    parser.add_argument("--field", choices=("name", "directory", "requirements", "lock"), default="name")
    args = parser.parse_args()

    name = args.profile or profile_for_preset(args.preset)
    profile = get_profile(name)
    values = {
        "name": profile.name,
        "directory": profile.directory,
        "requirements": profile.requirements,
        "lock": profile.lock,
    }
    print(values[args.field])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
