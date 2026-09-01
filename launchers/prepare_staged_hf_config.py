#!/usr/bin/env python3
"""Make staged Hugging Face checkpoints self-contained for offline jobs."""

import argparse
import json
from pathlib import Path


def localize_auto_map(config_path: Path) -> list[tuple[str, str]]:
    data = json.loads(config_path.read_text())
    auto_map = data.get("auto_map")
    if not isinstance(auto_map, dict):
        return []

    changed = []
    for key, value in auto_map.items():
        if not isinstance(value, str) or "--" not in value:
            continue
        _, local_reference = value.rsplit("--", 1)
        module_name = local_reference.split(".", 1)[0]
        if not (config_path.parent / f"{module_name}.py").is_file():
            continue
        auto_map[key] = local_reference
        changed.append((value, local_reference))

    if changed:
        config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    if not args.config.is_file():
        return
    for old, new in localize_auto_map(args.config):
        print(f"Localized Hugging Face auto_map: {old} -> {new}")


if __name__ == "__main__":
    main()
