#!/usr/bin/env python3
"""Make staged Hugging Face checkpoints self-contained for offline jobs."""

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
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


def patch_phi_transformers_compatibility(checkpoint_dir: Path) -> list[str]:
    """Patch Phi-4 remote code for Transformers 5 model initialization."""
    changed = []
    replacements = {
        "speech_conformer_encoder.py": [
            (
                "in_length = torch.tensor(feat_in, dtype=torch.float)",
                'in_length = torch.tensor(feat_in, dtype=torch.float, device="cpu")',
                "CPU shape calculation",
            ),
        ],
        "modeling_phi4mm.py": [
            (
                '_tied_weights_keys = ["lm_head.weight"]',
                '_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}',
                "Transformers 5 tied-weight mapping",
            ),
            (
                'task_type="CAUSAL_LM",',
                "task_type=None,",
                "generic inner-model LoRA adapters",
            ),
        ],
    }
    for filename, file_replacements in replacements.items():
        source_path = checkpoint_dir / filename
        if not source_path.is_file():
            continue
        source = source_path.read_text()
        for old, new, description in file_replacements:
            if old not in source:
                continue
            source = source.replace(old, new)
            changed.append(description)
        source_path.write_text(source)
    return changed


def transformers_major_version() -> int:
    try:
        return int(version("transformers").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    if not args.config.is_file():
        return
    for old, new in localize_auto_map(args.config):
        print(f"Localized Hugging Face auto_map: {old} -> {new}")
    if transformers_major_version() >= 5:
        for description in patch_phi_transformers_compatibility(args.config.parent):
            print(f"Patched Phi-4 remote code: {description}")


if __name__ == "__main__":
    main()
