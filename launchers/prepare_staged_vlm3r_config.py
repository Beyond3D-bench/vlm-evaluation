#!/usr/bin/env python3
"""Point a staged VLM-3R config at its node-local SigLIP snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} CONFIG_JSON SIGLIP_DIR")

    config_path = Path(sys.argv[1])
    siglip_dir = str(Path(sys.argv[2]).resolve())
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    config["mm_vision_tower"] = siglip_dir

    temporary_path = config_path.with_suffix(config_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
    temporary_path.replace(config_path)


if __name__ == "__main__":
    main()
