#!/usr/bin/env python3
"""
Quick diagnostic for per-frame point clouds.

If each frame cloud has a very similar center/extent and all frames overlap like a blob,
you may be accidentally merging camera-coordinate points instead of world-coordinate points.

Usage:
python check_world_vs_camera_coords.py --input_dir frame_pointclouds --pattern "*.ply" --max_frames 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--pattern", default="*.ply")
    ap.add_argument("--max_frames", type=int, default=50)
    args = ap.parse_args()

    paths = sorted(args.input_dir.glob(args.pattern))[: args.max_frames]
    if not paths:
        raise SystemExit("No files found.")

    centers = []
    extents = []

    for path in paths:
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            continue
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        centers.append(pts.mean(axis=0))
        extents.append(mx - mn)
        print(f"{path.name:50s} center={centers[-1]} extent={extents[-1]}")

    centers = np.vstack(centers)
    extents = np.vstack(extents)

    print("\nSummary:")
    print(f"center std across frames: {centers.std(axis=0)}")
    print(f"extent mean:              {extents.mean(axis=0)}")
    print(f"extent std:               {extents.std(axis=0)}")

    print("\nInterpretation:")
    print("- If centers barely move even though the camera moves a lot, check your coordinate system.")
    print("- For world-frame clouds, frame centers should change according to observed scene region/camera path.")
    print("- For camera-frame clouds wrongly merged together, everything often forms an enclosed shell/blob.")


if __name__ == "__main__":
    main()
