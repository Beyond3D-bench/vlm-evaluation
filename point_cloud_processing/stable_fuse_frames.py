#!/usr/bin/env python3
"""
Fuse one PLY per frame by keeping only temporally stable voxels.

This is useful for egocentric kitchen reconstruction:
- static surfaces appear repeatedly in similar 3D positions
- hands/tools/food-in-motion appear only briefly
- keeping voxels seen in >= min_frames suppresses transient clutter

Input assumption:
- each input PLY is already in the same world coordinate system.
- if the PLYs are in camera coordinates, this will NOT work correctly.

Example:
python stable_fuse_frames.py \
  --input_dir frame_pointclouds \
  --out stable_background.ply \
  --voxel 0.03 \
  --min_frames 5
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import open3d as o3d
from tqdm import tqdm


VoxelKey = Tuple[int, int, int]


def read_pcd(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Empty/unreadable cloud: {path}")
    return pcd


def voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    return np.floor(points / voxel).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--pattern", default="*.ply")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--voxel", type=float, required=True)
    ap.add_argument("--min_frames", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=0, help="0 means all frames.")
    ap.add_argument("--per_frame_voxel", type=float, default=0.0, help="Optional downsample before fusion.")
    ap.add_argument("--sor_neighbors", type=int, default=0, help="Optional per-frame SOR cleanup.")
    ap.add_argument("--sor_std", type=float, default=2.0)
    args = ap.parse_args()

    paths = sorted(args.input_dir.glob(args.pattern))
    if args.max_frames and args.max_frames > 0:
        paths = paths[: args.max_frames]
    if not paths:
        raise SystemExit(f"No files found in {args.input_dir} with pattern {args.pattern}")

    print(f"Found {len(paths)} frame clouds")

    # Accumulators per voxel.
    count_frames: Dict[VoxelKey, int] = defaultdict(int)
    sum_xyz: Dict[VoxelKey, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    sum_rgb: Dict[VoxelKey, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    count_points: Dict[VoxelKey, int] = defaultdict(int)
    has_any_color = False

    for path in tqdm(paths, desc="Reading/fusing frames"):
        pcd = read_pcd(path)

        if args.per_frame_voxel and args.per_frame_voxel > 0:
            pcd = pcd.voxel_down_sample(args.per_frame_voxel)

        if args.sor_neighbors and args.sor_neighbors > 0:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=args.sor_neighbors,
                std_ratio=args.sor_std,
            )

        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            continue

        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]

        colors = None
        if pcd.has_colors():
            cols = np.asarray(pcd.colors)[finite]
            colors = cols
            has_any_color = True

        keys_arr = voxel_keys(pts, args.voxel)

        # Within each frame, count a voxel once for temporal support.
        unique_keys, inverse = np.unique(keys_arr, axis=0, return_inverse=True)

        for ui, key_vec in enumerate(unique_keys):
            mask = inverse == ui
            key = tuple(int(x) for x in key_vec)
            pts_here = pts[mask]

            count_frames[key] += 1
            sum_xyz[key] += pts_here.sum(axis=0)
            count_points[key] += len(pts_here)

            if colors is not None:
                sum_rgb[key] += colors[mask].sum(axis=0)

    kept_pts: List[np.ndarray] = []
    kept_cols: List[np.ndarray] = []

    for key, frame_count in count_frames.items():
        if frame_count < args.min_frames:
            continue

        n = count_points[key]
        kept_pts.append(sum_xyz[key] / max(n, 1))

        if has_any_color:
            kept_cols.append(sum_rgb[key] / max(n, 1))

    if not kept_pts:
        raise SystemExit(
            "No voxels survived. Try lower --min_frames or larger --voxel."
        )

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(np.vstack(kept_pts))

    if has_any_color and len(kept_cols) == len(kept_pts):
        cols = np.vstack(kept_cols)
        cols = np.clip(cols, 0.0, 1.0)
        out_pcd.colors = o3d.utility.Vector3dVector(cols)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(args.out), out_pcd)
    if not ok:
        raise RuntimeError(f"Failed to write {args.out}")

    print(f"saved: {args.out}")
    print(f"kept voxels/points: {len(out_pcd.points):,}")
    print(f"min_frames: {args.min_frames}, voxel: {args.voxel}")


if __name__ == "__main__":
    main()
