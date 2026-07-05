#!/usr/bin/env python3
"""
Process a raw merged kitchen point cloud.

Main features:
- Reads PLY/PCD/etc. via Open3D
- Removes invalid points
- Optional voxel downsampling
- Optional statistical outlier removal
- Optional radius outlier removal
- Optional quantile crop
- Creates cut-open versions
- Creates thin cross-section slices
- Optional normal estimation
- Optional alpha-shape mesh reconstruction

Example:
python process_pointcloud.py \
  --input raw_kitchen.ply \
  --out_dir processed \
  --voxel 0.03 \
  --sor_neighbors 30 \
  --sor_std 2.0 \
  --make_cuts \
  --make_slices
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_cloud(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Could not read point cloud or file is empty: {path}")
    return pcd


def print_stats(name: str, pcd: o3d.geometry.PointCloud) -> None:
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print(f"[{name}] empty")
        return
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = pts.mean(axis=0)
    extent = mx - mn
    print(f"\n[{name}]")
    print(f"  points: {len(pts):,}")
    print(f"  min:    {mn}")
    print(f"  max:    {mx}")
    print(f"  center: {center}")
    print(f"  extent: {extent}")


def remove_nonfinite(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    mask = np.isfinite(pts).all(axis=1)
    if hasattr(pcd, "colors") and len(pcd.colors) == len(pts):
        colors = np.asarray(pcd.colors)
        mask = mask & np.isfinite(colors).all(axis=1)
    inds = np.where(mask)[0]
    return pcd.select_by_index(inds)


def quantile_crop(
    pcd: o3d.geometry.PointCloud,
    crop_x: Optional[Tuple[float, float]],
    crop_y: Optional[Tuple[float, float]],
    crop_z: Optional[Tuple[float, float]],
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    keep = np.ones(len(pts), dtype=bool)

    crops = [crop_x, crop_y, crop_z]
    for axis, crop in enumerate(crops):
        if crop is None:
            continue
        lo_q, hi_q = crop
        lo = np.quantile(pts[:, axis], lo_q)
        hi = np.quantile(pts[:, axis], hi_q)
        keep &= (pts[:, axis] >= lo) & (pts[:, axis] <= hi)
        print(f"  crop axis {axis}: quantiles [{lo_q}, {hi_q}] -> values [{lo:.4f}, {hi:.4f}]")

    return pcd.select_by_index(np.where(keep)[0])


def save_cloud(path: Path, pcd: o3d.geometry.PointCloud) -> None:
    ensure_dir(path.parent)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)
    if not ok:
        raise RuntimeError(f"Failed to write {path}")
    print(f"saved: {path} ({len(pcd.points):,} points)")


def make_cut_versions(pcd: o3d.geometry.PointCloud, out_dir: Path) -> None:
    """
    Make simple cut-open files:
    - keep lower 75% along each axis
    - keep upper 75% along each axis
    - keep middle 60% along each axis
    """
    ensure_dir(out_dir)
    pts = np.asarray(pcd.points)

    for axis_name, axis in [("x", 0), ("y", 1), ("z", 2)]:
        q25 = np.quantile(pts[:, axis], 0.25)
        q75 = np.quantile(pts[:, axis], 0.75)
        q20 = np.quantile(pts[:, axis], 0.20)
        q80 = np.quantile(pts[:, axis], 0.80)

        masks = {
            f"cut_{axis_name}_keep_low_75.ply": pts[:, axis] <= q75,
            f"cut_{axis_name}_keep_high_75.ply": pts[:, axis] >= q25,
            f"cut_{axis_name}_keep_mid_20_80.ply": (pts[:, axis] >= q20) & (pts[:, axis] <= q80),
        }

        for filename, mask in masks.items():
            sub = pcd.select_by_index(np.where(mask)[0])
            save_cloud(out_dir / filename, sub)


def make_slices(pcd: o3d.geometry.PointCloud, out_dir: Path, thickness_q: float = 0.05) -> None:
    """
    Create thin quantile slices around 25%, 50%, 75% along each axis.
    thickness_q=0.05 means each slice keeps roughly 5% of the cloud along that axis.
    """
    ensure_dir(out_dir)
    pts = np.asarray(pcd.points)
    half = thickness_q / 2.0

    for axis_name, axis in [("x", 0), ("y", 1), ("z", 2)]:
        values = pts[:, axis]
        for center_q in [0.25, 0.50, 0.75]:
            lo_q = max(0.0, center_q - half)
            hi_q = min(1.0, center_q + half)
            lo = np.quantile(values, lo_q)
            hi = np.quantile(values, hi_q)
            mask = (values >= lo) & (values <= hi)
            sub = pcd.select_by_index(np.where(mask)[0])
            save_cloud(out_dir / f"slice_{axis_name}_q{int(center_q*100):02d}.ply", sub)


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int) -> o3d.geometry.PointCloud:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(k=max_nn)
    return pcd


def reconstruct_alpha_mesh(
    pcd: o3d.geometry.PointCloud,
    alpha: float,
    out_path: Path,
) -> None:
    """
    Alpha shape can be useful for quick visual surface reconstruction.
    For raw egocentric kitchen clouds, use carefully: dynamic clutter can create bad surfaces.
    """
    print(f"Reconstructing alpha mesh with alpha={alpha} ...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    ok = o3d.io.write_triangle_mesh(str(out_path), mesh)
    if not ok:
        raise RuntimeError(f"Failed to write mesh {out_path}")
    print(f"saved mesh: {out_path} ({len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles)")


def parse_quantile_pair(values: Optional[list[float]]) -> Optional[Tuple[float, float]]:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("Crop argument must have exactly two numbers, e.g. --crop_x 0.05 0.95")
    lo, hi = values
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("Crop quantiles must satisfy 0 <= lo < hi <= 1")
    return (lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="Input point cloud, e.g. raw.ply")
    ap.add_argument("--out_dir", required=True, type=Path, help="Output directory")

    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size. 0 disables.")
    ap.add_argument("--sor_neighbors", type=int, default=0, help="Statistical outlier nb_neighbors. 0 disables.")
    ap.add_argument("--sor_std", type=float, default=2.0, help="Statistical outlier std_ratio.")
    ap.add_argument("--radius", type=float, default=0.0, help="Radius outlier search radius. 0 disables.")
    ap.add_argument("--radius_min_points", type=int, default=16, help="Radius outlier minimum neighbors.")

    ap.add_argument("--crop_x", nargs=2, type=float, default=None, metavar=("LO_Q", "HI_Q"))
    ap.add_argument("--crop_y", nargs=2, type=float, default=None, metavar=("LO_Q", "HI_Q"))
    ap.add_argument("--crop_z", nargs=2, type=float, default=None, metavar=("LO_Q", "HI_Q"))

    ap.add_argument("--make_cuts", action="store_true", help="Create cut-open PLY versions.")
    ap.add_argument("--make_slices", action="store_true", help="Create thin cross-section slices.")
    ap.add_argument("--slice_thickness_q", type=float, default=0.05)

    ap.add_argument("--estimate_normals", action="store_true")
    ap.add_argument("--normal_radius", type=float, default=0.10)
    ap.add_argument("--normal_max_nn", type=int, default=30)

    ap.add_argument("--alpha_mesh", type=float, default=0.0, help="If >0, create alpha-shape mesh with this alpha.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    pcd = read_cloud(args.input)
    print_stats("raw", pcd)
    pcd = remove_nonfinite(pcd)
    print_stats("nonfinite_removed", pcd)
    save_cloud(args.out_dir / "00_raw_copy.ply", pcd)

    current = pcd

    if args.voxel and args.voxel > 0:
        current = current.voxel_down_sample(args.voxel)
        print_stats("downsampled", current)
        save_cloud(args.out_dir / "01_downsampled.ply", current)

    if args.sor_neighbors and args.sor_neighbors > 0:
        current, ind = current.remove_statistical_outlier(
            nb_neighbors=args.sor_neighbors,
            std_ratio=args.sor_std,
        )
        print_stats("clean_sor", current)
        save_cloud(args.out_dir / "02_clean_sor.ply", current)

    if args.radius and args.radius > 0:
        current, ind = current.remove_radius_outlier(
            nb_points=args.radius_min_points,
            radius=args.radius,
        )
        print_stats("clean_radius", current)
        save_cloud(args.out_dir / "02b_clean_radius.ply", current)

    crop_x = parse_quantile_pair(args.crop_x)
    crop_y = parse_quantile_pair(args.crop_y)
    crop_z = parse_quantile_pair(args.crop_z)
    if crop_x is not None or crop_y is not None or crop_z is not None:
        print("\nApplying quantile crop:")
        current = quantile_crop(current, crop_x, crop_y, crop_z)
        print_stats("crop_quantile", current)
        save_cloud(args.out_dir / "03_crop_quantile.ply", current)

    if args.estimate_normals:
        current = estimate_normals(current, args.normal_radius, args.normal_max_nn)
        save_cloud(args.out_dir / "04_with_normals.ply", current)

    save_cloud(args.out_dir / "final_processed.ply", current)

    if args.make_cuts:
        make_cut_versions(current, args.out_dir / "cuts")

    if args.make_slices:
        make_slices(current, args.out_dir / "slices", thickness_q=args.slice_thickness_q)

    if args.alpha_mesh and args.alpha_mesh > 0:
        reconstruct_alpha_mesh(current, args.alpha_mesh, args.out_dir / "alpha_mesh.ply")


if __name__ == "__main__":
    main()
