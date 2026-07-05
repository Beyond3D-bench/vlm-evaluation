#!/usr/bin/env python3
"""
Simple Open3D viewer with smaller point size.

Examples:
python visualize_ply.py processed/final_processed.ply --point_size 1.0
python visualize_ply.py processed/cuts/cut_x_keep_low_75.ply processed/slices/slice_z_q50.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_files", nargs="+", type=Path)
    ap.add_argument("--point_size", type=float, default=1.0)
    ap.add_argument("--background", choices=["white", "black"], default="white")
    args = ap.parse_args()

    geoms = []
    for path in args.ply_files:
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            print(f"WARNING: empty/unreadable: {path}")
            continue
        print(f"{path}: {len(pcd.points):,} points, has_colors={pcd.has_colors()}")
        geoms.append(pcd)

    if not geoms:
        raise SystemExit("No readable point clouds.")

    vis = o3d.visualization.Visualizer()
    window_ok = vis.create_window(window_name="Point cloud viewer", width=1400, height=900)
    if not window_ok:
        raise SystemExit(
            "Open3D could not create a viewer window. This usually means the current "
            "session is headless or has no working OpenGL/GLFW display. Run this on a "
            "machine with a desktop display, connect with X11/VNC, or open the PLY in "
            "CloudCompare/MeshLab locally."
        )
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    if args.background == "white":
        opt.background_color = [1, 1, 1]
    else:
        opt.background_color = [0, 0, 0]

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
