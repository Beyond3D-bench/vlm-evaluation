#!/usr/bin/env python3
"""
Boost PLY vertex colors without changing geometry.

This works directly on binary_little_endian PLY files with red/green/blue
vertex properties, so it does not require Open3D.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


PLY_TO_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _read_header(raw: bytes) -> Tuple[List[str], int]:
    marker = b"end_header\n"
    idx = raw.find(marker)
    if idx < 0:
        marker = b"end_header\r\n"
        idx = raw.find(marker)
    if idx < 0:
        raise ValueError("Could not find PLY end_header")
    header_end = idx + len(marker)
    header_text = raw[:header_end].decode("ascii", errors="strict")
    return header_text.splitlines(), header_end


def _parse_vertex_layout(header_lines: List[str]):
    fmt = None
    vertex_count = None
    properties: List[Tuple[str, str]] = []
    in_vertex = False

    for line in header_lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and parts[0] == "property" and len(parts) == 3:
            properties.append((parts[1], parts[2]))

    if fmt != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian PLY is supported, got: {fmt}")
    if vertex_count is None or not properties:
        raise ValueError("Could not parse vertex element/properties from PLY header")
    for name in ("red", "green", "blue"):
        if name not in {prop_name for _, prop_name in properties}:
            raise ValueError(f"PLY has no vertex color property: {name}")
    return vertex_count, properties


def _vertex_dtype(properties: List[Tuple[str, str]]) -> np.dtype:
    fields = []
    for ply_type, name in properties:
        if ply_type not in PLY_TO_DTYPE:
            raise ValueError(f"Unsupported PLY property type: {ply_type} for {name}")
        fields.append((name, "<" + PLY_TO_DTYPE[ply_type]))
    return np.dtype(fields)


def _stretch_colors(
    colors: np.ndarray,
    low_percentile: float,
    high_percentile: float,
    gamma: float,
    brightness: float,
    per_channel: bool,
) -> np.ndarray:
    colors = np.clip(colors.astype(np.float64), 0.0, 1.0)

    if per_channel:
        lo = np.percentile(colors, low_percentile, axis=0)
        hi = np.percentile(colors, high_percentile, axis=0)
    else:
        lo = np.percentile(colors, low_percentile)
        hi = np.percentile(colors, high_percentile)

    denom = np.maximum(hi - lo, 1e-6)
    boosted = np.clip((colors - lo) / denom, 0.0, 1.0)
    if gamma <= 0:
        raise ValueError("--gamma must be > 0")
    boosted = np.power(boosted, gamma)
    return np.clip(boosted * brightness, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="Input colored binary PLY.")
    ap.add_argument("--output", required=True, type=Path, help="Output color-boosted PLY.")
    ap.add_argument("--low", type=float, default=1.0, help="Low percentile for color stretch.")
    ap.add_argument("--high", type=float, default=99.5, help="High percentile for color stretch.")
    ap.add_argument("--gamma", type=float, default=0.65, help="Gamma < 1 brightens mid-tones.")
    ap.add_argument("--brightness", type=float, default=1.25, help="Final brightness multiplier.")
    ap.add_argument(
        "--global_stretch",
        action="store_true",
        help="Use one global color range instead of separate RGB channel ranges.",
    )
    args = ap.parse_args()

    raw = args.input.read_bytes()
    header_lines, header_end = _read_header(raw)
    vertex_count, properties = _parse_vertex_layout(header_lines)
    dtype = _vertex_dtype(properties)
    vertex_bytes = vertex_count * dtype.itemsize
    vertex_blob = raw[header_end : header_end + vertex_bytes]
    rest = raw[header_end + vertex_bytes :]
    if len(vertex_blob) != vertex_bytes:
        raise ValueError("PLY ended before all vertex records were read")

    verts = np.frombuffer(vertex_blob, dtype=dtype, count=vertex_count).copy()
    colors = np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.float64) / 255.0
    before_min = colors.min(axis=0)
    before_max = colors.max(axis=0)
    before_mean = colors.mean(axis=0)

    boosted = _stretch_colors(
        colors,
        low_percentile=args.low,
        high_percentile=args.high,
        gamma=args.gamma,
        brightness=args.brightness,
        per_channel=not args.global_stretch,
    )
    boosted_u8 = np.rint(boosted * 255.0).clip(0, 255).astype(np.uint8)
    verts["red"] = boosted_u8[:, 0]
    verts["green"] = boosted_u8[:, 1]
    verts["blue"] = boosted_u8[:, 2]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw[:header_end] + verts.tobytes() + rest)

    after = boosted_u8.astype(np.float64) / 255.0
    print(f"read: {args.input} ({vertex_count:,} points)")
    print(f"wrote: {args.output}")
    print(f"RGB before min={before_min} mean={before_mean} max={before_max}")
    print(f"RGB after  min={after.min(axis=0)} mean={after.mean(axis=0)} max={after.max(axis=0)}")


if __name__ == "__main__":
    main()
