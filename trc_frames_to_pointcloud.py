#!/usr/bin/env python3
"""Parse TRC and export one point cloud file per frame.

By default, exports the first 10 frames to ASCII PLY files.

Usage:
    python trc_frames_to_pointcloud.py
    python trc_frames_to_pointcloud.py C:/APP/BD/BD08132/BD08132-Unnamed.trc --frames 10 --format ply
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def _split_fields(line: str) -> List[str]:
    if "\t" in line:
        return [p.strip() for p in line.strip().split("\t") if p.strip() != ""]
    return [p.strip() for p in line.strip().split() if p.strip() != ""]


def _safe_float(raw: str) -> float:
    try:
        return float(raw)
    except Exception:
        return 0.0


def _extract_xyz_triplets(row: List[str]) -> Tuple[List[Tuple[float, float, float]], int]:
    # TRC common layout: Frame#, Time, Timestamp, then XYZ triplets.
    # Some files also have a "TimeCode" header slot, but data columns may vary.
    if len(row) <= 3:
        return [], 0

    values = row[3:]
    usable = (len(values) // 3) * 3
    remainder = len(values) - usable

    points: List[Tuple[float, float, float]] = []
    for i in range(0, usable, 3):
        x = _safe_float(values[i])
        y = _safe_float(values[i + 1])
        z = _safe_float(values[i + 2])
        points.append((x, y, z))
    return points, remainder


def _write_xyz(path: Path, points: List[Tuple[float, float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _write_ply(path: Path, points: List[Tuple[float, float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def export_frames(trc_path: Path, out_dir: Path, n_frames: int, fmt: str) -> None:
    if not trc_path.exists():
        raise FileNotFoundError(f"TRC file not found: {trc_path}")

    lines = trc_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 4:
        raise ValueError("TRC file looks incomplete (fewer than 4 header lines).")

    header_1 = _split_fields(lines[0])
    header_2 = _split_fields(lines[1])
    header_3 = _split_fields(lines[2])
    header_4 = _split_fields(lines[3])

    data_start = 4
    while data_start < len(lines) and not lines[data_start].strip():
        data_start += 1

    rows = [_split_fields(line) for line in lines[data_start:] if line.strip()]
    if not rows:
        raise ValueError("No data rows found in TRC file.")

    out_dir.mkdir(parents=True, exist_ok=True)
    use_rows = rows[: max(1, n_frames)]

    print("=" * 80)
    print(f"TRC: {trc_path}")
    print("Header line 1:")
    print("  " + " | ".join(header_1))
    print("Header line 2:")
    print("  " + " | ".join(header_2))
    print("Header line 3:")
    print("  " + " | ".join(header_3))
    print("Columns (line 4):")
    print("  " + " | ".join(header_4 if header_4 else ["<missing>"]))
    print(f"Total data frames: {len(rows)}")
    print(f"Export frames    : {len(use_rows)}")
    print(f"Output dir       : {out_dir}")
    print("=" * 80)

    for idx, row in enumerate(use_rows, start=1):
        frame_id = row[0] if row else str(idx)
        points, remainder = _extract_xyz_triplets(row)
        file_name = f"frame_{idx:03d}_id_{frame_id}.{fmt}"
        out_file = out_dir / file_name

        if fmt == "ply":
            _write_ply(out_file, points)
        else:
            _write_xyz(out_file, points)

        print(
            f"[{idx:02d}] frame_id={frame_id} points={len(points)} remainder_cols={remainder} -> {out_file.name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse TRC and export each frame as a separate point cloud file."
    )
    parser.add_argument(
        "trc_path",
        nargs="?",
        default="C:/APP/BD/BD08132/BD08132-Unnamed.trc",
        help="Path to input TRC file.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of frames to export from the beginning (default: 10).",
    )
    parser.add_argument(
        "--out-dir",
        default="trc_frame_pointclouds",
        help="Output directory (default: trc_frame_pointclouds).",
    )
    parser.add_argument(
        "--format",
        choices=["ply", "xyz"],
        default="ply",
        help="Output point cloud format (default: ply).",
    )
    args = parser.parse_args()

    export_frames(
        trc_path=Path(args.trc_path),
        out_dir=Path(args.out_dir),
        n_frames=max(1, args.frames),
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
