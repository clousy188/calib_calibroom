#!/usr/bin/env python3
"""Simple TRC opener and preview tool.

Usage:
    python open_trc.py
    python open_trc.py C:/APP/BD/BD08132/BD08132-Unnamed.trc --preview 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def _split_fields(line: str) -> List[str]:
    # TRC files are usually tab-separated; some writers mix spaces/tabs.
    if "\t" in line:
        return [p.strip() for p in line.strip().split("\t") if p.strip() != ""]
    return [p.strip() for p in line.strip().split() if p.strip() != ""]


def parse_trc(trc_path: Path, preview_rows: int = 10) -> None:
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

    data_rows: List[List[str]] = []
    for raw in lines[data_start:]:
        if not raw.strip():
            continue
        data_rows.append(_split_fields(raw))

    print("=" * 80)
    print(f"TRC file: {trc_path}")
    print(f"Total lines: {len(lines)}")
    print(f"Data rows : {len(data_rows)}")
    print("=" * 80)

    print("Header line 1:")
    print("  " + " | ".join(header_1))

    if header_2 and header_3 and len(header_2) == len(header_3):
        print("Header key-values:")
        for k, v in zip(header_2, header_3):
            print(f"  {k:>20}: {v}")
    else:
        print("Header line 2:")
        print("  " + " | ".join(header_2))
        print("Header line 3:")
        print("  " + " | ".join(header_3))

    print("Columns:")
    print("  " + " | ".join(header_4 if header_4 else ["<missing>"]))

    if data_rows:
        first_len = len(data_rows[0])
        print(f"First data row columns: {first_len}")

        # Common layout: Frame#, Time, Timestamp, TimeCode, then XYZ triplets
        if first_len >= 4:
            marker_triplet_count = (first_len - 4) // 3
            remainder = (first_len - 4) % 3
            print(f"Estimated XYZ triplets: {marker_triplet_count} (remainder columns: {remainder})")

    print("-" * 80)
    print(f"Preview first {min(preview_rows, len(data_rows))} data rows:")
    for i, row in enumerate(data_rows[:preview_rows], start=1):
        print(f"[{i}] " + " | ".join(row[:16]) + (" | ..." if len(row) > 16 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Open and preview a .trc file.")
    parser.add_argument(
        "trc_path",
        nargs="?",
        default="C:/APP/BD/BD08132/BD08132-Unnamed.trc",
        help="Path to the TRC file.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="How many data rows to preview (default: 10).",
    )
    args = parser.parse_args()

    parse_trc(Path(args.trc_path), preview_rows=max(1, args.preview))


if __name__ == "__main__":
    main()
