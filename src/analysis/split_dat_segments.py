"""
Split tabular .dat recordings into N contiguous segments of equal duration in time.

Example: a 30-minute file ``001_0hz_1.dat`` with ``--parts 6`` yields six 5-minute files:
``001_0hz_1_1.dat`` … ``001_0hz_1_6.dat``.

Reading and time-scale rules match ``analyze_treadmill_wheel.load_data`` (sort, finite rows,
duplicate timestamps dropped by default). Output preserves header vs headerless layout and
all columns from the input table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.analysis.analyze_treadmill_wheel import (
    _infer_time_scale_to_seconds,
    _pick_column,
    _read_dat_table,
)


def _sniff_headerless(path: Path) -> bool:
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            if line.strip():
                first = line
                break
        else:
            raise ValueError(f"Input file is empty: {path}")
    parts = first.strip().split()
    if len(parts) < 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
        return True
    except ValueError:
        return False


def _rows_to_seconds(
    ts_raw: np.ndarray,
    time_unit: Literal["auto", "s", "ms"],
) -> np.ndarray:
    if time_unit == "ms":
        return ts_raw.astype(float) * 1e-3
    if time_unit == "s":
        return ts_raw.astype(float)
    return _infer_time_scale_to_seconds(ts_raw)[0]


def _prepare_table(
    path: Path,
    time_column: Optional[str],
    voltage_column: Optional[str],
    time_unit: Literal["auto", "s", "ms"],
) -> tuple[pd.DataFrame, str, np.ndarray, bool]:
    """Return cleaned dataframe, time column name, timestamps in seconds, headerless."""
    headerless = _sniff_headerless(path)
    df = _read_dat_table(path)
    t_col = _pick_column(df, time_column, "time")

    ts = pd.to_numeric(df[t_col], errors="coerce").to_numpy()
    v_col = _pick_column(df, voltage_column, "voltage")
    vs = pd.to_numeric(df[v_col], errors="coerce").to_numpy()
    bad = ~np.isfinite(ts) | ~np.isfinite(vs)
    df = df.loc[~bad].copy()
    ts = ts[~bad]

    if len(df) < 2:
        raise ValueError(f"Not enough finite rows in {path}")

    order = np.argsort(ts)
    df = df.iloc[order].reset_index(drop=True)
    ts = ts[order]

    dup = np.concatenate([[False], np.diff(ts) <= 0])
    if np.any(dup):
        df = df.loc[~dup].reset_index(drop=True)
        ts = ts[~dup]

    if len(df) < 2:
        raise ValueError(f"Not enough rows after dropping duplicate times in {path}")

    ts_s = _rows_to_seconds(ts, time_unit)
    if np.any(np.diff(ts_s) <= 0):
        raise ValueError(f"Timestamps not strictly increasing in seconds after conversion: {path}")

    return df, t_col, ts_s, headerless


def split_dat_file(
    path: Path,
    n_parts: int,
    outdir: Optional[Path] = None,
    time_column: Optional[str] = None,
    voltage_column: Optional[str] = None,
    time_unit: Literal["auto", "s", "ms"] = "auto",
) -> list[Path]:
    if n_parts < 2:
        raise ValueError("n_parts must be at least 2")

    df, _t_col, ts_s, headerless = _prepare_table(
        path, time_column, voltage_column, time_unit
    )

    t0 = float(ts_s[0])
    t1 = float(ts_s[-1])
    span = t1 - t0
    if span <= 0:
        raise ValueError(f"Non-positive time span in {path}")

    edges = np.linspace(t0, t1, n_parts + 1)
    dest_dir = outdir if outdir is not None else path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    stem = path.stem
    written: list[Path] = []
    for i in range(n_parts):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i < n_parts - 1:
            mask = (ts_s >= left) & (ts_s < right)
        else:
            mask = (ts_s >= left) & (ts_s <= right)

        chunk = df.loc[mask].copy()
        if chunk.empty:
            raise ValueError(
                f"Segment {i + 1}/{n_parts} of {path} has no samples "
                f"(window {left:.6g}–{right:.6g} s)."
            )

        out_path = dest_dir / f"{stem}_{i + 1}{path.suffix}"
        if headerless:
            chunk.to_csv(
                out_path,
                sep=" ",
                index=False,
                header=False,
                lineterminator="\n",
            )
        else:
            chunk.to_csv(
                out_path,
                sep=",",
                index=False,
                header=True,
                lineterminator="\n",
            )
        written.append(out_path)

    return written


def _gather_inputs(files: list[Path], input_dir: Optional[Path], recursive: bool) -> list[Path]:
    out: list[Path] = []
    for p in files:
        out.append(p.resolve())
    if input_dir is not None:
        root = input_dir.resolve()
        pattern = "**/*.dat" if recursive else "*.dat"
        out.extend(sorted(root.glob(pattern)))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main() -> None:
    p = argparse.ArgumentParser(description="Split .dat files into equal-time segments.")
    p.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input .dat files (optional if --input-dir is set)",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Split every *.dat in this directory",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="With --input-dir, search subfolders for *.dat",
    )
    p.add_argument(
        "-n",
        "--parts",
        type=int,
        default=6,
        metavar="N",
        help="Number of equal-duration segments (default: 6)",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as each input file)",
    )
    p.add_argument("--time-column", type=str, default=None)
    p.add_argument("--signal-column", type=str, default=None)
    p.add_argument(
        "--time-unit",
        choices=("auto", "s", "ms"),
        default="auto",
        help="Timestamp units (default: auto, same heuristic as wheel analyzer)",
    )

    args = p.parse_args()
    if args.input_dir is not None:
        idir = args.input_dir.resolve()
        if not idir.is_dir():
            p.error(f"--input-dir is not a directory or does not exist: {idir}")

    inputs = _gather_inputs(list(args.inputs), args.input_dir, args.recursive)
    if not inputs:
        if args.input_dir is not None:
            idir = args.input_dir.resolve()
            hint = (
                f"No .dat files matched in {idir}"
                + (" (recursive)" if args.recursive else " (non-recursive; try --recursive)")
                + "."
            )
            p.error(hint)
        p.error("Provide one or more input .dat paths, or use --input-dir <folder>.")

    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"Not a file: {path}")
        outs = split_dat_file(
            path,
            n_parts=args.parts,
            outdir=args.outdir,
            time_column=args.time_column,
            voltage_column=args.signal_column,
            time_unit=args.time_unit,  # type: ignore[arg-type]
        )
        rel = [o.name for o in outs]
        print(f"{path.name} -> {', '.join(rel)}")


if __name__ == "__main__":
    main()
