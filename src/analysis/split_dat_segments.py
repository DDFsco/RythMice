"""
Split a ~36-minute rhythmic-stimulation ``.dat`` recording into eighteen 2-minute segments.

Design (per session): twelve blocks ``[0, f1, 0, f2, …]`` where ``0`` is 4 minutes silence
(split into two files: pre-silence then post-silence) and ``f1``, ``f2`` are ``2`` or ``7``
meaning one 2-minute chunk at 2 Hz or 7 Hz. Silence files are tagged with the **upcoming**
stimulus (e.g. ``001_presilence2hz_1`` / ``001_possilence2hz_1`` before the first 2 Hz epoch in
session 1; ``001_presilence7hz_1`` before the first 7 Hz epoch). Session 1 uses
``[0, 2, 0, 7, 0, 2, 0, 7, 0, 2, 0, 7]``; session 2 uses
``[0, 7, 0, 2, 0, 7, 0, 2, 0, 7, 0, 2]``.

Reading and time-scale rules match ``analyze_treadmill_wheel.load_data`` (sort, finite rows,
duplicate timestamps dropped). Output preserves header vs headerless layout and all columns.
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

# Twelve condition codes per session: 0 = 4 min silence (two 2-min exports), 2 / 7 = one 2-min chunk.
SESSION_PATTERN: dict[int, list[int]] = {
    1: [0, 2, 0, 7, 0, 2, 0, 7, 0, 2, 0, 7],
    2: [0, 7, 0, 2, 0, 7, 0, 2, 0, 7, 0, 2],
}

SEGMENT_DURATION_S = 120.0
EXPECTED_TOTAL_S = 18 * SEGMENT_DURATION_S


def _normalize_subject_id(subject_id: str) -> str:
    s = subject_id.strip()
    if s.isdigit():
        return s.zfill(3)
    return s


def segment_stems_for_session(subject_id: str, session_id: int) -> list[str]:
    """Return eighteen output stems (no suffix), e.g. ``001_presilence2hz_1``, ``001_2hz_1``."""
    if session_id not in SESSION_PATTERN:
        raise ValueError(f"session_id must be 1 or 2, got {session_id}")
    sub = _normalize_subject_id(subject_id)
    pattern = SESSION_PATTERN[session_id]
    count_2hz = 0
    count_7hz = 0
    idx_pre_silence_2hz = 0
    idx_pre_silence_7hz = 0
    stems: list[str] = []
    for i, code in enumerate(pattern):
        if code == 0:
            following: Optional[int] = None
            for j in range(i + 1, len(pattern)):
                if pattern[j] in (2, 7):
                    following = pattern[j]
                    break
            if following is None:
                raise ValueError(
                    "Each silence block (0) must be followed by 2 or 7 later in the pattern."
                )
            if following == 2:
                idx_pre_silence_2hz += 1
                k = idx_pre_silence_2hz
                hz_tag = "2hz"
            else:
                idx_pre_silence_7hz += 1
                k = idx_pre_silence_7hz
                hz_tag = "7hz"
            stems.append(f"{sub}_presilence{hz_tag}_{k}")
            stems.append(f"{sub}_possilence{hz_tag}_{k}")
        elif code == 2:
            count_2hz += 1
            stems.append(f"{sub}_2hz_{count_2hz}")
        elif code == 7:
            count_7hz += 1
            stems.append(f"{sub}_7hz_{count_7hz}")
        else:
            raise ValueError(f"Invalid pattern code {code!r} (expected 0, 2, or 7)")
    if len(stems) != 18:
        raise RuntimeError(f"Internal error: expected 18 stems, got {len(stems)}")
    return stems


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


def split_dat_rhythm_session(
    path: Path,
    subject_id: str,
    session_id: int,
    outdir: Optional[Path] = None,
    time_column: Optional[str] = None,
    voltage_column: Optional[str] = None,
    time_unit: Literal["auto", "s", "ms"] = "auto",
    segment_seconds: float = SEGMENT_DURATION_S,
) -> list[Path]:
    """
    Split one continuous recording into eighteen segments aligned from the first timestamp.
    """
    df, _t_col, ts_s, headerless = _prepare_table(
        path, time_column, voltage_column, time_unit
    )
    t0 = float(ts_s[0])
    t_end_data = float(ts_s[-1])
    stems = segment_stems_for_session(subject_id, session_id)
    dest_dir = outdir if outdir is not None else path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for i, stem in enumerate(stems):
        left = t0 + i * segment_seconds
        right = t0 + (i + 1) * segment_seconds
        if i < len(stems) - 1:
            mask = (ts_s >= left) & (ts_s < right)
        else:
            mask = (ts_s >= left) & (ts_s <= right)

        chunk = df.loc[mask].copy()
        if chunk.empty:
            raise ValueError(
                f"Segment {i + 1}/{len(stems)} ({stem}) has no samples "
                f"(window {left:.6g}–{right:.6g} s)."
            )
        out_path = dest_dir / f"{stem}{path.suffix}"
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

    span = t_end_data - t0
    if span < EXPECTED_TOTAL_S - 1.0:
        raise ValueError(
            f"Recording span ({span:.1f} s) is shorter than expected "
            f"({EXPECTED_TOTAL_S:.0f} s for 18×{segment_seconds:.0f} s)."
        )
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Split one ~36 min session .dat into eighteen 2 min files "
            "(presilence2hz / possilence7hz / 2hz / 7hz, …) by subject and session."
        )
    )
    p.add_argument("subject_id", type=str, help="Subject id (e.g. 001); zero-padded if numeric")
    p.add_argument(
        "session_id",
        type=int,
        choices=(1, 2),
        help="Session 1: [0,2,0,7,...]; session 2: [0,7,0,2,...]",
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the full-session .dat file",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as input)",
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
    path = args.input.resolve()
    if not path.is_file():
        raise SystemExit(f"Input is not a file: {path}")

    outs = split_dat_rhythm_session(
        path,
        subject_id=args.subject_id,
        session_id=args.session_id,
        outdir=args.outdir.resolve() if args.outdir else None,
        time_column=args.time_column,
        voltage_column=args.signal_column,
        time_unit=args.time_unit,  # type: ignore[arg-type]
    )
    print(f"{path.name} -> {len(outs)} files")
    for o in outs:
        print(f"  {o.name}")


if __name__ == "__main__":
    main()
