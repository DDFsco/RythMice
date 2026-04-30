"""Aggregate treadmill wheel session_metrics under outputs/result and plot cohort comparisons.

Expects folder names like ``{mouse_id}_{freq}hz_{replicate}`` (e.g. ``001_7hz_1``),
or split segments ``{mouse_id}_{freq}hz_{replicate}_{segment}`` (e.g. ``001_7hz_1_3``).
Skips directories whose names start with ``test_``.

Writes:
  - ``cohort_sessions.csv`` — one row per session folder (incl. mean speed, valid run time,
    and total distance = mean speed × valid run time, deg)
  - ``cohort_paired_scatters.png`` — paired metrics (x vs y frequency conditions)
  - ``cohort_sound_vs_silence.png`` — distribution of metrics by frequency (0 Hz, 2 Hz, 7 Hz)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOLDER_RE = re.compile(r"^(\d+)_(\d+)hz_(\d+)$")
# Split recordings: same as above with an extra time-chunk index (from split_dat_segments).
FOLDER_SPLIT_RE = re.compile(r"^(\d+)_(\d+)hz_(\d+)_(\d+)$")


def _load_session_metrics(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _metrics_row(
    folder: Path,
    mouse: str,
    freq_hz: int,
    rep: int,
    data: dict[str, Any],
    segment: int,
) -> dict[str, Any]:
    eng = data.get("engagement") or {}
    spd = data.get("speed_valid_bouts_only") or {}
    mean_s = spd.get("mean_speed_deg_s")
    run_t = eng.get("total_running_time_valid_s")
    dist_deg: Optional[float] = None
    if mean_s is not None and run_t is not None:
        try:
            ms_f = float(mean_s)
            rt_f = float(run_t)
            if np.isfinite(ms_f) and np.isfinite(rt_f):
                dist_deg = ms_f * rt_f
        except (TypeError, ValueError):
            dist_deg = None
    return {
        "folder": folder.name,
        "mouse_id": mouse,
        "frequency_hz": freq_hz,
        "replicate": rep,
        "segment": segment,
        "mean_speed_deg_s": mean_s,
        "total_running_time_valid_s": run_t,
        "total_distance_valid_deg": dist_deg,
        "mean_bout_duration_valid_s": eng.get("mean_bout_duration_valid_s"),
        "n_valid_bouts": eng.get("n_valid_bouts"),
    }


def discover_sessions(result_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in sorted(result_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("test_"):
            continue
        m = FOLDER_SPLIT_RE.match(p.name)
        if m:
            mouse, fstr, rstr, sstr = m.group(1), m.group(2), m.group(3), m.group(4)
            segment = int(sstr)
        else:
            m = FOLDER_RE.match(p.name)
            if not m:
                continue
            mouse, fstr, rstr = m.group(1), m.group(2), m.group(3)
            segment = 1
        data = _load_session_metrics(p / "session_metrics.json")
        if data is None:
            continue
        rows.append(_metrics_row(p, mouse, int(fstr), int(rstr), data, segment))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _paired_points(
    df: pd.DataFrame,
    mouse: str,
    x_hz: int,
    y_hz: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Return (x_vals, y_vals, labels, recording_replicate) for paired sessions.

    ``recording_replicate`` is the trial/replicate index from the folder name (e.g. the
    ``1`` in ``001_7hz_1_3``); all segments of that replicate share one color in plots.
    """
    sub = df[df["mouse_id"] == mouse]
    group_keys = ["replicate", "segment"] if "segment" in sub.columns else ["replicate"]
    by_rep = sub.groupby(group_keys)
    xs, ys, rep_labels, recording_repls = [], [], [], []
    for key, g in by_rep:
        row_x = g[g["frequency_hz"] == x_hz]
        row_y = g[g["frequency_hz"] == y_hz]
        if len(row_x) != 1 or len(row_y) != 1:
            continue
        vx = row_x.iloc[0][metric]
        vy = row_y.iloc[0][metric]
        if vx is None or vy is None or (isinstance(vx, float) and np.isnan(vx)) or (isinstance(vy, float) and np.isnan(vy)):
            continue
        xs.append(float(vx))
        ys.append(float(vy))
        if isinstance(key, tuple):
            rec_rep, seg = int(key[0]), int(key[1])
            rep_labels.append(f"{rec_rep}s{seg}")
            recording_repls.append(rec_rep)
        else:
            rec_rep = int(key)
            rep_labels.append(str(rec_rep))
            recording_repls.append(rec_rep)
    return np.asarray(xs), np.asarray(ys), rep_labels, recording_repls


def _plot_paired_grid(df: pd.DataFrame, out_path: Path) -> None:
    mice = sorted(df["mouse_id"].unique())
    comparisons = [(0, 7), (0, 2), (2, 7)]
    metrics = [
        ("mean_speed_deg_s", "Mean speed (valid bouts, deg/s)"),
        (
            "total_distance_valid_deg",
            "Total distance (valid bouts)\nmean speed × valid run time (deg)",
        ),
        ("total_running_time_valid_s", "Engagement: total valid running time (s)"),
    ]
    fig, axes = plt.subplots(len(comparisons), len(metrics), figsize=(12, 10), squeeze=False)

    pairs = df[["mouse_id", "replicate"]].drop_duplicates().sort_values(["mouse_id", "replicate"])
    n_hues = max(len(pairs), 1)
    palette = plt.cm.tab10(np.linspace(0, 0.9, n_hues))
    color_by_mouse_rep: dict[tuple[str, int], np.ndarray] = {}
    for idx, row in enumerate(pairs.itertuples(index=False)):
        color_by_mouse_rep[(str(row.mouse_id), int(row.replicate))] = palette[idx % len(palette)]

    for i, (xh, yh) in enumerate(comparisons):
        for j, (mcol, mlabel) in enumerate(metrics):
            ax = axes[i][j]
            all_x: list[float] = []
            all_y: list[float] = []
            for mouse in mice:
                x_arr, y_arr, rep_labels, recording_repls = _paired_points(df, mouse, xh, yh, mcol)
                if len(x_arr):
                    for repl in sorted(set(recording_repls)):
                        m = np.array([r == repl for r in recording_repls], dtype=bool)
                        c = color_by_mouse_rep[(mouse, repl)]
                        ax.scatter(
                            x_arr[m],
                            y_arr[m],
                            color=c,
                            s=55,
                            label=f"mouse {mouse} rep {repl}",
                            zorder=3,
                        )
                    for x, y, rlab in zip(x_arr, y_arr, rep_labels):
                        ax.annotate(rlab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.8)
                    all_x.extend(x_arr.tolist())
                    all_y.extend(y_arr.tolist())
            if all_x and all_y:
                lo = min(min(all_x), min(all_y))
                hi = max(max(all_x), max(all_y))
                pad = (hi - lo) * 0.05 if hi > lo else 1.0
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5, zorder=1)
            ax.set_xlabel(f"{xh} Hz (silence)" if xh == 0 else f"{xh} Hz")
            ax.set_ylabel(f"{yh} Hz")
            ax.set_title(f"{xh} vs {yh} Hz\n{mlabel}")
            ax.grid(True, alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=min(len(by_label), 4),
            bbox_to_anchor=(0.5, 1.03),
        )
    fig.suptitle(
        "Paired session metrics (segment index in labels; color = mouse + recording replicate)",
        fontsize=12,
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sound_vs_silence(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["frequency_hz"].isin([0, 2, 7])].copy()
    label_map = {0: "0 Hz", 2: "2 Hz", 7: "7 Hz"}
    sub["group"] = sub["frequency_hz"].map(label_map)

    metrics = [
        ("mean_speed_deg_s", "Mean speed (deg/s)"),
        ("total_distance_valid_deg", "Total distance (mean speed × run time, deg)"),
        ("total_running_time_valid_s", "Total valid running time (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    groups = ["0 Hz", "2 Hz", "7 Hz"]
    positions = [1, 2, 3]
    colors = ["#a6cee3", "#b2df8a", "#fdb462"]
    for ax, (mcol, mtitle) in zip(axes, metrics):
        data = [sub.loc[sub["group"] == g, mcol].dropna().astype(float).values for g in groups]
        bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showmeans=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, rotation=12, ha="right")
        ax.set_ylabel(mtitle)
        ax.set_title(mtitle)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Cohort comparison by stimulus frequency (0 Hz, 2 Hz, 7 Hz)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("outputs/result"),
        help="Directory containing per-session output folders",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Where to write cohort CSV and figures (default: result-root)",
    )
    args = parser.parse_args()
    root = args.result_root.resolve()
    outdir = (args.outdir or root).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = discover_sessions(root)
    if df.empty:
        raise SystemExit(f"No sessions found under {root} (expected folders like 001_7hz_1 with session_metrics.json)")

    csv_path = outdir / "cohort_sessions.csv"
    df.sort_values(["mouse_id", "frequency_hz", "replicate", "segment"]).to_csv(csv_path, index=False)

    _plot_paired_grid(df, outdir / "cohort_paired_scatters.png")
    _plot_sound_vs_silence(df, outdir / "cohort_sound_vs_silence.png")
    print(f"Wrote {csv_path}")
    print(f"Wrote {outdir / 'cohort_paired_scatters.png'}")
    print(f"Wrote {outdir / 'cohort_sound_vs_silence.png'}")


if __name__ == "__main__":
    main()
