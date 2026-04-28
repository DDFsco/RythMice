"""Aggregate treadmill wheel session_metrics under outputs/result and plot cohort comparisons.

Expects folder names like ``{mouse_id}_{freq}hz_{replicate}`` (e.g. ``001_7hz_1``).
Skips directories whose names start with ``test_``.

Writes:
  - ``cohort_sessions.csv`` — one row per session folder
  - ``cohort_paired_scatters.png`` — paired metrics (x vs y frequency conditions)
  - ``cohort_sound_vs_silence.png`` — group comparison (0 Hz vs pooled 2+7 Hz)
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


def _load_session_metrics(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _metrics_row(folder: Path, mouse: str, freq_hz: int, rep: int, data: dict[str, Any]) -> dict[str, Any]:
    eng = data.get("engagement") or {}
    spd = data.get("speed_valid_bouts_only") or {}
    return {
        "folder": folder.name,
        "mouse_id": mouse,
        "frequency_hz": freq_hz,
        "replicate": rep,
        "mean_speed_deg_s": spd.get("mean_speed_deg_s"),
        "median_speed_deg_s": spd.get("median_speed_deg_s"),
        "total_running_time_valid_s": eng.get("total_running_time_valid_s"),
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
        m = FOLDER_RE.match(p.name)
        if not m:
            continue
        mouse, fstr, rstr = m.group(1), m.group(2), m.group(3)
        data = _load_session_metrics(p / "session_metrics.json")
        if data is None:
            continue
        rows.append(_metrics_row(p, mouse, int(fstr), int(rstr), data))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _paired_points(
    df: pd.DataFrame,
    mouse: str,
    x_hz: int,
    y_hz: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (x_vals, y_vals, replicate_ids) for sessions where both conditions exist."""
    sub = df[df["mouse_id"] == mouse]
    by_rep = sub.groupby("replicate")
    xs, ys, reps = [], [], []
    for rep, g in by_rep:
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
        reps.append(int(rep))
    return np.asarray(xs), np.asarray(ys), reps


def _plot_paired_grid(df: pd.DataFrame, out_path: Path) -> None:
    mice = sorted(df["mouse_id"].unique())
    comparisons = [(0, 7), (0, 2), (2, 7)]
    metrics = [
        ("mean_speed_deg_s", "Mean speed (valid bouts, deg/s)"),
        ("median_speed_deg_s", "Median speed (valid bouts, deg/s)"),
        ("total_running_time_valid_s", "Engagement: total valid running time (s)"),
    ]
    fig, axes = plt.subplots(len(comparisons), len(metrics), figsize=(12, 10), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(mice), 1)))

    for i, (xh, yh) in enumerate(comparisons):
        for j, (mcol, mlabel) in enumerate(metrics):
            ax = axes[i][j]
            all_x: list[float] = []
            all_y: list[float] = []
            for mi, mouse in enumerate(mice):
                x_arr, y_arr, reps = _paired_points(df, mouse, xh, yh, mcol)
                if len(x_arr):
                    ax.scatter(
                        x_arr,
                        y_arr,
                        color=colors[mi % len(colors)],
                        s=55,
                        label=f"mouse {mouse}",
                        zorder=3,
                    )
                    for x, y, r in zip(x_arr, y_arr, reps):
                        ax.annotate(str(r), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.8)
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
    fig.suptitle("Paired session metrics (replicate index matched within mouse)", fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sound_vs_silence(df: pd.DataFrame, out_path: Path) -> None:
    silence = df[df["frequency_hz"] == 0].copy()
    sound = df[df["frequency_hz"].isin([2, 7])].copy()
    silence["group"] = "Silence (0 Hz)"
    sound["group"] = "Sound (2 + 7 Hz)"
    both = pd.concat([silence, sound], ignore_index=True)

    metrics = [
        ("mean_speed_deg_s", "Mean speed (deg/s)"),
        ("median_speed_deg_s", "Median speed (deg/s)"),
        ("total_running_time_valid_s", "Total valid running time (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    groups = ["Silence (0 Hz)", "Sound (2 + 7 Hz)"]
    positions = [1, 2]
    for ax, (mcol, mtitle) in zip(axes, metrics):
        data = [both.loc[both["group"] == g, mcol].dropna().astype(float).values for g in groups]
        bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showmeans=True)
        for patch, color in zip(bp["boxes"], ["#a6cee3", "#b2df8a"]):
            patch.set_facecolor(color)
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, rotation=12, ha="right")
        ax.set_ylabel(mtitle)
        ax.set_title(mtitle)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Sound vs silence (all sessions; sound = pooled 2 Hz and 7 Hz)", fontsize=12)
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
    df.sort_values(["mouse_id", "frequency_hz", "replicate"]).to_csv(csv_path, index=False)

    _plot_paired_grid(df, outdir / "cohort_paired_scatters.png")
    _plot_sound_vs_silence(df, outdir / "cohort_sound_vs_silence.png")
    print(f"Wrote {csv_path}")
    print(f"Wrote {outdir / 'cohort_paired_scatters.png'}")
    print(f"Wrote {outdir / 'cohort_sound_vs_silence.png'}")


if __name__ == "__main__":
    main()
