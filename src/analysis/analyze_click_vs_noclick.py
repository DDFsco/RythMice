import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def _read_dat_table(path: Path) -> pd.DataFrame:
    """Read a LabJack-exported table using delimiter auto-detection."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Failed to read DAT/CSV file: {path}") from exc
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    return df


def _pick_column(df: pd.DataFrame, requested: Optional[str], role: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested {role} column '{requested}' not found in columns: {list(df.columns)}")
        return requested

    lower_map = {c.lower(): c for c in df.columns}
    if role == "time":
        for candidate in ("timestamp_s", "time_s", "time", "timestamp", "t"):
            if candidate in lower_map:
                return lower_map[candidate]
    if role == "signal":
        for candidate in ("speed_rps", "amplitude", "voltage", "signal", "value", "ch0", "channel0"):
            if candidate in lower_map:
                return lower_map[candidate]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError(f"Could not infer {role} column from file columns: {list(df.columns)}")
    return numeric_cols[0]


def _coerce_numeric(series: pd.Series, column_name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.notna().sum() == 0:
        raise ValueError(f"Column '{column_name}' has no numeric values.")
    return out


def _smooth_signal(signal: np.ndarray, smoothing_ms: float, median_dt_s: float) -> np.ndarray:
    if smoothing_ms <= 0:
        return signal
    win = int(round((smoothing_ms / 1000.0) / median_dt_s))
    win = max(win, 1)
    if win == 1:
        return signal
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(signal, kernel, mode="same")


def summarize_recording(
    data_path: Path,
    time_column: Optional[str],
    signal_column: Optional[str],
    sample_rate_hz: Optional[float],
    onset_s: Optional[float],
    offset_s: Optional[float],
    smoothing_ms: float,
) -> dict:
    df = _read_dat_table(data_path)
    t_col = _pick_column(df, time_column, role="time")
    s_col = _pick_column(df, signal_column, role="signal")

    time = _coerce_numeric(df[t_col], t_col).to_numpy()
    signal = _coerce_numeric(df[s_col], s_col).to_numpy()

    if np.all(np.isnan(time)):
        if sample_rate_hz is None:
            raise ValueError(
                f"No valid time values in '{data_path.name}'. Provide --sample-rate-hz to construct timestamps."
            )
        time = np.arange(len(signal), dtype=float) / float(sample_rate_hz)

    valid = np.isfinite(time) & np.isfinite(signal)
    time = time[valid]
    signal = signal[valid]
    if len(time) < 3:
        raise ValueError(f"Not enough valid samples in {data_path}")

    sort_idx = np.argsort(time)
    time = time[sort_idx]
    signal = signal[sort_idx]

    if onset_s is not None:
        keep = time >= onset_s
        time = time[keep]
        signal = signal[keep]
    if offset_s is not None:
        keep = time < offset_s
        time = time[keep]
        signal = signal[keep]
    if len(time) < 3:
        raise ValueError(f"Not enough samples after analysis window crop in {data_path}")

    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError(f"Time column in {data_path} has no positive step sizes.")
    median_dt_s = float(np.median(dt))
    smoothed = _smooth_signal(signal=signal, smoothing_ms=smoothing_ms, median_dt_s=median_dt_s)

    abs_signal = np.abs(smoothed)
    duration_s = float(time[-1] - time[0])
    mean_abs = float(np.mean(abs_signal))
    rms = float(np.sqrt(np.mean(smoothed**2)))
    auc_abs = float(np.trapezoid(abs_signal, time))
    running_index = float(auc_abs / duration_s) if duration_s > 0 else np.nan

    return {
        "n_samples": int(len(time)),
        "duration_s": duration_s,
        "median_dt_s": median_dt_s,
        "mean_abs_amplitude": mean_abs,
        "rms_amplitude": rms,
        "auc_abs_amplitude": auc_abs,
        "running_index": running_index,
        "trace_time_s": time,
        "trace_signal": smoothed,
    }


def _save_trace_plot(out_path: Path, title: str, time: np.ndarray, signal: np.ndarray) -> None:
    plt.figure(figsize=(10, 3))
    plt.plot(time, signal, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze wheel-running amplitude from DAT files and compare click vs no_click conditions."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with mouse_id, condition, data_file columns.")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Root path used to resolve relative data_file.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for tables, stats, and plots.")
    parser.add_argument("--time-column", type=str, default=None, help="Optional time column name in DAT file.")
    parser.add_argument("--signal-column", type=str, default=None, help="Optional amplitude/speed column name in DAT file.")
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=None,
        help="Use this sample rate if file has no usable time column.",
    )
    parser.add_argument("--onset-s", type=float, default=None, help="Optional analysis start time (seconds).")
    parser.add_argument("--offset-s", type=float, default=None, help="Optional analysis end time (seconds).")
    parser.add_argument("--smoothing-ms", type=float, default=100.0, help="Moving-average smoothing window in ms.")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    trace_dir = outdir / "trace_plots"
    trace_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    required_cols = {"mouse_id", "condition", "data_file"}
    if not required_cols.issubset(manifest.columns):
        raise ValueError(f"Manifest must include columns: {required_cols}")

    records = []
    for _, row in manifest.iterrows():
        mouse_id = str(row["mouse_id"])
        condition = str(row["condition"]).strip().lower()
        if condition not in {"click", "no_click"}:
            raise ValueError(f"Condition must be 'click' or 'no_click'. Found: {condition}")
        session_id = str(row["session_id"]) if "session_id" in manifest.columns else "na"

        data_path = Path(str(row["data_file"]))
        if not data_path.is_absolute():
            data_path = args.data_root / data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Data file does not exist: {data_path}")

        summary = summarize_recording(
            data_path=data_path,
            time_column=args.time_column,
            signal_column=args.signal_column,
            sample_rate_hz=args.sample_rate_hz,
            onset_s=args.onset_s,
            offset_s=args.offset_s,
            smoothing_ms=args.smoothing_ms,
        )
        records.append(
            {
                "mouse_id": mouse_id,
                "session_id": session_id,
                "condition": condition,
                "data_file": str(data_path),
                "n_samples": summary["n_samples"],
                "duration_s": summary["duration_s"],
                "mean_abs_amplitude": summary["mean_abs_amplitude"],
                "rms_amplitude": summary["rms_amplitude"],
                "auc_abs_amplitude": summary["auc_abs_amplitude"],
                "running_index": summary["running_index"],
            }
        )

        trace_name = f"{mouse_id}_{session_id}_{condition}.png".replace(" ", "_")
        _save_trace_plot(
            out_path=trace_dir / trace_name,
            title=f"{mouse_id} | {session_id} | {condition}",
            time=summary["trace_time_s"],
            signal=summary["trace_signal"],
        )

    rec_df = pd.DataFrame(records).sort_values(["mouse_id", "session_id", "condition"])
    rec_df.to_csv(outdir / "recording_summary.csv", index=False)

    mouse_df = (
        rec_df.groupby(["mouse_id", "condition"], as_index=False)["running_index"]
        .mean()
        .pivot(index="mouse_id", columns="condition", values="running_index")
        .reset_index()
    )
    if "click" not in mouse_df.columns or "no_click" not in mouse_df.columns:
        raise ValueError("Need both conditions for each mouse to compute paired comparison.")
    mouse_df["delta_click_minus_no_click"] = mouse_df["click"] - mouse_df["no_click"]
    mouse_df.to_csv(outdir / "mouse_level_paired.csv", index=False)

    valid = mouse_df.dropna(subset=["click", "no_click"])
    stats = {"n_mice_paired": int(len(valid))}
    if len(valid) >= 2:
        t_res = ttest_rel(valid["click"], valid["no_click"], nan_policy="omit")
        stats["paired_t_test_p"] = float(t_res.pvalue)
        stats["paired_t_test_stat"] = float(t_res.statistic)
        try:
            w_res = wilcoxon(valid["click"], valid["no_click"], zero_method="wilcox")
            stats["wilcoxon_p"] = float(w_res.pvalue)
            stats["wilcoxon_stat"] = float(w_res.statistic)
        except ValueError:
            stats["wilcoxon_p"] = np.nan
            stats["wilcoxon_stat"] = np.nan
    else:
        stats["paired_t_test_p"] = np.nan
        stats["paired_t_test_stat"] = np.nan
        stats["wilcoxon_p"] = np.nan
        stats["wilcoxon_stat"] = np.nan

    with open(outdir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    plt.figure(figsize=(6, 4))
    for _, row in valid.iterrows():
        plt.plot([0, 1], [row["no_click"], row["click"]], marker="o", alpha=0.7)
    plt.xticks([0, 1], ["no_click", "click"])
    plt.ylabel("Running index (AUC|amplitude| / duration)")
    plt.title("Paired mouse-level running comparison")
    plt.tight_layout()
    plt.savefig(outdir / "paired_mouse_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    box_data = [valid["no_click"].to_numpy(), valid["click"].to_numpy()]
    plt.boxplot(box_data, labels=["no_click", "click"])
    plt.ylabel("Running index (AUC|amplitude| / duration)")
    plt.title("Group summary")
    plt.tight_layout()
    plt.savefig(outdir / "group_boxplot.png", dpi=160)
    plt.close()

    print(f"Saved: {(outdir / 'recording_summary.csv').resolve()}")
    print(f"Saved: {(outdir / 'mouse_level_paired.csv').resolve()}")
    print(f"Saved: {(outdir / 'statistics.json').resolve()}")
    print(f"Saved plots in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
