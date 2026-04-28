"""
Analyze mouse treadmill / wheel recordings from .dat (tabular) files.

Pipeline: raw data → preprocessing → event detection → speed estimation
→ running bout detection → structured outputs + QC figures.

Engagement (valid bouts only) and speed (valid bouts only) are reported separately;
micro-bouts are retained as secondary metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from matplotlib.figure import Figure
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, medfilt, savgol_filter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WheelAnalysisConfig:
    """All tunable parameters (no magic numbers in the core logic)."""

    degrees_per_event: float = 18.0

    # Preprocessing
    smoothing_method: Literal["moving_average", "savgol", "none"] = "moving_average"
    smoothing_window_samples: int = 5
    savgol_window_length: int = 11
    savgol_polyorder: int = 2
    remove_baseline: bool = False
    baseline_window_samples: int = 501

    # Event detection
    event_method: Literal["threshold", "peak"] = "threshold"
    threshold_high: float = 5.0
    threshold_low: float = 1.5
    peak_prominence: float = 0.5
    peak_distance_samples: int = 10

    # Speed = event rate (counts/s on a uniform grid) × degrees_per_event, after Gaussian smoothing.
    # Raw voltage is not interpolated; only event timestamps are binned.
    speed_grid_hz: int = 100  # uniform binning grid: 100 or 1000
    sigma_analysis_s: float = 0.5  # Gaussian σ (s) on event-rate train for bouts / threshold / stats
    sigma_visualization_s: float = 1.0  # Gaussian σ (s) on event-rate train for plotting (typically ≥ analysis)
    movement_threshold_deg_s: float = 80.0

    # Bout rules
    min_valid_duration_s: float = 2.0
    min_valid_events: int = 5
    micro_max_duration_s: float = 2.0
    micro_max_events: int = 5
    valid_mean_speed_deg_s: float = 30.0
    merge_gap_s: float = 2.0

    # Column names (optional overrides)
    time_column: Optional[str] = None
    voltage_column: Optional[str] = None
    time_unit: Literal["auto", "s", "ms"] = "auto"

    # Debug: zoomed plot + CSV for a time range or around one detected event (mutually exclusive).
    debug_plot_start_s: Optional[float] = None
    debug_plot_end_s: Optional[float] = None
    debug_event_index: Optional[int] = None
    debug_event_padding_s: float = 0.5


@dataclass
class StimulusEpoch:
    """One stimulus interval for future alignment (sound vs silence, etc.)."""

    stimulus_on_s: float
    stimulus_off_s: float
    condition_label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def align_bouts_to_stimulus(
    bouts_df: pd.DataFrame,
    epochs: list[StimulusEpoch],
) -> pd.DataFrame:
    """
    Placeholder / extension hook: tag bouts that overlap each stimulus epoch.

    bouts_df must include start_time_s, end_time_s.
    Returns a copy with added columns epoch_index, epoch_label, overlap_fraction.
    """
    if bouts_df.empty or not epochs:
        out = bouts_df.copy()
        out["epoch_index"] = np.nan
        out["epoch_label"] = ""
        out["overlap_fraction"] = np.nan
        return out

    rows = []
    for _, b in bouts_df.iterrows():
        bs = float(b["start_time_s"])
        be = float(b["end_time_s"])
        best_idx = np.nan
        best_label = ""
        best_frac = 0.0
        dur = max(be - bs, np.finfo(float).eps)
        for i, ep in enumerate(epochs):
            o0 = max(bs, ep.stimulus_on_s)
            o1 = min(be, ep.stimulus_off_s)
            ov = max(0.0, o1 - o0)
            frac = ov / dur
            if frac > best_frac:
                best_frac = frac
                best_idx = i
                best_label = ep.condition_label
        r = b.to_dict()
        r["epoch_index"] = best_idx
        r["epoch_label"] = best_label
        r["overlap_fraction"] = best_frac
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# I/O and helpers
# ---------------------------------------------------------------------------


def _read_dat_table(path: Path) -> pd.DataFrame:
    """
    Load tabular exports. If the first data line is two numbers (no text header),
    read without treating line 1 as column names (common for LabJack .dat).
    """
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            first_nonempty = ""
            for line in f:
                if line.strip():
                    first_nonempty = line
                    break
            if not first_nonempty:
                raise ValueError(f"Input file is empty: {path}")

        parts = first_nonempty.strip().split()
        numeric_row0 = False
        if len(parts) >= 2:
            try:
                float(parts[0])
                float(parts[1])
                numeric_row0 = True
            except ValueError:
                numeric_row0 = False

        if numeric_row0:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                engine="python",
                header=None,
                usecols=[0, 1],
                names=["timestamp", "voltage"],
            )
        else:
            df = pd.read_csv(path, sep=None, engine="python")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to read DAT/CSV file: {path}") from exc
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    return df


def _pick_column(df: pd.DataFrame, requested: Optional[str], role: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested {role} column '{requested}' not in columns: {list(df.columns)}")
        return requested

    lower_map = {c.lower(): c for c in df.columns}
    if role == "time":
        for candidate in ("timestamp_s", "time_s", "time", "timestamp", "t"):
            if candidate in lower_map:
                return lower_map[candidate]
    if role == "voltage":
        for candidate in ("voltage", "signal", "value", "amplitude", "ch0", "channel0"):
            if candidate in lower_map:
                return lower_map[candidate]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        raise ValueError(f"Need at least two numeric columns in {list(df.columns)}")
    if role == "time":
        return numeric_cols[0]
    return numeric_cols[1]


def _infer_time_scale_to_seconds(time: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Auto-detect whether timestamps are in seconds or milliseconds; return seconds.

    Heuristic (seconds vs ms only):
    - LabJack / DAQ exports often use integer milliseconds: deltas are >= 1 in raw units.
    - High-rate CSVs in seconds use deltas < 1 (e.g. 0.001 s @ 1 kHz).

    If median(delta) >= 1 in raw units → treat as milliseconds.
    Otherwise → treat as seconds.

    Very coarse sampling (e.g. one sample every 2 s) stored in seconds can have delta >= 1;
    if that is common for your rig, pass pre-converted timestamps or extend this function.
    """
    t = np.asarray(time, dtype=float)
    finite = np.isfinite(t)
    if not np.any(finite):
        raise ValueError("No finite timestamps.")
    t_sorted = np.sort(t[finite])
    dt = np.diff(t_sorted)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Timestamps are not strictly usable for delta (no positive steps).")

    med_dt = float(np.median(dt))
    if med_dt >= 1.0:
        return time.astype(float) * 1e-3, "ms"
    return time.astype(float), "s"


def load_data(
    path: Path,
    time_column: Optional[str] = None,
    voltage_column: Optional[str] = None,
    time_unit: Literal["auto", "s", "ms"] = "auto",
) -> dict[str, Any]:
    """
    Load timestamp and voltage, validate, return times in seconds and metadata.
    """
    df = _read_dat_table(path)
    t_col = _pick_column(df, time_column, "time")
    v_col = _pick_column(df, voltage_column, "voltage")

    ts_raw = pd.to_numeric(df[t_col], errors="coerce").to_numpy()
    v_raw = pd.to_numeric(df[v_col], errors="coerce").to_numpy()

    if np.any(np.isnan(ts_raw)):
        n_bad = int(np.isnan(ts_raw).sum())
        raise ValueError(f"Timestamp column has {n_bad} NaN values; fix or drop rows before analysis.")

    if np.any(np.isnan(v_raw)):
        n_bad = int(np.isnan(v_raw).sum())
        raise ValueError(f"Voltage column has {n_bad} NaN values; fix or drop rows before analysis.")

    mask = np.isfinite(ts_raw) & np.isfinite(v_raw)
    ts_raw = ts_raw[mask]
    v_raw = v_raw[mask]

    if len(ts_raw) < 4:
        raise ValueError("Not enough samples after removing non-finite values.")

    order = np.argsort(ts_raw)
    ts_raw = ts_raw[order]
    v_raw = v_raw[order]

    duplicate_timestamps_dropped = 0
    if np.any(np.diff(ts_raw) <= 0):
        keep = np.concatenate([[True], np.diff(ts_raw) > 0])
        duplicate_timestamps_dropped = int(np.sum(~keep))
        ts_raw = ts_raw[keep]
        v_raw = v_raw[keep]

    if len(ts_raw) < 4:
        raise ValueError(
            "Not enough samples after removing duplicate timestamps (or file has ≤3 unique time points)."
        )

    if time_unit == "ms":
        ts_s = ts_raw.astype(float) * 1e-3
        inferred_unit = "ms"
    elif time_unit == "s":
        ts_s = ts_raw.astype(float)
        inferred_unit = "s"
    else:
        ts_s, inferred_unit = _infer_time_scale_to_seconds(ts_raw)

    if np.any(np.diff(ts_s) <= 0):
        raise ValueError("Timestamps are not strictly increasing in seconds (duplicate times).")

    return {
        "timestamp_s": ts_s,
        "voltage": v_raw,
        "time_column": t_col,
        "voltage_column": v_col,
        "inferred_time_unit": inferred_unit,
        "source_path": str(path.resolve()),
        "duplicate_timestamps_dropped": duplicate_timestamps_dropped,
    }


def preprocess_signal(
    timestamp_s: np.ndarray,
    voltage: np.ndarray,
    cfg: WheelAnalysisConfig,
) -> dict[str, np.ndarray]:
    """Denoise / optional baseline removal; always keep raw copy."""
    raw = np.asarray(voltage, dtype=float)
    x = raw.copy()

    if cfg.remove_baseline and len(x) >= cfg.baseline_window_samples:
        k = cfg.baseline_window_samples
        if k % 2 == 0:
            k += 1
        baseline = medfilt(x, kernel_size=min(k, len(x) // 2 * 2 + 1))
        x = x - baseline

    method = cfg.smoothing_method
    if method == "moving_average" and cfg.smoothing_window_samples > 1:
        w = int(cfg.smoothing_window_samples)
        kernel = np.ones(w, dtype=float) / float(w)
        x = np.convolve(x, kernel, mode="same")
    elif method == "savgol":
        wl = int(cfg.savgol_window_length)
        if wl % 2 == 0:
            wl += 1
        wl = max(3, min(wl, len(x) - (1 - len(x) % 2)))
        if wl >= 3 and wl <= len(x):
            x = savgol_filter(x, window_length=wl, polyorder=int(cfg.savgol_polyorder))

    return {"raw_voltage": raw, "processed_voltage": x}


def detect_events(
    timestamp_s: np.ndarray,
    signal: np.ndarray,
    cfg: WheelAnalysisConfig,
) -> dict[str, Any]:
    """
    Detect wheel rotation events using threshold hysteresis (default) or peaks.
    Returns event sample indices and event times (seconds).
    """
    t = np.asarray(timestamp_s, dtype=float)
    s = np.asarray(signal, dtype=float)
    n = len(s)

    if cfg.event_method == "peak":
        distance = max(1, int(cfg.peak_distance_samples))
        peaks, props = find_peaks(s, prominence=float(cfg.peak_prominence), distance=distance)
        times = t[peaks]
        return {
            "method": "peak",
            "event_indices": peaks,
            "event_times_s": times,
            "peak_properties": props,
        }

    high = float(cfg.threshold_high)
    low = float(cfg.threshold_low)
    if low >= high:
        raise ValueError("threshold_low must be < threshold_high for hysteresis.")

    state = 0
    crossings: list[int] = []
    for i in range(n):
        v = s[i]
        if state == 0:
            if v >= high:
                crossings.append(i)
                state = 1
        else:
            if v <= low:
                state = 0

    peaks_idx = np.array(crossings, dtype=int)
    times = t[peaks_idx]
    return {
        "method": "threshold",
        "event_indices": peaks_idx,
        "event_times_s": times,
        "peak_properties": None,
    }


def compute_speed_trace(
    timestamp_s: np.ndarray,
    event_times_s: np.ndarray,
    cfg: WheelAnalysisConfig,
) -> dict[str, np.ndarray]:
    """
    Continuous locomotor speed from **event counts** on a uniform time grid (no voltage interpolation).

    1. Histogram encoder events onto bins of width ``1 / speed_grid_hz``.
    2. ``event_rate = count / bin_width`` (events/s).
    3. Gaussian smoothing on the rate train with σ = ``sigma_analysis_s`` and σ = ``sigma_visualization_s``
       (seconds → scipy ``gaussian_filter1d`` in sample units).
    4. ``speed_deg_s = smoothed_rate_analysis * degrees_per_event`` (bouts, threshold, CSV primary speed).
       ``speed_deg_s_smooth_vis`` uses ``sigma_visualization_s``.
    """
    t = np.asarray(timestamp_s, dtype=float)
    ev = np.asarray(event_times_s, dtype=float)
    ev = np.sort(ev)

    hz = int(cfg.speed_grid_hz)
    if hz not in (100, 1000):
        raise ValueError("speed_grid_hz must be 100 or 1000.")
    bin_width = 1.0 / float(hz)

    t0_session = float(t[0])
    t1_session = float(t[-1])
    duration = max(t1_session - t0_session, bin_width)
    n_bins = max(1, int(np.ceil(duration / bin_width)))
    edges = t0_session + np.arange(n_bins + 1, dtype=float) * bin_width
    centers = (edges[:-1] + edges[1:]) / 2.0

    counts, _ = np.histogram(ev, bins=edges)
    counts_f = counts.astype(float)
    event_rate_raw = counts_f / bin_width

    sigma_a = float(cfg.sigma_analysis_s)
    sigma_v = float(cfg.sigma_visualization_s)
    if sigma_a <= 0 or sigma_v <= 0:
        raise ValueError("sigma_analysis_s and sigma_visualization_s must be positive.")

    sigma_samples_a = sigma_a / bin_width
    sigma_samples_v = sigma_v / bin_width
    rate_gauss_a = gaussian_filter1d(event_rate_raw, sigma=float(sigma_samples_a), mode="nearest")
    rate_gauss_v = gaussian_filter1d(event_rate_raw, sigma=float(sigma_samples_v), mode="nearest")

    deg_per_evt = float(cfg.degrees_per_event)
    speed_deg_s = rate_gauss_a * deg_per_evt
    speed_deg_vis = rate_gauss_v * deg_per_evt
    rev_s = speed_deg_s / 360.0

    active = speed_deg_s > float(cfg.movement_threshold_deg_s)
    half = bin_width / 2.0

    out: dict[str, Any] = {
        "window_center_s": centers,
        "window_halfwidth_s": np.full_like(centers, half),
        "bin_edges_s": edges,
        "bin_width_s": bin_width,
        "speed_grid_hz": float(hz),
        "event_count_per_bin": counts.astype(np.int64),
        "event_rate_per_s_raw": event_rate_raw,
        "event_rate_per_s_gaussian_analysis": rate_gauss_a,
        "event_rate_per_s_gaussian_viz": rate_gauss_v,
        "speed_deg_s": speed_deg_s,
        "speed_deg_s_smooth_vis": speed_deg_vis,
        "speed_rev_s": rev_s,
        "active_mask": active,
    }
    return out


def _bout_stats_from_span(
    t_start: float,
    t_end: float,
    event_times_s: np.ndarray,
    speed: dict[str, np.ndarray],
    degrees_per_event: float,
) -> dict[str, float]:
    ev = event_times_s[(event_times_s >= t_start) & (event_times_s <= t_end)]
    n_ev = int(len(ev))

    wc = speed["window_center_s"]
    half = float(speed["window_halfwidth_s"][0])
    m = (wc >= t_start - half) & (wc <= t_end + half)
    if np.any(m):
        spd = speed["speed_deg_s"][m]
        mean_s = float(np.mean(spd))
        peak_s = float(np.max(spd))
    else:
        dur = max(t_end - t_start, np.finfo(float).eps)
        mean_s = (n_ev * float(degrees_per_event)) / dur
        peak_s = mean_s

    return {
        "start_time_s": float(t_start),
        "end_time_s": float(t_end),
        "duration_s": float(t_end - t_start),
        "event_count": float(n_ev),
        "mean_speed_deg_s": mean_s,
        "peak_speed_deg_s": peak_s,
    }


def _initial_bouts_from_active(
    speed: dict[str, np.ndarray],
    timestamp_s: np.ndarray,
) -> list[tuple[float, float]]:
    """Merge consecutive active speed windows into [start, end] in session time."""
    active = speed["active_mask"]
    wc = speed["window_center_s"]
    half = float(speed["window_halfwidth_s"][0])
    if not np.any(active):
        return []

    segments: list[tuple[float, float]] = []
    in_run = False
    run_start = 0.0
    last_end = 0.0
    for i, is_a in enumerate(active):
        if is_a:
            left = float(wc[i] - half)
            right = float(wc[i] + half)
            if not in_run:
                in_run = True
                run_start = left
                last_end = right
            else:
                last_end = max(last_end, right)
        else:
            if in_run:
                segments.append((run_start, last_end))
                in_run = False
    if in_run:
        segments.append((run_start, last_end))

    t_lo = float(timestamp_s[0])
    t_hi = float(timestamp_s[-1])
    clipped = []
    for a, b in segments:
        clipped.append((max(a, t_lo), min(b, t_hi)))
    return clipped


def _merge_close_intervals(intervals: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if a - lb <= gap:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def detect_running_bouts(
    timestamp_s: np.ndarray,
    event_times_s: np.ndarray,
    speed: dict[str, np.ndarray],
    cfg: WheelAnalysisConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Bout detection uses local speed / event rate only (not raw voltage).

    Returns:
      bouts_df: one row per bout with classification valid vs micro
      running_state_per_center: 1 inside any merged bout else 0 (aligned to speed grid)
    """
    raw_spans = _initial_bouts_from_active(speed, timestamp_s)
    merged = _merge_close_intervals(raw_spans, float(cfg.merge_gap_s))

    rows = []
    for k, (ts, te) in enumerate(merged):
        st = _bout_stats_from_span(ts, te, event_times_s, speed, cfg.degrees_per_event)
        dur = st["duration_s"]
        n_ev = int(st["event_count"])
        mean_sp = st["mean_speed_deg_s"]

        is_micro = (dur < float(cfg.micro_max_duration_s)) or (n_ev < int(cfg.micro_max_events))
        is_valid = (
            (dur >= float(cfg.min_valid_duration_s))
            and (n_ev >= int(cfg.min_valid_events))
            and (mean_sp > float(cfg.valid_mean_speed_deg_s))
        )
        if is_micro or not is_valid:
            bout_type = "micro"
        else:
            bout_type = "valid"

        rows.append(
            {
                "bout_index": k + 1,
                "start_time_s": st["start_time_s"],
                "end_time_s": st["end_time_s"],
                "duration_s": st["duration_s"],
                "event_count": n_ev,
                "mean_speed_deg_s": st["mean_speed_deg_s"],
                "peak_speed_deg_s": st["peak_speed_deg_s"],
                "mean_speed_rev_s": st["mean_speed_deg_s"] / 360.0,
                "peak_speed_rev_s": st["peak_speed_deg_s"] / 360.0,
                "type": bout_type,
            }
        )

    bouts_df = pd.DataFrame(rows)
    if bouts_df.empty:
        running_state = np.zeros(len(speed["window_center_s"]), dtype=int)
        return bouts_df, running_state

    wc = speed["window_center_s"]
    state = np.zeros(len(wc), dtype=int)
    bout_id = np.zeros(len(wc), dtype=int)

    for _, b in bouts_df.iterrows():
        m = (wc >= b["start_time_s"]) & (wc <= b["end_time_s"])
        state[m] = 1
        bout_id[m] = int(b["bout_index"])

    return bouts_df, state


def refine_bout_classification(bouts_df: pd.DataFrame, cfg: WheelAnalysisConfig) -> pd.DataFrame:
    """Re-evaluate type flags after any external edits (identity if already correct)."""
    if bouts_df.empty:
        return bouts_df
    out = bouts_df.copy()
    types = []
    for _, b in out.iterrows():
        dur = float(b["duration_s"])
        n_ev = int(b["event_count"])
        mean_sp = float(b["mean_speed_deg_s"])
        is_micro = (dur < float(cfg.micro_max_duration_s)) or (n_ev < int(cfg.micro_max_events))
        is_valid = (
            (dur >= float(cfg.min_valid_duration_s))
            and (n_ev >= int(cfg.min_valid_events))
            and (mean_sp > float(cfg.valid_mean_speed_deg_s))
        )
        types.append("micro" if (is_micro or not is_valid) else "valid")
    out["type"] = types
    return out


def build_time_series_table(
    timestamp_s: np.ndarray,
    speed: dict[str, np.ndarray],
    running_state: np.ndarray,
    bouts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per speed window: center time, speeds, running state, bout id and type.
    Interpolate / map to original timestamps via merge_asof optional — here we output on the speed grid.
    """
    wc = speed["window_center_s"]
    bout_type_at_center = np.full(len(wc), "", dtype=object)
    bout_id = np.zeros(len(wc), dtype=int)

    if not bouts_df.empty:
        for _, b in bouts_df.iterrows():
            m = (wc >= b["start_time_s"]) & (wc <= b["end_time_s"])
            bout_id[m] = int(b["bout_index"])
            bout_type_at_center[m] = str(b["type"])

    row: dict[str, Any] = {
        "timestamp_s": wc,
        "speed_deg_s": speed["speed_deg_s"],
        "speed_rev_s": speed["speed_rev_s"],
        "running_state": running_state.astype(int),
        "bout_id": bout_id,
        "bout_type": bout_type_at_center,
    }
    sv = speed.get("speed_deg_s_smooth_vis")
    if sv is not None and len(sv) == len(wc):
        row["speed_deg_s_smooth_vis"] = sv
    if "event_count_per_bin" in speed and len(speed["event_count_per_bin"]) == len(wc):
        row["event_count_per_bin"] = speed["event_count_per_bin"]
        row["event_rate_per_s_raw"] = speed["event_rate_per_s_raw"]
        row["event_rate_per_s_gaussian_analysis"] = speed["event_rate_per_s_gaussian_analysis"]
        row["event_rate_per_s_gaussian_viz"] = speed["event_rate_per_s_gaussian_viz"]
    return pd.DataFrame(row)


def build_time_series_with_voltage(
    timestamp_s: np.ndarray,
    voltage: np.ndarray,
    speed: dict[str, np.ndarray],
    running_state: np.ndarray,
    bouts_df: pd.DataFrame,
) -> pd.DataFrame:
    df = build_time_series_table(timestamp_s, speed, running_state, bouts_df)
    v = np.asarray(voltage, dtype=float)
    t = np.asarray(timestamp_s, dtype=float)
    df["voltage_interp"] = np.interp(df["timestamp_s"].to_numpy(), t, v)
    return df


def session_summary(
    bouts_df: pd.DataFrame,
    speed: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Engagement + speed metrics; speed only from valid bouts."""
    if bouts_df.empty:
        return {
            "engagement": {
                "total_running_time_valid_s": 0.0,
                "n_valid_bouts": 0,
                "mean_bout_duration_valid_s": np.nan,
                "total_micro_bout_time_s": 0.0,
                "n_micro_bouts": 0,
            },
            "speed_valid_bouts_only": {
                "mean_speed_deg_s": np.nan,
                "median_speed_deg_s": np.nan,
                "peak_speed_deg_s": np.nan,
                "mean_speed_rev_s": np.nan,
                "peak_speed_rev_s": np.nan,
                "speed_histogram_counts": [],
                "speed_histogram_bin_edges_deg_s": [],
            },
        }

    valid = bouts_df[bouts_df["type"] == "valid"].copy()
    micro = bouts_df[bouts_df["type"] == "micro"].copy()

    eng = {
        "total_running_time_valid_s": float(valid["duration_s"].sum()) if len(valid) else 0.0,
        "n_valid_bouts": int(len(valid)),
        "mean_bout_duration_valid_s": float(valid["duration_s"].mean()) if len(valid) else np.nan,
        "total_micro_bout_time_s": float(micro["duration_s"].sum()) if len(micro) else 0.0,
        "n_micro_bouts": int(len(micro)),
    }

    wc = speed["window_center_s"]
    spd = speed["speed_deg_s"]
    mask_valid = np.zeros(len(wc), dtype=bool)
    for _, b in valid.iterrows():
        mask_valid |= (wc >= b["start_time_s"]) & (wc <= b["end_time_s"])

    if np.any(mask_valid):
        vspd = spd[mask_valid]
        mean_speed = float(np.mean(vspd))
        median_speed = float(np.median(vspd))
        peak_speed = float(np.max(vspd))
        hist_counts, edges = np.histogram(vspd, bins=min(30, max(5, len(vspd) // 3)))
    else:
        mean_speed = float(valid["mean_speed_deg_s"].mean()) if len(valid) else np.nan
        median_speed = float(np.median(valid["mean_speed_deg_s"])) if len(valid) else np.nan
        peak_speed = float(valid["peak_speed_deg_s"].max()) if len(valid) else np.nan
        hist_counts, edges = np.array([]), np.array([])

    spd_block = {
        "mean_speed_deg_s": mean_speed,
        "median_speed_deg_s": median_speed,
        "peak_speed_deg_s": peak_speed,
        "mean_speed_rev_s": mean_speed / 360.0 if np.isfinite(mean_speed) else np.nan,
        "peak_speed_rev_s": peak_speed / 360.0 if np.isfinite(peak_speed) else np.nan,
        "speed_histogram_counts": hist_counts.tolist(),
        "speed_histogram_bin_edges_deg_s": edges.tolist(),
    }

    return {"engagement": eng, "speed_valid_bouts_only": spd_block}


def plot_figure1_voltage_bouts(
    timestamp_s: np.ndarray,
    raw_voltage: np.ndarray,
    bouts_df: pd.DataFrame,
    out_path: Path,
    title: str = "Voltage and running bouts",
) -> None:
    fig = Figure(figsize=(12, 3.5))
    ax = fig.subplots()
    ax.plot(timestamp_s, raw_voltage, color="0.3", lw=0.6, label="Raw voltage")
    if not bouts_df.empty:
        for _, b in bouts_df.iterrows():
            c = "C2" if b["type"] == "valid" else "C1"
            ax.axvspan(b["start_time_s"], b["end_time_s"], color=c, alpha=0.25)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def plot_figure2_speed_trace(
    speed: dict[str, Any],
    running_state: np.ndarray,
    out_path: Path,
    title: str = "Speed trace",
    movement_threshold_deg_s: Optional[float] = None,
    sigma_analysis_s: Optional[float] = None,
    sigma_visualization_s: Optional[float] = None,
) -> None:
    fig = Figure(figsize=(12, 3.5))
    ax = fig.subplots()
    wc = speed["window_center_s"]
    sa = float(sigma_analysis_s) if sigma_analysis_s is not None else None
    sv_sig = float(sigma_visualization_s) if sigma_visualization_s is not None else None
    lbl_a = "Speed (deg/s), σ_analysis"
    if sa is not None:
        lbl_a = f"Speed (deg/s), σ_an={sa:g}s"
    ax.plot(wc, speed["speed_deg_s"], color="C0", lw=1.1, label=lbl_a)
    sv = speed.get("speed_deg_s_smooth_vis")
    if sv is not None and len(sv) == len(wc):
        lbl_v = "Speed (deg/s), σ_viz"
        if sv_sig is not None:
            lbl_v = f"Speed (deg/s), σ_viz={sv_sig:g}s"
        ax.plot(wc, sv, color="C1", lw=1.0, ls="--", alpha=0.85, label=lbl_v)
    if movement_threshold_deg_s is not None:
        ax.axhline(
            float(movement_threshold_deg_s),
            color="0.45",
            ls=":",
            lw=1.0,
            label=f"Movement thr={movement_threshold_deg_s:g}",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("deg/s")
    ax.set_title(title + " | event-rate Gaussian → deg/s")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def plot_figure3_event_qc(
    timestamp_s: np.ndarray,
    processed_voltage: np.ndarray,
    events: dict[str, Any],
    out_path: Path,
    title: str = "Event detection QC",
) -> None:
    fig = Figure(figsize=(12, 3.5))
    ax = fig.subplots()
    ax.plot(timestamp_s, processed_voltage, color="0.2", lw=0.7, label="Processed voltage")
    et = events["event_times_s"]
    if len(et):
        idx = events["event_indices"]
        ax.scatter(et, processed_voltage[idx], s=12, c="red", zorder=5, label="Events")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (processed)")
    ax.set_title(title + f" | method={events['method']}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def _resolve_debug_time_window(
    timestamp_s: np.ndarray,
    event_times_s: np.ndarray,
    cfg: WheelAnalysisConfig,
) -> tuple[float, float]:
    """Return (t0, t1) in seconds for debug zoom; raises if options are inconsistent."""
    has_win = cfg.debug_plot_start_s is not None or cfg.debug_plot_end_s is not None
    has_evt = cfg.debug_event_index is not None

    if has_win and has_evt:
        raise ValueError("Use either debug time window (start+end) or debug_event_index, not both.")

    t_lo = float(np.min(timestamp_s))
    t_hi = float(np.max(timestamp_s))

    if has_evt:
        i = int(cfg.debug_event_index)
        ev = np.asarray(event_times_s, dtype=float)
        if i < 0 or i >= len(ev):
            raise ValueError(f"debug_event_index={i} out of range (n_events={len(ev)}).")
        pad = float(cfg.debug_event_padding_s)
        tc = float(ev[i])
        t0 = max(t_lo, tc - pad)
        t1 = min(t_hi, tc + pad)
        if t1 <= t0:
            raise ValueError("Debug window collapsed after clamping; increase padding or check data.")
        return t0, t1

    if cfg.debug_plot_start_s is not None and cfg.debug_plot_end_s is not None:
        t0 = float(cfg.debug_plot_start_s)
        t1 = float(cfg.debug_plot_end_s)
        if t1 <= t0:
            raise ValueError("debug_plot_end_s must be greater than debug_plot_start_s.")
        t0 = max(t_lo, t0)
        t1 = min(t_hi, t1)
        if t1 <= t0:
            raise ValueError("Debug window does not overlap recording after clamping to data range.")
        return t0, t1

    if has_win:
        raise ValueError("Provide both debug_plot_start_s and debug_plot_end_s, or use debug_event_index.")

    raise ValueError("_resolve_debug_time_window called without debug options set.")


def plot_debug_time_window(
    timestamp_s: np.ndarray,
    raw_voltage: np.ndarray,
    processed_voltage: np.ndarray,
    events: dict[str, Any],
    speed: dict[str, np.ndarray],
    cfg: WheelAnalysisConfig,
    bouts_df: pd.DataFrame,
    t_start: float,
    t_end: float,
    out_path: Path,
    title: str = "Debug zoom",
) -> None:
    """
    Zoomed voltage (raw + processed), threshold lines, detected events, bout shading,
    and matching speed trace for the same interval.
    """
    t = np.asarray(timestamp_s, dtype=float)
    vr = np.asarray(raw_voltage, dtype=float)
    vp = np.asarray(processed_voltage, dtype=float)

    fig = Figure(figsize=(12, 6))
    ax_v, ax_s = fig.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )

    m = (t >= t_start) & (t <= t_end)
    if not np.any(m):
        ax_v.text(0.5, 0.5, "No samples in debug window", transform=ax_v.transAxes, ha="center")
    else:
        ax_v.plot(t[m], vr[m], color="0.55", lw=0.8, alpha=0.9, label="Raw voltage")
        ax_v.plot(t[m], vp[m], color="0.15", lw=1.0, label="Processed (event input)")

    if cfg.event_method == "threshold":
        ax_v.axhline(cfg.threshold_high, color="C3", ls="--", lw=1.0, label=f"High={cfg.threshold_high}")
        ax_v.axhline(cfg.threshold_low, color="C4", ls=":", lw=1.0, label=f"Low={cfg.threshold_low}")

    et = np.asarray(events["event_times_s"], dtype=float)
    ev_idx = np.asarray(events["event_indices"], dtype=int)
    in_win = (et >= t_start) & (et <= t_end)
    etw = et[in_win]
    idx_w = ev_idx[in_win]
    if len(etw):
        ax_v.scatter(etw, vp[idx_w], s=36, c="red", zorder=6, marker="v", label=f"Events (n={len(etw)})")
        global_idx = np.flatnonzero(in_win)
        for k, te in enumerate(etw):
            ax_v.axvline(te, color="red", alpha=0.2, lw=0.8)
            gi = int(global_idx[k])
            ax_v.annotate(
                str(gi),
                xy=(te, float(vp[idx_w[k]])),
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=7,
                color="darkred",
                ha="center",
            )

    if not bouts_df.empty:
        for _, b in bouts_df.iterrows():
            if float(b["end_time_s"]) < t_start or float(b["start_time_s"]) > t_end:
                continue
            c = "C2" if b["type"] == "valid" else "C1"
            ax_v.axvspan(b["start_time_s"], b["end_time_s"], color=c, alpha=0.12)

    ax_v.set_ylabel("Voltage (V)")
    ax_v.set_title(f"{title} | {t_start:.4f}–{t_end:.4f} s | method={events['method']}")
    ax_v.legend(loc="upper right", fontsize=7, ncol=2)

    wc = speed["window_center_s"]
    sm = (wc >= t_start) & (wc <= t_end)
    ax_s.plot(
        wc[sm],
        speed["speed_deg_s"][sm],
        color="C0",
        lw=1.1,
        label=f"Speed σ_an={float(cfg.sigma_analysis_s):g}s",
    )
    sv = speed.get("speed_deg_s_smooth_vis")
    if sv is not None and len(sv) == len(wc):
        sv_lbl = f"Speed σ_viz={float(cfg.sigma_visualization_s):g}s"
        ax_s.plot(wc[sm], sv[sm], color="C1", lw=1.0, ls="--", alpha=0.85, label=sv_lbl)
    ax_s.axhline(cfg.movement_threshold_deg_s, color="0.5", ls=":", lw=0.9, label=f"Move thr={cfg.movement_threshold_deg_s}")
    ax_s.set_ylabel("deg/s")
    ax_s.set_xlabel("Time (s)")
    ax_s.legend(loc="upper right", fontsize=7)

    ax_s.set_xlim(t_start, t_end)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def write_debug_window_csv(
    timestamp_s: np.ndarray,
    raw_voltage: np.ndarray,
    processed_voltage: np.ndarray,
    t_start: float,
    t_end: float,
    out_path: Path,
) -> int:
    """Write samples falling in [t_start, t_end]; returns row count."""
    t = np.asarray(timestamp_s, dtype=float)
    vr = np.asarray(raw_voltage, dtype=float)
    vp = np.asarray(processed_voltage, dtype=float)
    m = (t >= t_start) & (t <= t_end)
    n = int(np.sum(m))
    if n == 0:
        pd.DataFrame(columns=["timestamp_s", "raw_voltage", "processed_voltage"]).to_csv(out_path, index=False)
        return 0
    pd.DataFrame(
        {
            "timestamp_s": t[m],
            "raw_voltage": vr[m],
            "processed_voltage": vp[m],
        }
    ).to_csv(out_path, index=False)
    return n


def plot_figure4_bout_distributions(bouts_df: pd.DataFrame, out_path: Path, title: str = "Bout distributions") -> None:
    fig = Figure(figsize=(10, 3.5))
    axes = fig.subplots(1, 2)
    if bouts_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No bouts", ha="center")
    else:
        valid = bouts_df[bouts_df["type"] == "valid"]
        micro = bouts_df[bouts_df["type"] == "micro"]
        axes[0].hist(
            [valid["duration_s"].to_numpy(), micro["duration_s"].to_numpy()],
            stacked=False,
            label=["valid", "micro"],
            bins=max(8, int(np.sqrt(len(bouts_df)))),
            alpha=0.75,
        )
        axes[0].set_xlabel("Duration (s)")
        axes[0].legend(fontsize=8)

        axes[1].hist(
            [valid["mean_speed_deg_s"].to_numpy(), micro["mean_speed_deg_s"].to_numpy()],
            stacked=False,
            label=["valid", "micro"],
            bins=max(8, int(np.sqrt(len(bouts_df)))),
            alpha=0.75,
        )
        axes[1].set_xlabel("Mean speed (deg/s)")
        axes[1].legend(fontsize=8)
    axes[0].set_title("Duration")
    axes[1].set_title("Mean speed")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def run_pipeline(
    dat_path: Path,
    cfg: WheelAnalysisConfig,
    outdir: Path,
    stimulus_epochs: Optional[list[StimulusEpoch]] = None,
) -> dict[str, Any]:
    """End-to-end analysis; writes CSV/JSON and figures to outdir."""
    outdir.mkdir(parents=True, exist_ok=True)

    loaded = load_data(
        dat_path,
        cfg.time_column,
        cfg.voltage_column,
        time_unit=cfg.time_unit,
    )
    t = loaded["timestamp_s"]
    v_raw = loaded["voltage"]

    pre = preprocess_signal(t, v_raw, cfg)
    events = detect_events(t, pre["processed_voltage"], cfg)
    speed = compute_speed_trace(t, events["event_times_s"], cfg)
    bouts_df, run_state = detect_running_bouts(t, events["event_times_s"], speed, cfg)
    bouts_df = refine_bout_classification(bouts_df, cfg)

    ts_df = build_time_series_with_voltage(t, pre["raw_voltage"], speed, run_state, bouts_df)
    ts_df.to_csv(outdir / "timeseries_speed.csv", index=False)

    bouts_out = bouts_df.copy()
    if stimulus_epochs:
        bouts_out = align_bouts_to_stimulus(bouts_out, stimulus_epochs)
    bouts_out.to_csv(outdir / "bouts_summary.csv", index=False)

    summary = session_summary(bouts_df, speed)
    meta: dict[str, Any] = {
        "source": loaded["source_path"],
        "time_column": loaded["time_column"],
        "voltage_column": loaded["voltage_column"],
        "inferred_time_unit": loaded["inferred_time_unit"],
        "n_events": int(len(events["event_times_s"])),
        "duplicate_timestamps_dropped": int(loaded.get("duplicate_timestamps_dropped", 0)),
        "config": cfg.__dict__,
    }

    spd = summary["speed_valid_bouts_only"]
    counts = spd.get("speed_histogram_counts") or []
    edges = spd.get("speed_histogram_bin_edges_deg_s") or []
    if len(counts) and len(edges) >= 2:
        e = np.asarray(edges, dtype=float)
        c = np.asarray(counts, dtype=float)
        centers = (e[:-1] + e[1:]) / 2.0
        pd.DataFrame({"bin_center_deg_s": centers, "count": c}).to_csv(
            outdir / "speed_distribution_valid.csv", index=False
        )

    stem = dat_path.stem
    plot_figure1_voltage_bouts(t, pre["raw_voltage"], bouts_df, outdir / f"{stem}_fig1_voltage_bouts.png", title=stem)
    plot_figure2_speed_trace(
        speed,
        run_state,
        outdir / f"{stem}_fig2_speed.png",
        title=stem,
        movement_threshold_deg_s=float(cfg.movement_threshold_deg_s),
        sigma_analysis_s=float(cfg.sigma_analysis_s),
        sigma_visualization_s=float(cfg.sigma_visualization_s),
    )
    plot_figure3_event_qc(t, pre["processed_voltage"], events, outdir / f"{stem}_fig3_events.png", title=stem)
    plot_figure4_bout_distributions(bouts_df, outdir / f"{stem}_fig4_bout_hist.png", title=stem)

    debug_requested = cfg.debug_event_index is not None or (
        cfg.debug_plot_start_s is not None and cfg.debug_plot_end_s is not None
    )
    if debug_requested:
        tw0, tw1 = _resolve_debug_time_window(t, events["event_times_s"], cfg)
        plot_debug_time_window(
            t,
            pre["raw_voltage"],
            pre["processed_voltage"],
            events,
            speed,
            cfg,
            bouts_df,
            tw0,
            tw1,
            outdir / f"{stem}_fig_debug_zoom.png",
            title=stem,
        )
        n_dbg = write_debug_window_csv(
            t,
            pre["raw_voltage"],
            pre["processed_voltage"],
            tw0,
            tw1,
            outdir / f"{stem}_debug_window_voltage.csv",
        )
        meta["debug_window_s"] = [tw0, tw1]
        meta["debug_window_n_samples"] = n_dbg

    with open(outdir / "session_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, **summary}, f, indent=2)

    return {
        "loaded": loaded,
        "preprocess": pre,
        "events": events,
        "speed": speed,
        "bouts": bouts_df,
        "timeseries": ts_df,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Treadmill / wheel .dat analysis: bouts, speed, engagement.")
    parser.add_argument("--input", type=Path, required=True, help="Path to .dat / CSV file.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory.")

    parser.add_argument("--degrees-per-event", type=float, default=18.0)
    parser.add_argument(
        "--smoothing",
        choices=["moving_average", "savgol", "none"],
        default="moving_average",
    )
    parser.add_argument("--smoothing-window-samples", type=int, default=5)
    parser.add_argument("--savgol-window", type=int, default=11)
    parser.add_argument("--savgol-poly", type=int, default=2)
    parser.add_argument("--remove-baseline", action="store_true")

    parser.add_argument("--event-method", choices=["threshold", "peak"], default="threshold")
    parser.add_argument("--threshold-high", type=float, default=5.0)
    parser.add_argument("--threshold-low", type=float, default=1.5)
    parser.add_argument("--peak-prominence", type=float, default=0.5)
    parser.add_argument("--peak-distance-samples", type=int, default=10)

    parser.add_argument(
        "--speed-grid-hz",
        type=int,
        choices=[100, 1000],
        default=100,
        help="Uniform time-bin frequency for event counting (Hz): 100 or 1000.",
    )
    parser.add_argument(
        "--sigma-analysis-s",
        type=float,
        default=0.5,
        help="Gaussian σ (seconds) on event-rate train for bouts / threshold / exported primary speed.",
    )
    parser.add_argument(
        "--sigma-visualization-s",
        type=float,
        default=1.0,
        help="Gaussian σ (seconds) on event-rate train for visualization curve.",
    )
    parser.add_argument("--movement-threshold-deg-s", type=float, default=80.0)

    parser.add_argument("--min-valid-duration-s", type=float, default=2.0)
    parser.add_argument("--min-valid-events", type=int, default=5)
    parser.add_argument("--micro-max-duration-s", type=float, default=2.0)
    parser.add_argument("--micro-max-events", type=int, default=5)
    parser.add_argument("--valid-mean-speed-deg-s", type=float, default=30.0)
    parser.add_argument("--merge-gap-s", type=float, default=2.0)

    parser.add_argument("--time-column", type=str, default=None)
    parser.add_argument("--voltage-column", type=str, default=None)
    parser.add_argument("--time-unit", choices=["auto", "s", "ms"], default="auto")
    parser.add_argument(
        "--stimulus-epochs",
        type=Path,
        default=None,
        help="Optional JSON list of {stimulus_on_s, stimulus_off_s, condition_label} for alignment hook.",
    )
    parser.add_argument(
        "--debug-window-start",
        type=float,
        default=None,
        help="With --debug-window-end: plot zoom + CSV for this time range (seconds).",
    )
    parser.add_argument(
        "--debug-window-end",
        type=float,
        default=None,
        help="With --debug-window-start: end of debug zoom (seconds).",
    )
    parser.add_argument(
        "--debug-event-index",
        type=int,
        default=None,
        help="Alternative to debug window: zoom around event k (0-based) with padding.",
    )
    parser.add_argument(
        "--debug-event-padding-s",
        type=float,
        default=0.5,
        help="Half-width (seconds) around selected event when using --debug-event-index.",
    )

    args = parser.parse_args()

    if (args.debug_window_start is None) ^ (args.debug_window_end is None):
        parser.error("Provide both --debug-window-start and --debug-window-end, or neither.")
    if args.debug_event_index is not None and (
        args.debug_window_start is not None or args.debug_window_end is not None
    ):
        parser.error("Use either --debug-event-index or the debug window pair, not both.")

    cfg = WheelAnalysisConfig(
        degrees_per_event=args.degrees_per_event,
        smoothing_method=args.smoothing,
        smoothing_window_samples=args.smoothing_window_samples,
        savgol_window_length=args.savgol_window,
        savgol_polyorder=args.savgol_poly,
        remove_baseline=args.remove_baseline,
        event_method=args.event_method,
        threshold_high=args.threshold_high,
        threshold_low=args.threshold_low,
        peak_prominence=args.peak_prominence,
        peak_distance_samples=args.peak_distance_samples,
        speed_grid_hz=args.speed_grid_hz,
        sigma_analysis_s=args.sigma_analysis_s,
        sigma_visualization_s=args.sigma_visualization_s,
        movement_threshold_deg_s=args.movement_threshold_deg_s,
        min_valid_duration_s=args.min_valid_duration_s,
        min_valid_events=args.min_valid_events,
        micro_max_duration_s=args.micro_max_duration_s,
        micro_max_events=args.micro_max_events,
        valid_mean_speed_deg_s=args.valid_mean_speed_deg_s,
        merge_gap_s=args.merge_gap_s,
        time_column=args.time_column,
        voltage_column=args.voltage_column,
        time_unit=args.time_unit,
        debug_plot_start_s=args.debug_window_start,
        debug_plot_end_s=args.debug_window_end,
        debug_event_index=args.debug_event_index,
        debug_event_padding_s=args.debug_event_padding_s,
    )

    stimulus_epochs: Optional[list[StimulusEpoch]] = None
    if args.stimulus_epochs is not None:
        with open(args.stimulus_epochs, encoding="utf-8") as f:
            raw_eps = json.load(f)
        stimulus_epochs = []
        for ep in raw_eps:
            stimulus_epochs.append(
                StimulusEpoch(
                    stimulus_on_s=float(ep["stimulus_on_s"]),
                    stimulus_off_s=float(ep["stimulus_off_s"]),
                    condition_label=str(ep.get("condition_label", "")),
                    meta={k: v for k, v in ep.items() if k not in {"stimulus_on_s", "stimulus_off_s", "condition_label"}},
                )
            )

    run_pipeline(args.input, cfg, args.outdir, stimulus_epochs=stimulus_epochs)
    print(f"Done. Outputs in: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
