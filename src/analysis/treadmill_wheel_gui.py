"""
Interactive GUI: load wheel .dat, tune thresholds and bout parameters, re-run analysis
and refresh QC plots without leaving the app.

Run from repo root:
  python -m src.analysis.treadmill_wheel_gui
"""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from src.analysis.analyze_treadmill_wheel import (
    WheelAnalysisConfig,
    compute_speed_trace,
    detect_events,
    detect_running_bouts,
    load_data,
    preprocess_signal,
    refine_bout_classification,
    run_pipeline,
    session_summary,
)


# Hover help: what each control does and when to change it.
PARAM_HELP: dict[str, str] = {
    "file_path": (
        "Path to your LabJack / DAQ export: two columns (time, voltage).\n\n"
        "Change: pick a different recording; the file is re-read if its path or "
        "modification time changes."
    ),
    "time_unit": (
        "How timestamps in the file are scaled to seconds.\n"
        "• auto: median(Δt)≥1 in file units → milliseconds, else seconds.\n"
        "• s / ms: force seconds or milliseconds.\n\n"
        "Change: when auto mis-reads your clock (wrong duration or absurd sample rate)."
    ),
    "event_method": (
        "How each wheel rotation is counted.\n"
        "• threshold: hysteresis (high/low volts) — good for clean square-ish pulses.\n"
        "• peak: scipy find_peaks on smoothed voltage — good for rounded peaks.\n\n"
        "Change: if Fig 3 shows missed events (too few dots) or double counts (too many)."
    ),
    "threshold_high": (
        "Upper voltage for hysteresis crossing. Signal must rise to this level (from "
        "below low) to register one event.\n\n"
        "Change: raise if noise triggers events; lower if real rotations never reach "
        "the line. Must be > threshold_low."
    ),
    "threshold_low": (
        "Lower voltage to reset hysteresis so the next upward crossing can count again.\n\n"
        "Change: lower if the waveform never drops far enough to re-arm; raise if you "
        "get double counts on small ripples. Must stay below threshold_high."
    ),
    "peak_prominence": (
        "Minimum peak prominence (volt-like units) for find_peaks when Event method=peak.\n\n"
        "Change: increase to ignore small bumps; decrease if real peaks are not detected."
    ),
    "peak_distance_samples": (
        "Minimum index distance between peaks (samples) for Event method=peak.\n\n"
        "Change: increase if one rotation produces multiple peaks; decrease if peaks are "
        "merged and under-counted."
    ),
    "speed_window_s": (
        "Width (seconds) of the sliding window used to count events and convert to "
        "deg/s. Wider → smoother speed, more events per window.\n\n"
        "Change: widen (~0.2 s) if speed flickers between zero and high inside a real "
        "run; narrow if you need sharper time resolution."
    ),
    "speed_step_s": (
        "Step between consecutive speed window centers (seconds). Smaller → more "
        "points on the speed trace.\n\n"
        "Change: default 0.1 s with 0.2 s window gives overlapping windows and a "
        "continuous speed curve; reduce for finer time grid (slower)."
    ),
    "speed_visual_smooth_s": (
        "Extra moving average applied to the windowed speed for plots and CSV column "
        "speed_deg_s_smooth_vis only. Does NOT affect bout detection or movement "
        "threshold (those use the raw windowed speed).\n\n"
        "Change: set to 0 to disable; increase for a softer line on Fig 2."
    ),
    "movement_threshold_deg_s": (
        "Local speed above this marks the window as “active” for bout building.\n\n"
        "Change: lower if valid runs split into tiny gaps; raise if idle noise creates "
        "too many micro-bouts."
    ),
    "merge_gap_s": (
        "If two active bouts are separated by less than this (seconds), merge into one.\n\n"
        "Change: increase (e.g. 1.5–2.5 s) when the mouse briefly pauses mid-run but "
        "you want one continuous bout."
    ),
    "min_valid_duration_s": (
        "A bout must last at least this long (seconds) to count as valid (not micro), "
        "if it also passes event count and mean-speed checks.\n\n"
        "Change: lower slightly if true runs are short; raise to reject longer shuffles."
    ),
    "min_valid_events": (
        "Minimum detected events inside a bout for it to be valid.\n\n"
        "Change: lower if slow running yields too few counts; raise to reject twitchy "
        "segments with only 1–2 rotations."
    ),
    "micro_max_duration_s": (
        "Bouts shorter than this (seconds) are labeled micro (or fail valid rules).\n\n"
        "Change: usually tied to your definition of “flicker”; keep near 0.5–1 s unless "
        "you redefine micro vs valid."
    ),
    "micro_max_events": (
        "Bouts with fewer than this many events are treated as micro.\n\n"
        "Change: lower if slow wheels get misclassified as micro; raise to be stricter."
    ),
    "valid_mean_speed_deg_s": (
        "Mean speed (deg/s) within a bout must exceed this to be valid.\n\n"
        "Change: lower together with movement_threshold if everything becomes micro "
        "despite clear running; raise to require faster “real” runs only."
    ),
    "smoothing_window_samples": (
        "Moving-average length (samples) on voltage before event detection.\n\n"
        "Change: increase to reduce high-frequency noise (may blur fast edges); "
        "decrease if edges get too rounded and thresholds mis-fire."
    ),
    "degrees_per_event": (
        "Angular advance per counted event (default 18° per your rig).\n\n"
        "Change: only if your wheel/sensor geometry differs; affects deg/s and rev/s."
    ),
    "max_plot_points": (
        "Max points drawn for long voltage traces (decimation). Does not change saved "
        "CSV analysis — only GUI redraw speed.\n\n"
        "Change: lower on slow PCs; raise if the line looks too coarse."
    ),
    "run_button": (
        "Re-runs the pipeline with current parameters and refreshes all plots. "
        "Large files may take tens of seconds."
    ),
    "save_button": (
        "Writes the same outputs as the command-line tool (CSVs, JSON, PNGs) into "
        "the folder you choose, using the current parameter values."
    ),
}


class ToolTip:
    """Simple hover tooltip for any tk/ttk widget."""

    def __init__(self, widget: tk.Widget, text: str, wraplength: int = 340) -> None:
        self.widget = widget
        self.text = text.strip()
        self.wraplength = wraplength
        self._tip: Optional[tk.Toplevel] = None
        if not self.text:
            return
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event: Any = None) -> None:
        if self._tip is not None:
            return
        try:
            if not self.widget.winfo_viewable():
                return
        except tk.TclError:
            return
        x = int(self.widget.winfo_rootx() + 12)
        y = int(self.widget.winfo_rooty() + self.widget.winfo_height() + 6)
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        try:
            tw.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        lbl = tk.Label(
            tw,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            background="#ffffe0",
            foreground="#000000",
            font=("Segoe UI", 9),
            wraplength=self.wraplength,
            padx=10,
            pady=8,
        )
        lbl.pack()
        tw.update_idletasks()
        # Keep on screen
        sw = tw.winfo_screenwidth()
        sh = tw.winfo_screenheight()
        tw_w = tw.winfo_width()
        tw_h = tw.winfo_height()
        if x + tw_w > sw - 8:
            x = max(8, sw - tw_w - 8)
        if y + tw_h > sh - 8:
            y = max(8, int(self.widget.winfo_rooty() - tw_h - 6))
        tw.geometry(f"+{x}+{y}")

    def _hide(self, event: Any = None) -> None:
        if self._tip is not None:
            try:
                self._tip.destroy()
            except (tk.TclError, RuntimeError):
                pass
            self._tip = None


def _decimate(x: np.ndarray, y: np.ndarray, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_n:
        return x, y
    step = int(np.ceil(n / max_n))
    return x[::step], y[::step]


def _config_from_vars(vars_dict: dict[str, Any]) -> WheelAnalysisConfig:
    ev = vars_dict["event_method"].get().strip().lower()
    if ev not in ("threshold", "peak"):
        ev = "threshold"
    tu = vars_dict["time_unit"].get().strip().lower()
    if tu not in ("auto", "s", "ms"):
        tu = "auto"
    return WheelAnalysisConfig(
        degrees_per_event=float(vars_dict["degrees_per_event"].get()),
        smoothing_method="moving_average",
        smoothing_window_samples=int(float(vars_dict["smoothing_window_samples"].get())),
        event_method=ev,  # type: ignore[arg-type]
        threshold_high=float(vars_dict["threshold_high"].get()),
        threshold_low=float(vars_dict["threshold_low"].get()),
        peak_prominence=float(vars_dict["peak_prominence"].get()),
        peak_distance_samples=int(float(vars_dict["peak_distance_samples"].get())),
        speed_window_s=float(vars_dict["speed_window_s"].get()),
        speed_step_s=float(vars_dict["speed_step_s"].get()),
        speed_visual_smooth_s=float(vars_dict["speed_visual_smooth_s"].get()),
        movement_threshold_deg_s=float(vars_dict["movement_threshold_deg_s"].get()),
        min_valid_duration_s=float(vars_dict["min_valid_duration_s"].get()),
        min_valid_events=int(float(vars_dict["min_valid_events"].get())),
        micro_max_duration_s=float(vars_dict["micro_max_duration_s"].get()),
        micro_max_events=int(float(vars_dict["micro_max_events"].get())),
        valid_mean_speed_deg_s=float(vars_dict["valid_mean_speed_deg_s"].get()),
        merge_gap_s=float(vars_dict["merge_gap_s"].get()),
        time_unit=tu,  # type: ignore[arg-type]
    )


class TreadmillWheelGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Treadmill / wheel analyzer — live QC")
        root.geometry("1280x860")
        root.minsize(1024, 700)

        self._loaded_key: Optional[tuple[str, float]] = None
        self._loaded: Optional[dict[str, Any]] = None
        self._busy = False
        self._axes_full_limits: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
        self._mp_active = False
        self._mp_ax: Any = None
        self._mp_x0: Optional[float] = None
        self._mp_y0: Optional[float] = None
        self._mp_px0 = 0
        self._mp_py0 = 0
        self._mp_drag = False
        self._rp_active = False
        self._rp_ax: Any = None
        self._rp_px0 = 0
        self._rp_py0 = 0
        self._rp_drag = False
        self._pan_draw_timer: Optional[Any] = None
        self._destroying = False

        self._build_vars()
        self._build_layout()
        self.vars["time_unit"].trace_add("write", lambda *_: self._invalidate_cache())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_vars(self) -> None:
        self.vars: dict[str, Any] = {
            "file_path": tk.StringVar(value=""),
            "time_unit": tk.StringVar(value="auto"),
            "event_method": tk.StringVar(value="threshold"),
            "threshold_high": tk.StringVar(value="5"),
            "threshold_low": tk.StringVar(value="1.5"),
            "peak_prominence": tk.StringVar(value="0.5"),
            "peak_distance_samples": tk.StringVar(value="10"),
            "speed_window_s": tk.StringVar(value="0.2"),
            "speed_step_s": tk.StringVar(value="0.1"),
            "speed_visual_smooth_s": tk.StringVar(value="0"),
            "movement_threshold_deg_s": tk.StringVar(value="120"),
            "merge_gap_s": tk.StringVar(value="2"),
            "min_valid_duration_s": tk.StringVar(value="2.0"),
            "min_valid_events": tk.StringVar(value="5"),
            "micro_max_duration_s": tk.StringVar(value="2.0"),
            "micro_max_events": tk.StringVar(value="5"),
            "valid_mean_speed_deg_s": tk.StringVar(value="30"),
            "smoothing_window_samples": tk.StringVar(value="5"),
            "degrees_per_event": tk.StringVar(value="18"),
            "max_plot_points": tk.StringVar(value="80000"),
        }

    def _build_layout(self) -> None:
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(main, width=320)
        main.add(left, weight=0)

        right = ttk.Frame(main)
        main.add(right, weight=1)

        # --- Left: controls ---
        fp_lbl = ttk.Label(left, text="Data file (.dat / CSV)")
        fp_lbl.pack(anchor="w", pady=(0, 2))
        ToolTip(fp_lbl, PARAM_HELP["file_path"])
        fp_row = ttk.Frame(left)
        fp_row.pack(fill=tk.X)
        fp_entry = ttk.Entry(fp_row, textvariable=self.vars["file_path"], width=28)
        fp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ToolTip(fp_entry, PARAM_HELP["file_path"])
        ttk.Button(fp_row, text="Browse…", command=self._browse).pack(side=tk.LEFT, padx=(4, 0))

        run_btn = ttk.Button(left, text="Run analysis / refresh plots", command=self._on_run)
        run_btn.pack(fill=tk.X, pady=8)
        ToolTip(run_btn, PARAM_HELP["run_button"])
        save_btn = ttk.Button(left, text="Save full outputs to folder…", command=self._on_save_folder)
        save_btn.pack(fill=tk.X, pady=(0, 8))
        ToolTip(save_btn, PARAM_HELP["save_button"])

        self.status = ttk.Label(left, text="Load a file and click Run.", wraplength=300)
        self.status.pack(anchor="w", pady=4)
        ttk.Label(
            left,
            text="Tip: hover parameters for help. Plots: left-drag=pan, left-click=zoom in, right-click=zoom out; Reset zoom restores full view.",
            wraplength=300,
            font=("Segoe UI", 8),
            foreground="#555555",
        ).pack(anchor="w", pady=(0, 4))

        cf = ttk.LabelFrame(left, text="Parameters")
        cf.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(cf, highlightthickness=0)
        sb = ttk.Scrollbar(cf, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        def _row(r: int, label: str, varname: str) -> None:
            lb = ttk.Label(inner, text=label)
            lb.grid(row=r, column=0, sticky="w", padx=4, pady=2)
            en = ttk.Entry(inner, textvariable=self.vars[varname], width=14)
            en.grid(row=r, column=1, sticky="e", padx=4, pady=2)
            ht = PARAM_HELP.get(varname, "")
            if ht:
                ToolTip(lb, ht)
                ToolTip(en, ht)

        r = 0
        lbl_tu = ttk.Label(inner, text="Time unit")
        lbl_tu.grid(row=r, column=0, sticky="w", padx=4, pady=2)
        cb_tu = ttk.Combobox(
            inner,
            textvariable=self.vars["time_unit"],
            values=("auto", "s", "ms"),
            state="readonly",
            width=12,
        )
        cb_tu.grid(row=r, column=1, sticky="e", padx=4, pady=2)
        ToolTip(lbl_tu, PARAM_HELP["time_unit"])
        ToolTip(cb_tu, PARAM_HELP["time_unit"])
        r += 1
        lbl_em = ttk.Label(inner, text="Event method")
        lbl_em.grid(row=r, column=0, sticky="w", padx=4, pady=2)
        cb_em = ttk.Combobox(
            inner,
            textvariable=self.vars["event_method"],
            values=("threshold", "peak"),
            state="readonly",
            width=12,
        )
        cb_em.grid(row=r, column=1, sticky="e", padx=4, pady=2)
        ToolTip(lbl_em, PARAM_HELP["event_method"])
        ToolTip(cb_em, PARAM_HELP["event_method"])
        r += 1
        _row(r, "threshold_high (V)", "threshold_high")
        r += 1
        _row(r, "threshold_low (V)", "threshold_low")
        r += 1
        _row(r, "peak_prominence", "peak_prominence")
        r += 1
        _row(r, "peak_distance_samples", "peak_distance_samples")
        r += 1
        _row(r, "speed_window_s", "speed_window_s")
        r += 1
        _row(r, "speed_step_s", "speed_step_s")
        r += 1
        _row(r, "speed_smooth_vis (s, 0=off)", "speed_visual_smooth_s")
        r += 1
        _row(r, "movement_threshold (deg/s)", "movement_threshold_deg_s")
        r += 1
        _row(r, "merge_gap_s", "merge_gap_s")
        r += 1
        _row(r, "min_valid_duration_s", "min_valid_duration_s")
        r += 1
        _row(r, "min_valid_events", "min_valid_events")
        r += 1
        _row(r, "micro_max_duration_s", "micro_max_duration_s")
        r += 1
        _row(r, "micro_max_events", "micro_max_events")
        r += 1
        _row(r, "valid_mean_speed (deg/s)", "valid_mean_speed_deg_s")
        r += 1
        _row(r, "smoothing_window_samples", "smoothing_window_samples")
        r += 1
        _row(r, "degrees_per_event", "degrees_per_event")
        r += 1
        _row(r, "max_plot_points (decimate)", "max_plot_points")

        # --- Right: matplotlib (voltage full width; speed | events; duration | mean speed) ---
        self.fig = Figure(figsize=(11, 9.5), dpi=100, layout="constrained")
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 0.9])
        self.ax1 = self.fig.add_subplot(gs[0, :])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[1, 1])
        self.ax4 = self.fig.add_subplot(gs[2, 0])
        self.ax5 = self.fig.add_subplot(gs[2, 1])

        zoom_bar = ttk.Frame(right)
        zoom_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        ttk.Button(zoom_bar, text="Reset zoom (all plots)", command=self._reset_figure_zoom).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Label(
            zoom_bar,
            text=(
                "Plots: left-drag = pan | left-click = zoom in | right-click = zoom out. "
                "Toolbar pan/zoom may conflict—use Home or turn tool off."
            ),
            font=("Segoe UI", 8),
            foreground="#444444",
        ).pack(side=tk.LEFT)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.mpl_connect("button_press_event", self._on_mpl_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mpl_motion)
        self.canvas.mpl_connect("button_release_event", self._on_mpl_release)

        mpl_widget = self.canvas.get_tk_widget()

        def _wheel_params_only(e: Any) -> None:
            w: Any = e.widget
            while w is not None:
                if w is mpl_widget:
                    return
                try:
                    w = w.master
                except tk.TclError:
                    break
            w = e.widget
            while w is not None:
                if w is cf:
                    canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
                    return
                try:
                    w = w.master
                except tk.TclError:
                    break

        self.root.bind_all("<MouseWheel>", _wheel_params_only, add="+")

    def _schedule_pan_redraw(self) -> None:
        """Throttle canvas redraw during pan: each motion updates limits, but draw runs at most ~40 Hz."""
        if self._pan_draw_timer is not None:
            self.root.after_cancel(self._pan_draw_timer)
        self._pan_draw_timer = self.root.after(24, self._flush_pan_redraw)

    def _flush_pan_redraw(self) -> None:
        self._pan_draw_timer = None
        self.canvas.draw_idle()

    def _flush_pan_immediate(self) -> None:
        if self._pan_draw_timer is not None:
            self.root.after_cancel(self._pan_draw_timer)
            self._pan_draw_timer = None
        self.canvas.draw_idle()

    def _zoom_ax_at(self, ax: Any, xdata: float, ydata: float, *, zoom_in: bool) -> None:
        """Zoom in (narrow) or out (wide) keeping (xdata, ydata) fixed on screen."""
        base = 1.35
        scale = 1.0 / base if zoom_in else base
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        w = cur_xlim[1] - cur_xlim[0]
        h = cur_ylim[1] - cur_ylim[0]
        if w <= 0 or h <= 0:
            return
        new_w = w * scale
        new_h = h * scale
        relx = (cur_xlim[1] - xdata) / w
        rely = (cur_ylim[1] - ydata) / h
        ax.set_xlim(xdata - new_w * (1 - relx), xdata + new_w * relx)
        ax.set_ylim(ydata - new_h * (1 - rely), ydata + new_h * rely)

    def _on_mpl_press(self, event: Any) -> None:
        if event.inaxes is None:
            return
        if event.button == 1:
            self._rp_active = False
            if event.xdata is None or event.ydata is None:
                return
            self._mp_active = True
            self._mp_ax = event.inaxes
            self._mp_x0 = float(event.xdata)
            self._mp_y0 = float(event.ydata)
            self._mp_px0 = int(event.x)
            self._mp_py0 = int(event.y)
            self._mp_drag = False
        elif event.button == 3:
            self._mp_active = False
            self._mp_ax = None
            self._mp_drag = False
            if event.xdata is None or event.ydata is None:
                return
            self._rp_active = True
            self._rp_ax = event.inaxes
            self._rp_px0 = int(event.x)
            self._rp_py0 = int(event.y)
            self._rp_drag = False

    def _on_mpl_motion(self, event: Any) -> None:
        if self._rp_active and self._rp_ax is not None:
            if event.inaxes == self._rp_ax:
                if math.hypot(event.x - self._rp_px0, event.y - self._rp_py0) >= 5:
                    self._rp_drag = True
            return
        if not self._mp_active or self._mp_ax is None:
            return
        if event.inaxes != self._mp_ax or event.xdata is None or self._mp_x0 is None or self._mp_y0 is None:
            return
        dist_px = math.hypot(event.x - self._mp_px0, event.y - self._mp_py0)
        if dist_px < 5:
            return
        self._mp_drag = True
        dx = float(event.xdata) - self._mp_x0
        dy = float(event.ydata) - self._mp_y0
        ax = self._mp_ax
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(x0 - dx, x1 - dx)
        ax.set_ylim(y0 - dy, y1 - dy)
        self._mp_x0 = float(event.xdata)
        self._mp_y0 = float(event.ydata)
        self._schedule_pan_redraw()

    def _on_mpl_release(self, event: Any) -> None:
        if event.button == 3:
            if not self._rp_active:
                return
            ax = self._rp_ax
            was_drag = self._rp_drag
            self._rp_active = False
            self._rp_ax = None
            self._rp_drag = False
            if was_drag or ax is None:
                return
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            if math.hypot(event.x - self._rp_px0, event.y - self._rp_py0) > 8:
                return
            self._zoom_ax_at(ax, float(event.xdata), float(event.ydata), zoom_in=False)
            self.canvas.draw_idle()
            return

        if event.button != 1 or not self._mp_active:
            return
        ax = self._mp_ax
        was_drag = self._mp_drag
        self._mp_active = False
        self._mp_ax = None
        try:
            if was_drag:
                self._flush_pan_immediate()
                return
            if ax is None:
                return
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            dist_px = math.hypot(event.x - self._mp_px0, event.y - self._mp_py0)
            if dist_px > 8:
                return
            self._zoom_ax_at(ax, float(event.xdata), float(event.ydata), zoom_in=True)
            self.canvas.draw_idle()
        finally:
            self._mp_drag = False

    def _snapshot_axes_limits(self) -> None:
        self._axes_full_limits.clear()
        for ax in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5):
            self._axes_full_limits[id(ax)] = (ax.get_xlim(), ax.get_ylim())

    def _reset_figure_zoom(self) -> None:
        for ax in (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5):
            lims = self._axes_full_limits.get(id(ax))
            if lims is not None:
                ax.set_xlim(lims[0])
                ax.set_ylim(lims[1])
        self.canvas.draw_idle()

    def _browse(self) -> None:
        p = filedialog.askopenfilename(
            title="Select .dat or CSV",
            filetypes=[("Data", "*.dat *.csv *.txt"), ("All", "*.*")],
        )
        if p:
            self.vars["file_path"].set(p)
            self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._loaded_key = None
        self._loaded = None

    def _get_loaded(self) -> dict[str, Any]:
        """Load data from the current file path (call from main thread only)."""
        raw = self.vars["file_path"].get().strip()
        if not raw:
            raise ValueError("Choose a data file first.")
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        tu = self.vars["time_unit"].get().strip().lower()
        if tu not in ("auto", "s", "ms"):
            tu = "auto"
        key = (str(path.resolve()), path.stat().st_mtime, tu)
        if self._loaded_key != key or self._loaded is None:
            self._loaded = load_data(path, time_column=None, voltage_column=None, time_unit=tu)  # type: ignore[arg-type]
            self._loaded_key = key
        return self._loaded

    def _on_close(self) -> None:
        """Tear down Matplotlib *before* root.destroy so Tk PhotoImages/vars are not finalized later off-loop."""
        if self._destroying:
            return
        self._destroying = True
        try:
            if self._pan_draw_timer is not None:
                self.root.after_cancel(self._pan_draw_timer)
        except (tk.TclError, ValueError, RuntimeError):
            pass
        self._pan_draw_timer = None

        try:
            self.root.unbind_all("<MouseWheel>")
        except tk.TclError:
            pass

        try:
            self.toolbar.destroy()
        except (tk.TclError, AttributeError, RuntimeError):
            pass

        try:
            self.canvas.get_tk_widget().destroy()
        except (tk.TclError, AttributeError, RuntimeError):
            pass

        try:
            plt.close(self.fig)
        except Exception:
            pass

        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _on_run(self) -> None:
        if self._busy:
            return
        try:
            file_path = self.vars["file_path"].get().strip()
            if not file_path:
                raise ValueError("Choose a data file first.")
            if not Path(file_path).is_file():
                raise FileNotFoundError(f"File not found: {file_path}")
            max_pts = int(float(self.vars["max_plot_points"].get()))
            cfg = _config_from_vars(self.vars)
            tu = self.vars["time_unit"].get().strip().lower()
            if tu not in ("auto", "s", "ms"):
                tu = "auto"
            cfg.time_unit = tu  # type: ignore[assignment]
        except (ValueError, OSError) as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self._busy = True
        self.status.config(text="Running…")
        self.root.update_idletasks()

        fp = file_path

        def work() -> None:
            err: Optional[BaseException] = None
            payload: Optional[dict[str, Any]] = None
            try:
                loaded = load_data(
                    Path(fp),
                    time_column=None,
                    voltage_column=None,
                    time_unit=cfg.time_unit,  # type: ignore[arg-type]
                )
                t = loaded["timestamp_s"]
                v = loaded["voltage"]
                t0 = float(t[0])

                pre = preprocess_signal(t, v, cfg)
                events = detect_events(t, pre["processed_voltage"], cfg)
                speed = compute_speed_trace(t, events["event_times_s"], cfg)
                bouts_df, run_state = detect_running_bouts(t, events["event_times_s"], speed, cfg)
                bouts_df = refine_bout_classification(bouts_df, cfg)
                summ = session_summary(bouts_df, speed)

                payload = {
                    "t": t,
                    "t0": t0,
                    "v_raw": pre["raw_voltage"],
                    "v_proc": pre["processed_voltage"],
                    "events": events,
                    "speed": speed,
                    "run_state": run_state,
                    "bouts_df": bouts_df,
                    "summary": summ,
                    "n_events": len(events["event_times_s"]),
                    "max_pts": max_pts,
                    "cfg": cfg,
                    "duplicate_timestamps_dropped": int(loaded.get("duplicate_timestamps_dropped", 0)),
                    "source_name": Path(fp).name,
                }
            except BaseException as e:
                err = e
            self.root.after(0, lambda pl=payload, er=err: self._finish_run(pl, er))

        threading.Thread(target=work, daemon=True).start()

    def _finish_run(self, payload: Optional[dict[str, Any]], err: Optional[BaseException]) -> None:
        try:
            if err is not None:
                self.status.config(text=f"Error: {err}")
                messagebox.showerror("Analysis failed", str(err))
                return

            assert payload is not None
            t = payload["t"]
            t0 = payload["t0"]
            tt = t - t0
            v_raw = payload["v_raw"]
            v_proc = payload["v_proc"]
            events = payload["events"]
            speed = payload["speed"]
            run_state = payload["run_state"]
            bouts_df = payload["bouts_df"]
            summ = payload["summary"]
            max_pts = payload["max_pts"]
            cfg = payload["cfg"]

            eng = summ["engagement"]
            spd_sum = summ["speed_valid_bouts_only"]
            mean_spd = float(spd_sum["mean_speed_deg_s"])
            med_spd = float(spd_sum["median_speed_deg_s"])
            spd_tail = ""
            if np.isfinite(mean_spd) and np.isfinite(med_spd):
                spd_tail = f" | mean {mean_spd:.1f} | median {med_spd:.1f} deg/s"
            elif np.isfinite(mean_spd):
                spd_tail = f" | mean {mean_spd:.1f} deg/s"
            elif np.isfinite(med_spd):
                spd_tail = f" | median {med_spd:.1f} deg/s"
            dup = int(payload.get("duplicate_timestamps_dropped", 0))
            status_msg = (
                f"Events: {payload['n_events']} | valid bouts: {eng['n_valid_bouts']} | "
                f"micro bouts: {eng['n_micro_bouts']} | valid run time: {eng['total_running_time_valid_s']:.1f}s"
                f"{spd_tail}"
            )
            if dup:
                status_msg += f" | dropped {dup} duplicate time row(s)"
            self.status.config(text=status_msg)

            # Fig 1
            ax = self.ax1
            ax.clear()
            tx, vy = _decimate(tt, v_raw, max_pts)
            ax.plot(tx, vy, color="0.3", lw=0.5, label="Raw voltage")
            if not bouts_df.empty:
                for _, b in bouts_df.iterrows():
                    c = "C2" if b["type"] == "valid" else "C1"
                    ax.axvspan(float(b["start_time_s"]) - t0, float(b["end_time_s"]) - t0, color=c, alpha=0.25)
            ax.set_xlabel("Time (s) from start")
            ax.set_ylabel("Voltage")
            ax.set_title("Voltage + bouts (green=valid, orange=micro)")
            ax.legend(loc="upper right", fontsize=7)

            # Fig 2 — continuous windowed speed (+ optional viz-only smooth)
            ax = self.ax2
            ax.clear()
            wc = speed["window_center_s"] - t0
            ax.plot(wc, speed["speed_deg_s"], color="C0", lw=1.0, label="Speed (sliding window)")
            sv = speed.get("speed_deg_s_smooth_vis")
            if sv is not None and len(sv) == len(wc):
                ax.plot(wc, sv, color="C1", lw=0.95, ls="--", alpha=0.85, label="Smoothed (viz only)")
            ax.axhline(
                float(cfg.movement_threshold_deg_s),
                color="0.45",
                ls=":",
                lw=0.9,
                label=f"Move thr={cfg.movement_threshold_deg_s:g}",
            )
            ax.set_xlabel("Time (s) from start")
            ax.set_ylabel("deg/s")
            ax.set_title("Speed (continuous windowed)")
            ax.legend(loc="upper right", fontsize=7)

            # Fig 3
            ax = self.ax3
            ax.clear()
            tx, vp = _decimate(tt, v_proc, max_pts)
            ax.plot(tx, vp, color="0.2", lw=0.5, label="Processed voltage")
            et = np.asarray(events["event_times_s"], dtype=float) - t0
            if len(et):
                idx = np.asarray(events["event_indices"], dtype=int)
                ax.scatter(et, v_proc[idx], s=10, c="red", zorder=5, label="Events")
            if cfg.event_method == "threshold":
                ax.axhline(cfg.threshold_high, color="C3", ls="--", lw=0.9)
                ax.axhline(cfg.threshold_low, color="C4", ls=":", lw=0.9)
            ax.set_xlabel("Time (s) from start")
            ax.set_ylabel("Voltage (processed)")
            ax.set_title(f"Event QC | {events['method']}")
            ax.legend(loc="upper right", fontsize=7)

            # Fig 4–5: bout histograms
            for ax in (self.ax4, self.ax5):
                ax.clear()
            if bouts_df.empty:
                self.ax4.text(0.5, 0.5, "No bouts", ha="center", va="center", transform=self.ax4.transAxes)
                self.ax5.text(0.5, 0.5, "No bouts", ha="center", va="center", transform=self.ax5.transAxes)
            else:
                valid = bouts_df[bouts_df["type"] == "valid"]
                micro = bouts_df[bouts_df["type"] == "micro"]
                bins_d = max(8, int(np.sqrt(len(bouts_df))))
                self.ax4.hist(
                    [valid["duration_s"].to_numpy(), micro["duration_s"].to_numpy()],
                    bins=bins_d,
                    label=["valid", "micro"],
                    alpha=0.75,
                )
                self.ax4.set_xlabel("Duration (s)")
                self.ax4.set_ylabel("Count")
                self.ax4.set_title("Bout duration")
                self.ax4.legend(fontsize=7)
                self.ax5.hist(
                    [valid["mean_speed_deg_s"].to_numpy(), micro["mean_speed_deg_s"].to_numpy()],
                    bins=bins_d,
                    label=["valid", "micro"],
                    alpha=0.75,
                )
                self.ax5.set_xlabel("Mean speed (deg/s)")
                self.ax5.set_ylabel("Count")
                self.ax5.set_title("Bout mean speed")
                self.ax5.legend(fontsize=7)

            self.fig.suptitle(str(payload.get("source_name", "session")), fontsize=10)
            self.canvas.draw_idle()
            self._snapshot_axes_limits()
        finally:
            self._busy = False

    def _on_save_folder(self) -> None:
        raw = self.vars["file_path"].get().strip()
        if not raw:
            messagebox.showinfo("Save", "Choose a data file first.")
            return
        out = filedialog.askdirectory(title="Select output folder")
        if not out:
            return
        try:
            cfg = _config_from_vars(self.vars)
            cfg.time_unit = self.vars["time_unit"].get().strip().lower()  # type: ignore[assignment]
            if cfg.time_unit not in ("auto", "s", "ms"):
                cfg.time_unit = "auto"
            run_pipeline(Path(raw), cfg, Path(out))
            messagebox.showinfo("Save", f"Saved outputs to:\n{out}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


def main() -> None:
    root = tk.Tk()
    TreadmillWheelGui(root)
    try:
        root.mainloop()
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
