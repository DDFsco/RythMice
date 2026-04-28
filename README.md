# RythMice

Software and scripts for the mouse locomotion rhythm study described in:
`Rhythm_Mouse_Locomotion_Proposal_v4-2.pdf`.

## What this project contains

- Stimulus generation for periodic and random click trains (2-12 Hz by default)
- Session timeline generation (baseline, ON/OFF blocks, post-silence)
- Analysis utilities for speed modulation and periodic vs random comparisons
- DAT-based wheel-running analysis for click vs no-click conditions
- Config files that match the current proposal and can be edited as the protocol evolves

## Initial folder layout

- `config/` study and stimulus settings
- `matlab/` MATLAB-first stimulus/playback scripts
- `src/stimuli/` audio stimulus generation
- `src/protocol/` session plan creation
- `src/analysis/` encoder/session analysis
- `data/` raw/intermediate outputs (created by you during experiments)
- `outputs/` generated stimuli and reports

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Generate click stimuli:
   - `python src/stimuli/generate_click_stimuli.py --config config/stimulus_config.json --outdir outputs/stimuli`
4. Generate a session schedule:
   - `python src/protocol/build_session_plan.py --config config/session_config.json --outdir outputs/session_plans`
5. Analyze one session (after you export encoder data):
   - `python src/analysis/analyze_speed.py --encoder data/raw/example_encoder.csv --events outputs/session_plans/session_001.csv --outdir outputs/analysis`

## Click vs no-click pipeline (DAT files)

Use this when LabJack records wheel amplitude traces and you want a direct behavior comparison across conditions.

1. Prepare a manifest CSV from `config/analysis_manifest_template.csv`:
   - Required columns: `mouse_id`, `condition`, `data_file`
   - Optional: `session_id`
   - `condition` must be `click` or `no_click`
2. Place your `.dat` files under `data/raw/` (or anywhere and use absolute paths in manifest).
3. Run the analysis:
   - `python src/analysis/analyze_click_vs_noclick.py --manifest config/analysis_manifest_template.csv --data-root . --outdir outputs/analysis_click`
4. If your DAT columns are unusual, specify them:
   - `python src/analysis/analyze_click_vs_noclick.py --manifest <manifest.csv> --outdir outputs/analysis_click --time-column Time --signal-column CH0`
5. If your DAT has no time column, pass sample rate:
   - `python src/analysis/analyze_click_vs_noclick.py --manifest <manifest.csv> --outdir outputs/analysis_click --sample-rate-hz 1000`

Outputs in `outputs/analysis_click`:
- `recording_summary.csv`: per-recording running metrics
- `mouse_level_paired.csv`: mouse-level click vs no-click table + delta
- `statistics.json`: paired t-test and Wilcoxon p-values
- `paired_mouse_comparison.png`: paired lines per mouse
- `group_boxplot.png`: group summary plot
- `trace_plots/*.png`: quick QC plots for each recording

## MATLAB workflow (recommended for your setup)

If you will record with LabJack tools and run audio in MATLAB:

1. Generate all periodic/random click WAVs:
   - `generate_click_wavs('outputs/stimuli_matlab', [2 4 6 8 10 12], 120, 44100, 5, 0.75, 42)`
2. Quick loop test of one file:
   - `play_wav_loop('outputs/stimuli_matlab/periodic_8hz.wav', 10, 0)`
3. Run a full randomized ON/OFF session:
   - `run_session_playlist('outputs/stimuli_matlab', [2 6 8 12], true, 120, 300, 1200, 600, 7)`
4. Export LabJack speed data to CSV and run analysis:
   - `python src/analysis/analyze_speed.py --encoder data/raw/example_encoder.csv --events outputs/session_plans_matlab/<session_file>.csv --outdir outputs/analysis`

## Data conventions (recommended)

- Encoder CSV columns:
  - `timestamp_s` (float seconds)
  - `speed_rps` (float revolutions/sec)
- Events CSV columns:
  - `start_s`, `end_s`, `block_type`, `stimulus_name`, `frequency_hz`, `is_random`

## Treadmill / wheel `.dat` analysis (events, bouts, speed)

Use `src/analysis/analyze_treadmill_wheel.py` when LabJack (or similar) logs **timestamp + voltage** and you need **wheel events**, **running speed (deg/s and rev/s)**, **running bouts** (valid vs micro), and QC figures. The locomotor speed pipeline is summarized below; raw voltage is **not** interpolated or resampled for speed—only **encoder event times** feed the rate estimate.

### Speed estimation (implementation)

Pipeline (`compute_speed_trace`):

1. **Encoder events** — Event timestamps are detected from the voltage trace using the chosen hysteresis (`threshold_high` / `threshold_low`) or **peak** detection (`peak_prominence`, `peak_distance_samples`), after optional voltage smoothing (`smoothing_*`).
2. **Uniform time grid** — The recording interval `[t_first, t_last]` is divided into contiguous bins of width **`1 / speed_grid_hz`** seconds. Allowed values: **`speed_grid_hz = 100`** (10 ms bins, default) or **`1000`** (1 ms bins). Bin edges align at `t_first`.
3. **Event-count train** — Events are **binned** with `numpy.histogram` (counts per bin). No interpolation of the analog voltage onto this grid.
4. **Instantaneous event rate** — Per bin: **`event_rate (events/s) = count_in_bin / bin_width`**.
5. **Gaussian smoothing on the rate** — The rate vector is smoothed along time with **`scipy.ndimage.gaussian_filter1d`** (`mode="nearest"` at edges). Width **σ is specified in seconds**; internally it is converted to **samples** as `σ / bin_width`. Two independent Gaussian passes on the **same** raw rate train:
   - **`sigma_analysis_s`** (default **0.5 s**) → `event_rate_per_s_gaussian_analysis` → primary angular speed **`speed_deg_s`** and **`speed_rev_s`**. Used for **movement threshold**, **active bouts**, **session / valid-bout statistics**, and **`speed_distribution_valid.csv`**.
   - **`sigma_visualization_s`** (default **1.0 s**, usually ≥ analysis) → **`speed_deg_s_smooth_vis`** for a smoother trace on **Fig 2** (and the same column in exports). Does not change bout rules relative to `speed_deg_s`.
6. **Angular speed** — **`speed_deg_s = smoothed_event_rate × degrees_per_event`** (and the same for the visualization curve).
7. **Thresholding & bouts** — A bin is **active** when `speed_deg_s` (analysis) **>** `movement_threshold_deg_s` (default **80 deg/s**). Contiguous active regions are merged with `merge_gap_s`, then classified as valid vs micro using duration, event count, and `valid_mean_speed_deg_s`.
8. **QC figures** — **Fig 3** shows **processed voltage + event markers** only. **Fig 2** overlays analysis speed and (by default) the wider-σ visualization speed.

CLI flags: `--speed-grid-hz`, `--sigma-analysis-s`, `--sigma-visualization-s`, `--movement-threshold-deg-s`, `--degrees-per-event`. All must keep **`sigma_*` > 0**.

### Prerequisites

- Python 3.10+ and `pip install -r requirements.txt`
- On Windows, if `python` opens the Microsoft Store, install Python from [python.org](https://www.python.org/downloads/) or run `winget install Python.Python.3.12`, or disable **Settings → Apps → Advanced app settings → App execution aliases** for `python.exe`.

### Basic run

From the repository root (`MiceRyth/`):

```bash
python -m src.analysis.analyze_treadmill_wheel --input path/to/recording.dat --outdir outputs/wheel_run_001
```

### Interactive GUI (tune thresholds and refresh plots)

From the repo root:

```bash
python -m src.analysis.treadmill_wheel_gui
```

On Windows, if `python` opens the **Microsoft Store** or says *Python was not found*, use the launcher (picks a real install under `%LOCALAPPDATA%\Programs\Python\` and skips the `WindowsApps` stub):

```powershell
.\scripts\run_treadmill_wheel_gui.ps1
```

Or call the interpreter explicitly, for example:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m src.analysis.treadmill_wheel_gui
```

Browse to a `.dat` file, edit parameters (thresholds, bout rules, etc.), click **Run analysis / refresh plots**. The five-panel figure updates after each run (large traces are decimated for speed; full-resolution exports use **Save full outputs to folder…**, which calls the same pipeline as the CLI).

### Input file format

- Tab- or space-separated table with **two numeric columns**: time, voltage.
- **No header row** is supported: if the first line is two numbers, it is read as data (not column names).
- With a text header (e.g. `time voltage`), the script still auto-detects columns when possible.
- Optional: `--time-column` and `--voltage-column` to force column names.
- Time axis: default **auto** treats `median(Δt) ≥ 1` in file units as **milliseconds**, otherwise **seconds**. Override with `--time-unit s` or `--time-unit ms` if auto-detection is wrong.

### Useful parameters (all optional)

| Goal | Flags (examples) |
|------|-------------------|
| Event detection (hysteresis, default) | `--threshold-high`, `--threshold-low` (volts; **high > low**) |
| Event detection (peaks) | `--event-method peak`, `--peak-prominence`, `--peak-distance-samples` |
| Voltage smoothing (before events) | `--smoothing moving_average\|savgol\|none`, `--smoothing-window-samples` |
| Speed grid | `--speed-grid-hz` **`100`** or **`1000`** (bin width = 1 / Hz) |
| Gaussian σ on **event rate** | `--sigma-analysis-s` (default **0.5** s; bouts & `speed_deg_s`), `--sigma-visualization-s` (default **1.0** s; `speed_deg_s_smooth_vis` & Fig 2) |
| Movement threshold (on analysis speed) | `--movement-threshold-deg-s` (default **80**) |
| Bout merging / validity | `--merge-gap-s` (default 2), `--min-valid-duration-s`, `--min-valid-events`, `--valid-mean-speed-deg-s` |
| Micro-bout cutoffs | `--micro-max-duration-s`, `--micro-max-events` |
| Wheel geometry | `--degrees-per-event` (default **18**) |

If **micro-bouts dominate** but you see long **0–5 V** oscillation blocks in the plots, try **increasing** `--merge-gap-s`, **increasing** `--sigma-analysis-s` / `--sigma-visualization-s`, and **lowering** `--movement-threshold-deg-s` / `--valid-mean-speed-deg-s` slightly, then re-run.

### Debug zoom (voltage + speed in a time range)

Either a fixed window (seconds, same time base as after load):

```bash
python -m src.analysis.analyze_treadmill_wheel --input recording.dat --outdir out --debug-window-start 120 --debug-window-end 125
```

Or around the *k*-th detected event (0-based):

```bash
python -m src.analysis.analyze_treadmill_wheel --input recording.dat --outdir out --debug-event-index 100 --debug-event-padding-s 0.5
```

Extra outputs: `*_fig_debug_zoom.png`, `*_debug_window_voltage.csv`, and `meta.debug_window_*` in `session_metrics.json`.

### Stimulus alignment (optional)

Pass a JSON array of epochs; bouts CSV gains overlap columns:

```bash
python -m src.analysis.analyze_treadmill_wheel --input recording.dat --outdir out --stimulus-epochs epochs.json
```

Example `epochs.json`:

```json
[
  {"stimulus_on_s": 0, "stimulus_off_s": 60, "condition_label": "sound"},
  {"stimulus_on_s": 120, "stimulus_off_s": 180, "condition_label": "silence"}
]
```

### Outputs (in `--outdir`)

- `bouts_summary.csv`: per-bout times, duration, event count, mean/peak speed, `type` (`valid` / `micro`)
- `timeseries_speed.csv`: one row per bin **center** (`timestamp_s`): **`speed_deg_s`** (σ_analysis), **`speed_deg_s_smooth_vis`** (σ_viz), **`speed_rev_s`**, **`event_count_per_bin`**, **`event_rate_per_s_raw`**, **`event_rate_per_s_gaussian_analysis`**, **`event_rate_per_s_gaussian_viz`**, **`running_state`**, **`bout_id`**, **`bout_type`**, **`voltage_interp`** (linear interp of raw voltage at bin centers)
- `session_metrics.json`: engagement (valid bouts), speed stats (valid bouts only), micro summaries, config echo
- `speed_distribution_valid.csv`: histogram of speeds inside valid bouts
- `*_fig1_voltage_bouts.png` … `*_fig4_bout_hist.png`: QC figures

### Example: bundled test file

PowerShell helper (falls back to `%LOCALAPPDATA%\Programs\Python\Python312\python.exe` if `python` is missing):

```powershell
.\scripts\run_test_nosound_wheel.ps1
```

Equivalent manual command (thresholds depend on your voltage scale; adjust for your rig):

```bash
python -m src.analysis.analyze_treadmill_wheel --input src/rawdata/test_nosound.dat --outdir outputs/wheel_test_nosound --time-unit s --threshold-high 5.111 --threshold-low 5.105 --movement-threshold-deg-s 5 --valid-mean-speed-deg-s 5 --smoothing-window-samples 3
```

## Next additions (after first dry run)

- DAQ adapter for direct `.dat` ingestion
- Optional video/pose timestamp alignment hooks
- Automated per-animal and cohort-level report generation
