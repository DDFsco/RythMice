import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


def epoch_mean_speed(encoder_df: pd.DataFrame, start_s: float, end_s: float) -> float:
    mask = (encoder_df["timestamp_s"] >= start_s) & (encoder_df["timestamp_s"] < end_s)
    if not np.any(mask):
        return np.nan
    return float(encoder_df.loc[mask, "speed_rps"].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze speed modulation for one session.")
    parser.add_argument("--encoder", type=Path, required=True, help="CSV with timestamp_s and speed_rps.")
    parser.add_argument("--events", type=Path, required=True, help="Session CSV from build_session_plan.py.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for analysis tables.")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    encoder = pd.read_csv(args.encoder)
    events = pd.read_csv(args.events)

    required_encoder_cols = {"timestamp_s", "speed_rps"}
    if not required_encoder_cols.issubset(encoder.columns):
        raise ValueError(f"Encoder CSV must include columns: {required_encoder_cols}")

    on_blocks = events[events["block_type"] == "stim_on"].copy()
    off_blocks = events[events["block_type"] == "matched_silence"].copy().reset_index(drop=True)
    on_blocks = on_blocks.reset_index(drop=True)

    n = min(len(on_blocks), len(off_blocks))
    on_blocks = on_blocks.iloc[:n].copy()
    off_blocks = off_blocks.iloc[:n].copy()

    rows = []
    for i in range(n):
        on = on_blocks.iloc[i]
        off = off_blocks.iloc[i]
        on_mean = epoch_mean_speed(encoder, float(on["start_s"]), float(on["end_s"]))
        off_mean = epoch_mean_speed(encoder, float(off["start_s"]), float(off["end_s"]))
        rows.append(
            {
                "trial_idx": i,
                "frequency_hz": on["frequency_hz"],
                "is_random": int(on["is_random"]),
                "on_mean_speed_rps": on_mean,
                "off_mean_speed_rps": off_mean,
                "delta_on_minus_off_rps": on_mean - off_mean,
            }
        )

    trial_df = pd.DataFrame(rows)
    trial_df.to_csv(outdir / "trial_level_modulation.csv", index=False)

    summary_rows = []
    grouped = trial_df.groupby(["frequency_hz", "is_random"], dropna=False)
    for (freq, is_random), g in grouped:
        valid = g.dropna(subset=["on_mean_speed_rps", "off_mean_speed_rps"])
        if len(valid) >= 2:
            stat = ttest_rel(valid["on_mean_speed_rps"], valid["off_mean_speed_rps"], nan_policy="omit")
            p_value = float(stat.pvalue)
        else:
            p_value = np.nan
        summary_rows.append(
            {
                "frequency_hz": freq,
                "is_random": is_random,
                "n_trials": len(valid),
                "mean_delta_on_minus_off_rps": float(valid["delta_on_minus_off_rps"].mean()) if len(valid) else np.nan,
                "p_value_on_vs_off_paired_t": p_value,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["frequency_hz", "is_random"])
    summary_df.to_csv(outdir / "summary_by_condition.csv", index=False)

    periodic = summary_df[summary_df["is_random"] == 0].copy()
    random = summary_df[summary_df["is_random"] == 1].copy()
    regularity = periodic.merge(random, on="frequency_hz", suffixes=("_periodic", "_random"))
    if not regularity.empty:
        regularity["delta_periodic_minus_random"] = (
            regularity["mean_delta_on_minus_off_rps_periodic"] - regularity["mean_delta_on_minus_off_rps_random"]
        )
    regularity.to_csv(outdir / "regularity_comparison.csv", index=False)

    print(f"Saved trial output to: {(outdir / 'trial_level_modulation.csv').resolve()}")
    print(f"Saved summary output to: {(outdir / 'summary_by_condition.csv').resolve()}")
    print(f"Saved regularity output to: {(outdir / 'regularity_comparison.csv').resolve()}")


if __name__ == "__main__":
    main()
