import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class StimCondition:
    stimulus_name: str
    frequency_hz: float
    is_random: int


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_conditions(tempos: List[float], include_random: bool) -> List[StimCondition]:
    conds: List[StimCondition] = []
    for hz in tempos:
        conds.append(StimCondition(f"periodic_{int(hz)}hz.wav", hz, 0))
        if include_random:
            conds.append(StimCondition(f"random_{int(hz)}hz.wav", hz, 1))
    return conds


def main() -> None:
    parser = argparse.ArgumentParser(description="Create randomized ON/OFF session timeline.")
    parser.add_argument("--config", type=Path, required=True, help="Path to session JSON config.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for CSV session plans.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    session_id = str(cfg["session_id"])
    baseline_s = float(cfg["baseline_silence_s"])
    stim_on_s = float(cfg["stim_on_s"])
    stim_off_s = float(cfg["stim_off_s"])
    post_s = float(cfg["post_silence_s"])
    tempo_subset = [float(x) for x in cfg["tempo_subset_hz"]]
    include_random = bool(cfg["include_random_controls"])
    max_stimuli = int(cfg["max_stimuli_in_session"])
    seed = int(cfg.get("random_seed", 0))

    rng = np.random.default_rng(seed)
    conditions = build_conditions(tempo_subset, include_random)
    rng.shuffle(conditions)
    conditions = conditions[:max_stimuli]

    out_csv = outdir / f"{session_id}.csv"
    t = 0.0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["start_s", "end_s", "block_type", "stimulus_name", "frequency_hz", "is_random"],
        )
        writer.writeheader()

        writer.writerow(
            {
                "start_s": t,
                "end_s": t + baseline_s,
                "block_type": "baseline_silence",
                "stimulus_name": "silence",
                "frequency_hz": "",
                "is_random": "",
            }
        )
        t += baseline_s

        for c in conditions:
            writer.writerow(
                {
                    "start_s": t,
                    "end_s": t + stim_on_s,
                    "block_type": "stim_on",
                    "stimulus_name": c.stimulus_name,
                    "frequency_hz": c.frequency_hz,
                    "is_random": c.is_random,
                }
            )
            t += stim_on_s

            writer.writerow(
                {
                    "start_s": t,
                    "end_s": t + stim_off_s,
                    "block_type": "matched_silence",
                    "stimulus_name": "silence",
                    "frequency_hz": c.frequency_hz,
                    "is_random": c.is_random,
                }
            )
            t += stim_off_s

        writer.writerow(
            {
                "start_s": t,
                "end_s": t + post_s,
                "block_type": "post_silence",
                "stimulus_name": "silence",
                "frequency_hz": "",
                "is_random": "",
            }
        )

    print(f"Wrote session plan: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
