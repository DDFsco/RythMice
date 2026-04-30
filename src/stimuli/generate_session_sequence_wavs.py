"""Concatenate silence and periodic click trains into long session WAVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

import numpy as np

try:
    from .generate_click_stimuli import (
        load_config,
        make_click_kernel,
        periodic_click_train,
        write_wav_mono,
    )
except ImportError:
    from generate_click_stimuli import (
        load_config,
        make_click_kernel,
        periodic_click_train,
        write_wav_mono,
    )


def silence_segment(duration_s: float, sample_rate: int) -> np.ndarray:
    n = int(round(duration_s * sample_rate))
    return np.zeros(n, dtype=np.float32)


def sound_segment(
    freq_hz: float, duration_s: float, sample_rate: int, kernel: np.ndarray
) -> np.ndarray:
    n = int(round(duration_s * sample_rate))
    return periodic_click_train(n, sample_rate, freq_hz, kernel)


def concat_session(
    codes: Sequence[int],
    silence_s: float,
    sound_s: float,
    sample_rate: int,
    click_ms: float,
    amplitude: float,
    rng_seed: int,
) -> np.ndarray:
    np.random.seed(rng_seed)
    kernel = make_click_kernel(click_ms, sample_rate, amplitude)

    chunks: List[np.ndarray] = []
    for code in codes:
        if code == 0:
            chunks.append(silence_segment(silence_s, sample_rate))
        elif code in (2, 7):
            chunks.append(sound_segment(float(code), sound_s, sample_rate, kernel))
        else:
            raise ValueError(f"Unsupported segment code {code}; use 0, 2, or 7.")
    return np.concatenate(chunks, dtype=np.float32)


DEFAULT_PLANS: List[List[int]] = [
    [0, 2, 0, 7, 0, 2, 0, 7, 0, 2],
    [0, 7, 0, 2, 0, 7, 0, 2, 0, 7],
    [0, 2, 0, 7, 0, 2, 0, 7, 0, 2],
    [0, 7, 0, 2, 0, 7, 0, 2, 0, 7],
    [0, 2, 0, 7, 0, 2, 0, 7, 0, 2],
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build long WAVs from 0=silence, 2=2Hz periodic, 7=7Hz periodic blocks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "stimulus_config.json",
        help="JSON with sample_rate_hz, click_duration_ms, click_amplitude, random_seed.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs" / "stimuli_sessions",
        help="Directory for session WAV files and manifest.",
    )
    parser.add_argument("--silence-s", type=float, default=240.0, help="Duration for code 0 (s).")
    parser.add_argument("--sound-s", type=float, default=120.0, help="Duration for codes 2 and 7 (s).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sr = int(cfg["sample_rate_hz"])
    click_ms = float(cfg["click_duration_ms"])
    amp = float(cfg["click_amplitude"])
    base_seed = int(cfg.get("random_seed", 42))

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Same macro-sequence repeats use identical audio (A vs B template).
    seed_for_plan = [
        base_seed + 1000,  # A
        base_seed + 2000,  # B
        base_seed + 1000,
        base_seed + 2000,
        base_seed + 1000,
    ]

    manifest_path = args.outdir / "session_sequence_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=[
                "session_index",
                "filename",
                "path",
                "sequence_codes",
                "silence_segment_s",
                "sound_segment_s",
                "total_duration_s",
                "sample_rate_hz",
                "rng_seed",
            ],
        )
        writer.writeheader()

        for i, codes in enumerate(DEFAULT_PLANS, start=1):
            sig = concat_session(codes, args.silence_s, args.sound_s, sr, click_ms, amp, seed_for_plan[i - 1])
            name = f"session_{i:02d}_sequence.wav"
            out_path = args.outdir / name
            write_wav_mono(out_path, sig, sr)
            total_s = len(sig) / sr
            writer.writerow(
                {
                    "session_index": i,
                    "filename": name,
                    "path": str(out_path.resolve()),
                    "sequence_codes": " ".join(str(c) for c in codes),
                    "silence_segment_s": args.silence_s,
                    "sound_segment_s": args.sound_s,
                    "total_duration_s": round(total_s, 3),
                    "sample_rate_hz": sr,
                    "rng_seed": seed_for_plan[i - 1],
                }
            )
            print(f"Wrote {out_path} ({total_s/60:.2f} min)")

    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
