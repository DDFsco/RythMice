import argparse
import csv
import json
import math
import wave
from pathlib import Path

import numpy as np


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_wav_mono(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(signal, -1.0, 1.0)
    int16 = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())


def make_click_kernel(click_duration_ms: float, sample_rate: int, amplitude: float) -> np.ndarray:
    click_samples = max(1, int(round(sample_rate * click_duration_ms / 1000.0)))
    noise = np.random.uniform(-1.0, 1.0, click_samples)
    window = np.hanning(click_samples)
    kernel = amplitude * noise * window
    return kernel.astype(np.float32)


def periodic_click_train(total_samples: int, sample_rate: int, freq_hz: float, kernel: np.ndarray) -> np.ndarray:
    signal = np.zeros(total_samples, dtype=np.float32)
    step = max(1, int(round(sample_rate / freq_hz)))
    for idx in range(0, total_samples, step):
        end = min(total_samples, idx + len(kernel))
        signal[idx:end] += kernel[: end - idx]
    return signal


def poisson_click_train(
    total_samples: int,
    sample_rate: int,
    rate_hz: float,
    kernel: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    signal = np.zeros(total_samples, dtype=np.float32)
    t = 0.0
    total_s = total_samples / sample_rate
    while t < total_s:
        interval = rng.exponential(1.0 / rate_hz)
        t += interval
        idx = int(math.floor(t * sample_rate))
        if idx >= total_samples:
            break
        end = min(total_samples, idx + len(kernel))
        signal[idx:end] += kernel[: end - idx]
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate periodic and random click stimuli.")
    parser.add_argument("--config", type=Path, required=True, help="Path to stimulus JSON config.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for WAV files.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sr = int(cfg["sample_rate_hz"])
    duration_s = float(cfg["epoch_duration_s"])
    click_ms = float(cfg["click_duration_ms"])
    amp = float(cfg["click_amplitude"])
    freqs = [float(x) for x in cfg["tempo_frequencies_hz"]]
    include_silence = bool(cfg.get("include_silence_file", True))
    seed = int(cfg.get("random_seed", 0))
    rng = np.random.default_rng(seed)

    np.random.seed(seed)
    kernel = make_click_kernel(click_ms, sr, amp)
    total_samples = int(round(sr * duration_s))

    manifest_path = outdir / "stimulus_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=[
                "stimulus_name",
                "path",
                "frequency_hz",
                "is_random",
                "duration_s",
                "sample_rate_hz",
            ],
        )
        writer.writeheader()

        for hz in freqs:
            periodic = periodic_click_train(total_samples, sr, hz, kernel)
            periodic_name = f"periodic_{int(hz)}hz.wav"
            periodic_path = outdir / periodic_name
            write_wav_mono(periodic_path, periodic, sr)
            writer.writerow(
                {
                    "stimulus_name": periodic_name,
                    "path": str(periodic_path.resolve()),
                    "frequency_hz": hz,
                    "is_random": 0,
                    "duration_s": duration_s,
                    "sample_rate_hz": sr,
                }
            )

            random_sig = poisson_click_train(total_samples, sr, hz, kernel, rng)
            random_name = f"random_{int(hz)}hz.wav"
            random_path = outdir / random_name
            write_wav_mono(random_path, random_sig, sr)
            writer.writerow(
                {
                    "stimulus_name": random_name,
                    "path": str(random_path.resolve()),
                    "frequency_hz": hz,
                    "is_random": 1,
                    "duration_s": duration_s,
                    "sample_rate_hz": sr,
                }
            )

        if include_silence:
            silence_name = "silence.wav"
            silence_path = outdir / silence_name
            write_wav_mono(silence_path, np.zeros(total_samples, dtype=np.float32), sr)
            writer.writerow(
                {
                    "stimulus_name": silence_name,
                    "path": str(silence_path.resolve()),
                    "frequency_hz": "",
                    "is_random": "",
                    "duration_s": duration_s,
                    "sample_rate_hz": sr,
                }
            )

    print(f"Wrote stimuli to: {outdir.resolve()}")
    print(f"Wrote manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
