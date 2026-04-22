"""T1.3 — Metric collection utilities for benchmark experiments.

Provides:
  - VRAM monitoring via nvidia-smi parsing
  - Token throughput calculation
  - Latency statistics
  - JSON result persistence
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from inference import config


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------

def get_vram_usage_mb() -> list[int]:
    """Return current VRAM usage in MB per GPU via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return [int(x.strip()) for x in out.strip().split("\n") if x.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []


def get_vram_peak_mb() -> int:
    """Return peak VRAM usage across all GPUs."""
    values = get_vram_usage_mb()
    return max(values) if values else 0


def sample_vram(interval: float = 1.0) -> list[int]:
    """Sample VRAM usage once every `interval` seconds, return list of MB values."""
    return get_vram_usage_mb()


# ---------------------------------------------------------------------------
# Token throughput & latency
# ---------------------------------------------------------------------------

def calc_tokens_per_second(n_tokens: int, elapsed_s: float) -> float:
    """Calculate tokens/s throughput."""
    if elapsed_s <= 0:
        return 0.0
    return round(n_tokens / elapsed_s, 2)


def calc_latency_stats(latencies: list[float]) -> dict:
    """Compute latency statistics from a list of per-token latencies.

    Args:
        latencies: list of latency values in milliseconds.

    Returns:
        dict with mean, p50, p95, p99, min, max, stddev.
    """
    if not latencies:
        return {}

    import statistics

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def percentile(pct: float) -> float:
        idx = int(n * pct / 100.0)
        return sorted_lat[min(idx, n - 1)]

    return {
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(percentile(50), 2),
        "p95_ms": round(percentile(95), 2),
        "p99_ms": round(percentile(99), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "stddev_ms": round(statistics.stdev(latencies), 2) if n > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_result(experiment_name: str, data: dict) -> Path:
    """Save benchmark result as JSON to results/<experiment>_<timestamp>.json.

    Args:
        experiment_name: Short name for the experiment (e.g. 'quantization').
        data: Result dictionary to serialize.

    Returns:
        Path to the written file.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = config.RESULTS_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Result saved: {filepath}")
    return filepath


def load_results(experiment_prefix: str) -> list[dict]:
    """Load all result JSON files matching an experiment prefix.

    Args:
        experiment_prefix: Prefix of the filename (e.g. 'quantization').

    Returns:
        List of result dictionaries.
    """
    results = []
    for f in sorted(config.RESULTS_DIR.glob(f"{experiment_prefix}_*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results
