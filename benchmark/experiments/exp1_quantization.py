"""T2.1 — Quantization comparison experiment (llama-bench backend).

Compare q4_0 / q8_0 / f16 with a fixed configuration.
llama-bench loads the model once and runs N repetitions internally,
measuring prompt-processing (pp) and token-generation (tg) throughput.

Usage:
    python -m benchmark.experiments.exp1_quantization [--dry_run] [--reps 3]
"""

import argparse
import time

from inference import config
from inference.run_inference import run_bench
from benchmark.metrics import get_vram_usage_mb, get_vram_peak_mb, save_result


def run_quantization_experiment(quant: str, repetitions: int, dry_run: bool = False) -> dict:
    """Run llama-bench for a given quantization and collect VRAM.

    Args:
        quant: Quantization key (e.g. 'q4_0', 'q8_0', 'f16').
        repetitions: Number of bench repetitions.
        dry_run: If True, only print commands without executing.

    Returns:
        dict with pp_tps, tg_tps, vram_mb metrics.
    """
    print(f"  Sampling baseline VRAM...")
    vram_before = get_vram_usage_mb()

    bench_result = run_bench(
        quant=quant,
        n_gpu_layers=-1,   # full GPU offload for fair comparison
        n_gen=config.BENCH_N_GEN,
        n_prompt=config.BENCH_N_PROMPT,
        ctx_size=2048,
        repetitions=repetitions,
        dry_run=dry_run,
    )

    vram_after = get_vram_usage_mb()
    vram_peak = max(
        max(vram_after) if vram_after else 0,
        max(vram_before) if vram_before else 0,
    )

    return {
        "quant": quant,
        "desc": config.QUANTIZATIONS[quant]["desc"],
        "repetitions": repetitions,
        "pp_tps": bench_result.get("pp_tps"),
        "pp_tps_stddev": bench_result.get("pp_tps_stddev"),
        "tg_tps": bench_result.get("tg_tps"),
        "tg_tps_stddev": bench_result.get("tg_tps_stddev"),
        "wall_time_s": bench_result.get("wall_time_s"),
        "vram_mb": {
            "peak": vram_peak,
            "before": vram_before,
            "after": vram_after,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Quantization comparison experiment")
    parser.add_argument("--reps", type=int, default=config.BENCH_REPETITIONS,
                        help="llama-bench repetitions per quantization")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 1: Quantization Comparison  (llama-bench)")
    print("=" * 60)

    results = {}
    for quant_key in config.QUANTIZATIONS:
        print(f"\nTesting {quant_key} ({config.QUANTIZATIONS[quant_key]['desc']})...")
        results[quant_key] = run_quantization_experiment(quant_key, args.reps, args.dry_run)

    report = {
        "experiment": "quantization_comparison",
        "backend": "llama-bench",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_gen": config.BENCH_N_GEN,
            "n_prompt": config.BENCH_N_PROMPT,
            "n_gpu_layers": -1,
            "repetitions": args.reps,
        },
        "results": results,
    }

    save_result("quantization", report)

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Quant':<8} {'PP t/s':<12} {'TG t/s':<12} {'TG stddev':<12} {'Peak VRAM':<12}")
    print("-" * 56)
    for qk, rv in results.items():
        pp = rv["pp_tps"] or "N/A"
        tg = rv["tg_tps"] or "N/A"
        tg_std = rv["tg_tps_stddev"] or "N/A"
        vram = rv["vram_mb"]["peak"] or "N/A"
        print(f"{qk:<8} {str(pp):<12} {str(tg):<12} {str(tg_std):<12} {str(vram):<12}")

    print("\n[DONE] Experiment 1 complete.")


if __name__ == "__main__":
    main()
