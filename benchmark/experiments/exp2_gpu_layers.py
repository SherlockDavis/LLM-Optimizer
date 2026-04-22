"""T2.2 — GPU layer offload experiment (llama-bench backend).

Test throughput across different GPU offload levels with fixed q4_0 quantization.
llama-bench loads the model once and runs N repetitions internally.

Layers tested: 0 / 20 / 40 / -1 (full offload)

Usage:
    python -m benchmark.experiments.exp2_gpu_layers [--dry_run] [--reps 3] [--quant q4_0]
"""

import argparse
import time

from inference import config
from inference.run_inference import run_bench
from benchmark.metrics import get_vram_usage_mb, save_result


def run_gpu_layer_experiment(
    n_layers: int,
    quant: str = "q4_0",
    repetitions: int = config.BENCH_REPETITIONS,
    dry_run: bool = False,
) -> dict:
    """Run llama-bench at a specific GPU layer count and collect VRAM.

    Args:
        n_layers: Number of layers to offload to GPU (-1 = all).
        quant: Quantization variant.
        repetitions: Number of bench repetitions.
        dry_run: If True, only print commands without executing.

    Returns:
        dict with pp_tps, tg_tps, vram_mb metrics.
    """
    layer_label = "full" if n_layers == -1 else str(n_layers)
    print(f"  Sampling baseline VRAM...")
    vram_before = get_vram_usage_mb()

    bench_result = run_bench(
        quant=quant,
        n_gpu_layers=n_layers,
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
        "n_gpu_layers": n_layers,
        "label": layer_label,
        "quant": quant,
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
    parser = argparse.ArgumentParser(description="GPU layer offload experiment")
    parser.add_argument("--reps", type=int, default=config.BENCH_REPETITIONS,
                        help="llama-bench repetitions per configuration")
    parser.add_argument("--quant", type=str, default="q4_0",
                        choices=list(config.QUANTIZATIONS.keys()),
                        help="Quantization variant")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 2: GPU Layer Offload Comparison  (llama-bench)")
    print("=" * 60)

    results = {}
    for n_layers in config.GPU_LAYER_OPTIONS:
        label = "full" if n_layers == -1 else str(n_layers)
        print(f"\nTesting n_gpu_layers={label}...")
        results[label] = run_gpu_layer_experiment(
            n_layers=n_layers,
            quant=args.quant,
            repetitions=args.reps,
            dry_run=args.dry_run,
        )

    report = {
        "experiment": "gpu_layers_comparison",
        "backend": "llama-bench",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "quantization": args.quant,
            "n_gen": config.BENCH_N_GEN,
            "n_prompt": config.BENCH_N_PROMPT,
            "ctx_size": 2048,
            "repetitions": args.reps,
        },
        "results": results,
    }

    save_result("gpu_layers", report)

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'GPU Layers':<12} {'PP t/s':<12} {'TG t/s':<12} {'TG stddev':<12} {'Peak VRAM':<12}")
    print("-" * 60)
    for label, rv in results.items():
        pp = rv["pp_tps"] or "N/A"
        tg = rv["tg_tps"] or "N/A"
        tg_std = rv["tg_tps_stddev"] or "N/A"
        vram = rv["vram_mb"]["peak"] or "N/A"
        print(f"{label:<12} {str(pp):<12} {str(tg):<12} {str(tg_std):<12} {str(vram):<12}")

    print("\n[DONE] Experiment 2 complete.")


if __name__ == "__main__":
    main()
