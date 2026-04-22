"""T2.3 — Context length impact experiment (llama-bench backend).

Measure throughput across different context window sizes using fixed q4_0 quantization
and full GPU offload.  The KV-cache allocation grows with ctx_size, so this experiment
shows how context window size affects both throughput and VRAM usage.

Context lengths tested: 512 / 1024 / 2048 tokens

Usage:
    python -m benchmark.experiments.exp3_context_len [--dry_run] [--reps 3] [--n_gpu_layers -1]
"""

import argparse
import time

from inference import config
from inference.run_inference import run_bench
from benchmark.metrics import get_vram_usage_mb, save_result


def run_context_length_experiment(
    ctx_len: int,
    n_gpu_layers: int = -1,
    repetitions: int = config.BENCH_REPETITIONS,
    dry_run: bool = False,
) -> dict:
    """Run llama-bench at a specific context length and collect VRAM.

    n_prompt is set to ctx_len // 2 so the prompt fills half the context window,
    making differences in KV-cache management visible.

    Args:
        ctx_len: Context window size in tokens.
        n_gpu_layers: Number of layers to offload (-1 = all).
        repetitions: Number of bench repetitions.
        dry_run: If True, only print commands without executing.

    Returns:
        dict with pp_tps, tg_tps, vram_mb metrics.
    """
    # Scale n_prompt with ctx_len so we test realistic fill ratios
    n_prompt = max(64, ctx_len // 2)
    n_gen = config.BENCH_N_GEN  # keep generation length constant

    print(f"  Sampling baseline VRAM...")
    vram_before = get_vram_usage_mb()

    bench_result = run_bench(
        quant="q4_0",
        n_gpu_layers=n_gpu_layers,
        n_gen=n_gen,
        n_prompt=n_prompt,
        ctx_size=ctx_len,
        repetitions=repetitions,
        dry_run=dry_run,
    )

    vram_after = get_vram_usage_mb()
    vram_peak = max(
        max(vram_after) if vram_after else 0,
        max(vram_before) if vram_before else 0,
    )

    return {
        "ctx_len": ctx_len,
        "n_prompt_used": n_prompt,
        "n_gen": n_gen,
        "n_gpu_layers": n_gpu_layers,
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
    parser = argparse.ArgumentParser(description="Context length impact experiment")
    parser.add_argument("--reps", type=int, default=config.BENCH_REPETITIONS,
                        help="llama-bench repetitions per context length")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="GPU offload layers (-1 = all)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 3: Context Length Impact  (llama-bench)")
    print("=" * 60)

    results = {}
    for ctx_len in config.CONTEXT_LENGTHS:
        print(f"\nTesting ctx_len={ctx_len}...")
        results[str(ctx_len)] = run_context_length_experiment(
            ctx_len=ctx_len,
            n_gpu_layers=args.n_gpu_layers,
            repetitions=args.reps,
            dry_run=args.dry_run,
        )

    report = {
        "experiment": "context_length_comparison",
        "backend": "llama-bench",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "quantization": "q4_0",
            "n_gpu_layers": args.n_gpu_layers,
            "n_gen": config.BENCH_N_GEN,
            "repetitions": args.reps,
        },
        "results": results,
    }

    save_result("context_len", report)

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Ctx Len':<10} {'n_prompt':<10} {'PP t/s':<12} {'TG t/s':<12} {'TG stddev':<12} {'Peak VRAM':<12}")
    print("-" * 68)
    for label, rv in results.items():
        n_p = rv["n_prompt_used"]
        pp = rv["pp_tps"] or "N/A"
        tg = rv["tg_tps"] or "N/A"
        tg_std = rv["tg_tps_stddev"] or "N/A"
        vram = rv["vram_mb"]["peak"] or "N/A"
        print(f"{label:<10} {str(n_p):<10} {str(pp):<12} {str(tg):<12} {str(tg_std):<12} {str(vram):<12}")

    print("\n[DONE] Experiment 3 complete.")


if __name__ == "__main__":
    main()
