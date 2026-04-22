"""T1.2 — Run inference via llama-cli and return structured results.

Usage:
    python inference/run_inference.py --model q4_0 --n_gpu_layers 40 [--dry_run]
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from . import config


def find_model(quant: str) -> Path:
    """Find the first matching GGUF model file by quantization key."""
    suffix = config.QUANTIZATIONS[quant]["suffix"]
    for f in config.MODELS_DIR.glob("*.gguf"):
        if suffix in f.name.upper():
            return f
    raise FileNotFoundError(f"No model found for quantization '{quant}' in {config.MODELS_DIR}")


def parse_metrics(stdout: str, stderr: str, elapsed: float) -> dict:
    """Extract tokens/s and other metrics from llama-cli output.

    Handles llama.cpp output formats:
      - '[ Prompt: X t/s | Generation: Y t/s ]'  (new interactive/chat format)
      - 'xxx.xx tokens/sec' / 'xx.xx tok/s'
      - 'prompt_eval_time=xxx.ms', 'eval_time=xxx.ms'
    """
    combined = stdout + stderr

    # --- tokens_per_second (generation speed) ---
    tokens_per_second = None
    prompt_tps = None

    # Priority 1: new llama.cpp chat format '[ Prompt: X t/s | Generation: Y t/s ]'
    m = re.search(
        r"Prompt:\s*(\d+\.?\d*)\s*t/s\s*\|\s*Generation:\s*(\d+\.?\d*)\s*t/s",
        combined, re.IGNORECASE
    )
    if m:
        prompt_tps = float(m.group(1))
        tokens_per_second = float(m.group(2))

    # Priority 2: classic formats
    if tokens_per_second is None:
        for pat in [
            r"(\d+\.?\d*)\s*tokens?/s(?:ec)?",
            r"(\d+\.?\d*)\s*tok/s",
        ]:
            m = re.search(pat, combined, re.IGNORECASE)
            if m:
                tokens_per_second = float(m.group(1))
                break

    # --- prompt_tokens & decode_tokens ---
    prompt_tokens = None
    decode_tokens = None

    # Fallback: try extracting from 'N tokens, xxx.xxx ms' style
    fm = re.findall(r"(\d+)\s*tokens?,\s+([\d.]+)\s+ms", combined)
    if len(fm) >= 2:
        prompt_tokens = int(fm[0][0])
        decode_tokens = int(fm[1][0])
    elif len(fm) == 1:
        decode_tokens = int(fm[0][0])

    return {
        "tokens_per_second": tokens_per_second,
        "prompt_tps": prompt_tps,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "wall_time_s": round(elapsed, 3),
    }


def run_inference(
    quant: str = "q4_0",
    n_gpu_layers: int = 40,
    n_predict: int = config.DEFAULT_N_PREDICT,
    prompt: str = config.DEFAULT_PROMPT,
    temp: float = config.DEFAULT_TEMP,
    ctx_size: int = 2048,
    top_p: float = config.DEFAULT_TOP_P,
    repeat_penalty: float = config.DEFAULT_REPEAT_PENALTY,
    dry_run: bool = False,
) -> dict:
    """Run a single inference pass and return structured results.

    Args:
        quant: Quantization key (e.g. 'q4_0', 'q8_0', 'f16').
        n_gpu_layers: Number of layers to offload to GPU. -1 = all.
        n_predict: Number of tokens to generate.
        prompt: Input prompt text.
        temp: Sampling temperature.
        ctx_size: Context window size in tokens.
        top_p: Top-p sampling parameter.
        repeat_penalty: Repetition penalty.
        dry_run: If True, only print the command without executing.

    Returns:
        dict with keys: model, quant, n_gpu_layers, prompt, output,
                        tokens_per_second, prompt_tokens, decode_tokens, wall_time_s.
    """
    # In dry_run mode, skip model file lookup and use a placeholder path
    if dry_run:
        suffix = config.QUANTIZATIONS[quant]["suffix"]
        model_path = config.MODELS_DIR / f"placeholder_{suffix}.gguf"
    else:
        model_path = find_model(quant)

    cmd = [
        str(config.LLAMA_CLI),
        "--model", str(model_path),
        "--prompt", prompt,
        "--n_predict", str(n_predict),
        "--temp", str(temp),
        "--ctx-size", str(ctx_size),
        "--n-gpu-layers", str(n_gpu_layers),
        "--top-p", str(top_p),
        "--repeat-penalty", str(repeat_penalty),
        "--no-display-prompt",
        "--single-turn",  # run one turn then exit (batch mode)
    ]

    result = {
        "model": model_path.name,
        "quant": quant,
        "n_gpu_layers": n_gpu_layers,
        "prompt": prompt,
        "command": " ".join(cmd),
    }

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        result.update({
            "output": "(dry run — no output)",
            "tokens_per_second": None,
            "prompt_tps": None,
            "prompt_tokens": None,
            "decode_tokens": None,
            "wall_time_s": None,
        })
        return result

    if not config.LLAMA_CLI.exists():
        raise FileNotFoundError(f"llama-cli not found at {config.LLAMA_CLI}")

    print(f"  Running inference: quant={quant}, gpu_layers={n_gpu_layers}")
    start = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        stdin=subprocess.DEVNULL,  # prevent blocking on stdin
    )
    elapsed = time.time() - start

    metrics = parse_metrics(proc.stdout, proc.stderr, elapsed)
    result["output"] = proc.stdout.strip()
    result["stderr_excerpt"] = proc.stderr.strip()[:200]
    result.update(metrics)

    return result


def run_bench(
    quant: str = "q4_0",
    n_gpu_layers: int = -1,
    n_gen: int = config.BENCH_N_GEN,
    n_prompt: int = config.BENCH_N_PROMPT,
    ctx_size: int = 2048,
    repetitions: int = config.BENCH_REPETITIONS,
    dry_run: bool = False,
) -> dict:
    """Run llama-bench for throughput benchmarking.

    Loads model once and runs N repetitions, measuring prompt-processing (pp)
    and token-generation (tg) throughput in tokens/second.

    Args:
        quant: Quantization key (e.g. 'q4_0', 'q8_0', 'f16').
        n_gpu_layers: Number of layers to offload to GPU (-1 = all).
        n_gen: Number of tokens to generate per iteration.
        n_prompt: Number of prompt tokens per iteration.
        ctx_size: Context window size in tokens.
        repetitions: Number of timing repetitions.
        dry_run: If True, only print the command without executing.

    Returns:
        dict with pp_tps (prompt t/s), tg_tps (generation t/s), and stddevs.
    """
    if dry_run:
        suffix = config.QUANTIZATIONS[quant]["suffix"]
        model_path = config.MODELS_DIR / f"placeholder_{suffix}.gguf"
    else:
        model_path = find_model(quant)

    # llama-bench uses range-format parsing; negative values (e.g. -1 for "all layers")
    # are not supported — use 999 as an equivalent "offload everything" sentinel.
    ngl_bench = 999 if n_gpu_layers < 0 else n_gpu_layers

    cmd = [
        str(config.LLAMA_BENCH),
        "-m", str(model_path),
        "-ngl", str(ngl_bench),
        "-n", str(n_gen),
        "-p", str(n_prompt),
        "-r", str(repetitions),
        "-o", "jsonl",
    ]

    result = {
        "model": model_path.name,
        "quant": quant,
        "n_gpu_layers": n_gpu_layers,
        "n_gen": n_gen,
        "n_prompt": n_prompt,
        "ctx_size": ctx_size,
        "repetitions": repetitions,
        "command": " ".join(cmd),
    }

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        result.update({
            "pp_tps": None, "pp_tps_stddev": None,
            "tg_tps": None, "tg_tps_stddev": None,
            "wall_time_s": None,
        })
        return result

    if not config.LLAMA_BENCH.exists():
        raise FileNotFoundError(f"llama-bench not found at {config.LLAMA_BENCH}")

    print(f"  Running bench: quant={quant}, ngl={n_gpu_layers}, "
          f"n_gen={n_gen}, n_prompt={n_prompt}, reps={repetitions}")
    start = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        stdin=subprocess.DEVNULL,
    )
    elapsed = time.time() - start

    # Parse JSONL output — each line is one test type (pp or tg)
    pp_data: dict = {}
    tg_data: dict = {}
    for line in proc.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            n_p = data.get("n_prompt", 0)
            n_g = data.get("n_gen", 0)
            if n_p > 0 and n_g == 0:
                pp_data = data
            elif n_g > 0 and n_p == 0:
                tg_data = data
        except json.JSONDecodeError:
            pass

    result.update({
        "wall_time_s": round(elapsed, 3),
        "pp_tps": round(pp_data["avg_ts"], 2) if "avg_ts" in pp_data else None,
        "pp_tps_stddev": round(pp_data.get("stddev_ts", 0), 2) if pp_data else None,
        "tg_tps": round(tg_data["avg_ts"], 2) if "avg_ts" in tg_data else None,
        "tg_tps_stddev": round(tg_data.get("stddev_ts", 0), 2) if tg_data else None,
        "stderr_excerpt": proc.stderr.strip()[:200],
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="Run inference via llama-cli")
    parser.add_argument("--model", type=str, default="q4_0", help="Quantization key")
    parser.add_argument("--n_gpu_layers", type=int, default=40, help="GPU offload layers")
    parser.add_argument("--n_predict", type=int, default=config.DEFAULT_N_PREDICT)
    parser.add_argument("--prompt", type=str, default=config.DEFAULT_PROMPT)
    parser.add_argument("--temp", type=float, default=config.DEFAULT_TEMP)
    parser.add_argument("--ctx_size", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=config.DEFAULT_TOP_P)
    parser.add_argument("--repeat_penalty", type=float, default=config.DEFAULT_REPEAT_PENALTY)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    result = run_inference(
        quant=args.model,
        n_gpu_layers=args.n_gpu_layers,
        n_predict=args.n_predict,
        prompt=args.prompt,
        temp=args.temp,
        ctx_size=args.ctx_size,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        dry_run=args.dry_run,
    )
    print("\n--- Results ---")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
