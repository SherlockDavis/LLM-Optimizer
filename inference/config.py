"""T1.1 — Central configuration for inference and benchmark experiments."""

import os
from pathlib import Path

# ---------- Project paths ----------
PROJECT_DIR = Path(__file__).resolve().parent.parent
LLAMA_CLI = PROJECT_DIR / "llama.cpp" / "build" / "bin" / "llama-cli.exe"
LLAMA_BENCH = PROJECT_DIR / "llama.cpp" / "build" / "bin" / "llama-bench.exe"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Ensure results dir exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- Quantization labels ----------
QUANTIZATIONS = {
    "q4_0": {"suffix": "Q4_0", "desc": "4-bit quantized"},
    "q8_0": {"suffix": "Q8_0", "desc": "8-bit quantized"},
    "f16":  {"suffix": "F16",  "desc": "16-bit float"},
}

# ---------- GPU layer defaults ----------
GPU_LAYER_OPTIONS = [0, 20, 40, -1]  # -1 = full offload

# ---------- Context lengths ----------
CONTEXT_LENGTHS = [512, 1024, 2048]

# ---------- Default inference params ----------
DEFAULT_PROMPT = "Explain the concept of attention mechanism in neural networks."
DEFAULT_N_PREDICT = 256
DEFAULT_TEMP = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPEAT_PENALTY = 1.1

# ---------- Experiment defaults ----------
QUANT_RUNS = 5          # legacy: number of runs per quantization for averaging
BENCH_REPETITIONS = 3   # llama-bench repetitions per configuration
BENCH_N_GEN = 128       # tokens to generate per bench iteration
BENCH_N_PROMPT = 128    # prompt tokens per bench iteration
