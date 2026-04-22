"""
plot_results.py — 读取 results/ 目录下最新的 JSON 结果文件，生成对比图表。

使用方式：
    python -m analysis.plot_results
    python -m analysis.plot_results --out_dir results/plots
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 无显示环境下也能保存图片
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── 项目根目录 ────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


# ── 工具函数 ──────────────────────────────────────────────────────

def find_latest(prefix: str) -> Path | None:
    """在 results/ 下找到最新的 <prefix>_*.json 文件。"""
    pattern = str(RESULTS_DIR / f"{prefix}_*.json")
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bar_with_err(ax, labels, values, errors, color, label_text, width=0.35, offset=0):
    """绘制带误差棒的柱状图，自动过滤 None 值。"""
    x = np.arange(len(labels))
    bar_vals = [v if v is not None else 0 for v in values]
    bar_errs = [e if e is not None else 0 for e in errors]
    bars = ax.bar(x + offset, bar_vals, width, label=label_text, color=color,
                  yerr=bar_errs, capsize=4, alpha=0.85)
    # 在 None 的柱子上标注 N/A
    for i, (v, bar) in enumerate(zip(values, bars)):
        if v is None:
            ax.text(bar.get_x() + bar.get_width() / 2, 2,
                    "N/A", ha="center", va="bottom", fontsize=8, color="red")
    return bars


# ── 子图1：量化对比 ───────────────────────────────────────────────

def plot_quantization(ax_pp, ax_tg, data: dict):
    """在两个子图上分别绘制量化对比的 PP/TG 性能。"""
    results = data["results"]
    quants = list(results.keys())

    pp_vals = [results[q].get("pp_tps") for q in quants]
    pp_errs = [results[q].get("pp_tps_stddev") or 0 for q in quants]
    tg_vals = [results[q].get("tg_tps") for q in quants]
    tg_errs = [results[q].get("tg_tps_stddev") or 0 for q in quants]

    x = np.arange(len(quants))
    colors_pp = ["#4C9BE8", "#7DC9A0", "#F5A623"]
    colors_tg = ["#3A7BD5", "#57B894", "#E8842A"]

    # PP 图
    for i, (q, v, e) in enumerate(zip(quants, pp_vals, pp_errs)):
        h = v if v is not None else 0
        ax_pp.bar(x[i], h, color=colors_pp[i], yerr=e if v is not None else 0,
                  capsize=5, alpha=0.85, label=q)
        if v is None:
            ax_pp.text(x[i], 5, "N/A", ha="center", va="bottom", fontsize=9, color="red")
    ax_pp.set_title("Prefill Speed (PP t/s) — Quantization", fontsize=11)
    ax_pp.set_ylabel("tokens / second")
    ax_pp.set_xticks(x)
    ax_pp.set_xticklabels([q.upper() for q in quants])
    ax_pp.legend()
    ax_pp.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # TG 图
    for i, (q, v, e) in enumerate(zip(quants, tg_vals, tg_errs)):
        h = v if v is not None else 0
        ax_tg.bar(x[i], h, color=colors_tg[i], yerr=e if v is not None else 0,
                  capsize=5, alpha=0.85, label=q)
        if v is None:
            ax_tg.text(x[i], 0.5, "N/A", ha="center", va="bottom", fontsize=9, color="red")
    ax_tg.set_title("Decode Speed (TG t/s) — Quantization", fontsize=11)
    ax_tg.set_ylabel("tokens / second")
    ax_tg.set_xticks(x)
    ax_tg.set_xticklabels([q.upper() for q in quants])
    ax_tg.legend()


# ── 子图2：GPU 层数对比 ───────────────────────────────────────────

def plot_gpu_layers(ax_pp, ax_tg, data: dict):
    """在两个子图上分别绘制 GPU 层数对比的 PP/TG 性能。"""
    results = data["results"]
    order = ["0", "20", "40", "full"]
    labels = [k for k in order if k in results]

    pp_vals = [results[k].get("pp_tps") for k in labels]
    pp_errs = [results[k].get("pp_tps_stddev") or 0 for k in labels]
    tg_vals = [results[k].get("tg_tps") for k in labels]
    tg_errs = [results[k].get("tg_tps_stddev") or 0 for k in labels]

    x = np.arange(len(labels))
    color_pp = "#4C9BE8"
    color_tg = "#3A7BD5"

    ax_pp.bar(x, pp_vals, color=color_pp, yerr=pp_errs, capsize=5, alpha=0.85)
    ax_pp.plot(x, pp_vals, "o--", color="#1A5BA0", linewidth=1.5)
    ax_pp.set_title("Prefill Speed (PP t/s) — GPU Layers", fontsize=11)
    ax_pp.set_ylabel("tokens / second")
    ax_pp.set_xticks(x)
    ax_pp.set_xticklabels([f"ngl={l}" for l in labels])
    ax_pp.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ax_tg.bar(x, tg_vals, color=color_tg, yerr=tg_errs, capsize=5, alpha=0.85)
    ax_tg.plot(x, tg_vals, "o--", color="#1A3A6A", linewidth=1.5)
    ax_tg.set_title("Decode Speed (TG t/s) — GPU Layers", fontsize=11)
    ax_tg.set_ylabel("tokens / second")
    ax_tg.set_xticks(x)
    ax_tg.set_xticklabels([f"ngl={l}" for l in labels])


# ── 子图3：上下文长度对比 ─────────────────────────────────────────

def plot_context_len(ax_pp, ax_tg, data: dict):
    """在两个子图上分别绘制上下文长度对比的 PP/TG 性能。"""
    results = data["results"]
    ctx_keys = sorted(results.keys(), key=int)

    pp_vals = [results[k].get("pp_tps") for k in ctx_keys]
    pp_errs = [results[k].get("pp_tps_stddev") or 0 for k in ctx_keys]
    tg_vals = [results[k].get("tg_tps") for k in ctx_keys]
    tg_errs = [results[k].get("tg_tps_stddev") or 0 for k in ctx_keys]

    x = np.arange(len(ctx_keys))
    color_pp = "#7DC9A0"
    color_tg = "#57B894"

    ax_pp.bar(x, pp_vals, color=color_pp, yerr=pp_errs, capsize=5, alpha=0.85)
    ax_pp.plot(x, pp_vals, "o--", color="#2E7D5A", linewidth=1.5)
    ax_pp.set_title("Prefill Speed (PP t/s) — Context Length", fontsize=11)
    ax_pp.set_ylabel("tokens / second")
    ax_pp.set_xticks(x)
    ax_pp.set_xticklabels([f"ctx={k}" for k in ctx_keys])
    ax_pp.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ax_tg.bar(x, tg_vals, color=color_tg, yerr=tg_errs, capsize=5, alpha=0.85)
    ax_tg.plot(x, tg_vals, "o--", color="#1E5E40", linewidth=1.5)
    ax_tg.set_title("Decode Speed (TG t/s) — Context Length", fontsize=11)
    ax_tg.set_ylabel("tokens / second")
    ax_tg.set_xticks(x)
    ax_tg.set_xticklabels([f"ctx={k}" for k in ctx_keys])


# ── 主入口 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成 LLM 推理 benchmark 对比图表")
    parser.add_argument("--out_dir", type=str, default=str(RESULTS_DIR / "plots"),
                        help="图表输出目录（默认 results/plots/）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载最新结果
    quant_file = find_latest("quantization")
    gpu_file = find_latest("gpu_layers")
    ctx_file = find_latest("context_len")

    missing = []
    if not quant_file:
        missing.append("quantization_*.json")
    if not gpu_file:
        missing.append("gpu_layers_*.json")
    if not ctx_file:
        missing.append("context_len_*.json")
    if missing:
        print(f"[ERROR] 找不到结果文件：{missing}")
        return

    quant_data = load_json(quant_file)
    gpu_data = load_json(gpu_file)
    ctx_data = load_json(ctx_file)

    print(f"[INFO] 量化对比  → {quant_file.name}")
    print(f"[INFO] GPU 层数  → {gpu_file.name}")
    print(f"[INFO] 上下文长度 → {ctx_file.name}")

    # ── 图1：量化对比（2行×1列）────────────────────────────────────
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig1.suptitle("Experiment 1: Quantization Comparison\n(Qwen2.5-7B, RTX 3060 Laptop)",
                  fontsize=13, fontweight="bold")
    plot_quantization(ax1, ax2, quant_data)
    fig1.tight_layout()
    p1 = out_dir / "exp1_quantization.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    print(f"[SAVED] {p1}")

    # ── 图2：GPU 层数对比 ────────────────────────────────────────
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))
    fig2.suptitle("Experiment 2: GPU Layers Comparison\n(Q4_0, Qwen2.5-7B, RTX 3060 Laptop)",
                  fontsize=13, fontweight="bold")
    plot_gpu_layers(ax3, ax4, gpu_data)
    fig2.tight_layout()
    p2 = out_dir / "exp2_gpu_layers.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"[SAVED] {p2}")

    # ── 图3：上下文长度对比 ──────────────────────────────────────
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(10, 5))
    fig3.suptitle("Experiment 3: Context Length Comparison\n(Q4_0, Full GPU, Qwen2.5-7B)",
                  fontsize=13, fontweight="bold")
    plot_context_len(ax5, ax6, ctx_data)
    fig3.tight_layout()
    p3 = out_dir / "exp3_context_len.png"
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)
    print(f"[SAVED] {p3}")

    # ── 图4：总览大图（3组×2指标）───────────────────────────────
    fig4, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig4.suptitle("LLM Inference Optimization — Full Benchmark Overview\n"
                  "Qwen2.5-7B-Instruct · llama.cpp · RTX 3060 Laptop (6 GB)",
                  fontsize=14, fontweight="bold", y=0.98)
    plot_quantization(axes[0][0], axes[0][1], quant_data)
    plot_gpu_layers(axes[1][0], axes[1][1], gpu_data)
    plot_context_len(axes[2][0], axes[2][1], ctx_data)
    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    p4 = out_dir / "overview.png"
    fig4.savefig(p4, dpi=150)
    plt.close(fig4)
    print(f"[SAVED] {p4}")

    print("\n全部图表已生成！")


if __name__ == "__main__":
    main()
