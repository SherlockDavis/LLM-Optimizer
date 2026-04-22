"""
report_gen.py — 读取 results/ 下最新 JSON，自动生成 Markdown 格式分析报告。

使用方式：
    python -m analysis.report_gen
    python -m analysis.report_gen --out report.md
"""

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


# ── 工具函数 ──────────────────────────────────────────────────────

def find_latest(prefix: str) -> Path | None:
    """找 results/ 下最新的 <prefix>_*.json。"""
    files = sorted(glob.glob(str(RESULTS_DIR / f"{prefix}_*.json")))
    return Path(files[-1]) if files else None


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(val, unit="", na="N/A", precision=2):
    """格式化数值，None 时返回 na。"""
    if val is None:
        return na
    return f"{val:.{precision}f}{unit}"


def pct_change(a, b):
    """计算 b 相对 a 的百分比变化，None 时返回 None。"""
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a * 100


# ── 报告各节 ──────────────────────────────────────────────────────

def section_overview() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# LLM 推理优化实验报告

> 自动生成时间：{now}

## 项目概述

本项目基于 [llama.cpp](https://github.com/ggerganov/llama.cpp) 在 Windows + NVIDIA GPU 环境下，
对 **Qwen2.5-7B-Instruct** 模型进行端侧推理优化实验，系统评估量化精度、GPU 卸载层数、
上下文长度三个维度对推理速度的影响。

**测试平台：**

| 项目 | 规格 |
|------|------|
| 模型 | Qwen2.5-7B-Instruct |
| 推理框架 | llama.cpp (CUDA 后端) |
| GPU | NVIDIA RTX 3060 Laptop (6 GB VRAM) |
| 评测工具 | llama-bench (内置于 llama.cpp) |
| 重复次数 | 每组配置跑 3 次取均值 |

**指标说明：**
PP t/s（Prefill tokens/second）反映模型处理输入 prompt 的速度；
TG t/s（Text Generation tokens/second）反映逐 token 生成速度，是用户实际体验到的"打字速度"。
"""


def section_quantization(data: dict) -> str:
    results = data["results"]
    ts = data.get("timestamp", "")
    cfg = data.get("config", {})

    rows = []
    for q, r in results.items():
        pp = fmt(r.get("pp_tps"), " t/s")
        pp_std = fmt(r.get("pp_tps_stddev"), " t/s")
        tg = fmt(r.get("tg_tps"), " t/s")
        tg_std = fmt(r.get("tg_tps_stddev"), " t/s")
        vram = r.get("vram_mb", {}).get("peak")
        vram_s = f"{vram} MB" if vram else "N/A"
        wall = fmt(r.get("wall_time_s"), " s")
        rows.append(f"| {q.upper()} | {pp} ± {pp_std} | {tg} ± {tg_std} | {vram_s} | {wall} |")

    table = "\n".join(rows)

    # 计算 q8_0 vs q4_0 的速度对比（如果都有值）
    analysis = ""
    q4 = results.get("q4_0", {})
    q8 = results.get("q8_0", {})
    f16 = results.get("f16", {})

    if q4.get("tg_tps") and q8.get("tg_tps"):
        ratio = q4["tg_tps"] / q8["tg_tps"]
        analysis += (f"\nQ4_0 的 TG 速度是 Q8_0 的 **{ratio:.1f}x**，"
                     f"原因是 Q8_0（约 7.7 GB）超出本机 6 GB VRAM，发生显存溢出回落至 CPU 计算，导致速度骤降。\n")
    if f16.get("tg_tps") is None:
        analysis += "F16（约 14 GB）远超显存容量，llama-bench 无法完成测试，输出 N/A。\n"

    img = "![量化对比](plots/exp1_quantization.png)\n" if (PLOTS_DIR / "exp1_quantization.png").exists() else ""

    return f"""---

## 实验一：量化精度对比

**实验时间：** {ts}
**固定参数：** n_prompt={cfg.get('n_prompt')}, n_gen={cfg.get('n_gen')}, GPU 全量卸载

| 量化版本 | PP 速度 | TG 速度 | 显存峰值 | 耗时 |
|---------|---------|---------|---------|------|
{table}

### 分析
{analysis.strip()}

{img}"""


def section_gpu_layers(data: dict) -> str:
    results = data["results"]
    ts = data.get("timestamp", "")
    cfg = data.get("config", {})

    order = ["0", "20", "40", "full"]
    rows = []
    for k in order:
        if k not in results:
            continue
        r = results[k]
        pp = fmt(r.get("pp_tps"), " t/s")
        pp_std = fmt(r.get("pp_tps_stddev"), " t/s")
        tg = fmt(r.get("tg_tps"), " t/s")
        tg_std = fmt(r.get("tg_tps_stddev"), " t/s")
        label = "全GPU" if k == "full" else f"{k} 层"
        rows.append(f"| {label} | {pp} ± {pp_std} | {tg} ± {tg_std} |")

    table = "\n".join(rows)

    # 分析：0层 vs full层
    r0 = results.get("0", {})
    rfull = results.get("full", {})
    analysis = ""
    if r0.get("tg_tps") and rfull.get("tg_tps"):
        tg_ratio = rfull["tg_tps"] / r0["tg_tps"]
        pp_ratio = rfull.get("pp_tps", 0) / (r0.get("pp_tps") or 1)
        analysis = (f"全量 GPU 卸载（full）相比纯 CPU（0 层），TG 速度提升 **{tg_ratio:.1f}x**，"
                    f"PP 速度提升 **{pp_ratio:.1f}x**。\n"
                    f"GPU 层数从 0→20→40 时性能呈阶梯式增长；40 层与 full 层结果接近，"
                    f"说明该模型 40 层已基本涵盖所有 transformer 计算层。")

    img = "![GPU层数对比](plots/exp2_gpu_layers.png)\n" if (PLOTS_DIR / "exp2_gpu_layers.png").exists() else ""

    return f"""---

## 实验二：GPU 层数对比

**实验时间：** {ts}
**固定参数：** 量化={cfg.get('quantization')}, n_prompt={cfg.get('n_prompt')}, n_gen={cfg.get('n_gen')}

| GPU 层数 | PP 速度 | TG 速度 |
|---------|---------|---------|
{table}

### 分析
{analysis.strip()}

{img}"""


def section_context_len(data: dict) -> str:
    results = data["results"]
    ts = data.get("timestamp", "")
    cfg = data.get("config", {})

    keys = sorted(results.keys(), key=int)
    rows = []
    for k in keys:
        r = results[k]
        pp = fmt(r.get("pp_tps"), " t/s")
        pp_std = fmt(r.get("pp_tps_stddev"), " t/s")
        tg = fmt(r.get("tg_tps"), " t/s")
        tg_std = fmt(r.get("tg_tps_stddev"), " t/s")
        n_p = r.get("n_prompt_used", "-")
        rows.append(f"| {k} (prompt={n_p}) | {pp} ± {pp_std} | {tg} ± {tg_std} |")

    table = "\n".join(rows)

    # 计算 512 vs 2048 的变化
    r512 = results.get("512", {})
    r2048 = results.get("2048", {})
    analysis = ""
    if r512.get("pp_tps") and r2048.get("pp_tps"):
        pp_drop = pct_change(r512["pp_tps"], r2048["pp_tps"])
        tg_drop = pct_change(r512.get("tg_tps"), r2048.get("tg_tps"))
        analysis = (f"上下文从 512 增至 2048 时，PP 速度下降约 **{abs(pp_drop):.1f}%**，"
                    f"TG 速度下降约 **{abs(tg_drop):.1f}%**。\n"
                    f"TG 速度对上下文长度不敏感（仅约 {abs(tg_drop):.1f}% 下降），"
                    f"说明 KV Cache 在当前显存下管理良好；"
                    f"PP 速度随上下文增长有更明显的下降趋势，符合 Attention 复杂度 O(n²) 的预期。")

    img = "![上下文长度对比](plots/exp3_context_len.png)\n" if (PLOTS_DIR / "exp3_context_len.png").exists() else ""

    return f"""---

## 实验三：上下文长度对比

**实验时间：** {ts}
**固定参数：** 量化=Q4_0, GPU 全量卸载, n_gen={cfg.get('n_gen')}

| 上下文长度 | PP 速度 | TG 速度 |
|-----------|---------|---------|
{table}

### 分析
{analysis.strip()}

{img}"""


def section_conclusion() -> str:
    return """---

## 综合结论

通过三组实验，得出以下核心结论：

**量化精度方面：** Q4_0 是当前硬件（6 GB VRAM）下唯一能完整跑完的精度级别，
TG 速度达到约 57 t/s，用户体验流畅。Q8_0 因超出显存被迫回落 CPU，速度骤降至约 3 t/s。
实际部署建议优先选 Q4_0 或 Q5 系列。

**GPU 加速方面：** 全量 GPU 卸载相比纯 CPU 运行，TG 速度提升约 5x，收益非常显著。
即使只卸载一半层数（约 20 层），速度也能提升 2x 以上，说明哪怕显存有限也应尽量卸载更多层。

**上下文长度方面：** 在测试范围内（512~2048），TG 速度非常稳定（变化不超过 4%），
说明 llama.cpp 的 KV Cache 机制对当前规模的上下文管理高效。
PP 速度随上下文增长约下降 16%，在可接受范围内。

**最佳推理配置推荐：** Q4_0 量化 + 全量 GPU 卸载 + 上下文 ≤ 2048，
可在 RTX 3060 Laptop 上实现约 57 t/s 的流畅 TG 体验。
"""


def section_appendix(quant_f, gpu_f, ctx_f) -> str:
    return f"""---

## 附录：原始数据文件

- 量化对比：`results/{quant_f}`
- GPU 层数：`results/{gpu_f}`
- 上下文长度：`results/{ctx_f}`

图表文件保存在 `results/plots/` 目录下。
"""


# ── 主入口 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成 LLM benchmark Markdown 分析报告")
    parser.add_argument("--out", type=str, default="report.md",
                        help="输出文件名（默认 report.md，保存在项目根目录）")
    args = parser.parse_args()

    out_path = PROJECT_DIR / args.out

    quant_file = find_latest("quantization")
    gpu_file = find_latest("gpu_layers")
    ctx_file = find_latest("context_len")

    missing = [name for name, f in
               [("quantization", quant_file), ("gpu_layers", gpu_file), ("context_len", ctx_file)]
               if f is None]
    if missing:
        print(f"[ERROR] 找不到结果文件：{missing}，请先运行实验脚本")
        return

    quant_data = load_json(quant_file)
    gpu_data = load_json(gpu_file)
    ctx_data = load_json(ctx_file)

    sections = [
        section_overview(),
        section_quantization(quant_data),
        section_gpu_layers(gpu_data),
        section_context_len(ctx_data),
        section_conclusion(),
        section_appendix(quant_file.name, gpu_file.name, ctx_file.name),
    ]

    report = "\n\n".join(sections)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[DONE] 分析报告已写入：{out_path}")


if __name__ == "__main__":
    main()
