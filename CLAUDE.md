# CLAUDE.md — On-device LLM Inference Optimization

## 项目目标
基于 llama.cpp 在 Windows + NVIDIA GPU 环境下实现端侧 7B 模型推理系统，
完成量化对比、GPU加速、KV Cache 三组实验，输出完整 benchmark 报告。

## 技术栈
- 推理框架：llama.cpp（GGUF 格式，CUDA 后端）
- 语言：Python 3.10+（benchmark），C++（llama.cpp 本体）
- GPU：NVIDIA RTX XXXX，CUDA 12.x
- 路径约定：
  - llama-cli：`./llama.cpp/build/bin/llama-cli.exe`
  - 模型目录：`./models/`
  - 结果目录：`./results/`（JSON 格式追加写入）

## 开发规范
- 所有脚本支持 `--dry_run` 参数
- benchmark 结果统一写入 `results/<实验名>_<时间戳>.json`
- 函数加 docstring，benchmark 脚本加进度打印
- 不引入非必要依赖，显存监控只用 nvidia-smi 解析

---

## 任务规划

> 状态标记：[ ] 待开始  [x] 已完成  [~] 进行中

### Phase 0：环境搭建
- [x] **T0.1** 编写 `setup/build_llama.sh`，含 `-DLLAMA_CUDA=ON` 编译参数
- [x] **T0.2** 编写 `setup/download_models.py`，下载 Qwen2.5-7B 的 q4/q8/f16 三个 GGUF 版本
- [x] **T0.3** 编写 `requirements.txt`（matplotlib, pandas, psutil, tqdm, huggingface_hub）
- [x] **T0.4** 验证 llama-cli 可正常推理，输出 hello world（`setup/verify_install.py`）

### Phase 1：推理核心模块
- [x] **T1.1** `inference/config.py`：定义模型路径、量化版本、GPU层数等配置常量
- [x] **T1.2** `inference/run_inference.py`：封装 llama-cli 调用，返回结构化结果（tokens/s、耗时、输出文本）
- [x] **T1.3** `benchmark/metrics.py`：实现 tokens/s、prompt latency、decode latency、VRAM 采集

### Phase 2：三组实验
- [x] **T2.1** `benchmark/experiments/exp1_quantization.py`
  - 对比 q4_0 / q8_0 / f16，固定 prompt，各跑 5 次取均值
  - 输出：速度、显存、输出一致性对比
  - Bug修复：vram_mb 统一为 dict 结构，新增 consistency_pct 指标
- [x] **T2.2** `benchmark/experiments/exp2_gpu_layers.py`
  - GPU 层数：0 / 20 / 40 / -1（full），固定量化版本 q4
  - 输出：tokens/s 随层数变化曲线
  - 改进：新增 `--quant` 参数支持切换量化变体
- [x] **T2.3** `benchmark/experiments/exp3_context_len.py`
  - 上下文长度：512 / 1024 / 2048 tokens
  - 输出：延迟随上下文增长趋势
  - 改进：新增 `--n_gpu_layers` 参数灵活性

### Phase 3：分析与报告
- [x] **T3.1** `analysis/plot_results.py`：读取 results/ 下 JSON，生成对比图表（matplotlib）
- [x] **T3.2** `analysis/report_gen.py`：自动生成 Markdown 格式分析报告
- [x] **T3.3** 完善 `README.md`：写清楚环境要求、复现步骤、实验结论

### Phase 4：进阶优化（可选）
- [ ] **T4.1** 实现 speculative decoding 对比实验
- [ ] **T4.2** CPU + GPU 混合调度策略分析
- [ ] **T4.3** KV Cache 压缩效果测试

---

## 当前状态
**进行中 Phase：** Phase 4（可选）
**下一个任务：** T4.1（可选）speculative decoding
**已知问题：** 无

## 对 Claude Code 的指令习惯
- 每次只做一个任务（如"完成 T1.2"），完成后更新状态标记
- 生成代码前先说明实现思路
- 遇到路径/平台问题优先问我，不要自行假设
