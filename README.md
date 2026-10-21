# 洗发水带货分镜脚本 AI / Shampoo Video Script Generator

基于 Gemma 3 12B 的带货短视频分镜脚本生成项目：SFT 微调 → DPO 偏好对齐 → 可复现评测 → Gradio 演示。

> **English summary**: A Gemma 3 12B pipeline for shampoo short-video scripts, with QLoRA SFT, DPO on cleaned preference pairs, a 30-prompt reproducible evaluation harness, and a Gradio demo. This revision hardens the P0 engineering path: entry points actually run, DPO checkpoint composition is explicit, training data is cleaned and gated, and evaluation no longer treats truncation as quality.

---

## 这次 P0 改了什么

1. **启动链路修好**：`scripts/run_app.py` 不再 import 一个只在 `__main__` 里存在的 `demo`；Gradio app 改成 `build_app()` 工厂；旧的 `form_ui.py` / `script_utils.py` 只保留兼容包装。
2. **Demo 接上真正的模型**：新增 `src/inference/backends.py`，支持 Ollama 和 `base + SFT + DPO` 的 transformers 后端；不再把 base 模型当成微调模型展示。
3. **修 DPO checkpoint 组合**：明确 `base → SFT → merge → DPO` 的加载顺序；训练保存 `sft_adapter/`、`dpo_adapter/` 和 `composition.json`，新增 `scripts/merge_adapters.py`。
4. **修依赖/配置/入口的三套系统**：`requirements.txt` 只放核心工具，训练依赖拆到 `requirements-train.txt`；`config/finetune_config.yaml` 统一为 Gemma 3；`scripts/run_finetune.py` 真正调度 Gemma 3 SFT/DPO。
5. **数据清洗 + 质量门**：`scripts/clean_data.py` 过滤空口播、0 秒镜头、时间轴断裂、空音乐等问题，并从清洗后的 SFT train/val 分别构造 DPO，修掉原 DPO val 泄漏。
6. **30-prompt 可复现评测**：新增 `evaluation/prompts_30.json`、`scripts/run_eval.py`、重写的 `src/evaluation/metrics.py`；记录 seed、prompt hash、git commit、延迟、finish_reason，输出 `raw.jsonl` / `summary.json` / `report.md`。
7. **测试与 CI**：23 个 stdlib `unittest`，覆盖 schema、数据清洗、DPO 构造、指标、prompt 文件、入口 import；GitHub Actions 跑测试 + 数据检查 + mock 评测。
8. **旧代码隔离**：原 3-prompt 评测、旧 unsloth 脚本移到 `legacy/` 和 `evaluation/legacy/`，避免继续误导。

---

## 目录

```text
shampoo_video_script/
├── config/finetune_config.yaml   # SFT/DPO 唯一配置
├── data/
│   ├── train/ val/               # 原始数据（保留，用于审计）
│   └── clean/                    # 清洗后的 SFT/DPO 数据 + 质量报告
├── evaluation/
│   ├── prompts_30.json           # 固定的 30 条评测 prompt
│   ├── REPORT.md                 # 本次 P0 的数据/评测说明
│   ├── results/                  # 运行产物（gitignore）
│   └── legacy/                   # 旧 3-prompt 评测（仅历史）
├── legacy/                       # 旧 unsloth / 3-prompt 脚本（不维护）
├── scripts/
│   ├── clean_data.py             # 数据清洗 + 构造 DPO
│   ├── run_eval.py               # 30-prompt 评测入口
│   ├── run_app.py                # Gradio 入口
│   ├── run_inference.py          # 单次推理
│   ├── run_finetune.py           # SFT/DPO 调度
│   └── merge_adapters.py         # SFT+DPO 合并为完整模型
├── src/
│   ├── script_schema.py          # JSON/时间轴/口播/字段校验
│   ├── data/clean.py             # 清洗与 DPO 构造
│   ├── evaluation/metrics.py     # 客观指标
│   ├── inference/backends.py     # Ollama / transformers 后端
│   └── finetune/gemma3/          # 正式 SFT/DPO 脚本
└── tests/
```

---

## Quickstart

### 1. 核心环境（数据清洗 / 评测 / UI，无需 GPU）

```bash
./setup.sh                 # 或手动：pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Windows 用 `setup.bat`。

### 2. 清洗数据

```bash
python scripts/clean_data.py
```

输出：

```text
data/clean/
├── shampoo_train.json   # 528 条
├── shampoo_val.json     # 59 条
├── dpo_train.json       # 528 对
├── dpo_val.json         # 59 对
├── quality_report.json
└── QUALITY_REPORT.md
```

质量门默认：≥5 镜、五字段非空、时间轴连续、时长 > 0；口播语速默认只警告不剔除。只保留 ≥10 镜的高质量子集：

```bash
python scripts/clean_data.py --min-shots 10 --out-dir data/clean_strict
```

### 3. 训练（GPU 服务器）

```bash
pip install -r requirements-train.txt

# SFT（QLoRA）
python scripts/run_finetune.py --stage sft \
  --base-model /root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master

# DPO（在 SFT adapter 上继续）
python scripts/run_finetune.py --stage dpo \
  --base-model /root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master \
  --sft-adapter outputs/gemma3_sft/final
```

DPO 输出目录：

```text
outputs/gemma3_dpo/final/
├── sft_adapter/
├── dpo_adapter/
├── tokenizer/
└── composition.json
```

**不要**把 `dpo_adapter` 单独加载到 base 上；它是基于 merged SFT 训练的增量。

### 4. 合并 SFT + DPO（推荐用于最终评测/部署）

```bash
python scripts/merge_adapters.py \
  --base-model /root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master \
  --sft-adapter outputs/gemma3_sft/final \
  --dpo-adapter outputs/gemma3_dpo/final/dpo_adapter \
  --output outputs/shampoo_sft_dpo_merged \
  --device cpu
```

12B bf16 合并很吃内存；24GB 卡建议直接用 `TransformersBackend` 的两段 adapter 加载，或换 4B 模型。

### 5. 30-prompt 评测

先自检 harness（不需要模型）：

```bash
python scripts/run_eval.py --backend mock --out-dir evaluation/results/mock_30
```

真实模型：

```bash
python scripts/run_eval.py --backend transformers \
  --base-model /root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master \
  --sft-adapter outputs/gemma3_sft/final \
  --dpo-adapter outputs/gemma3_dpo/final/dpo_adapter \
  --max-new-tokens 700 \
  --out-dir evaluation/results/sft_dpo
```

输出 `raw.jsonl`、`scored.json`、`summary.json`、`report.md`、`manifest.json`。每个 run 都记录 seed、prompt hash、system prompt hash 和 git commit。

> 24GB + 12B bf16 下 700 token 仍可能截断；这时应换 4B 模型、增加显存，或实现分段续写，而不是把 `max_new_tokens` 盲目调到 1500。

### 6. App

```bash
# 本地 Ollama
OLLAMA_MODEL=gemma_3_12B_q80:latest python scripts/run_app.py

# 服务器：直接加载 base + SFT + DPO
SCRIPT_BACKEND=transformers \
SCRIPT_BASE_MODEL=/root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master \
SCRIPT_SFT_ADAPTER=outputs/gemma3_sft/final \
SCRIPT_DPO_ADAPTER=outputs/gemma3_dpo/final/dpo_adapter \
python scripts/run_app.py --host 0.0.0.0 --port 9999

# 或加载 merge 后的完整模型
SCRIPT_BACKEND=transformers \
SCRIPT_MODEL=outputs/shampoo_sft_dpo_merged \
python scripts/run_app.py
```

`SCRIPT_USE_CACHE=1` 只在显存足够时开启；24GB + 12B bf16 默认关闭。

---

## 数据质量结论（本次实际跑出来的）

原始 SFT 数据 1,275 条（其中 12 条重复输出，去重后 1,263 条参与质量门），清洗后保留 528 条 train / 59 条 val，通过率约 41.8%。被剔除的主要原因：

- 空/占位口播：1,023 / 10,334 个镜头（9.9%）
- 音乐缺失或“无”：2,034 / 10,334（19.7%）
- 0 秒镜头：177
- 时间轴断裂 / 重叠：216 / 35
- 口播超过 4 字/秒：78.0%（中位数 5.75 字/秒）
- 只有 27.1% 的样本 ≥10 镜
- ≥10 镜 + 五字段 + 时间连续（语速只警告）：95 / 1,275（7.5%）；若再把语速 ≤4 字/秒作为硬门槛：0 / 1,275

清洗后仍有 **82.2%（3,328 / 4,049）** 的镜头口播超过 4 字/秒；528 条里只有 **95 条 ≥10 镜（18.0%）**。也就是说，数据问题不只是“脏”，而是脚本本身的节奏和结构不符合 app prompt；下一步需要重写口播/重定时，而不是只靠过滤。

这些不是“模型调参”能解决的，必须先清洗。详细报告见 `data/clean/QUALITY_REPORT.md` 和 `evaluation/REPORT.md`。

---

## 评测指标

`src/evaluation/metrics.py` 把以下几类信号分开，不再用一个“JSON 合法”混在一起：

- `json_valid`：能否解析出完整 JSON（截断不算合法）
- `schema_valid`：五字段、时间轴、空口播、空音乐等机械校验
- `complete_script`：schema 合法且 ≥ `min_shots`
- `truncated`：`finish_reason == length`，或明显未闭合 JSON
- `gaps` / `overlaps` / `zero_duration`
- `pacing_violations`：按 3–4 字/秒
- `keyword_coverage`：只从合法 JSON 内容里统计，避免“正文里提到关键词”骗分
- `forbidden_brand_hits`：prompt 未指定品牌时，检查是否编造品牌
- `latency_p50/p95`

**旧 3-prompt 报告不再代表新结论**，已移到 `evaluation/legacy/`。用新指标重打旧日志的结果是：

| 模型 | JSON 合法 | 完整脚本（≥10 镜） | 截断 | 平均镜头 |
|---|---:|---:|---:|---:|
| BASE | 0/3 | 0/3 | 3/3 | 0（截断无法解析） |
| SFT | 3/3 | 0/3 | 0/3 | 6.67 |
| DPO | 0/3 | 0/3 | 3/3 | 0（截断无法解析） |

这比“DPO 4.75 分、显著优于 SFT”更接近事实：**旧评测里 DPO/base 的 JSON 是坏的，不能直接用于产品。**

---

## 已知限制

1. 528 条 SFT 仍然偏小；质量门只保证机械格式，不保证创意质量。
2. 30 条 prompt 能覆盖主要场景，但还不是统计显著的大规模评测；下一步应加人工盲评/LLM-as-judge，并报告置信区间。
3. 12B + 24GB 仍然是工程瓶颈；如果 `max_new_tokens=700` 仍截断，优先换 4B 或 48GB 卡，而不是继续在产品里接受截断。
4. DPO rejected 仍是程序注入的硬伤，不是真实人类偏好；它主要教“避免低级错误”，不教“写出更好的广告”。
5. 数据集含真实品牌、价格、主播 persona；商用前需补 LICENSE、data card、model card 和广告合规检查（尤其“防脱/生发”这类宣称）。

---

## 验收命令

```bash
python -m unittest discover -s tests -v       # 23 passed
python scripts/clean_data.py                  # 528 / 59 + DPO 528 / 59
python scripts/run_eval.py --backend mock --out-dir evaluation/results/mock_30
python -c "from scripts.run_app import build_app; print(callable(build_app))"
```
