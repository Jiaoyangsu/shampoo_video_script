# P0 修复与数据/评测报告

本报告记录 2026-10 这次 P0 修复的**可复现证据**。它替代原来的 3-prompt 报告（已移到
`evaluation/legacy/REPORT_3PROMPT.md`）。

> 说明：本机没有 GPU / transformers，未在本机复跑 12B 模型。下面的“真实结果”包括：
> 数据清洗、旧日志重打分、30-prompt harness 自检；真实模型 30-prompt 运行命令在最后，
> 需要在 AutoDL/GPU 机器上执行。

---

## 1. 修复清单

| 问题 | 修复 |
|---|---|
| `run_app.py` import 不存在的 `demo` | `build_app()` 工厂；`run_app` 只 import 工厂 |
| `form_ui.py` 缺 `import os` | 改为 `chat_ui` 兼容包装 |
| 异常无限递归重试 | 有限次重试；失败返回结构化错误 |
| Demo 用 Ollama base 冒充微调模型 | 新增 Ollama / transformers 双后端；支持 SFT+DPO |
| DPO 只保存新 adapter，eval 只 load DPO | 保存 `sft_adapter/` + `dpo_adapter/` + `composition.json`；新增 merge 脚本 |
| `transformers<4.49` 不支持 Gemma 3 | 训练依赖改为 `transformers>=4.50,<4.52` |
| config 是 DeepSeek-1.5B | 统一为 Gemma 3 配置 |
| 旧 unsloth/3-prompt 脚本误导 | 移到 `legacy/` |
| 数据空口播/0 秒/时间断裂 | `scripts/clean_data.py` 质量门 |
| DPO val 来自 SFT train | DPO train/val 分别从清洗后的 SFT train/val 构造 |
| `temperature` 无 `do_sample` | `TransformersBackend` 显式 `do_sample=True` |
| 旧 metrics 把截断当合法/把字段完整当时间单调 | 重写 `src/evaluation/metrics.py` |

---

## 2. 数据清洗结果（真实运行）

命令：

```bash
python scripts/clean_data.py
```

质量门：≥5 镜、五字段非空、时间轴连续、时长 > 0；口播 3–4 字/秒默认只警告。

| 数据集 | 原始 | 保留 | 剔除 | 通过率 |
|---|---:|---:|---:|---:|
| SFT train | 1,275 | **528** | 735 | 41.8% |
| SFT val | 142 | **59** | 83 | 41.5% |
| DPO train | — | **528 对** | 0 | — |
| DPO val | — | **59 对** | 0 | — |

> 12 条重复输出被去重，所以 `528 + 735 = 1,263`，不是 1,275。

原始数据的机械问题（清洗前）：

| 检查项 | 数量 |
|---|---:|
| 空/占位口播 | 1,023 / 10,334 镜头（9.9%） |
| 音乐缺失/“无” | 2,034 / 10,334（19.7%） |
| 0 秒镜头 | 177 |
| 时间轴断裂 / 重叠 | 216 / 35 |
| 口播 >4 字/秒 | 7,477 / 9,580（78.0%）；中位 5.75 字/秒 |
| ≥10 镜 | 346 / 1,275（27.1%） |
| ≥10 镜 + 五字段 + 连续（语速只警告） | 95 / 1,275（7.5%） |
| 再加语速 ≤4 字/秒硬门槛 | 0 / 1,275 |

清洗后的残留问题：

| 指标 | 清洗后 |
|---|---:|
| 空口播 / 0 秒 / 时间断裂 | 0（已被质量门过滤） |
| 口播 >4 字/秒 | **3,328 / 4,049（82.2%）** |
| ≥10 镜 | **95 / 528（18.0%）** |

所以清洗解决的是“空/断/0 秒”这类硬伤，**没有解决节奏和结构**；后者需要重写口播/重新分配时间轴，不能靠过滤。

完整报告：`data/clean/QUALITY_REPORT.md`；机器可读：`data/clean/quality_report.json`。

**重要**：清洗只解决“能不能拍”的机械问题，不解决“创意好不好”。528 条仍偏小，
下一步应补充人工精选/改写，而不是继续放大 DPO 轮数。

---

## 2.5 30 秒重写 pilot（真实运行）

针对“82.2% 语速超标 / 只有 18% ≥10 镜”的问题，新增了 `scripts/rewrite_data.py`：

- 输入：`data/clean`
- 模型：本地 `qwen2.5:14b`（Ollama JSON schema）
- 目标：30 秒、10–12 镜、每镜 8–12 字、≤4 字/秒
- 后处理：按目标时长确定性重排时间轴；口播太长导致放不下的样本会被拒绝
- 输出：`data/rewrite_30s/`

pilot 结果：

| split | 接受 | 拒绝 | 机械校验 |
|---|---:|---:|---|
| train | 50 | 0 | 10–12 镜、30s、0 语速超标 |
| val | 15 | 0 | 10–12 镜、30s、0 语速超标 |

抽检表：`data/rewrite_30s/train_review.csv`、`val_review.csv`。
当前 flags：`generic_cta` 33/50、`short_visual` 23/50、`duplicate_voiceover` 3/50。
这说明它已经是可拍的格式草稿，但**创意质量仍需人工抽检**；抽检后再扩到 200 条并重训。

## 3. 旧日志重打分（真实运行）

用新指标重打 `evaluation/logs/cmp_*.log`：

| 模型 | JSON 合法 | Schema 合法 | 完整脚本（≥10 镜） | 截断 | 平均镜头 |
|---|---:|---:|---:|---:|---:|
| BASE | 0/3 | 0/3 | 0/3 | 3/3 | 0 |
| SFT | 3/3 | 3/3 | 0/3 | 0/3 | 6.67 |
| DPO | 0/3 | 0/3 | 0/3 | 3/3 | 0 |

结论：

- 旧报告说“DPO 业务质量 4.75，显著优于 SFT 2.25”，但 DPO 的 3 条输出全部是**截断的非法 JSON**；
  对产品来说这是硬失败，不是“内容密度更高”。
- SFT 的优势是格式：3/3 合法，但平均只有 6.67 镜，不满足 app prompt 的 ≥10。
- 旧 DPO 日志的风格（fenced JSON、`0:00`、`CU`）也更像 base，而不是 SFT；与
  “DPO adapter 单独 load 到 base”的代码路径一致。

重打分结果（本地生成，gitignore）：`evaluation/results/legacy_rescored/*.json`。

---

## 4. 30-prompt 评测集

文件：`evaluation/prompts_30.json`。30 条，覆盖：

- 品牌：海飞丝、清扬、潘婷、霸王
- 人群：男士、儿童、婴儿、孕产、学生党、上班族
- 功效：去屑、控油、蓬松、修护、防脱（合规约束）
- 场景：旅行、健身房、夏季、冬季、直播、新品、对比
- 风险：白牌不得虚构品牌、防脱不得医疗承诺、英文输入要求中文 JSON

每条包含：`id`、`category`、`prompt`、`expected_brand`、`required_keywords`、
`forbidden_brands`、`min_shots`、`target_duration_sec`、`notes`。

### Harness 自检（mock，不代表模型质量）

```bash
python scripts/run_eval.py --backend mock --out-dir evaluation/results/mock_30
```

30 条全部跑通；mock 故意一半合法、一半截断，得到：

| 指标 | 值 |
|---|---:|
| JSON 合法率 | 50.0% |
| 完整脚本率 | 50.0% |
| 截断率 | 50.0% |
| 平均镜头数 | 5.0 |

这验证了“截断会被识别为失败”，而不是被当成合法 JSON。

### 真实模型运行（待 GPU）

```bash
python scripts/run_eval.py --backend transformers \
  --base-model /root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master \
  --sft-adapter /root/autodl-tmp/outputs/gemma3_sft/final \
  --dpo-adapter /root/autodl-tmp/outputs/gemma3_dpo/final/dpo_adapter \
  --max-new-tokens 700 \
  --out-dir evaluation/results/sft_dpo
```

建议同时跑：

1. `--merged-model`：最终部署组合。
2. `--backend ollama --model ...`：Demo 实际模型。
3. base（不带 adapter）：确认 SFT/DPO 的增量。

---

## 5. 指标定义

| 指标 | 含义 |
|---|---|
| `json_valid` | 完整 JSON 可解析；截断为 false |
| `schema_valid` | 五字段、时间轴、空口播/空音乐等机械校验 |
| `complete_script` | schema 合法且镜头数 ≥ prompt 的 `min_shots` |
| `truncated` | `finish_reason=length` 或明显未闭合 JSON |
| `gaps` / `overlaps` | 时间轴断裂/重叠 |
| `zero_duration` | 0 秒或负时长镜头 |
| `pacing_violations` | 去掉标点后 >4 字/秒的镜头数 |
| `keyword_coverage` | 只从合法 JSON 内容统计，避免正文“提到关键词”骗分 |
| `forbidden_brand_hits` | prompt 未指定品牌时，输出中出现的品牌 |
| `latency_p50/p95` | 端到端生成延迟 |

---

## 6. 验收

```bash
python -m unittest discover -s tests -v       # 27 passed
python scripts/clean_data.py                  # 528 / 59 + DPO 528 / 59
python scripts/run_eval.py --backend mock --out-dir evaluation/results/mock_30
```

通过标准：

- 测试全绿；
- `data/clean` 可重复生成，`quality_report.json` hash 稳定；
- mock 30 条能完整跑完并输出 report；
- `scripts/run_app.py` 在安装 `requirements.txt` 后可 import/启动；
- 真实 30-prompt 结果出来后，报告 `summary.json` 而不是只贴 3 条样本。
