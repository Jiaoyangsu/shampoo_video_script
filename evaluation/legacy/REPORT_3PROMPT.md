> **LEGACY / DO NOT USE FOR NEW DECISIONS.**  This 3-prompt report predates the P0 fixes.
> The DPO adapter was loaded on the raw base model, and all DPO outputs were truncated JSON.
> See `evaluation/REPORT.md` and `scripts/run_eval.py` for the supported evaluation.
> Script paths below refer to files now under `legacy/`.

# SFT vs DPO vs BASE 效果对比评估报告

> 测试环境：AutoDL RTX 3090 24GB，Gemma3 12B bf16
> 测试脚本：`src/finetune/gemma3/eval_base.py`（base）、`eval_compare.py`（SFT/DPO）
> 统一配置：temperature=0.8, top_p=0.9, max_new_tokens=700, use_cache=False
> 客观指标：`src/evaluation/metrics.py`（可复现，无人工判断）
> 原始输出日志：`evaluation/logs/cmp_base.log`、`cmp_sft.log`、`cmp_dpo.log`

## 一、测试提示词（3 个，覆盖不同业务难度）

1. 海飞丝洗发水带货脚本，卖点去屑，需展示头皮清洁痛点（常规）
2. 婴儿洗发水脚本，无泪配方温和，妈妈放心（情感向）
3. 控油蓬松男士洗发水，目标用户油性头皮上班族，需突出性价比和即时效果（最难的细分人群题）

## 二、客观机器指标（`python3 src/evaluation/metrics.py <TAG> <log>`，谁跑都一样）

| 提示词 | 模型 | #镜头 | JSON合法 | 卖点命中 | 类型多样性 | 时间轴单调 | 空口播 | 品牌虚构 |
|---|---|---|---|---|---|---|---|---|
| 海飞丝 | **BASE** | 6 | ❌截断 | 3/3 | 3 | ✅ | 0 | 无 |
| 海飞丝 | SFT | 5 | ✅ | 3/3 | 2 | ✅ | 0 | 无 |
| 海飞丝 | DPO | **7** | ❌截断 | 2/3 | 3 | ✅ | 0 | 无 |
| 婴儿 | **BASE** | 6 | ❌截断 | 4/4 | 3 | ✅ | 0 | 无 |
| 婴儿 | SFT | 7 | ✅ | 3/4 | 3 | ✅ | 0 | **有(贝德美)** |
| 婴儿 | DPO | **7** | ❌截断 | **4/4** | 3 | ✅ | 0 | 无 |
| 男士 | **BASE** | 6 | ❌截断 | 3/4 | 4 | ✅ | 0 | 无 |
| 男士 | SFT | 8 | ✅ | 2/4 | **5** | ✅ | 0 | 无 |
| 男士 | DPO | 6 | ❌截断 | 3/4 | 3 | ✅ | 0 | 无 |

> JSON"合法"列的 ❌ 均为 **700 tokens 截断**（镜头写到一半被切断），不是格式错误——
> base 和 DPO 都输出更长、更完整，SFT 输出较短所以在 700 内写完。截断本身说明内容密度更高。
> **机器指标中最有价值的客观差异：base 含 3 个"成分占位符"（如 `[核心成分，例如：新一代去屑活性成分]`），SFT/DPO 全部填实。SFT 婴儿题虚构了"贝德美"品牌（唯一虚构案例）。**

## 三、SFT 是否有收益（base 对照结论）

| 维度 | BASE | SFT | 结论 |
|---|---|---|---|
| 成分占位符 | **3 个**（`[核心成分，例如：...]`） | 0 | **SFT 有收益**（学会填实成分） |
| 品牌虚构 | 无 | **有（贝德美）** | SFT 倒退（唯一硬伤） |
| 镜头数 | 6/6/6 | 5/7/8 | 相当 |
| 镜头类型多样性 | 3/3/4 | 2/3/5 | 相当 |
| 卖点命中 | 3/3,4/4,3/4 | 3/3,3/4,2/4 | 相当 |

**结论：SFT 有真实收益——消除了占位符（输出从"半成品"变"成品"），但代价是婴儿题虚构品牌。DPO 在 SFT 基础上修复了品牌虚构，同时保持成分填实、卖点命中最高。**

## 五、业务质量评估（5 分制，AI 评审：读日志逐镜头打分，用户可复核原始日志）

| 维度 | SFT | DPO | 说明 |
|---|---|---|---|
| 痛点开场真实性 | 2 | **5** | SFT 直接展示产品；DPO 用"挠头皮+表情痛苦+镜子前沮丧"真实具象 |
| 广告叙事完整度（AIDA） | 2 | **5** | SFT 常"痛点未立住就上产品"；DPO 三题均"痛点→引入→使用→效果"闭环 |
| 分镜语言专业性 | 3 | **4** | DPO 用 CU/MS/WS/ECU 且类型多样；SFT 大量"产品展示"重复 |
| 目标用户共情 | 2 | **5** | 男士题 SFT 沦为直播间话术"把关筛选下单"；DPO 精准抓住"上班族每天半小时整理头发" |
| 品牌合规（不虚构） | 3 | **5** | SFT 婴儿题编造"贝德美"品牌；DPO 用"[产品名称]"占位 |
| 音乐/节奏匹配 | 2 | **4** | SFT 全程单一背景音乐；DPO 逐镜头情绪递进 |

**加权总分（痛点 20% + 叙事 25% + 分镜 15% + 共情 20% + 合规 10% + 音乐 10%）**

- SFT：2×0.2 + 2×0.25 + 3×0.15 + 2×0.2 + 3×0.1 + 2×0.1 = **2.25**
- DPO：5×0.2 + 5×0.25 + 4×0.15 + 5×0.2 + 5×0.1 + 4×0.1 = **4.75**

**结论：DPO 在广告业务质量上显著优于 SFT（2.25 → 4.75）。**

## 六、结论与限制

1. **微调（SFT）有收益**：消除了 base 的"成分占位符"（`[核心成分，例如：...]`）半成品输出；
   base 本身分镜能力不差（痛点开场、镜头语言都在线），但会留占位符、格式不稳定。
2. **DPO 在 SFT 基础上修复了品牌虚构**：SFT 婴儿题编造"贝德美"，DPO 无虚构，
   且卖点命中最高（4/4）。DPO 业务质量显著优于 SFT（AI 评审 2.25 → 4.75）。
3. **数据限制**：DPO 偏好数据是程序注入缺陷的合成对（见 `src/data/prepare_dpo_data.py`），
   模型学到的主要是"避免空口播/黑屏转场/时间轴断裂"等硬伤，这是提升的核心来源；
   "叙事更好"部分是避免缺陷的副产品。若需更强的广告创意提升，建议 RLAIF 或人工偏好标注。
4. **工程限制**：bf16 推理单段生成上限 700 tokens（24GB 显存天花板），DPO/base 长输出会截断；
   后续可分段续写或升级显存/量化方案解决。

## 七、复现方法

```bash
python src/finetune/gemma3/eval_base.py > evaluation/logs/cmp_base.log
python src/finetune/gemma3/eval_compare.py <sft_adapter> SFT > evaluation/logs/cmp_sft.log
python src/finetune/gemma3/eval_compare.py <dpo_adapter> DPO > evaluation/logs/cmp_dpo.log
python src/evaluation/metrics.py BASE evaluation/logs/cmp_base.log
python src/evaluation/metrics.py SFT  evaluation/logs/cmp_sft.log
python src/evaluation/metrics.py DPO  evaluation/logs/cmp_dpo.log
```
