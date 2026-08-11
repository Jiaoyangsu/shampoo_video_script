# 洗发水带货分镜脚本 AI

基于 Gemma 3 12B 的带货短视频分镜脚本生成模型，支持 SFT 微调、DPO 偏好对齐与 JSON 格式推理验证。

## 项目结构

```
shampoo_video_script/
├── config/                  # 训练配置
├── data/
│   ├── train/               # SFT + DPO 训练数据
│   │   ├── shampoo_train.json    # SFT 数据
│   │   ├── dpo_train.json        # DPO 偏好对（1275 对）
│   ├── val/
│   │   ├── shampoo_val.json
│   │   └── dpo_val.json          # DPO 验证集（142 对）
├── src/
│   ├── app/                 # Gradio UI 应用
│   ├── data/                # 数据准备脚本
│   ├── finetune/
│   │   ├── gemma3/          # 服务器跑通的正式脚本（推荐使用）
│   │   │   ├── finetune_sft.py    # SFT QLoRA 训练
│   │   │   ├── finetune_dpo.py    # DPO 训练（precompute ref）
│   │   │   ├── eval_compare.py    # SFT vs DPO 推理对比
│   │   │   └── inference_test.py  # JSON 格式推理验证
│   │   ├── dpo_train.py     # 旧版 unsloth DPO（legacy，未验证）
│   │   └── trainer.py       # 旧版 unsloth SFT（legacy，未验证）
│   └── inference/           # 推理模块
├── scripts/                 # 运行入口
├── outputs/                 # 训练输出（gitignore）
├── models/                  # 模型权重（gitignore）
└── requirements.txt
```

## 服务器环境（AutoDL RTX 3090 24GB）

- Python：miniconda3（/root/miniconda3）
- torch 2.3.0+cu121、transformers 4.51.3、peft 0.15.0、trl 0.14.0、bitsandbytes 0.44.1
- base 模型：`/root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master`（ModelScope 下载）
- SFT 输出：`/root/autodl-tmp/outputs/gemma3_sft/final`（bf16 推理可用）
- DPO 输出：`/root/autodl-tmp/outputs/gemma3_dpo/final`（bf16 推理可用）
- 训练数据：`/root/sft_data/`

## 已知关键问题（务必先读）

### 1. bitsandbytes 量化与 gemma3 不兼容（NaN）
4bit/8bit 量化加载 gemma3 推理会产生 NaN logits（`multinomial` 采样崩溃）。
**训练可以用 4bit（QLoRA），推理必须用 bf16 全精度**。
bf16 下 12B ≈ 23GB，恰好塞进 3090 的 24GB。

### 2. bf16 推理显存天花板
bf16 + 24GB 卡只剩 ~1GB 余量：
- `use_cache=False` 时单次生成上限约 700 tokens，超出会 OOM
- `use_cache=True` 会按 gemma3 的 max_position_embeddings(131072) 预分配 KV cache，直接爆显存
- `cache_implementation="dynamic"` 对 gemma3 无效（transformers 4.51 强制 hybrid）
- `merge_and_unload()` 合并 adapter 也 OOM（23GB + 合并缓冲）
- **结论**：bf16 下只能 `use_cache=False` + 短生成（≤700 tokens），长脚本需分段续写或接受截断

### 3. DPO 输出带 JSON 围栏
DPO 训练数据用 `json.dumps(..., indent=2)`，模型学会输出 ` ```json ` 围栏。
解析时需用正则剥围栏：`re.search(r"```json\s*(.*?)\s*```", text, re.S)`

### 4. DPO 偏好数据是合成缺陷（非人工标注）
`src/data/prepare_dpo_data.py`：rejected 是程序注入的缺陷
（空口播 / 黑屏转场 / 时间轴断裂 / 镜头类型乱序）。
DPO 学到的是"避免低级硬伤"，不是"写出更好广告"。若需真偏好，用 RLAIF 或人工标注。

### 5. 教训：训练前必须审计数据
本次 DPO 训练前未审查偏好数据生成逻辑，导致发现硬伤时已浪费算力。
**流程要求**：训练前先 `inspect` 数据（样本拆解、chosen/rejected 差异、格式与 SFT 一致性），确认无误再启动。

## 训练流程

```bash
# 1. SFT（QLoRA，bf16 推理）
python src/finetune/gemma3/finetune_sft.py

# 2. DPO（base + SFT adapter + 新 LoRA，precompute ref）
python src/finetune/gemma3/finetune_dpo.py

# 3. 对比验证（SFT vs DPO）
python src/finetune/gemma3/eval_compare.py <sft_adapter_path> SFT
python src/finetune/gemma3/eval_compare.py <dpo_adapter_path> DPO
```

## Gemma3 prompt 模板

```
<bos><start_of_turn>user\n{SYSTEM_PROMPT}\n\n{input}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn><eos>
```
