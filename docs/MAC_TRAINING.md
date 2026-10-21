# Mac 训练可行性记录

测试机器：MacBook Pro，Apple M5 Pro，24GB 统一内存，macOS，Python 3.12。

## 结论

| 目标 | Mac 24GB 是否可行 | 说明 |
|---|---|---|
| Gemma 3 12B QLoRA（原 AutoDL 方案） | **不可行** | 12B bf16 权重本身约 24GB；项目依赖 bitsandbytes 4bit + Unsloth/CUDA 路径，MPS 不支持这条链路 |
| 0.5B 小模型 LoRA（MLX） | **可行** | 100 iters 约 54 秒；只适合验证 pipeline，不能代表 12B 效果 |
| 4B 级模型 LoRA（MPS/MLX） | 理论可行，未在本机验证 | 需要更仔细的显存/序列长度配置 |

> 重要：Mac 上训 0.5B 的结果 **不是** 你 AutoDL 上 12B SFT/DPO 的结果，不能拿来替代。

## 实际跑过的 0.5B smoke test

- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 训练框架：`mlx-lm 0.32.2`（Apple Silicon 原生，比 PyTorch MPS 快很多）
- 数据：`data/clean/shampoo_train.json` 528 条，`shampoo_val.json` 59 条
- 配置：LoRA，batch 1，max-seq-length 768，lr 1e-4，`--mask-prompt`，100 iters
- 训练：100 iters 约 **54 秒**；train loss **1.398**，val loss **1.325**
- 推理：MLX，max_tokens=1024，temperature 0.7

### 30-prompt 结果（真实）

| 指标 | 值 |
|---|---:|
| JSON 合法率 | **10.0%** |
| 完整脚本率（≥10 镜） | **3.3%** |
| 截断率 | **90.0%** |
| 平均镜头数 | 0.9 |
| 延迟 P50 | 7.4s |

3 条样本的单独结果更好（2/3 合法、2/3 ≥10 镜），但 30 条整体说明：0.5B 会大量超长输出并在 1024 token 处截断，不能作为可用模型。

### 为什么 MPS + transformers 没有继续

同一个 0.5B 用 PyTorch MPS + transformers 跑 1890 token 的样本：

- 前 55 步约 1 秒/步；
- 之后降到约 30–45 秒/步（疑似 MPS 内存/交换压力）；
- 完整 1 epoch 需要数十分钟，不适合继续。

MLX 路径明显更适合 Mac；但 12B 仍然不现实。

## 如果要在 Mac 上继续

1. 只做 **0.5B–3B** 的格式/pipeline 验证，不追求效果。
2. 用 MLX，不要用 CUDA/bitsandbytes/unsloth 路径。
3. 把 `max_seq_length` 控制在 768–1024；更长的脚本会显著变慢。
4. 真正要复现 12B SFT/DPO：回 AutoDL 3090/4090/48GB，或租 GPU。

## 真正的 12B 训练入口

```bash
python scripts/run_finetune.py --stage sft \
  --base-model /root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master

python scripts/run_finetune.py --stage dpo \
  --base-model /root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master \
  --sft-adapter outputs/gemma3_sft/final
```
