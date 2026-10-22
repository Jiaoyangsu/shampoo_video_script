# 当前训练状态（2026-10-22）

## 已完成

- 15 条盲评抽检：`11 edit / 4 reject`，`pass + edit = 73.3%`。
- 3 条 train + 1 条 val 被人工 review 剔除。
- 当前 30s rewrite 数据：**54 train / 14 val**（不是 200–300）。
- DPO 对：54 / 14，全部由 rewrite 数据构造。
- 30-prompt 评测 harness、Mac 0.5B SFT/DPO smoke 脚本已就绪。

## 未完成，原因

### 1. 扩到 200–300 条

本机 Ollama `qwen2.5:14b` 重写一条 30s/10–12 镜脚本约 25–65 秒；
尝试双实例并行时出现空输出/超时。当前只稳定产出 57 条 accepted（review 后 54）。
按这个吞吐，补到 200 条还需要约 1–2 小时稳定 GPU 时间。

### 2. 12B SFT/DPO

Mac 24GB 不能跑 Gemma 3 12B QLoRA：

- 12B bf16 权重约 24GB；
- bitsandbytes 4bit / Unsloth 是 CUDA 路径，MPS 不支持；
- 没有可用的 AutoDL 实例（README 里的旧 SSH 端口全部 connection refused）。

### 3. 0.5B smoke SFT

在 Mac MPS 上跑了 50 步作为连通性测试，训练在约 50 步后发散
（loss/grad_norm 出现 NaN），不可作为模型结果。DPO 只验证了 2 步能跑通，
没有做有效训练。

## 需要什么

要真正完成“200–300 条 + 12B SFT/DPO”：

1. 一台可用 GPU（3090/4090/48GB）或 AutoDL SSH；
2. 或接受先用 0.5B–3B 做 pipeline 验证，不把结果当 12B 效果；
3. 或继续在 Mac 上分批跑数据重写（约 1–2 小时/100 条），训练仍回 GPU。

不要把这批 54 条当作最终训练集；它只是 review 后的 pilot。
