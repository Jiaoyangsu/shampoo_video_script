# Base 对照实验：执行说明

## 目标
证明 SFT 是否有收益：将未微调的 Gemma3 12B base 与 SFT、DPO 在同一批 3 个 prompt 下对比。

## 预计耗时
- 模型加载：~1 分钟
- 3 个 prompt 推理（各 700 tokens）：3-6 分钟
- 指标计算：秒级
- **总计约 10-15 分钟 GPU 时间**

## 步骤（AutoDL 开机后）

### 1. 上传脚本到服务器
```bash
scp -P 37563 src/finetune/gemma3/eval_base.py root@connect.nmb2.seetacloud.com:/root/sft_data/
```

### 2. 跑 base 推理
```bash
cd /root/sft_data && nohup python3 eval_base.py > cmp_base.log 2>&1 &
# 等待出现 ALL_TEST_DONE
```

### 3. 拉回日志（本地执行）
```bash
scp -P 37563 root@connect.nmb2.seetacloud.com:/root/sft_data/cmp_base.log ~/Desktop/shampoo_video_script/evaluation/logs/
```

### 4. 本地计算三项客观指标
```bash
python3 src/evaluation/metrics.py BASE  evaluation/logs/cmp_base.log
python3 src/evaluation/metrics.py SFT   evaluation/logs/cmp_sft.log
python3 src/evaluation/metrics.py DPO   evaluation/logs/cmp_dpo.log
```

### 5. 结论判定
对比 `#shots / kw / types / emptyVO / brand` 五列：
- base ≈ SFT：SFT 无收益（DPO 收益来自 base 本身）
- SFT 明显优于 base：SFT 有收益
- 更新 `evaluation/REPORT.md` 的"SFT 收益"一节

### 6. 关机省费用
AutoDL 网页关机。

## 已就绪文件
- `src/finetune/gemma3/eval_base.py` — base 推理（与 eval_compare.py 同配置，无 adapter）
- `src/evaluation/metrics.py` — 客观指标（卖点命中/镜头数/类型多样性/时间轴单调/品牌虚构/空口播）
- `evaluation/logs/cmp_sft.log`、`cmp_dpo.log` — 已有日志
