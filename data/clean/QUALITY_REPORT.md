# 数据清洗报告

生成方式：`python scripts/clean_data.py`

## 配置
```json
{
  "min_shots": 5,
  "require_music": true,
  "max_chars_per_second": 4.0,
  "enforce_pacing": false,
  "seed": 3407,
  "max_dpo_pairs": null
}
```

## SFT
| 数据集 | 原始 | 保留 | 剔除 | 通过率 |
|---|---:|---:|---:|---:|
| train | 1275 | **528** | 735 | 41.8% |
| val | 142 | **59** | 83 | 41.5% |

注：完全重复的输出在质量门前被去重；因此 `保留 + 剔除` 可能小于原始条数。

跨 split 重复 input 被从 val 移除：0 条。

## DPO
| 数据集 | 对数 | 跳过 |
|---|---:|---:|
| train | **528** | 0 |
| val | **59** | 0 |

缺陷类型分布（train）：
```json
{
  "empty_voiceover": 77,
  "missing_music": 68,
  "zero_duration": 83,
  "broken_time": 79,
  "wrong_shot_type": 65,
  "shuffle_order": 75,
  "duplicate_shot": 70,
  "bad_pacing": 11
}
```

## 说明
- 质量门检查机械可验证项：五字段非空、时间轴有效且连续、时长 > 0。
- 口播语速默认只计数/警告；加 `--enforce-pacing` 才把超过阈值的样本剔除。
- 通过质量门不代表创意质量好；创意质量仍需人工/LLM 盲评。
- DPO 的 chosen 来自清洗后的 SFT train/val，rejected 由程序注入**至少一个**新增硬伤，且要求硬伤数严格增加。
