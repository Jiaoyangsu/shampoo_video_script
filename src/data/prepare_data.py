"""
数据准备模块：过滤洗发水数据、划分训练/验证集
"""
import json
import os
import random

SHAMPOO_KEYWORDS = [
    "洗发水", "洗发露", "shampoo", "护发素", "发膜", "护发精华",
    "头皮", "去屑", "控油", "蓬松", "柔顺", "顺滑", "润发",
    "干发", "油头", "头发", "发丝", "发根", "毛躁", "修复", "烫染",
]

NON_SHAMPOO_KEYWORDS = [
    "沐浴露", "沐浴", "身体乳", "洗手液", "眉笔", "口红",
    "眼影", "粉底", "面膜", "面霜", "精华液", "洗衣液", "洗洁精",
]


def is_shampoo(text: str) -> bool:
    text_lower = text.lower()
    has = any(kw.lower() in text_lower for kw in SHAMPOO_KEYWORDS)
    not_other = not any(kw.lower() in text_lower for kw in NON_SHAMPOO_KEYWORDS)
    return has and not_other


def filter_shampoo_data(input_path: str, output_dir: str, val_ratio: float = 0.1, seed: int = 3407):
    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [item for item in data if is_shampoo(item["input"])]
    print(f"  原始数据: {len(data)} → 洗发水数据: {len(filtered)}")

    random.shuffle(filtered)
    split = int(len(filtered) * (1 - val_ratio))
    train, val = filtered[:split], filtered[split:]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "shampoo_train.json")
    val_path = os.path.join(output_dir, "shampoo_val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"  训练集: {len(train)}, 验证集: {len(val)}")
    print(f"  已保存到: {output_dir}")
    return train_path, val_path
