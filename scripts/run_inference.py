#!/usr/bin/env python3
"""
一键推理入口
"""
import os
import sys
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference.generate import ScriptGenerator


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    gen = ScriptGenerator("models/shampoo_lora")

    questions = [
        "请为海飞丝去屑洗发水设计带货短视频脚本，核心卖点是去屑和清凉感，目标用户是有头屑困扰的年轻人。",
        "请为霸王防脱洗发水生成带货分镜脚本，突出中药成分和防脱生发效果，适合压力大脱发的上班族。",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        result = gen.generate_json(q)
        print(json.dumps(result, ensure_ascii=False, indent=2))
