#!/usr/bin/env python3
"""
一键微调入口
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.finetune.trainer import run_finetune


if __name__ == "__main__":
    # 切换到项目根目录
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    config_path = "config/finetune_config.yaml"
    run_finetune(config_path)
