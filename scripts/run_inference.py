#!/usr/bin/env python3
"""One-off inference CLI.

Examples
--------
python scripts/run_inference.py --merged-model /path/to/merged --prompt "..."
python scripts/run_inference.py --base-model /path/to/base \
  --sft-adapter /path/to/sft --dpo-adapter /path/to/dpo
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.generate import ScriptGenerator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-model")
    parser.add_argument("--base-model")
    parser.add_argument("--sft-adapter")
    parser.add_argument("--dpo-adapter")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="repeatable; defaults to two shampoo examples",
    )
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    generator = ScriptGenerator(
        base_model=args.base_model,
        merged_model=args.merged_model,
        sft_adapter=args.sft_adapter,
        dpo_adapter=args.dpo_adapter,
        use_cache=args.use_cache,
        max_new_tokens=args.max_new_tokens,
    )
    prompts = args.prompts or [
        "请为海飞丝去屑洗发水设计 30 秒带货短视频脚本，核心卖点是去屑和清凉感，目标用户是有头屑困扰的年轻人。",
        "请为霸王防脱洗发水生成 30 秒带货分镜脚本，突出头皮护理和发根强韧，适合压力大的上班族；不要做医疗功效承诺。",
    ]
    for prompt in prompts:
        print("=" * 60)
        print("Q:", prompt)
        print("=" * 60)
        result = generator.generate_json(prompt)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
