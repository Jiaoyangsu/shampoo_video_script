#!/usr/bin/env python3
"""Run the Gemma 3 SFT and/or DPO scripts.

The old entry point imported the legacy unsloth ``trainer.py`` and a config
whose base model was DeepSeek-R1-Distill-Qwen-1.5B.  This dispatcher runs the
scripts that actually match the documented Gemma 3 pipeline.

Examples:
    python scripts/run_finetune.py --stage sft --base-model /path/to/gemma3
    python scripts/run_finetune.py --stage dpo --sft-adapter outputs/gemma3_sft/final
    python scripts/run_finetune.py --stage all --base-model /path/to/gemma3
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "sft": PROJECT_ROOT / "src" / "finetune" / "gemma3" / "finetune_sft.py",
    "dpo": PROJECT_ROOT / "src" / "finetune" / "gemma3" / "finetune_dpo.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=["sft", "dpo", "all"], default="sft")
    args, passthrough = parser.parse_known_args()

    stages = ["sft", "dpo"] if args.stage == "all" else [args.stage]
    for stage in stages:
        stage_args = list(passthrough)
        if stage == "sft":
            # SFT does not use adapter flags; strip them so `--stage all` works.
            filtered = []
            skip_next = False
            for item in stage_args:
                if skip_next:
                    skip_next = False
                    continue
                if item in {"--sft-adapter", "--dpo-adapter"}:
                    skip_next = True
                    continue
                if item.startswith(("--sft-adapter=", "--dpo-adapter=")):
                    continue
                filtered.append(item)
            stage_args = filtered
        command = [sys.executable, str(SCRIPTS[stage])] + stage_args
        print("=" * 70)
        print("Running", stage, ":", " ".join(command))
        print("=" * 70)
        result = subprocess.run(command, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
