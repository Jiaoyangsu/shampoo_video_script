#!/usr/bin/env python3
"""Merge SFT + DPO LoRA adapters into one deployable model.

This is the canonical way to build the model that ``eval_compare.py`` used to
*claim* it was evaluating.  Order matters: base -> SFT -> merge -> DPO -> merge.

The merge is memory hungry.  For Gemma 3 12B in bf16 you need roughly 24GB for
the weights plus merge buffers; if the 3090 OOMs, run on CPU with enough RAM or
use the 4B model.  For 24GB GPUs, prefer loading the two adapters separately at
inference time (``TransformersBackend`` supports that).

Example:
    python scripts/merge_adapters.py \
      --base-model /root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master \
      --sft-adapter /root/autodl-tmp/outputs/gemma3_sft/final \
      --dpo-adapter /root/autodl-tmp/outputs/gemma3_dpo/final/dpo_adapter \
      --output /root/autodl-tmp/models/shampoo_sft_dpo_merged \
      --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--dpo-adapter", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, args.dtype)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer from", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base in", args.dtype, "on", args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    print("Applying SFT adapter:", args.sft_adapter)
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()

    if args.dpo_adapter:
        print("Applying DPO adapter on top of merged SFT:", args.dpo_adapter)
        model = PeftModel.from_pretrained(model, args.dpo_adapter)
        model = model.merge_and_unload()

    print("Saving merged model to", output)
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    (output / "merge_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_model": args.base_model,
                "sft_adapter": args.sft_adapter,
                "dpo_adapter": args.dpo_adapter,
                "dtype": args.dtype,
                "device": args.device,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Done. Use this directory as SCRIPT_MODEL / --merged-model.")


if __name__ == "__main__":
    main()
