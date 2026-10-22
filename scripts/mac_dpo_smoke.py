#!/usr/bin/env python3
"""Small-model DPO smoke test on Apple Silicon (MPS + TRL).

This is a pipeline check, not the 12B Gemma run.  It mirrors the project's
SFT-then-DPO composition: TRL merges the SFT adapter, trains a new DPO LoRA,
and uses the merged SFT model as reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.backends import read_system_prompt  # noqa: E402


def format_dpo(rows, tokenizer, system_prompt):
    prompts, chosen, rejected = [], [], []
    for row in rows:
        prompts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["input"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        chosen.append(json.dumps(row["chosen"], ensure_ascii=False) + tokenizer.eos_token)
        rejected.append(json.dumps(row["rejected"], ensure_ascii=False) + tokenizer.eos_token)
    return prompts, chosen, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft-adapter", default="outputs/mac_sft_05b")
    parser.add_argument("--train", default="data/rewrite_30s/dpo_train.json")
    parser.add_argument("--val", default="data/rewrite_30s/dpo_val.json")
    parser.add_argument("--output", default="outputs/mac_dpo_05b")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    system_prompt = read_system_prompt()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = json.loads(Path(args.train).read_text(encoding="utf-8"))
    val_rows = json.loads(Path(args.val).read_text(encoding="utf-8"))
    tr_p, tr_c, tr_r = format_dpo(train_rows, tokenizer, system_prompt)
    va_p, va_c, va_r = format_dpo(val_rows, tokenizer, system_prompt)
    train_ds = Dataset.from_dict({"prompt": tr_p, "chosen": tr_c, "rejected": tr_r})
    val_ds = Dataset.from_dict({"prompt": va_p, "chosen": va_c, "rejected": va_r})
    print("pairs", len(train_ds), len(val_ds), flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to("mps")
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    cfg = DPOConfig(
        output_dir=args.output,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        beta=args.beta,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        use_cpu=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        precompute_ref_log_probs=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("saved", args.output)


if __name__ == "__main__":
    main()
