#!/usr/bin/env python3
"""Small-model SFT smoke test on Apple Silicon (MPS + TRL).

This is NOT the 12B Gemma run.  It validates the data/prompt/loss path on a
0.5B model when no CUDA GPU is available.
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


def build_text(row, tokenizer, system_prompt):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["input"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    answer = json.dumps(row["output"], ensure_ascii=False) + tokenizer.eos_token
    return prompt + answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train", default="data/rewrite_30s/train.json")
    parser.add_argument("--output", default="outputs/mac_sft_05b")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    system_prompt = read_system_prompt()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = json.loads(Path(args.train).read_text(encoding="utf-8"))
    data = [
        {"text": build_text(row, tokenizer, system_prompt)}
        for row in rows
    ]
    dataset = Dataset.from_list(data)
    print("rows", len(dataset), flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to("mps")
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    cfg = SFTConfig(
        output_dir=args.output,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        use_cpu=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_text_field="text",
        completion_only_loss=False,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("saved", args.output)


if __name__ == "__main__":
    main()
