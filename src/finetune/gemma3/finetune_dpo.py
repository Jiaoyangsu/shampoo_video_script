"""Gemma 3 DPO training on top of the SFT adapter.

Important: TRL 0.14 merges the existing SFT adapter when ``peft_config`` is
passed.  The resulting DPO adapter is therefore a delta on top of *merged SFT*.
Saving only that adapter and loading it on the raw base (what the old
``eval_compare.py`` did) silently drops SFT.

This script saves both adapters plus a composition manifest:

    final/
      sft_adapter/
      dpo_adapter/
      composition.json

Use ``scripts/merge_adapters.py`` for a single merged model, or let
``TransformersBackend`` load SFT then DPO.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.finetune.config import load_config, read_system_prompt, resolve_path  # noqa: E402

SYSTEM_PROMPT = read_system_prompt()


def format_dpo(examples, tokenizer):
    prompts = []
    for item in examples["input"]:
        messages = [
            {
                "role": "user",
                "content": "%s\n\n%s" % (SYSTEM_PROMPT.strip(), item.strip()),
            }
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    chosen = [
        json.dumps(value, ensure_ascii=False) + tokenizer.eos_token
        for value in examples["chosen"]
    ]
    rejected = [
        json.dumps(value, ensure_ascii=False) + tokenizer.eos_token
        for value in examples["rejected"]
    ]
    return {"prompt": prompts, "chosen": chosen, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--sft-adapter", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    dpo_cfg = cfg["dpo"]

    base_model = args.base_model or os.environ.get("SCRIPT_BASE_MODEL") or model_cfg["base_model"]
    sft_adapter = (
        args.sft_adapter
        or os.environ.get("SCRIPT_SFT_ADAPTER")
        or str(PROJECT_ROOT / "outputs" / "gemma3_sft" / "final")
    )
    if not sft_adapter:
        raise SystemExit("--sft-adapter (or SCRIPT_SFT_ADAPTER) is required")
    if not Path(sft_adapter).exists():
        raise SystemExit("SFT adapter not found: %s" % sft_adapter)

    train_file = resolve_path(args.train_file or data_cfg["dpo_train_file"])
    val_file = resolve_path(args.val_file or data_cfg["dpo_val_file"])
    output_dir = args.output_dir or dpo_cfg["output_dir"]
    epochs = args.epochs if args.epochs is not None else int(dpo_cfg["epochs"])
    learning_rate = args.learning_rate or float(dpo_cfg["learning_rate"])
    beta = args.beta if args.beta is not None else float(dpo_cfg["beta"])
    max_length = args.max_length or int(dpo_cfg["max_length"])
    max_prompt_length = args.max_prompt_length or int(dpo_cfg["max_prompt_length"])

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    os.makedirs(output_dir, exist_ok=True)

    print("Loading tokenizer:", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("Loading base + SFT adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model = PeftModel.from_pretrained(model, sft_adapter)

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )

    print("Loading DPO data:", train_file, val_file)
    train_raw = load_dataset("json", data_files=train_file, split="train")
    val_raw = load_dataset("json", data_files=val_file, split="train")
    train_ds = train_raw.map(
        format_dpo,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=train_raw.column_names,
    )
    val_ds = val_raw.map(
        format_dpo,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=val_raw.column_names,
    )
    print("train=%d val=%d" % (len(train_ds), len(val_ds)))

    dpo_config = DPOConfig(
        per_device_train_batch_size=int(dpo_cfg["batch_size"]),
        gradient_accumulation_steps=int(dpo_cfg["grad_accum"]),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        warmup_ratio=float(dpo_cfg["warmup_ratio"]),
        weight_decay=0.01,
        logging_steps=int(dpo_cfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(dpo_cfg["eval_steps"]),
        save_strategy="steps",
        save_steps=int(dpo_cfg["save_steps"]),
        save_total_limit=int(dpo_cfg["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        lr_scheduler_type="linear",
        seed=3407,
        data_seed=3407,
        output_dir=output_dir,
        report_to="none",
        optim="adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=int(dpo_cfg["precompute_ref_batch_size"]),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    trainer.train()

    final_path = Path(output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    dpo_adapter_dir = final_path / "dpo_adapter"
    sft_adapter_dir = final_path / "sft_adapter"
    print("Saving DPO adapter to", dpo_adapter_dir)
    if dpo_adapter_dir.exists():
        shutil.rmtree(dpo_adapter_dir)
    trainer.save_model(str(dpo_adapter_dir))
    if sft_adapter_dir.exists():
        shutil.rmtree(sft_adapter_dir)
    shutil.copytree(sft_adapter, sft_adapter_dir)
    tokenizer.save_pretrained(str(final_path / "tokenizer"))
    (final_path / "composition.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_model": base_model,
                "sft_adapter": "sft_adapter",
                "dpo_adapter": "dpo_adapter",
                "load_order": ["base", "sft", "merge", "dpo"],
                "note": (
                    "The DPO adapter was trained on merged SFT weights. "
                    "Load SFT, merge, then load DPO. Do not load DPO alone on base."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved composition to", final_path)


if __name__ == "__main__":
    main()
