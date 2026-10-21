"""Gemma 3 SFT QLoRA training (config-driven).

Key fixes vs the old script:
* uses ``tokenizer.apply_chat_template`` instead of hand-written ``<bos>`` /
  ``<start_of_turn>`` tokens (which could double-BOS);
* masks the prompt and trains only on the JSON answer when TRL's completion-only
  collator is available;
* runs validation during training and keeps the best checkpoint;
* reads paths/params from ``config/finetune_config.yaml`` with CLI overrides.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.finetune.config import load_config, read_system_prompt, resolve_path  # noqa: E402

SYSTEM_PROMPT = read_system_prompt()


def build_example(example, tokenizer):
    messages = [
        {
            "role": "user",
            "content": "%s\n\n%s" % (SYSTEM_PROMPT.strip(), example["input"].strip()),
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    answer = json.dumps(example["output"], ensure_ascii=False)
    return {
        "text": prompt + answer + tokenizer.eos_token,
        "prompt": prompt,
        "answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    sft_cfg = cfg["sft"]

    base_model = args.base_model or os.environ.get("SCRIPT_BASE_MODEL") or model_cfg["base_model"]
    train_file = resolve_path(args.train_file or data_cfg["train_file"])
    val_file = resolve_path(args.val_file or data_cfg["val_file"])
    output_dir = args.output_dir or sft_cfg["output_dir"]
    max_seq_length = args.max_seq_length or int(model_cfg["max_seq_length"])
    epochs = args.epochs if args.epochs is not None else int(sft_cfg["epochs"])
    batch_size = args.batch_size or int(sft_cfg["batch_size"])
    grad_accum = args.grad_accum or int(sft_cfg["grad_accum"])
    learning_rate = args.learning_rate or float(sft_cfg["learning_rate"])

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    os.makedirs(output_dir, exist_ok=True)

    print("Loading tokenizer:", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if model_cfg.get("load_in_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )

    print("Loading datasets:", train_file, val_file)
    train_ds = load_dataset("json", data_files=train_file, split="train")
    val_ds = load_dataset("json", data_files=val_file, split="train")
    train_ds = train_ds.map(
        build_example,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        build_example,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=val_ds.column_names,
    )
    print("train=%d val=%d" % (len(train_ds), len(val_ds)))

    data_collator = None
    try:
        from trl import DataCollatorForCompletionOnlyLM

        # Train only on the assistant JSON; the prompt tokens become -100.
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template="<start_of_turn>model\n", tokenizer=tokenizer
        )
    except Exception as exc:  # pragma: no cover - version dependent
        print("Completion-only collator unavailable (%s); training on full text" % exc)

    sft_config = SFTConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_ratio=float(sft_cfg["warmup_ratio"]),
        weight_decay=float(sft_cfg["weight_decay"]),
        logging_steps=int(sft_cfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(sft_cfg["eval_steps"]),
        save_strategy="steps",
        save_steps=int(sft_cfg["save_steps"]),
        save_total_limit=int(sft_cfg["save_total_limit"]),
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
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=lambda example: example["text"],
        data_collator=data_collator,
    )

    print("Starting SFT training...")
    trainer.train()
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print("Saved adapter to", final_path)


if __name__ == "__main__":
    main()
