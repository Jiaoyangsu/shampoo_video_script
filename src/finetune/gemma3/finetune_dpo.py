import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

BASE = "/root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master"
SFT_ADAPTER = "/root/autodl-tmp/outputs/gemma3_sft/final"
OUTPUT_DIR = "/root/autodl-tmp/outputs/gemma3_dpo"
DATA_DIR = "/root/sft_data"

SYSTEM_PROMPT = "你是一位经验丰富的带货短视频导演。你的任务是分析用户指令，生成可直接用于拍摄的高质量带货短视频分镜脚本。\n脚本中的每个镜头必须包含：time_axis（时间轴）、shot_type（镜头类型）、visual_description（画面描述）、voiceover（口播）、music（背景音乐）。\n输出格式为 JSON 列表。只输出 JSON，禁止任何额外文本。"

BOS = "<bos>"
USER_START = "<start_of_turn>user\n"
USER_END = "<end_of_turn>\n"
MODEL_START = "<start_of_turn>model\n"
MODEL_END = "<end_of_turn>"
EOS = "<eos>"


def build_prompt(p):
    return f"{BOS}{USER_START}{SYSTEM_PROMPT}\n\n{p}{USER_END}{MODEL_START}"


def build_response(r):
    return f"{r}{MODEL_END}{EOS}"


def format_dpo(examples):
    prompts, chosen, rejected = [], [], []
    for p, c, r in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        prompts.append(build_prompt(p))
        chosen.append(build_response(c))
        rejected.append(build_response(r))
    return {"prompt": prompts, "chosen": chosen, "rejected": rejected}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (4bit QLoRA)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto", attn_implementation="eager"
    )
    print("Loading SFT adapter...")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    train_raw = load_dataset("json", data_files=f"{DATA_DIR}/dpo_train.json", split="train")
    val_raw = load_dataset("json", data_files=f"{DATA_DIR}/dpo_val.json", split="train")
    train_ds = train_raw.map(format_dpo, batched=True, remove_columns=train_raw.column_names)
    val_ds = val_raw.map(format_dpo, batched=True, remove_columns=val_raw.column_names)
    print(f"train={len(train_ds)} val={len(val_ds)}")
    print("Sample prompt:", train_ds[0]["prompt"][:120])
    print("Sample chosen:", train_ds[0]["chosen"][:120])

    dpo_config = DPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-6,
        beta=0.1,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        optim="adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        max_length=1536,
        max_prompt_length=768,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
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

    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
