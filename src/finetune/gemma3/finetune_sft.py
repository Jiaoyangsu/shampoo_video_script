import json
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

MODEL_PATH = "/root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master"
DATA_DIR = "/root/sft_data"
OUTPUT_DIR = "/root/autodl-tmp/outputs/gemma3_sft"

SYSTEM_PROMPT = "你是一位经验丰富的带货短视频导演。你的任务是分析用户指令，生成可直接用于拍摄的高质量带货短视频分镜脚本。\n脚本中的每个镜头必须包含：time_axis（时间轴）、shot_type（镜头类型）、visual_description（画面描述）、voiceover（口播）、music（背景音乐）。\n输出格式为 JSON 列表。只输出 JSON，禁止任何额外文本。"

def build_example(ex):
    out_str = json.dumps(ex["output"], ensure_ascii=False)
    return {
        "text": f"<bos><start_of_turn>user\n{SYSTEM_PROMPT}\n\n{ex['input']}<end_of_turn>\n<start_of_turn>model\n{out_str}<end_of_turn><eos>"
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with QLoRA 4bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    train_ds = load_dataset("json", data_files=f"{DATA_DIR}/shampoo_train.json", split="train")
    val_ds = load_dataset("json", data_files=f"{DATA_DIR}/shampoo_val.json", split="train")
    train_ds = train_ds.map(build_example, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(build_example, remove_columns=val_ds.column_names)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    sft_config = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        optim="adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        max_seq_length=2048,
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
        formatting_func=lambda x: x["text"],
    )

    print("Starting training...")
    trainer.train()

    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
