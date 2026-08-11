"""
SFT 微调模块
"""
import json
import os
import sys

import yaml
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

SYSTEM_PROMPT = """你是一位经验丰富的带货短视频导演。你的任务是分析用户指令，生成可直接用于拍摄的高质量带货短视频分镜脚本。
脚本中的每个镜头必须包含：time_axis（时间轴）、shot_type（镜头类型）、visual_description（画面描述）、voiceover（口播）、music（背景音乐）。
输出格式为 JSON 列表。只输出 JSON，禁止任何额外文本。"""


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_prompt_template(system_prompt: str) -> str:
    return system_prompt + "\n\n### Instruction:\n{INPUT}\n\n### Response:\n{OUTPUT}"


def format_prompts(examples, eos_token, template: str):
    texts = []
    for inp, out in zip(examples["input"], examples["output"]):
        out_str = json.dumps(out, ensure_ascii=False, indent=2)
        texts.append(template.replace("{INPUT}", inp).replace("{OUTPUT}", out_str) + eos_token)
    return {"text": texts}


def run_finetune(config_path: str = "../config/finetune_config.yaml"):
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    print("=" * 50)
    print(f"基础模型: {model_cfg['name']}")
    print(f"LoRA: r={lora_cfg['r']}, alpha={lora_cfg['alpha']}, dropout={lora_cfg['dropout']}")
    print(f"训练参数: batch={train_cfg['batch_size']}, "
          f"accumulation={train_cfg['gradient_accumulation']}, "
          f"epochs={train_cfg['epochs']}, lr={train_cfg['learning_rate']}")
    print(f"数据: {data_cfg['train_file']}")
    print("=" * 50)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=model_cfg["use_4bit"],
    )

    EOS_TOKEN = tokenizer.eos_token
    template = build_prompt_template(SYSTEM_PROMPT)

    train_ds = load_dataset("json", data_files=data_cfg["train_file"], split="train")
    val_ds = load_dataset("json", data_files=data_cfg["val_file"], split="train")

    train_ds = train_ds.map(format_prompts, batched=True,
                            fn_kwargs={"eos_token": EOS_TOKEN, "template": template})
    val_ds = val_ds.map(format_prompts, batched=True,
                        fn_kwargs={"eos_token": EOS_TOKEN, "template": template})

    print(f"\n训练集: {len(train_ds)} | 验证集: {len(val_ds)}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=model_cfg["max_seq_length"],
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=train_cfg["batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation"],
            num_train_epochs=train_cfg["epochs"],
            warmup_ratio=train_cfg["warmup_ratio"],
            learning_rate=train_cfg["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=train_cfg["logging_steps"],
            eval_strategy="steps",
            eval_steps=train_cfg["eval_steps"],
            save_strategy="steps",
            save_steps=train_cfg["save_steps"],
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=train_cfg["weight_decay"],
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=train_cfg["output_dir"],
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    trainer.train()

    save_path = "models/shampoo_lora"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n模型已保存: {save_path}")

    return save_path


if __name__ == "__main__":
    run_finetune()
