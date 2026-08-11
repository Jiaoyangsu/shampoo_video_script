"""
洗发水带货分镜 DPO 训练
基于已微调的 LoRA 权重继续做偏好对齐
"""
import json
import os
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import DPOTrainer
from transformers import TrainingArguments

MODEL_PATH = "models/shampoo_lora"            # 已经微调过的 LoRA 权重
USE_4BIT = True
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-6
BETA = 0.1                                     # DPO 温度参数


def process_dpo_data():
    with open("data/train/dpo_train.json") as f:
        train_raw = json.load(f)
    with open("data/val/dpo_val.json") as f:
        val_raw = json.load(f)
    return train_raw, val_raw


if __name__ == "__main__":
    train_data, val_data = process_dpo_data()
    print(f"DPO 数据: {len(train_data)} 训练, {len(val_data)} 验证")

    # 加载基座模型 + LoRA 权重
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=USE_4BIT,
    )

    # DPO 需要 reference model
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=USE_4BIT,
    )

    # 添加 LoRA（跟微调时一致）
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    def tokenize_dpo(batch):
        result = tokenizer(
            batch["prompt"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH
        )
        chosen_ids = tokenizer(
            [p + c for p, c in zip(batch["prompt"], batch["chosen"])],
            padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH,
        )["input_ids"]
        rejected_ids = tokenizer(
            [p + r for p, r in zip(batch["prompt"], batch["rejected"])],
            padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH,
        )["input_ids"]
        result["chosen_input_ids"] = chosen_ids
        result["rejected_input_ids"] = rejected_ids
        return result

    train_ds = Dataset.from_list(train_data).map(tokenize_dpo, batched=True)
    val_ds = Dataset.from_list(val_data).map(tokenize_dpo, batched=True)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        beta=BETA,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            num_train_epochs=3,
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            output_dir="outputs/dpo_outputs",
            report_to="none",
            remove_unused_columns=False,
        ),
    )

    trainer.train()

    save_path = "models/shampoo_dpo"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"DPO 模型已保存: {save_path}")
