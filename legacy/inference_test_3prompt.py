import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master"
ADAPTER = "/root/autodl-tmp/outputs/gemma3_dpo/final"

SYSTEM_PROMPT = "你是一位经验丰富的带货短视频导演。你的任务是分析用户指令，生成可直接用于拍摄的高质量带货短视频分镜脚本。\n脚本中的每个镜头必须包含：time_axis（时间轴）、shot_type（镜头类型）、visual_description（画面描述）、voiceover（口播）、music（背景音乐）。\n输出格式为 JSON 列表。只输出 JSON，禁止任何额外文本。"

print("Loading base model (bf16)...")
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
)
print("Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

test_prompts = [
    "帮我生成一个海飞丝洗发水的带货短视频分镜脚本，核心卖点是去屑，需要展示头皮清洁的痛点。",
    "设计一个婴儿洗发水视频脚本，强调无泪配方和温和，妈妈放心。",
]

for p in test_prompts:
    text = f"<bos><start_of_turn>user\n{SYSTEM_PROMPT}\n\n{p}<end_of_turn>\n<start_of_turn>model\n"
    ids = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=700,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
        )
    gen = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    gen = gen.replace("<end_of_turn>", "").replace("<eos>", "").strip()

    print("=" * 60)
    print("PROMPT:", p)
    print("RAW OUTPUT:")
    print(gen)
    print()
    try:
        parsed = json.loads(gen)
        print("JSON VALID: True, shots:", len(parsed))
        if parsed:
            print("First shot keys:", list(parsed[0].keys()))
    except json.JSONDecodeError as e:
        print("JSON VALID: False ->", e)
    print()

print("ALL_TEST_DONE")
