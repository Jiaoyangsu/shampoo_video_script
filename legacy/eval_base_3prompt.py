import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master"

SYSTEM_PROMPT = "你是一位经验丰富的带货短视频导演。你的任务是分析用户指令，生成可直接用于拍摄的高质量带货短视频分镜脚本。\n脚本中的每个镜头必须包含：time_axis（时间轴）、shot_type（镜头类型）、visual_description（画面描述）、voiceover（口播）、music（背景音乐）。\n输出格式为 JSON 列表。只输出 JSON，禁止任何额外文本。"

test_prompts = [
    "帮我生成一个海飞丝洗发水的带货短视频分镜脚本，核心卖点是去屑，需要展示头皮清洁的痛点。",
    "设计一个婴儿洗发水视频脚本，强调无泪配方和温和，妈妈放心。",
    "做一个控油蓬松男士洗发水的带货脚本，目标用户是油性头皮、发量扁塌的上班族，预算有限，需要突出性价比和即时效果。",
]


def strip_fence(text):
    m = re.search(r"```json\s*(.*?)\s*```", text, re.S)
    return m.group(1) if m else text


def try_parse(text):
    js = strip_fence(text)
    try:
        return json.loads(js), js
    except json.JSONDecodeError:
        return None, js


print("===== MODEL: BASE (no adapter) =====")
tokenizer = AutoTokenizer.from_pretrained(BASE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Loading base (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
)
model.eval()

for p in test_prompts:
    text = f"<bos><start_of_turn>user\n{SYSTEM_PROMPT}\n\n{p}<end_of_turn>\n<start_of_turn>model\n"
    ids = tokenizer(text, return_tensors="pt").to("cuda")
    torch.cuda.empty_cache()
    with torch.no_grad():
        out = model.generate(
            ids["input_ids"],
            max_new_tokens=700,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
        )
    raw = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    raw = raw.replace("<end_of_turn>", "").replace("<eos>", "").strip()
    print("\n" + "=" * 70)
    print(f"PROMPT: {p}")
    print(f"RAW:")
    print(raw)
    print("-" * 70)
    parsed, js = try_parse(raw)
    if parsed is not None:
        print(f"JSON VALID: True, shots={len(parsed)}")
        for s in parsed:
            print(f"  [{s.get('time_axis')}] {s.get('shot_type')} | {s.get('visual_description','')[:38]}... | VO: {s.get('voiceover','')[:26]}")
    else:
        n = len(re.findall(r"\{[^{}]*\}", js))
        print(f"JSON VALID: False, complete_shot_objs={n}")
    print("=" * 70)

print("ALL_TEST_DONE")
