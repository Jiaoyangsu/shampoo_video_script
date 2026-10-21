"""
为洗发水分镜生成 DPO 偏好数据对
chosen = 原始训练数据中的好脚本
rejected = 故意注入缺陷的坏脚本（镜头类型错误/时间轴断裂/缺字段/顺序乱）
"""
import json
import random
import copy

random.seed(3407)

SHOT_TYPES = ["产品引入", "痛点展示", "产品概述", "产品亮点", "使用演示", "产品促销"]


def build_prompt(input_text: str, system_prompt: str) -> str:
    return system_prompt + f"\n\n### Instruction:\n{input_text}\n\n### Response:\n"


def inject_defects(output: list, defect_type: str = None) -> list:
    """对脚本注入特定缺陷，生成 rejected 版本"""
    out = copy.deepcopy(output)
    if not out:
        return out

    defect = defect_type or random.choice([
        "wrong_type", "broken_time", "empty_voiceover", "shuffle",
    ])

    if defect == "wrong_type":
        bad_types = ["产品特写", "广告插入", "品牌介绍", "用户访谈", "黑屏转场"]
        for shot in out:
            if random.random() < 0.4:
                shot["shot_type"] = random.choice(bad_types)

    elif defect == "broken_time":
        if len(out) >= 2:
            i = random.randint(0, len(out) - 2)
            out[i]["time_axis"] = "00:30-00:35"
            out[i + 1]["time_axis"] = "00:02-00:06"

    elif defect == "empty_voiceover":
        for shot in out:
            if random.random() < 0.5:
                shot["voiceover"] = ""

    elif defect == "shuffle":
        shot_types = [s["shot_type"] for s in out]
        random.shuffle(shot_types)
        for i, s in enumerate(out):
            s["shot_type"] = shot_types[i]

    return out


def main():
    with open("/Users/jiaoyangsu/Desktop/VideoScriptAI/finetune/datasets/shampoo_all.json") as f:
        data = json.load(f)

    dpo_pairs = []
    for item in data:
        prompt = build_prompt(item["input"], "")
        chosen = json.dumps(item["output"], ensure_ascii=False, indent=2)

        defect = random.choice(["wrong_type", "broken_time", "empty_voiceover", "shuffle"])
        rejected = json.dumps(
            inject_defects(item["output"], defect),
            ensure_ascii=False, indent=2,
        )

        dpo_pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    random.shuffle(dpo_pairs)
    split = int(len(dpo_pairs) * 0.9)

    train = dpo_pairs[:split]
    val = dpo_pairs[split:]

    out_dir = "/Users/jiaoyangsu/Desktop/VideoScriptAI/finetune/datasets"
    with open(f"{out_dir}/dpo_train.json", "w") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(f"{out_dir}/dpo_val.json", "w") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"DPO 数据准备完成")
    print(f"  训练集: {len(train)} 对")
    print(f"  验证集: {len(val)} 对")
    print(f"\n缺陷类型示例:")
    for i in range(3):
        print(f"\nPrompt: {data[i]['input'][:60]}...")
        print(f"Chosen: {json.dumps(data[i]['output'], ensure_ascii=False, indent=2)[:200]}...")
        rejected = inject_defects(data[i]["output"])
        print(f"Rejected ({'缺陷'}): {json.dumps(rejected, ensure_ascii=False, indent=2)[:200]}...")


if __name__ == "__main__":
    main()
