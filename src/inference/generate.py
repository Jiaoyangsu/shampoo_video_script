"""
模型推理模块
"""
import json
from unsloth import FastLanguageModel

SYSTEM_PROMPT = """你是一位经验丰富的带货短视频导演。你的任务是分析用户指令，生成可直接用于拍摄的高质量带货短视频分镜脚本。
脚本中的每个镜头必须包含：time_axis（时间轴）、shot_type（镜头类型）、visual_description（画面描述）、voiceover（口播）、music（背景音乐）。
输出格式为 JSON 列表。只输出 JSON，禁止任何额外文本。"""


class ScriptGenerator:
    def __init__(self, model_path: str, max_seq_length: int = 2048):
        print(f"加载模型: {model_path}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(self.model)
        self.prompt_template = SYSTEM_PROMPT + "\n\n### Instruction:\n{INPUT}\n\n### Response:\n"
        self.max_seq_length = max_seq_length

    def generate(self, product_input: str, max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_p: float = 0.95) -> str:
        prompt = self.prompt_template.replace("{INPUT}", product_input)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = self.tokenizer.batch_decode(outputs)[0]
        if "### Response:" in response:
            response = response.split("### Response:")[1]
        return response.strip()

    def generate_json(self, product_input: str, **kwargs) -> list:
        raw = self.generate(product_input, **kwargs)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise


if __name__ == "__main__":
    gen = ScriptGenerator("../models/shampoo_lora")

    test_cases = [
        "请为海飞丝去屑洗发水设计带货短视频脚本，核心卖点是去屑和清凉感，目标用户是有头屑困扰的年轻人。",
        "请为霸王防脱洗发水生成带货分镜脚本，突出中药成分和防脱生发效果，适合压力大脱发的上班族。",
    ]
    for q in test_cases:
        print(f"\n{'='*60}")
        print(f"问题: {q[:60]}...")
        print(f"{'='*60}")
        result = gen.generate_json(q)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:800])
