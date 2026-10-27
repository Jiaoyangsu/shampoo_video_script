import unittest

from src.finetune.gemma3.finetune_dpo import format_dpo
from src.finetune.gemma3.finetune_sft import build_example
from tests.helpers import make_script


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "CHAT::" + messages[0]["content"]

    eos_token = "<eos>"


class TrainingPromptTests(unittest.TestCase):
    def test_sft_prompt_contains_duration(self):
        example = {"input": "产品A", "output": make_script(10), "target_duration_sec": 30}
        text = build_example(example, FakeTokenizer())["text"]
        self.assertIn("目标时长：30 秒", text)
        self.assertIn("10–12", text)

    def test_dpo_uses_chat_template_not_instruction_template(self):
        examples = {
            "input": ["产品A"],
            "chosen": [make_script(10)],
            "rejected": [make_script(10)],
        }
        out = format_dpo(examples, FakeTokenizer())
        self.assertIn("CHAT::", out["prompt"][0])
        self.assertIn("产品A", out["prompt"][0])
        self.assertIn("目标时长：30 秒", out["prompt"][0])
        self.assertIn("10–12", out["prompt"][0])
        self.assertNotIn("### Instruction", out["prompt"][0])
        self.assertTrue(out["chosen"][0].endswith("<eos>"))
        self.assertTrue(out["rejected"][0].endswith("<eos>"))


if __name__ == "__main__":
    unittest.main()
