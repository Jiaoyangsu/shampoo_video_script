import json
import unittest
from pathlib import Path

from src.evaluation.metrics import load_prompts


class PromptTests(unittest.TestCase):
    def test_prompt_file_is_valid(self):
        path = Path("evaluation/prompts_30.json")
        prompts = load_prompts(path)
        self.assertEqual(len(prompts), 30)
        self.assertEqual(len({p["id"] for p in prompts}), 30)
        for prompt in prompts:
            self.assertTrue(prompt["required_keywords"])
            self.assertEqual(prompt["min_shots"], 10)
            if prompt.get("expected_brand"):
                self.assertNotIn(prompt["expected_brand"], prompt["forbidden_brands"])

    def test_system_prompt_is_short_enough(self):
        text = Path("src/app/system_prompt.md").read_text(encoding="utf-8")
        self.assertLess(len(text), 1200)

    def test_prompt_file_is_json(self):
        data = json.loads(Path("evaluation/prompts_30.json").read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)


if __name__ == "__main__":
    unittest.main()
