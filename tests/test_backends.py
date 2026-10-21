import unittest

from src.inference.backends import MockBackend, build_chat_prompt
from src.script_schema import extract_json


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT::" + messages[0]["content"]


class BackendTests(unittest.TestCase):
    def test_mock_valid(self):
        result = MockBackend("valid").generate("sys", "user")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(len(extract_json(result.text)), 10)

    def test_mock_truncated(self):
        result = MockBackend("truncated").generate("sys", "user")
        self.assertEqual(result.finish_reason, "length")
        with self.assertRaises(Exception):
            extract_json(result.text)

    def test_chat_prompt_contains_system_and_user(self):
        prompt = build_chat_prompt(FakeTokenizer(), "SYSTEM", "USER")
        self.assertIn("SYSTEM", prompt)
        self.assertIn("USER", prompt)


if __name__ == "__main__":
    unittest.main()
