import unittest

from src.script_schema import (
    ScriptValidationError,
    extract_json,
    validate_script,
)
from tests.helpers import make_script


class SchemaTests(unittest.TestCase):
    def test_valid_script_passes(self):
        result = validate_script(make_script(10), min_shots=10)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.metrics["n_shots"], 10)
        self.assertEqual(result.metrics["gaps"], 0)
        self.assertEqual(result.metrics["overlaps"], 0)

    def test_bare_fenced_and_prose_json(self):
        script = make_script(10)
        import json

        raw = json.dumps(script, ensure_ascii=False)
        self.assertEqual(len(extract_json(raw)), 10)
        self.assertEqual(len(extract_json("```json\n%s\n```" % raw)), 10)
        self.assertEqual(len(extract_json("这是脚本：\n%s\n谢谢" % raw)), 10)

    def test_truncated_json_is_rejected(self):
        import json

        raw = json.dumps(make_script(10), ensure_ascii=False)
        with self.assertRaises(ScriptValidationError):
            extract_json(raw[: len(raw) // 2])

    def test_zero_duration_and_gap_are_errors(self):
        script = make_script(10)
        script[2]["time_axis"] = "00:06-00:06"
        script[3]["time_axis"] = "00:12-00:15"
        result = validate_script(script, min_shots=5)
        self.assertFalse(result.ok)
        self.assertGreater(result.metrics["zero_duration"], 0)
        self.assertGreater(result.metrics["gaps"], 0)

    def test_empty_voiceover_and_music_are_errors(self):
        script = make_script(10)
        script[0]["voiceover"] = "（无）"
        script[1]["music"] = "无"
        result = validate_script(script, min_shots=5)
        self.assertFalse(result.ok)
        self.assertGreater(result.metrics["empty_voiceover"], 0)
        self.assertGreater(result.metrics["empty_music"], 0)

    def test_pacing_warning_can_be_non_fatal(self):
        script = make_script(10)
        script[0]["voiceover"] = "这" * 20  # 20 chars / 3 sec
        result = validate_script(
            script,
            min_shots=5,
            max_chars_per_second=4.0,
            enforce_pacing=False,
        )
        self.assertTrue(result.ok, result.errors)
        self.assertGreater(result.metrics["pacing_violations"], 0)


if __name__ == "__main__":
    unittest.main()
