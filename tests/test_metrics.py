import json
import unittest

from src.evaluation.metrics import score_output, summarize
from tests.helpers import make_script


class MetricsTests(unittest.TestCase):
    def spec(self, **overrides):
        spec = {
            "id": "P01",
            "category": "品牌/去屑",
            "prompt": "test",
            "expected_brand": "海飞丝",
            "required_keywords": ["去屑", "头皮"],
            "forbidden_brands": ["贝德美"],
            "min_shots": 10,
            "target_duration_sec": [20, 45],
        }
        spec.update(overrides)
        return spec

    def test_valid_output(self):
        script = make_script(10)
        script[0]["voiceover"] = "海飞丝去屑洗发水护理头皮"
        raw = json.dumps(script, ensure_ascii=False)
        score = score_output(self.spec(), raw, finish_reason="stop", latency_s=1.2)
        self.assertTrue(score["json_valid"])
        self.assertTrue(score["schema_valid"])
        self.assertTrue(score["complete_script"])
        self.assertFalse(score["truncated"])
        self.assertEqual(score["forbidden_brand_hits"], [])
        self.assertTrue(score["expected_brand_mentioned"])

    def test_truncated_output_is_not_complete(self):
        raw = json.dumps(make_script(10), ensure_ascii=False)[:100]
        score = score_output(self.spec(), raw, finish_reason="length")
        self.assertFalse(score["json_valid"])
        self.assertFalse(score["complete_script"])
        self.assertTrue(score["truncated"])

    def test_forbidden_brand(self):
        script = make_script(10)
        script[0]["voiceover"] = "这款贝德美很好用"
        raw = json.dumps(script, ensure_ascii=False)
        score = score_output(self.spec(), raw)
        self.assertEqual(score["forbidden_brand_hits"], ["贝德美"])

    def test_summary(self):
        script = make_script(10)
        raw = json.dumps(script, ensure_ascii=False)
        records = [score_output(self.spec(), raw), score_output(self.spec(), raw[:50])]
        summary = summarize(records)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["json_valid_rate"], 0.5)
        self.assertEqual(summary["by_category"]["品牌/去屑"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
