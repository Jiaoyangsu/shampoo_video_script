import unittest

from src.data.rewrite import retime_script, user_prompt, validate_rewrite
from src.script_schema import validate_script
from tests.helpers import make_script


class RewriteTests(unittest.TestCase):
    def test_prompt_has_hard_constraints(self):
        prompt = user_prompt("海飞丝去屑洗发水", target_duration=30)
        for phrase in ["10-12", "00:00", "4 字/秒", "JSON"]:
            self.assertIn(phrase, prompt)

    def test_retime_fits_30s_and_respects_pacing(self):
        script = make_script(10)
        # Make voiceovers deliberately long; retime should expand the timeline.
        for shot in script:
            shot["voiceover"] = "这是一段比较长的口播文案"
        retimed = retime_script(script, target_duration=30, max_chars_per_second=4.0)
        self.assertIsNotNone(retimed)
        self.assertEqual(len(retimed), 10)
        result = validate_script(
            retimed, min_shots=10, require_music=True, enforce_pacing=True
        )
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.metrics["total_duration_sec"], 30)

    def test_retime_rejects_impossible_script(self):
        script = make_script(10)
        for shot in script:
            shot["voiceover"] = "超" * 80
        self.assertIsNone(retime_script(script, target_duration=30))

    def test_validate_rejects_invented_brand(self):
        script = make_script(10)
        for i, shot in enumerate(script):
            shot["shot_type"] = ["产品引入", "痛点展示", "产品概述", "产品卖点", "使用演示"][i % 5]
            shot["time_axis"] = "%02d:%02d-%02d:%02d" % (i * 3 // 60, i * 3 % 60, (i * 3 + 3) // 60, (i * 3 + 3) % 60)
        script[0]["voiceover"] = "贝德美很好用"
        result = validate_rewrite(script, "海飞丝去屑洗发水")
        self.assertFalse(result.ok)
        self.assertTrue(any("invented brand" in e for e in result.errors))


if __name__ == "__main__":
    unittest.main()
