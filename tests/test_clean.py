import unittest

from src.data.clean import DEFECT_TYPES, build_dpo_rows, clean_sft_rows, inject_defect
from src.script_schema import count_hard_defects
from tests.helpers import make_row, make_script


class CleanTests(unittest.TestCase):
    def test_quality_gate_filters_short_and_empty_rows(self):
        good = make_row(10)
        bad = make_row(2)
        empty = make_row(10)
        empty["output"][0]["voiceover"] = ""
        result = clean_sft_rows([good, bad, empty], min_shots=5)
        self.assertEqual(result.kept, 1, result.rejected)
        self.assertEqual(len(result.rejected), 2)

    def test_duplicate_output_is_removed(self):
        row = make_row(10)
        result = clean_sft_rows([row, row], min_shots=5)
        self.assertEqual(result.kept, 1)
        self.assertEqual(result.duplicate_rows, 1)

    def test_all_defect_types_create_more_hard_defects(self):
        script = make_script(10, chars=8)
        chosen_count = count_hard_defects(script)
        import random

        for defect in DEFECT_TYPES:
            rejected = inject_defect(script, defect, random.Random(0))
            self.assertIsNotNone(rejected, defect)
            self.assertNotEqual(rejected, script, defect)
            self.assertGreater(
                count_hard_defects(rejected),
                chosen_count,
                "%s did not increase defects: %s -> %s"
                % (defect, chosen_count, count_hard_defects(rejected)),
            )

    def test_dpo_pairs_are_valid_and_different(self):
        rows = [make_row(10, "产品A"), make_row(10, "产品B")]
        pairs, stats = build_dpo_rows(rows, seed=1)
        self.assertEqual(len(pairs), 2, stats)
        for pair in pairs:
            self.assertNotEqual(pair["chosen"], pair["rejected"])
            self.assertGreater(pair["rejected_defects"], pair["chosen_defects"])
            self.assertIn(pair["defect"], DEFECT_TYPES)
            self.assertEqual(pair["chosen"], rows[pairs.index(pair)]["output"])


if __name__ == "__main__":
    unittest.main()


class LeakageTests(unittest.TestCase):
    def test_dpo_val_comes_from_sft_val(self):
        import json
        import tempfile
        from pathlib import Path

        from src.data.clean import run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [make_row(10, "train product")]
            val = [make_row(10, "val product")]
            (root / "train.json").write_text(json.dumps(train, ensure_ascii=False), encoding="utf-8")
            (root / "val.json").write_text(json.dumps(val, ensure_ascii=False), encoding="utf-8")
            report = run(
                sft_train_path=root / "train.json",
                sft_val_path=root / "val.json",
                out_dir=root / "out",
                min_shots=5,
            )
            self.assertEqual(report["dpo"]["train"]["pairs"], 1)
            self.assertEqual(report["dpo"]["val"]["pairs"], 1)
            dpo_train = json.loads((root / "out" / "dpo_train.json").read_text(encoding="utf-8"))
            dpo_val = json.loads((root / "out" / "dpo_val.json").read_text(encoding="utf-8"))
            self.assertEqual(dpo_train[0]["input"], "train product")
            self.assertEqual(dpo_val[0]["input"], "val product")
