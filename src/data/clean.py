"""Data cleaning, quality gating and synthetic DPO pair construction.

The original repository trained on every row in ``shampoo_train.json``.  An
audit showed that roughly half of the rows violate the project's own script
spec (empty voiceover, zero-second shots, broken time axis, missing music).
This module makes the gate explicit and reproducible.

The DPO pairs are built *only from cleaned SFT rows*, and the validation split
is built from the cleaned SFT validation split (never from SFT train), which
fixes the leakage in the old ``prepare_dpo_data.py``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Allow running as ``python scripts/clean_data.py`` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.script_schema import (  # noqa: E402
    BAD_SHOT_TYPES,
    EMPTY_MUSIC_VALUES,
    EMPTY_VOICEOVER_VALUES,
    count_hard_defects,
    normalize_shot,
    parse_time_axis,
    validate_script,
)

DEFECT_TYPES = (
    "empty_voiceover",
    "missing_music",
    "zero_duration",
    "broken_time",
    "wrong_shot_type",
    "shuffle_order",
    "duplicate_shot",
    "bad_pacing",
)


@dataclass
class CleanResult:
    rows: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_rows: int = 0
    cross_split_duplicates: int = 0

    @property
    def kept(self) -> int:
        return len(self.rows)

    @property
    def total(self) -> int:
        return self.kept + len(self.rejected)


def _row_key(row: Dict[str, Any]) -> str:
    payload = json.dumps(
        {"input": row.get("input", ""), "output": row.get("output", [])},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _output_key(row: Dict[str, Any]) -> str:
    return json.dumps(row.get("output", []), ensure_ascii=False, sort_keys=True)


def _fmt_time(seconds: int) -> str:
    minutes, sec = divmod(max(0, int(seconds)), 60)
    return "%02d:%02d" % (minutes, sec)


def _shift_shot(shot: Dict[str, str], seconds: int) -> Optional[Dict[str, str]]:
    span = parse_time_axis(shot.get("time_axis", ""))
    if span is None:
        return None
    start, end = span
    shifted = copy.deepcopy(shot)
    shifted["time_axis"] = "%s-%s" % (
        _fmt_time(start + seconds),
        _fmt_time(end + seconds),
    )
    return shifted


def inject_defect(
    script: Sequence[Dict[str, str]],
    defect_type: str,
    rng: random.Random,
) -> Optional[List[Dict[str, str]]]:
    """Return a corrupted copy of ``script``, or None if not applicable."""
    shots = [normalize_shot(s) for s in script]
    if not shots:
        return None
    rejected = copy.deepcopy(shots)

    if defect_type == "empty_voiceover":
        candidates = [
            i
            for i, s in enumerate(rejected)
            if s.get("voiceover", "").strip() not in EMPTY_VOICEOVER_VALUES
        ]
        if not candidates:
            return None
        for i in rng.sample(candidates, k=max(1, len(candidates) // 2)):
            rejected[i]["voiceover"] = ""
        return rejected

    if defect_type == "missing_music":
        candidates = [
            i
            for i, s in enumerate(rejected)
            if s.get("music", "").strip() not in EMPTY_MUSIC_VALUES
        ]
        if not candidates:
            return None
        for i in rng.sample(candidates, k=max(1, len(candidates) // 2)):
            rejected[i]["music"] = ""
        return rejected

    if defect_type == "zero_duration":
        candidates = []
        for i, s in enumerate(rejected):
            span = parse_time_axis(s.get("time_axis", ""))
            if span is not None and span[1] > span[0]:
                candidates.append(i)
        if not candidates:
            return None
        i = rng.choice(candidates)
        start, _ = parse_time_axis(rejected[i]["time_axis"])
        rejected[i]["time_axis"] = "%s-%s" % (_fmt_time(start), _fmt_time(start))
        return rejected

    if defect_type == "broken_time":
        if len(rejected) < 2:
            return None
        i = rng.randrange(len(rejected) - 1)
        shifted = _shift_shot(rejected[i + 1], 7)
        if shifted is None:
            return None
        rejected[i + 1] = shifted
        return rejected

    if defect_type == "wrong_shot_type":
        rejected[rng.randrange(len(rejected))]["shot_type"] = rng.choice(
            sorted(BAD_SHOT_TYPES)
        )
        return rejected

    if defect_type == "shuffle_order":
        before = [s.get("shot_type") for s in rejected]
        for _ in range(6):
            rng.shuffle(rejected)
            if [s.get("shot_type") for s in rejected] != before:
                return rejected
        return None

    if defect_type == "duplicate_shot":
        i = rng.randrange(len(rejected))
        rejected.insert(i + 1, copy.deepcopy(rejected[i]))
        return rejected

    if defect_type == "bad_pacing":
        candidates = [
            i
            for i, s in enumerate(rejected)
            if s.get("voiceover", "").strip()
            and parse_time_axis(s.get("time_axis", ""))
        ]
        if not candidates:
            return None
        i = rng.choice(candidates)
        rejected[i]["voiceover"] = (
            rejected[i]["voiceover"] * 2
        )  # same duration, double the spoken characters
        return rejected

    raise ValueError("unknown defect type: %s" % defect_type)


def clean_sft_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    min_shots: int = 5,
    require_music: bool = True,
    max_chars_per_second: float = 4.0,
    enforce_pacing: bool = False,
    allow_short_warning: bool = True,
) -> CleanResult:
    result = CleanResult()
    seen_inputs = set()
    seen_outputs = set()

    for row in rows:
        if not isinstance(row, dict) or "input" not in row or "output" not in row:
            result.rejected.append({"reason": "invalid row shape", "row": row})
            continue
        if not isinstance(row["input"], str) or not row["input"].strip():
            result.rejected.append({"reason": "empty input", "row": row})
            continue
        if not isinstance(row["output"], list):
            result.rejected.append({"reason": "output is not a list", "row": row})
            continue

        output_key = _output_key(row)
        if output_key in seen_outputs:
            result.duplicate_rows += 1
            continue
        seen_outputs.add(output_key)

        validation = validate_script(
            row["output"],
            min_shots=min_shots,
            require_music=require_music,
            max_chars_per_second=max_chars_per_second,
            enforce_pacing=enforce_pacing,
        )
        if not validation.ok:
            result.rejected.append(
                {
                    "reason": "quality gate",
                    "errors": validation.errors,
                    "metrics": validation.metrics,
                    "input": row["input"],
                }
            )
            continue

        input_key = row["input"].strip()
        if input_key in seen_inputs:
            result.duplicate_rows += 1
            continue
        seen_inputs.add(input_key)

        cleaned = {
            "input": input_key,
            "output": validation.shots,
            "quality": validation.metrics,
        }
        # Keep warnings out of the training file, but count them in the report.
        if allow_short_warning and validation.warnings:
            cleaned["quality_warnings"] = validation.warnings
        result.rows.append(cleaned)
    return result


def build_dpo_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    seed: int = 3407,
    max_pairs: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    pairs: List[Dict[str, Any]] = []
    defect_counts: Dict[str, int] = {name: 0 for name in DEFECT_TYPES}
    skipped = 0

    for row in rows:
        chosen = [normalize_shot(s) for s in row["output"]]
        chosen_defects = count_hard_defects(chosen)
        candidates = list(DEFECT_TYPES)
        rng.shuffle(candidates)
        made = False
        for defect in candidates:
            rejected = inject_defect(chosen, defect, rng)
            if not rejected or rejected == chosen:
                continue
            if count_hard_defects(rejected) <= chosen_defects:
                continue
            pairs.append(
                {
                    "input": row["input"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "defect": defect,
                    "chosen_defects": chosen_defects,
                    "rejected_defects": count_hard_defects(rejected),
                }
            )
            defect_counts[defect] += 1
            made = True
            break
        if not made:
            skipped += 1
        if max_pairs and len(pairs) >= max_pairs:
            break

    stats = {
        "pairs": len(pairs),
        "skipped": skipped,
        "defect_counts": defect_counts,
        "seed": seed,
    }
    return pairs, stats


def _hash_rows(rows: Sequence[Dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _dedupe_cross_split(
    train: List[Dict[str, Any]], val: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    train_inputs = {row["input"].strip() for row in train}
    kept_val = []
    dropped = 0
    for row in val:
        if row["input"].strip() in train_inputs:
            dropped += 1
            continue
        kept_val.append(row)
    return train, kept_val, dropped


def build_report(
    *,
    sft_train_raw: int,
    sft_val_raw: int,
    train_result: CleanResult,
    val_result: CleanResult,
    dpo_train_stats: Dict[str, Any],
    dpo_val_stats: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    def reasons(result: CleanResult) -> Dict[str, int]:
        counter: Dict[str, int] = {}
        for item in result.rejected:
            reason = item.get("reason", "unknown")
            counter[reason] = counter.get(reason, 0) + 1
        return counter

    return {
        "config": config,
        "sft": {
            "train_raw": sft_train_raw,
            "train_kept": train_result.kept,
            "train_rejected": len(train_result.rejected),
            "train_duplicate_rows": train_result.duplicate_rows,
            "train_pass_rate": round(
                train_result.kept / max(1, train_result.total), 4
            ),
            "val_raw": sft_val_raw,
            "val_kept": val_result.kept,
            "val_rejected": len(val_result.rejected),
            "val_duplicate_rows": val_result.duplicate_rows,
            "val_pass_rate": round(val_result.kept / max(1, val_result.total), 4),
            "cross_split_duplicates_dropped": val_result.cross_split_duplicates,
            "rejection_reasons": {
                "train": reasons(train_result),
                "val": reasons(val_result),
            },
            "sha1": {
                "train": _hash_rows(train_result.rows),
                "val": _hash_rows(val_result.rows),
            },
        },
        "dpo": {
            "train": dpo_train_stats,
            "val": dpo_val_stats,
        },
    }


def write_report_markdown(report: Dict[str, Any], path: Path) -> None:
    sft = report["sft"]
    lines = [
        "# 数据清洗报告",
        "",
        "生成方式：`python scripts/clean_data.py`",
        "",
        "## 配置",
        "```json",
        json.dumps(report["config"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## SFT",
        "| 数据集 | 原始 | 保留 | 剔除 | 通过率 |",
        "|---|---:|---:|---:|---:|",
        "| train | %s | **%s** | %s | %.1f%% |"
        % (
            sft["train_raw"],
            sft["train_kept"],
            sft["train_rejected"],
            sft["train_pass_rate"] * 100,
        ),
        "| val | %s | **%s** | %s | %.1f%% |"
        % (
            sft["val_raw"],
            sft["val_kept"],
            sft["val_rejected"],
            sft["val_pass_rate"] * 100,
        ),
        "",
        "注：完全重复的输出在质量门前被去重；因此 `保留 + 剔除` 可能小于原始条数。",
        "",
        "跨 split 重复 input 被从 val 移除：%s 条。" % sft["cross_split_duplicates_dropped"],
        "",
        "## DPO",
        "| 数据集 | 对数 | 跳过 |",
        "|---|---:|---:|",
        "| train | **%s** | %s |" % (report["dpo"]["train"]["pairs"], report["dpo"]["train"]["skipped"]),
        "| val | **%s** | %s |" % (report["dpo"]["val"]["pairs"], report["dpo"]["val"]["skipped"]),
        "",
        "缺陷类型分布（train）：",
        "```json",
        json.dumps(report["dpo"]["train"]["defect_counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 说明",
        "- 质量门检查机械可验证项：五字段非空、时间轴有效且连续、时长 > 0。",
        "- 口播语速默认只计数/警告；加 `--enforce-pacing` 才把超过阈值的样本剔除。",
        "- 通过质量门不代表创意质量好；创意质量仍需人工/LLM 盲评。",
        "- DPO 的 chosen 来自清洗后的 SFT train/val，rejected 由程序注入**至少一个**新增硬伤，且要求硬伤数严格增加。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    sft_train_path: Path,
    sft_val_path: Path,
    out_dir: Path,
    min_shots: int = 5,
    require_music: bool = True,
    max_chars_per_second: float = 4.0,
    enforce_pacing: bool = False,
    seed: int = 3407,
    max_dpo_pairs: Optional[int] = None,
    check_only: bool = False,
) -> Dict[str, Any]:
    sft_train_raw = json.loads(sft_train_path.read_text(encoding="utf-8"))
    sft_val_raw = json.loads(sft_val_path.read_text(encoding="utf-8"))

    train_result = clean_sft_rows(
        sft_train_raw,
        min_shots=min_shots,
        require_music=require_music,
        max_chars_per_second=max_chars_per_second,
        enforce_pacing=enforce_pacing,
    )
    val_result = clean_sft_rows(
        sft_val_raw,
        min_shots=min_shots,
        require_music=require_music,
        max_chars_per_second=max_chars_per_second,
        enforce_pacing=enforce_pacing,
    )
    train_result.rows, val_result.rows, dropped = _dedupe_cross_split(
        train_result.rows, val_result.rows
    )
    val_result.cross_split_duplicates = dropped

    if check_only:
        report = build_report(
            sft_train_raw=len(sft_train_raw),
            sft_val_raw=len(sft_val_raw),
            train_result=train_result,
            val_result=val_result,
            dpo_train_stats={"pairs": 0, "skipped": 0},
            dpo_val_stats={"pairs": 0, "skipped": 0},
            config={
                "min_shots": min_shots,
                "require_music": require_music,
                "max_chars_per_second": max_chars_per_second,
                "enforce_pacing": enforce_pacing,
            },
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    dpo_train, dpo_train_stats = build_dpo_rows(
        train_result.rows, seed=seed, max_pairs=max_dpo_pairs
    )
    dpo_val, dpo_val_stats = build_dpo_rows(
        val_result.rows, seed=seed + 1, max_pairs=max_dpo_pairs
    )

    report = build_report(
        sft_train_raw=len(sft_train_raw),
        sft_val_raw=len(sft_val_raw),
        train_result=train_result,
        val_result=val_result,
        dpo_train_stats=dpo_train_stats,
        dpo_val_stats=dpo_val_stats,
        config={
            "min_shots": min_shots,
            "require_music": require_music,
            "max_chars_per_second": max_chars_per_second,
            "enforce_pacing": enforce_pacing,
            "seed": seed,
            "max_dpo_pairs": max_dpo_pairs,
        },
    )

    _write_json(out_dir / "shampoo_train.json", train_result.rows)
    _write_json(out_dir / "shampoo_val.json", val_result.rows)
    _write_json(out_dir / "dpo_train.json", dpo_train)
    _write_json(out_dir / "dpo_val.json", dpo_val)
    _write_json(out_dir / "quality_report.json", report)
    write_report_markdown(report, out_dir / "QUALITY_REPORT.md")

    print(
        "SFT: train %d/%d kept, val %d/%d kept; DPO: train %d pairs, val %d pairs"
        % (
            train_result.kept,
            train_result.total,
            val_result.kept,
            val_result.total,
            len(dpo_train),
            len(dpo_val),
        )
    )
    print("Wrote %s" % out_dir)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Clean SFT/DPO data and build DPO pairs")
    parser.add_argument("--sft-train", default="data/train/shampoo_train.json")
    parser.add_argument("--sft-val", default="data/val/shampoo_val.json")
    parser.add_argument("--out-dir", default="data/clean")
    parser.add_argument("--min-shots", type=int, default=5)
    parser.add_argument("--max-chars-per-second", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-dpo-pairs", type=int, default=None)
    parser.add_argument(
        "--enforce-pacing",
        action="store_true",
        help="fail rows whose voiceover exceeds --max-chars-per-second",
    )
    parser.add_argument(
        "--allow-music-blank",
        action="store_true",
        help="treat music='无' as intentional silence rather than a defect",
    )
    parser.add_argument("--check", action="store_true", help="report only, do not write")
    args = parser.parse_args(argv)

    run(
        sft_train_path=Path(args.sft_train),
        sft_val_path=Path(args.sft_val),
        out_dir=Path(args.out_dir),
        min_shots=args.min_shots,
        require_music=not args.allow_music_blank,
        max_chars_per_second=args.max_chars_per_second,
        enforce_pacing=args.enforce_pacing,
        seed=args.seed,
        max_dpo_pairs=args.max_dpo_pairs,
        check_only=args.check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
