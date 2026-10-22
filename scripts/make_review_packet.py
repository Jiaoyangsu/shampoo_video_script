#!/usr/bin/env python3
"""Create a small, stratified, blind review packet from the rewrite output."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_flags(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["id"]: row.get("flags", "") for row in csv.DictReader(handle)}


def pick(rows: List[Dict[str, Any]], flags: Dict[str, str], quotas: Dict[str, int], rng: random.Random):
    buckets = {"clean": [], "generic": [], "short": [], "other": []}
    for row in rows:
        flag = flags.get(row["id"], "")
        if not flag:
            buckets["clean"].append(row)
        elif flag == "generic_cta":
            buckets["generic"].append(row)
        elif flag == "short_visual":
            buckets["short"].append(row)
        else:
            buckets["other"].append(row)
    selected = []
    for name, count in quotas.items():
        pool = buckets.get(name, [])
        rng.shuffle(pool)
        selected.extend(pool[:count])
    # If a bucket is short, fill from the remaining pool.
    remaining = [row for row in rows if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sum(quotas.values()) - len(selected))])
    return selected


def shot_block(shot: Dict[str, str], index: int) -> str:
    return (
        "**%d. `%s` — %s**\n\n"
        "- 画面：%s\n"
        "- 口播：%s\n"
        "- 音乐：%s\n"
        % (
            index,
            shot.get("time_axis", ""),
            shot.get("shot_type", ""),
            shot.get("visual_description", ""),
            shot.get("voiceover", ""),
            shot.get("music", ""),
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/rewrite_30s")
    parser.add_argument("--out", default="data/rewrite_30s/review_packet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    root = Path(args.dir)
    rng = random.Random(args.seed)

    selected = []
    selections = [
        ("train", {"clean": 4, "generic": 3, "short": 2, "other": 1}),
        ("val", {"clean": 2, "generic": 2, "short": 1, "other": 0}),
    ]
    for split, quotas in selections:
        rows = read_jsonl(root / ("%s_accepted.jsonl" % split))
        flags = read_flags(root / ("%s_review.csv" % split))
        for row in pick(rows, flags, quotas, rng):
            selected.append((split, row, flags.get(row["id"], "")))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md_path = out.with_suffix(".md")
    csv_path = out.with_suffix(".csv")
    flags_path = out.parent / (out.name + "_flags.csv")

    lines = [
        "# Rewrite 30s — 抽检包",
        "",
        "共 %d 条：10 条 train + 5 条 val。**先盲评，不要先看 flags。**" % len(selected),
        "",
        "评分维度：事实/品牌、口播自然度、画面可拍性、节奏叙事、转化力；1=差，5=好。",
        "",
        "结论：`pass` 可直接保留；`edit` 小改后保留；`reject` 重写或丢弃。",
        "",
    ]
    for index, (split, row, flag) in enumerate(selected, 1):
        lines += [
            "---",
            "",
            "## %02d. `%s` (%s)" % (index, row["id"], split),
            "",
            "**原始产品信息**：%s" % row["input"],
            "",
            "**目标**：%ss / %s 镜" % (row.get("target_duration_sec", 30), row.get("target_shots", "")),
            "",
        ]
        for shot_index, shot in enumerate(row["output"], 1):
            lines.append(shot_block(shot, shot_index))
        lines += [
            "**评分**：事实 __/5　口播 __/5　画面 __/5　叙事 __/5　转化 __/5",
            "",
            "**结论**：`pass` / `edit` / `reject`　　备注：________________",
            "",
        ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "id",
                "split",
                "n_shots",
                "input",
                "voiceovers",
                "visual_descriptions",
                "fact_score",
                "copy_score",
                "visual_score",
                "story_score",
                "conversion_score",
                "decision",
                "notes",
            ]
        )
        for split, row, _ in selected:
            shots = row["output"]
            writer.writerow(
                [
                    row["id"],
                    split,
                    len(shots),
                    row["input"],
                    " | ".join(s.get("voiceover", "") for s in shots),
                    " | ".join(s.get("visual_description", "") for s in shots),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    with flags_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "split", "flags_for_after_review"])
        for split, row, flag in selected:
            writer.writerow([row["id"], split, flag])

    print("Wrote", md_path)
    print("Wrote", csv_path)
    print("Wrote", flags_path)
    print("IDs:", ", ".join(row["id"] for _, row, _ in selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
