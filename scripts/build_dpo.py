#!/usr/bin/env python3
"""Build DPO pairs from an SFT dataset (default: the 30s rewrite set)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean import build_dpo_rows  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/rewrite_30s/train.json")
    parser.add_argument("--val", default="data/rewrite_30s/val.json")
    parser.add_argument("--out-dir", default="data/rewrite_30s")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, path in (("train", args.train), ("val", args.val)):
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        pairs, pair_stats = build_dpo_rows(rows, seed=args.seed + (0 if split == "train" else 1))
        (out / ("dpo_%s.json" % split)).write_text(
            json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stats[split] = pair_stats
        print("%s: %d pairs, %d skipped" % (split, len(pairs), pair_stats["skipped"]))
    (out / "dpo_quality.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
