#!/usr/bin/env python3
"""Rewrite cleaned product briefs into 30-second, 10-12 shot scripts.

Pilot:
    python scripts/rewrite_data.py --limit 10 --out-dir data/rewrite_30s

Full train split (resumable):
    python scripts/rewrite_data.py \
      --source data/clean/shampoo_train.json \
      --out-dir data/rewrite_30s \
      --limit 200 --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rewrite import (  # noqa: E402
    OllamaRewriteClient,
    convert_for_training,
    detect_explicit_duration,
    rewrite_one,
)


def _row_id(row: Dict[str, Any], index: int) -> str:
    payload = json.dumps(row.get("input", ""), ensure_ascii=False)
    return "rw-%04d-%s" % (index, hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8])


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_report(
    *,
    source: str,
    accepted: Sequence[Dict[str, Any]],
    rejected: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    failure_reasons: Dict[str, int] = {}
    for row in rejected:
        attempts = row.get("attempts") or []
        errors = attempts[-1].get("errors", []) if attempts else ["unknown"]
        for error in errors[:3]:
            key = error.split(":")[0][:80]
            failure_reasons[key] = failure_reasons.get(key, 0) + 1
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "config": config,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accept_rate": round(len(accepted) / max(1, len(accepted) + len(rejected)), 4),
        "failure_reasons": failure_reasons,
        "accepted_attempts": {
            "1": sum(1 for row in accepted if len(row.get("attempts", [])) == 1),
            "2": sum(1 for row in accepted if len(row.get("attempts", [])) == 2),
            "3": sum(1 for row in accepted if len(row.get("attempts", [])) >= 3),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/clean/shampoo_train.json")
    parser.add_argument("--out-dir", default="data/rewrite_30s")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default=os.environ.get("REWRITE_MODEL", "qwen2.5:14b"))
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--target-duration", type=int, default=30)
    parser.add_argument("--min-shots", type=int, default=10)
    parser.add_argument("--max-shots", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args(argv)

    source = PROJECT_ROOT / args.source if not Path(args.source).is_absolute() else Path(args.source)
    source_rows = json.loads(source.read_text(encoding="utf-8"))
    selected = source_rows[args.offset : args.offset + args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = out_dir / ("%s_accepted.jsonl" % args.split)
    rejected_path = out_dir / ("%s_rejected.jsonl" % args.split)

    accepted = _read_jsonl(accepted_path) if args.resume else []
    rejected = _read_jsonl(rejected_path) if args.resume else []
    done_ids = {row["id"] for row in accepted} | {row["id"] for row in rejected}
    if not args.resume:
        accepted_path.unlink(missing_ok=True)
        rejected_path.unlink(missing_ok=True)

    client = OllamaRewriteClient(model=args.model, host=args.host)
    config = {
        "model": args.model,
        "target_duration_sec": args.target_duration,
        "min_shots": args.min_shots,
        "max_shots": args.max_shots,
        "max_attempts": args.max_attempts,
        "source": str(source),
    }
    started = datetime.now(timezone.utc)

    for index, row in enumerate(selected, 1):
        row_id = _row_id(row, args.offset + index - 1)
        if row_id in done_ids:
            print("[%d/%d] skip %s" % (index, len(selected), row_id))
            continue
        explicit_duration = detect_explicit_duration(row.get("input", ""))
        if explicit_duration and abs(explicit_duration - args.target_duration) > 5:
            item = {
                "id": row_id,
                "source_index": args.offset + index - 1,
                "input": row["input"],
                "attempts": [
                    {
                        "attempt": 0,
                        "ok": False,
                        "errors": [
                            "duration conflict: brief asks %ss, target is %ss"
                            % (explicit_duration, args.target_duration)
                        ],
                    }
                ],
                "model": args.model,
            }
            rejected.append(item)
            _append_jsonl(rejected_path, item)
            print("  SKIP duration conflict %ss" % explicit_duration, flush=True)
            continue
        print("[%d/%d] rewrite %s" % (index, len(selected), row_id), flush=True)
        result = rewrite_one(
            row,
            client,
            target_duration=args.target_duration,
            min_shots=args.min_shots,
            max_shots=args.max_shots,
            max_attempts=args.max_attempts,
        )
        base = {
            "id": row_id,
            "source_index": args.offset + index - 1,
            "input": row["input"],
            "target_duration_sec": args.target_duration,
            "target_shots": args.min_shots,
            "attempts": result["attempts"],
            "model": args.model,
        }
        if result["ok"]:
            item = dict(base)
            item["output"] = result["output"]
            item["target_shots"] = result["quality"].get("n_shots", args.min_shots)
            item["quality"] = result["quality"]
            accepted.append(item)
            _append_jsonl(accepted_path, item)
            print("  OK shots=%s duration=%ss" % (
                result["quality"].get("n_shots"),
                result["quality"].get("total_duration_sec"),
            ))
        else:
            item = dict(base)
            item["quality"] = result.get("quality", {})
            item["raw"] = (result.get("raw") or "")[:2000]
            rejected.append(item)
            _append_jsonl(rejected_path, item)
            print("  REJECT %s" % (result["attempts"][-1]["errors"][:2] if result["attempts"] else "unknown"))

    report = build_report(
        source=str(source),
        accepted=accepted,
        rejected=rejected,
        config=config,
    )
    report["elapsed_seconds"] = round(
        (datetime.now(timezone.utc) - started).total_seconds(), 1
    )
    _write_json(out_dir / ("%s_quality.json" % args.split), report)
    _write_json(out_dir / ("%s.json" % args.split), convert_for_training(accepted))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Wrote", out_dir / ("%s.json" % args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
