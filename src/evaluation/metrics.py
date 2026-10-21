"""Objective metrics for generated shampoo scripts.

The important design change from the original ``metrics.py`` is that JSON
validity, schema validity, truncation and content checks are separate signals.
A long but truncated script is not "better"; it is a format failure.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.script_schema import (  # noqa: E402
    ScriptValidationError,
    extract_json,
    validate_script,
)


def load_prompts(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("prompt file must contain a JSON list")
    seen = set()
    prompts = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each prompt must be an object")
        for key in ("id", "category", "prompt"):
            if not item.get(key):
                raise ValueError("prompt missing %s: %r" % (key, item))
        if item["id"] in seen:
            raise ValueError("duplicate prompt id: %s" % item["id"])
        seen.add(item["id"])
        item.setdefault("required_keywords", [])
        item.setdefault("forbidden_brands", [])
        item.setdefault("min_shots", 10)
        item.setdefault("target_duration_sec", [20, 45])
        for key in ("required_keywords", "forbidden_brands"):
            if not isinstance(item[key], list) or not all(
                isinstance(value, str) and value.strip() for value in item[key]
            ):
                raise ValueError("%s must be a list of non-empty strings: %s" % (key, item["id"]))
        prompts.append(item)
    return prompts


def _all_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_all_text(v) for v in value)
    if isinstance(value, dict):
        return "\n".join(_all_text(v) for v in value.values())
    return str(value)


def _keyword_hits(text: str, keywords: Sequence[str]) -> Dict[str, bool]:
    lower = text.lower()
    return {kw: kw.lower() in lower for kw in keywords}


def _brand_hits(text: str, brands: Sequence[str]) -> List[str]:
    lower = text.lower()
    return [brand for brand in brands if brand.lower() in lower]


def _looks_truncated(raw: str) -> bool:
    """Best-effort inference when the backend does not report finish_reason."""
    stripped = (raw or "").strip()
    if not stripped:
        return False
    if "time_axis" not in stripped and "shot_type" not in stripped:
        return False
    return stripped[-1] not in "]}"


def score_output(
    prompt_spec: Dict[str, Any],
    raw: str,
    *,
    finish_reason: Optional[str] = None,
    latency_s: Optional[float] = None,
    model: Optional[str] = None,
    max_chars_per_second: float = 4.0,
) -> Dict[str, Any]:
    """Score one model output.  Never raises: parse errors become metrics."""
    required = list(prompt_spec.get("required_keywords", []))
    forbidden = list(prompt_spec.get("forbidden_brands", []))
    expected_brand = prompt_spec.get("expected_brand")

    parsed = None
    parse_error = None
    try:
        parsed = extract_json(raw)
    except (ScriptValidationError, ValueError, TypeError) as exc:
        parse_error = str(exc)

    validation = validate_script(
        parsed if parsed is not None else [],
        min_shots=1,
        require_music=True,
        max_chars_per_second=max_chars_per_second,
        enforce_pacing=False,
        target_duration_sec=tuple(prompt_spec.get("target_duration_sec", [20, 45])),
    )
    m = validation.metrics
    content_text = _all_text(parsed) if parsed is not None else ""
    all_text = raw + "\n" + content_text
    # Keyword coverage is a content metric: do not give credit for keywords
    # that only appeared in prose around a truncated/invalid JSON payload.
    keyword_hits = _keyword_hits(content_text, required)
    brand_hits = _brand_hits(all_text, forbidden)
    expected_hit = bool(
        expected_brand and expected_brand.lower() in content_text.lower()
    )

    json_valid = parsed is not None
    schema_valid = bool(json_valid and validation.ok)
    min_shots = int(prompt_spec.get("min_shots", 10))
    complete = bool(schema_valid and m.get("n_shots", 0) >= min_shots)
    truncated = finish_reason == "length" or (not json_valid and _looks_truncated(raw))

    return {
        "id": prompt_spec["id"],
        "category": prompt_spec.get("category", ""),
        "model": model or "",
        "prompt": prompt_spec["prompt"],
        "finish_reason": finish_reason,
        "latency_s": round(latency_s, 3) if latency_s is not None else None,
        "raw_chars": len(raw or ""),
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "complete_script": complete,
        "truncated": truncated,
        "parse_error": parse_error,
        "n_shots": m.get("n_shots", 0),
        "has_min_shots": m.get("n_shots", 0) >= min_shots,
        "total_duration_sec": m.get("total_duration_sec", 0),
        "gaps": m.get("gaps", 0),
        "overlaps": m.get("overlaps", 0),
        "zero_duration": m.get("zero_duration", 0),
        "empty_voiceover": m.get("empty_voiceover", 0),
        "empty_music": m.get("empty_music", 0),
        "pacing_violations": m.get("pacing_violations", 0),
        "mean_chars_per_second": m.get("mean_chars_per_second"),
        "distinct_shot_types": m.get("distinct_shot_types", 0),
        "keyword_hits": keyword_hits,
        "keyword_coverage": (
            round(sum(keyword_hits.values()) / len(keyword_hits), 4)
            if keyword_hits
            else None
        ),
        "missing_keywords": [k for k, hit in keyword_hits.items() if not hit],
        "forbidden_brand_hits": brand_hits,
        "expected_brand": expected_brand,
        "expected_brand_mentioned": expected_hit if expected_brand else None,
        "validation_errors": validation.errors,
        "validation_warnings": validation.warnings,
    }


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n": 0}

    def rate(key: str) -> float:
        return round(sum(1 for r in records if r.get(key)) / len(records), 4)

    latencies = sorted(
        r["latency_s"] for r in records if r.get("latency_s") is not None
    )
    keyword_values = [
        r["keyword_coverage"]
        for r in records
        if r.get("keyword_coverage") is not None
    ]
    duration_values = [r.get("total_duration_sec", 0) for r in records]
    shot_values = [r.get("n_shots", 0) for r in records]

    by_category: Dict[str, Dict[str, Any]] = {}
    categories = sorted({r.get("category", "") for r in records})
    for category in categories:
        subset = [r for r in records if r.get("category", "") == category]
        by_category[category] = {
            "n": len(subset),
            "json_valid_rate": round(
                sum(1 for r in subset if r["json_valid"]) / len(subset), 4
            ),
            "schema_valid_rate": round(
                sum(1 for r in subset if r["schema_valid"]) / len(subset), 4
            ),
            "complete_script_rate": round(
                sum(1 for r in subset if r["complete_script"]) / len(subset), 4
            ),
            "mean_shots": round(
                sum(r["n_shots"] for r in subset) / len(subset), 2
            ),
        }

    return {
        "n": len(records),
        "json_valid_rate": rate("json_valid"),
        "schema_valid_rate": rate("schema_valid"),
        "complete_script_rate": rate("complete_script"),
        "truncation_rate": rate("truncated"),
        "has_min_shots_rate": rate("has_min_shots"),
        "time_continuous_rate": round(
            sum(
                1
                for r in records
                if r["json_valid"] and r["gaps"] == 0 and r["overlaps"] == 0
            )
            / len(records),
            4,
        ),
        "pacing_ok_rate": round(
            sum(1 for r in records if r["pacing_violations"] == 0) / len(records),
            4,
        ),
        "empty_voiceover_rate": round(
            sum(1 for r in records if r["empty_voiceover"] > 0) / len(records), 4
        ),
        "forbidden_brand_rate": round(
            sum(1 for r in records if r["forbidden_brand_hits"]) / len(records), 4
        ),
        "mean_shots": round(statistics.mean(shot_values), 2),
        "mean_duration_sec": round(statistics.mean(duration_values), 2),
        "mean_keyword_coverage": (
            round(statistics.mean(keyword_values), 4) if keyword_values else None
        ),
        "latency_p50_s": (
            round(statistics.median(latencies), 3) if latencies else None
        ),
        "latency_p95_s": (
            round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 3)
            if latencies
            else None
        ),
        "by_category": by_category,
    }


def render_markdown(
    summary: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    *,
    model_label: str = "",
    title: str = "30-prompt 评测报告",
) -> str:
    lines = [
        "# %s" % title,
        "",
        "模型：`%s`" % (model_label or "unknown"),
        "",
        "## 总览",
        "| 指标 | 值 |",
        "|---|---:|",
        "| Prompt 数 | %s |" % summary.get("n", 0),
        "| JSON 合法率 | %.1f%% |" % (summary.get("json_valid_rate", 0) * 100),
        "| Schema 合法率 | %.1f%% |" % (summary.get("schema_valid_rate", 0) * 100),
        "| 完整脚本率（含 >=10 镜） | %.1f%% |"
        % (summary.get("complete_script_rate", 0) * 100),
        "| 截断率 | %.1f%% |" % (summary.get("truncation_rate", 0) * 100),
        "| 时间轴连续率 | %.1f%% |" % (summary.get("time_continuous_rate", 0) * 100),
        "| 语速达标率 | %.1f%% |" % (summary.get("pacing_ok_rate", 0) * 100),
        "| 出现禁用品牌率 | %.1f%% |" % (summary.get("forbidden_brand_rate", 0) * 100),
        "| 平均镜头数 | %s |" % summary.get("mean_shots", 0),
        "| 平均时长 | %ss |" % summary.get("mean_duration_sec", 0),
        "| 关键词覆盖率 | %s |"
        % (
            "%.1f%%" % (summary.get("mean_keyword_coverage") * 100)
            if summary.get("mean_keyword_coverage") is not None
            else "n/a"
        ),
        "| 延迟 P50 / P95 | %ss / %ss |"
        % (summary.get("latency_p50_s"), summary.get("latency_p95_s")),
        "",
        "## 分类",
        "| 类别 | n | JSON | Schema | 完整脚本 | 平均镜头 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, item in summary.get("by_category", {}).items():
        lines.append(
            "| %s | %s | %.1f%% | %.1f%% | %.1f%% | %s |"
            % (
                category,
                item["n"],
                item["json_valid_rate"] * 100,
                item["schema_valid_rate"] * 100,
                item["complete_script_rate"] * 100,
                item["mean_shots"],
            )
        )

    failures = [
        r
        for r in records
        if not r.get("complete_script") or r.get("forbidden_brand_hits")
    ][:12]
    if failures:
        lines += ["", "## 典型失败样本", ""]
        for r in failures:
            reasons = []
            if not r["json_valid"]:
                reasons.append("JSON 解析失败")
            if r["truncated"]:
                reasons.append("截断")
            if not r["has_min_shots"]:
                reasons.append("镜头数不足 (%s)" % r["n_shots"])
            if r["forbidden_brand_hits"]:
                reasons.append("品牌: %s" % ",".join(r["forbidden_brand_hits"]))
            if r["gaps"] or r["overlaps"]:
                reasons.append("时间轴断裂/重叠")
            lines.append(
                "- `%s` [%s] %s"
                % (r["id"], r["category"], "；".join(reasons) or "未通过")
            )
    lines.append("")
    return "\n".join(lines)


def _legacy_blocks(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    blocks = []
    for block in text.split("=" * 70):
        if "PROMPT:" not in block:
            continue
        prompt_match = re.search(r"PROMPT: (.+)", block)
        raw_match = re.search(r"RAW:\n(.*?)(?=\n-{70})", block, re.S)
        if prompt_match and raw_match:
            blocks.append(
                {
                    "prompt": prompt_match.group(1).strip(),
                    "raw": raw_match.group(1).strip(),
                }
            )
    return blocks


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score generated scripts")
    parser.add_argument("--run", help="raw.jsonl produced by scripts/run_eval.py")
    parser.add_argument("--prompts", default="evaluation/prompts_30.json")
    parser.add_argument("--out", help="write markdown report here")
    parser.add_argument("--model-label", default="")
    parser.add_argument("--legacy-log", help="score an old cmp_*.log file")
    args = parser.parse_args(argv)

    if args.legacy_log:
        blocks = _legacy_blocks(Path(args.legacy_log))
        records = []
        for block in blocks:
            spec = {
                "id": "legacy-%d" % len(records),
                "category": "legacy",
                "prompt": block["prompt"],
                "required_keywords": [],
                "forbidden_brands": [],
                "min_shots": 10,
            }
            records.append(score_output(spec, block["raw"]))
        summary = summarize(records)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if not args.run:
        parser.error("provide --run or --legacy-log")
    prompts = {p["id"]: p for p in load_prompts(Path(args.prompts))}
    records = []
    for line in Path(args.run).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        spec = prompts[item["id"]]
        records.append(
            score_output(
                spec,
                item["raw"],
                finish_reason=item.get("finish_reason"),
                latency_s=item.get("latency_s"),
                model=item.get("model"),
            )
        )
    summary = summarize(records)
    report = render_markdown(summary, records, model_label=args.model_label)
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        (out_path.parent / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_path.parent / "scored.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
