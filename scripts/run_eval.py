#!/usr/bin/env python3
"""Reproducible 30-prompt evaluation harness.

Examples
--------
# 1) Smoke-test the harness without a GPU/model:
python scripts/run_eval.py --backend mock --limit 3 --out-dir evaluation/results/mock

# 2) Evaluate the fine-tuned model on AutoDL (base + SFT + DPO adapters):
python scripts/run_eval.py \
  --backend transformers \
  --base-model /root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master \
  --sft-adapter /root/autodl-tmp/outputs/gemma3_sft/final \
  --dpo-adapter /root/autodl-tmp/outputs/gemma3_dpo/final/dpo_adapter \
  --max-new-tokens 700 \
  --out-dir evaluation/results/sft_dpo

# 3) Evaluate a merged model (recommended for final comparison):
python scripts/run_eval.py --backend transformers \
  --merged-model /root/autodl-tmp/models/shampoo_sft_dpo_merged \
  --out-dir evaluation/results/merged

# 4) Evaluate the local Ollama demo model:
python scripts/run_eval.py --backend ollama --model gemma_3_12B_q80:latest \
  --out-dir evaluation/results/ollama_base
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    load_prompts,
    render_markdown,
    score_output,
    summarize,
)
from src.finetune.prompting import build_user_content  # noqa: E402
from src.inference.backends import (  # noqa: E402
    BackendConfigError,
    MockBackend,
    OllamaBackend,
    TransformersBackend,
    build_backend_from_env,
    read_system_prompt,
)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT)
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_done_ids(raw_path: Path) -> set:
    done = set()
    if not raw_path.exists():
        return done
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["id"])
        except Exception:
            continue
    return done


def build_backend(args: argparse.Namespace):
    if args.backend == "mock":
        return MockBackend(args.mock_mode)
    if args.backend == "ollama":
        if not args.model:
            raise BackendConfigError("--model is required for --backend ollama")
        return OllamaBackend(args.model, host=args.host)
    if args.backend == "transformers":
        return TransformersBackend(
            base_model=args.base_model,
            merged_model=args.merged_model,
            sft_adapter=args.sft_adapter,
            dpo_adapter=args.dpo_adapter,
            use_cache=args.use_cache,
            max_new_tokens=args.max_new_tokens,
        )
    return build_backend_from_env()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    prompts = load_prompts(Path(args.prompts))
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        prompts = [p for p in prompts if p["id"] in wanted]
    if args.limit:
        prompts = prompts[: args.limit]

    system_prompt = read_system_prompt(args.system_prompt)
    backend = build_backend(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    done = _load_done_ids(raw_path) if args.resume else set()
    if not args.resume and raw_path.exists():
        raw_path.unlink()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": args.backend,
        "model_label": args.label,
        "prompts_file": str(args.prompts),
        "prompts_sha256": _sha256(Path(args.prompts).read_text(encoding="utf-8")),
        "system_prompt_sha256": _sha256(system_prompt),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "limit": args.limit,
        "ids": args.ids,
        "command": " ".join(sys.argv),
    }
    _write_json(out_dir / "manifest.json", manifest)

    records: List[Dict[str, Any]] = []
    with raw_path.open("a", encoding="utf-8") as raw_file:
        for index, spec in enumerate(prompts, 1):
            if spec["id"] in done:
                print("[%d/%d] skip %s (already done)" % (index, len(prompts), spec["id"]))
                continue
            print("[%d/%d] %s %s" % (index, len(prompts), spec["id"], spec["category"]))
            started = time.time()
            error = None
            try:
                target_duration = spec.get("target_duration_sec", 30)
                if isinstance(target_duration, (list, tuple)):
                    target_duration = int(sum(target_duration) / len(target_duration))
                user_prompt = build_user_content(
                    spec["prompt"],
                    duration=int(target_duration),
                    shots="%s–12" % spec.get("min_shots", 10),
                )
                result = backend.generate(
                    system_prompt,
                    user_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed + index if args.seed is not None else None,
                )
                text = result.text
                finish_reason = result.finish_reason
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
                meta = result.meta
            except Exception as exc:  # keep the run alive; record the failure
                text = ""
                finish_reason = "error"
                prompt_tokens = None
                completion_tokens = None
                meta = {}
                error = "%s: %s" % (type(exc).__name__, exc)
                if args.fail_fast:
                    raise
            latency = time.time() - started
            item = {
                "id": spec["id"],
                "category": spec["category"],
                "prompt": spec["prompt"],
                "raw": text,
                "finish_reason": finish_reason,
                "latency_s": round(latency, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": args.label,
                "backend": args.backend,
                "seed": args.seed + index if args.seed is not None else None,
                "error": error,
                "meta": meta,
            }
            raw_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            raw_file.flush()
            records.append(item)
            if error:
                print("  ERROR: %s" % error)

    # Re-read the full file so --resume scores completed rows too.
    all_items = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {p["id"]: p for p in load_prompts(Path(args.prompts))}
    scored = []
    for item in all_items:
        spec = by_id.get(item["id"])
        if not spec:
            continue
        scored.append(
            score_output(
                spec,
                item.get("raw", ""),
                finish_reason=item.get("finish_reason"),
                latency_s=item.get("latency_s"),
                model=item.get("model"),
            )
        )

    summary = summarize(scored)
    report = render_markdown(
        summary,
        scored,
        model_label=args.label or args.backend,
        title="%d-prompt 评测报告" % len(scored),
    )
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "scored.json", scored)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    print("\n" + report)
    print("Wrote %s" % out_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run reproducible script evaluation")
    parser.add_argument("--prompts", default="evaluation/prompts_30.json")
    parser.add_argument("--out-dir", default="evaluation/results/run")
    parser.add_argument("--backend", default="env", choices=["env", "ollama", "transformers", "mock"])
    parser.add_argument("--model", help="Ollama model name")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--base-model")
    parser.add_argument("--merged-model")
    parser.add_argument("--sft-adapter")
    parser.add_argument("--dpo-adapter")
    parser.add_argument("--label", default="")
    parser.add_argument("--system-prompt", default="src/app/system_prompt.md")
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None, help="comma-separated prompt ids")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--mock-mode", default="mixed", choices=["valid", "truncated", "invalid", "mixed"])
    args = parser.parse_args(argv)

    if args.backend == "env":
        # Respect SCRIPT_BACKEND/OLLAMA_MODEL if provided, otherwise default to
        # Ollama so the command matches the app's default.
        os.environ.setdefault("SCRIPT_BACKEND", "ollama")
        if args.model:
            os.environ["OLLAMA_MODEL"] = args.model
        if args.host:
            os.environ["OLLAMA_HOST"] = args.host
        for attr, env_name in (
            ("merged_model", "SCRIPT_MODEL"),
            ("base_model", "SCRIPT_BASE_MODEL"),
            ("sft_adapter", "SCRIPT_SFT_ADAPTER"),
            ("dpo_adapter", "SCRIPT_DPO_ADAPTER"),
        ):
            if getattr(args, attr):
                os.environ[env_name] = getattr(args, attr)
    try:
        run(args)
    except BackendConfigError as exc:
        print("Backend configuration error: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
