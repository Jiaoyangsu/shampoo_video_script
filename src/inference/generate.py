"""Inference helper used by ``scripts/run_inference.py``.

This replaces the old unsloth-only implementation, which passed ``temperature``
without ``do_sample=True`` (so sampling options were ignored) and forced
``use_cache=True`` on a 24GB card (the exact OOM path documented in README).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.inference.backends import Backend, TransformersBackend, read_system_prompt
from src.script_schema import ScriptValidationError, extract_json, validate_script


class ScriptGenerator:
    def __init__(
        self,
        backend: Optional[Backend] = None,
        *,
        base_model: Optional[str] = None,
        merged_model: Optional[str] = None,
        sft_adapter: Optional[str] = None,
        dpo_adapter: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_cache: bool = False,
        max_new_tokens: int = 700,
    ) -> None:
        self.backend = backend or TransformersBackend(
            base_model=base_model,
            merged_model=merged_model,
            sft_adapter=sft_adapter,
            dpo_adapter=dpo_adapter,
            use_cache=use_cache,
            max_new_tokens=max_new_tokens,
        )
        self.system_prompt = system_prompt or read_system_prompt()
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        product_input: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = 42,
    ) -> str:
        result = self.backend.generate(
            self.system_prompt,
            product_input,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        return result.text

    def generate_json(self, product_input: str, **kwargs: Any) -> List[Dict[str, str]]:
        raw = self.generate(product_input, **kwargs)
        parsed = extract_json(raw)
        validation = validate_script(
            parsed, min_shots=1, require_music=True, enforce_pacing=False
        )
        if not validation.ok:
            raise ScriptValidationError(
                "model output failed validation: %s" % "; ".join(validation.errors[:8])
            )
        return validation.shots


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-model", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--sft-adapter", default=None)
    parser.add_argument("--dpo-adapter", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    generator = ScriptGenerator(
        base_model=args.base_model,
        merged_model=args.merged_model,
        sft_adapter=args.sft_adapter,
        dpo_adapter=args.dpo_adapter,
        use_cache=args.use_cache,
        max_new_tokens=args.max_new_tokens,
    )
    prompts = [
        args.prompt
        or "请为海飞丝去屑洗发水设计 30 秒带货短视频脚本，核心卖点是去屑和清凉感。"
    ]
    for prompt in prompts:
        print(json.dumps(generator.generate_json(prompt), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
