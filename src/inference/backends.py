"""Inference backends for the Gradio app and the evaluation harness.

Two production backends are supported:

* ``ollama``      -- call a local Ollama server (easy local demo).
* ``transformers`` -- load a merged model, or base + SFT adapter + DPO adapter.

The transformers path deliberately loads SFT *and then* DPO.  The old
``eval_compare.py`` loaded only the DPO adapter on top of the base model, which
silently dropped the SFT weights.  See ``scripts/merge_adapters.py`` for the
canonical offline merge.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "app" / "system_prompt.md"


class BackendConfigError(RuntimeError):
    """Raised when a backend is not configured correctly."""


@dataclass
class GenerationResult:
    text: str
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def read_system_prompt(path: Optional[str] = None) -> str:
    prompt_path = Path(path) if path else DEFAULT_SYSTEM_PROMPT_PATH
    if not prompt_path.exists():
        raise FileNotFoundError("system prompt not found: %s" % prompt_path)
    return prompt_path.read_text(encoding="utf-8")


def build_chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    """Build the same prompt shape used by SFT/DPO training.

    Gemma 3's chat template does not have a reliable separate system role in
    every transformers version, so the system prompt is prepended to the user
    turn.  This is exactly what the training scripts do.
    """
    messages = [
        {
            "role": "user",
            "content": "%s\n\n%s" % (system_prompt.strip(), user_prompt.strip()),
        }
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


class Backend:
    name = "base"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = 700,
        temperature: float = 0.8,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        raise NotImplementedError


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        timeout: int = 900,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = 700,
        temperature: float = 0.8,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        options: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 50,
            "num_predict": max_new_tokens,
            "repetition_penalty": 1.05,
        }
        if seed is not None:
            options["seed"] = seed
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": options,
        }
        request = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise BackendConfigError(
                "cannot reach Ollama at %s: %s" % (self.host, exc)
            ) from exc
        return GenerationResult(
            text=data.get("response", ""),
            finish_reason=data.get("done_reason"),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            meta={"backend": "ollama", "model": self.model},
        )


class TransformersBackend(Backend):
    name = "transformers"

    def __init__(
        self,
        *,
        base_model: Optional[str] = None,
        merged_model: Optional[str] = None,
        sft_adapter: Optional[str] = None,
        dpo_adapter: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        use_cache: bool = False,
        trust_remote_code: bool = False,
        max_new_tokens: int = 700,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on GPU env
            raise BackendConfigError(
                "transformers backend requires torch/transformers/peft; "
                "install requirements-train.txt"
            ) from exc

        if not merged_model and not base_model:
            raise BackendConfigError(
                "set SCRIPT_MODEL (merged model) or SCRIPT_BASE_MODEL + adapters"
            )
        if merged_model and (sft_adapter or dpo_adapter):
            raise BackendConfigError(
                "choose either a merged model or base + adapters, not both"
            )
        if dpo_adapter and not sft_adapter and not merged_model:
            raise BackendConfigError(
                "a DPO adapter is a delta on merged SFT; also pass SCRIPT_SFT_ADAPTER "
                "or use a merged model. Loading DPO alone on base is the bug this "
                "pipeline is designed to prevent."
            )
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.use_cache = use_cache

        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        model_path = merged_model or base_model
        assert model_path is not None
        self.tokenizer = AutoTokenizer.from_pretrained(
            merged_model or base_model,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
        )

        if not merged_model:
            if sft_adapter:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, sft_adapter)
                # Merge SFT into the base so the DPO delta is applied on top of
                # exactly the model it was trained against.
                if dpo_adapter:
                    self.model = self.model.merge_and_unload()
            if dpo_adapter:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, dpo_adapter)
        self.model.eval()
        self.meta = {
            "backend": "transformers",
            "model": model_path,
            "sft_adapter": sft_adapter,
            "dpo_adapter": dpo_adapter,
            "use_cache": use_cache,
        }

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        prompt = build_chat_prompt(self.tokenizer, system_prompt, user_prompt)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        limit = max_new_tokens or self.max_new_tokens

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": limit,
            "use_cache": self.use_cache,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature and temperature > 0:
            generate_kwargs.update(
                {"do_sample": True, "temperature": temperature, "top_p": top_p}
            )
        else:
            generate_kwargs["do_sample"] = False
        with torch.no_grad():
            output = self.model.generate(**generate_kwargs)
        generated = output[0][input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        last_token = int(generated[-1]) if len(generated) else None
        hit_eos = last_token == self.tokenizer.eos_token_id
        finish_reason = "stop" if hit_eos else ("length" if len(generated) >= limit else "stop")
        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=int(input_ids.shape[1]),
            completion_tokens=int(generated.shape[0]),
            meta=dict(self.meta),
        )


class MockBackend(Backend):
    """Deterministic backend used by tests and ``--backend mock`` self-checks."""

    name = "mock"

    def __init__(self, mode: str = "mixed") -> None:
        self.mode = mode
        self.calls = 0

    def _valid_script(self) -> str:
        shots = []
        vo = [
            "头皮出油有头屑真的很尴尬",
            "早上刚洗下午就塌了",
            "这款洗发水泡沫很绵密",
            "清洁头皮的同时也很温和",
            "洗完之后头发很清爽",
            "发根蓬松不贴头皮",
            "香味是淡淡的花香",
            "坚持使用头屑少了很多",
            "价格也很适合学生党",
            "点击下方链接就能下单",
        ]
        for index in range(10):
            start = index * 3
            end = start + 3
            shots.append(
                {
                    "time_axis": "%02d:%02d-%02d:%02d" % (start // 60, start % 60, end // 60, end % 60),
                    "shot_type": "痛点展示" if index == 0 else "产品展示",
                    "visual_description": "第%d个镜头的画面描述，展示产品使用场景。" % (index + 1),
                    "voiceover": vo[index],
                    "music": "轻快背景音乐",
                }
            )
        return json.dumps(shots, ensure_ascii=False)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = 700,
        temperature: float = 0.8,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        self.calls += 1
        script = self._valid_script()
        if self.mode == "valid":
            text = script
        elif self.mode == "truncated":
            text = script[: int(len(script) * 0.72)]
        elif self.mode == "invalid":
            text = "好的，这是脚本：\n" + script[:80]
        else:
            text = script if self.calls % 2 else script[: int(len(script) * 0.72)]
        finish_reason = "length" if len(text) < len(script) else "stop"
        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=0,
            completion_tokens=len(text),
            meta={"backend": "mock", "mode": self.mode},
        )


def build_backend_from_env() -> Backend:
    """Build a backend from environment variables.

    OLLAMA_MODEL=gemma_3_12B_q80:latest
    SCRIPT_BACKEND=ollama|transformers
    SCRIPT_MODEL=/path/to/merged_model
    SCRIPT_BASE_MODEL=/path/to/gemma-3-12b-it
    SCRIPT_SFT_ADAPTER=/path/to/gemma3_sft/final
    SCRIPT_DPO_ADAPTER=/path/to/gemma3_dpo/final/dpo_adapter
    SCRIPT_USE_CACHE=0
    """
    backend = os.environ.get("SCRIPT_BACKEND", "ollama").strip().lower()
    max_new_tokens = int(os.environ.get("SCRIPT_MAX_NEW_TOKENS", "700"))
    if backend == "ollama":
        model = os.environ.get("OLLAMA_MODEL", "").strip()
        if not model:
            raise BackendConfigError("OLLAMA_MODEL is required for the Ollama backend")
        return OllamaBackend(
            model=model,
            host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        )
    if backend == "transformers":
        return TransformersBackend(
            base_model=os.environ.get("SCRIPT_BASE_MODEL") or None,
            merged_model=os.environ.get("SCRIPT_MODEL") or None,
            sft_adapter=os.environ.get("SCRIPT_SFT_ADAPTER") or None,
            dpo_adapter=os.environ.get("SCRIPT_DPO_ADAPTER") or None,
            use_cache=os.environ.get("SCRIPT_USE_CACHE", "0") == "1",
            max_new_tokens=max_new_tokens,
        )
    if backend == "mock":
        return MockBackend(os.environ.get("SCRIPT_MOCK_MODE", "mixed"))
    raise BackendConfigError("unknown SCRIPT_BACKEND: %s" % backend)
