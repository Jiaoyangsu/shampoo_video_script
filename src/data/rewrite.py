"""Rewrite existing product briefs into 30-second, 10-12 shot scripts.

The source data is mechanically clean but too dense: 82% of shots exceed
4 chars/sec and only 18% have >=10 shots.  This module asks a strong model to
rewrite the *voiceover and time axis* while preserving the product brief, then
rejects anything that does not pass the 30s / 10-12 shot / <=4 chars/sec gate.

It is intentionally separate from ``clean.py``: cleaning fixes broken data;
rewriting changes the creative content and therefore requires human spot-checks.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.script_schema import (
    BAD_SHOT_TYPES,
    ScriptValidation,
    parse_time_axis,
    validate_script,
)

KNOWN_BRANDS = [
    "海飞丝",
    "清扬",
    "潘婷",
    "飘柔",
    "沙宣",
    "力士",
    "多芬",
    "施华蔻",
    "霸王",
    "滋源",
    "蜂花",
    "拉芳",
    "阿道夫",
    "贝德美",
    "百雀羚",
    "飞虎牌",
    "红色小象",
    "戴可思",
    "SPES",
    "KONO",
    "满婷",
]

ALLOWED_SHOT_TYPES = {
    "产品引入",
    "痛点展示",
    "产品概述",
    "产品卖点",
    "使用演示",
    "产品促销",
}

SYSTEM_PROMPT = """你是一位资深带货短视频编导，擅长把产品介绍改写成可拍摄的 30 秒分镜脚本。
你只输出 JSON，不输出解释、Markdown 或代码围栏。"""

SHOT_SCHEMA = {
    "type": "array",
    "minItems": 10,
    "maxItems": 12,
    "items": {
        "type": "object",
        "properties": {
            "time_axis": {"type": "string"},
            "shot_type": {"type": "string"},
            "visual_description": {"type": "string"},
            "voiceover": {"type": "string"},
            "music": {"type": "string"},
        },
        "required": [
            "time_axis",
            "shot_type",
            "visual_description",
            "voiceover",
            "music",
        ],
        "additionalProperties": False,
    },
}


def user_prompt(
    product_brief: str,
    *,
    target_duration: int = 30,
    min_shots: int = 10,
    max_shots: int = 12,
    feedback: Optional[str] = None,
) -> str:
    feedback_block = ""
    if feedback:
        feedback_block = "\n上一次输出未通过校验，原因：%s\n请针对这些问题重写。\n" % feedback
    return f"""请把下面的产品信息改写成一条中文带货短视频分镜脚本。
{feedback_block}
硬性要求：
1. 只输出 JSON 列表，不要输出任何解释、前后缀或 Markdown 代码围栏。
2. 镜头数必须在 {min_shots}-{max_shots} 个之间。
3. 每个镜头必须包含且只包含这 5 个字符串字段：
   time_axis, shot_type, visual_description, voiceover, music。
4. time_axis 格式为 "MM:SS-MM:SS"，从 00:00 开始，连续、不重叠、无 0 秒镜头。
5. 总时长控制在 {target_duration - 2}-{target_duration + 2} 秒；每个镜头 2-4 秒。
6. 每个镜头 voiceover 控制在 8-12 个汉字；语速不超过 4 字/秒。
7. shot_type 只能从这些值里选：{", ".join(sorted(ALLOWED_SHOT_TYPES))}；
   至少使用 4 种不同 shot_type，不要所有镜头都写“产品展示”。
8. 不得编造产品信息中没有的品牌、价格、销量、成分、认证；
   不得使用“治疗、生发、根治、100% 有效”等医疗或绝对化表述。
9. 保留产品信息中的产品名、核心卖点和目标人群。
10. 促销口播要结合本产品的具体卖点，不要每条都用“限时优惠，速来抢购”这类通用句；
    同一段脚本内不要重复使用同一句 CTA。
11. visual_description 至少 20 个汉字，写清人物、动作、场景和镜头运动；
    不要只写“产品特写”“头发特写”这种过短描述。
12. 避免“效果显著、深层滋养修复、专业设计配方”这类空泛词；
    尽量引用产品信息里的成分、使用场景和目标人群。

产品信息：
{product_brief.strip()}
"""


@dataclass
class RewriteResult:
    ok: bool
    shots: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "shots": self.shots,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def _brands_in(text: str) -> List[str]:
    lower = (text or "").lower()
    return [brand for brand in KNOWN_BRANDS if brand.lower() in lower]


def _fmt(seconds: int) -> str:
    minutes, sec = divmod(max(0, int(seconds)), 60)
    return "%02d:%02d" % (minutes, sec)


def retime_script(
    script: Any,
    *,
    target_duration: int = 30,
    max_chars_per_second: float = 4.0,
) -> Optional[List[Dict[str, str]]]:
    """Rebuild a continuous time axis so the script fits the target duration.

    The model is good at writing shots but bad at arithmetic.  If the shots
    already contain 10-12 well-formed beats, we can make the timeline exact
    instead of rejecting the whole sample.  Returns None if the voiceover is
    simply too long to fit (which must go back to the model).
    """
    from src.script_schema import as_script_list, normalize_shot, spoken_char_count

    try:
        shots = [normalize_shot(s) for s in as_script_list(script)]
    except Exception:
        return None
    if not shots:
        return None

    durations = []
    for shot in shots:
        chars = spoken_char_count(shot.get("voiceover", ""))
        durations.append(max(1, int((chars + max_chars_per_second - 0.001) // max_chars_per_second)))

    minimum = sum(durations)
    if minimum > target_duration + 2:
        return None

    # Distribute the remaining seconds proportionally; every shot keeps at
    # least its minimum duration, so the pacing constraint remains satisfied.
    extra = target_duration - minimum
    if extra > 0:
        total_min = sum(durations)
        remainders = []
        for i, value in enumerate(durations):
            exact = extra * value / total_min
            add = int(exact)
            durations[i] += add
            remainders.append((exact - add, i))
        left = target_duration - sum(durations)
        for _, i in sorted(remainders, reverse=True)[:left]:
            durations[i] += 1

    cursor = 0
    out = []
    for shot, duration in zip(shots, durations):
        start = cursor
        end = cursor + duration
        cursor = end
        shot = dict(shot)
        shot["time_axis"] = "%s-%s" % (_fmt(start), _fmt(end))
        out.append(shot)
    if cursor != target_duration:
        return None
    return out


def validate_rewrite(
    script: Any,
    product_brief: str,
    *,
    target_duration: int = 30,
    min_shots: int = 10,
    max_shots: int = 12,
    max_chars_per_second: float = 4.0,
) -> RewriteResult:
    validation: ScriptValidation = validate_script(
        script,
        min_shots=min_shots,
        require_music=True,
        max_chars_per_second=max_chars_per_second,
        enforce_pacing=True,
        target_duration_sec=(target_duration - 2, target_duration + 2),
    )
    errors = list(validation.errors)
    warnings = list(validation.warnings)
    m = dict(validation.metrics)

    shots = validation.shots
    if len(shots) > max_shots:
        errors.append("has %d shots, maximum is %d" % (len(shots), max_shots))
    if shots:
        first = parse_time_axis(shots[0].get("time_axis", ""))
        last = parse_time_axis(shots[-1].get("time_axis", ""))
        if first is None or first[0] != 0:
            errors.append("first shot must start at 00:00")
        if last is not None and last[1] > target_duration + 3:
            errors.append(
                "last shot ends at %ds, target is %ds" % (last[1], target_duration)
            )
    total = m.get("total_duration_sec", 0)
    if total and not (target_duration - 2 <= total <= target_duration + 2):
        errors.append(
            "total duration %ds outside %d-%ds"
            % (total, target_duration - 2, target_duration + 2)
        )

    shot_types = [s.get("shot_type", "") for s in shots]
    if len(set(shot_types)) < 4:
        errors.append("uses fewer than 4 distinct shot_type values")
    for index, shot_type in enumerate(shot_types):
        if shot_type in BAD_SHOT_TYPES:
            errors.append("shot %d uses bad shot_type %r" % (index, shot_type))
        if shot_type not in ALLOWED_SHOT_TYPES:
            warnings.append(
                "shot %d shot_type %r is outside the preferred enum"
                % (index, shot_type)
            )

    input_brands = _brands_in(product_brief)
    output_brands = _brands_in(json.dumps(shots, ensure_ascii=False))
    unknown = [b for b in output_brands if b not in input_brands]
    if unknown:
        errors.append("invented brand(s): %s" % ", ".join(sorted(set(unknown))))
    # If the input names a brand, the rewrite should keep it.
    for brand in input_brands:
        if brand not in output_brands:
            warnings.append("input brand %r is missing from output" % brand)

    return RewriteResult(
        ok=not errors,
        shots=shots,
        errors=errors,
        warnings=warnings,
        metrics=m,
    )


class OllamaRewriteClient:
    def __init__(
        self,
        model: str = "qwen2.5:14b",
        host: str = "http://127.0.0.1:11434",
        timeout: int = 600,
        temperature: float = 0.6,
        top_p: float = 0.9,
        num_predict: int = 2200,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict

    def generate(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        format_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        options: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
            "repeat_penalty": 1.05,
        }
        if seed is not None:
            options["seed"] = seed
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": format_schema or "json",
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
            raise RuntimeError("cannot reach Ollama at %s: %s" % (self.host, exc)) from exc
        return data.get("response", "")


def parse_model_json(text: str) -> Any:
    """Parse the model output, accepting a fenced JSON block or a JSON object."""
    from src.script_schema import extract_json

    return extract_json(text)


def rewrite_one(
    row: Dict[str, Any],
    client: OllamaRewriteClient,
    *,
    target_duration: int = 30,
    min_shots: int = 10,
    max_shots: int = 12,
    max_attempts: int = 3,
    base_seed: int = 3407,
) -> Dict[str, Any]:
    brief = row["input"]
    feedback = None
    attempts = []
    for attempt in range(1, max_attempts + 1):
        prompt = user_prompt(
            brief,
            target_duration=target_duration,
            min_shots=min_shots,
            max_shots=max_shots,
            feedback=feedback,
        )
        started = time.time()
        raw = ""
        retimed = None
        try:
            raw = client.generate(
                prompt, seed=base_seed + attempt * 17, format_schema=SHOT_SCHEMA
            )
            parsed = parse_model_json(raw)
            retimed = retime_script(
                parsed,
                target_duration=target_duration,
                max_chars_per_second=4.0,
            )
            if retimed is not None:
                parsed = retimed
            quality = validate_rewrite(
                parsed,
                brief,
                target_duration=target_duration,
                min_shots=min_shots,
                max_shots=max_shots,
            )
        except Exception as exc:
            quality = RewriteResult(ok=False, errors=["%s: %s" % (type(exc).__name__, exc)])
            parsed = None
        attempts.append(
            {
                "attempt": attempt,
                "latency_s": round(time.time() - started, 3),
                "raw_chars": len(raw),
                "ok": quality.ok,
                "normalized_timing": retimed is not None,
                "errors": quality.errors[:20],
                "warnings": quality.warnings[:20],
                "metrics": quality.metrics,
            }
        )
        if quality.ok:
            return {
                "ok": True,
                "output": quality.shots,
                "attempts": attempts,
                "quality": quality.metrics,
                "raw": raw,
            }
        feedback = "；".join(quality.errors[:6]) or "JSON 结构不符合要求"
    return {
        "ok": False,
        "output": None,
        "attempts": attempts,
        "quality": attempts[-1].get("metrics", {}) if attempts else {},
        "raw": raw,
    }


def convert_for_training(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only fields needed by SFT/DPO."""
    out = []
    for row in rows:
        out.append(
            {
                "input": row["input"],
                "output": row["output"],
                "target_duration_sec": row.get("target_duration_sec", 30),
                "target_shots": len(row["output"]),
            }
        )
    return out
