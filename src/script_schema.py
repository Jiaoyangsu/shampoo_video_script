"""Shared schema and validation for shampoo short-video scripts.

This module is deliberately dependency-free (standard library only) so that it
can be used by the data-cleaning scripts, the offline evaluator, the training
scripts and the Gradio app without pulling in torch/transformers.

A script is a JSON list of shots.  Every shot must contain:

    time_axis, shot_type, visual_description, voiceover, music

The validator is intentionally strict about the things that can be checked
mechanically (required fields, time axis, empty voiceover, pacing) and leaves
creative quality to human/LLM review.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

REQUIRED_FIELDS: Tuple[str, ...] = (
    "time_axis",
    "shot_type",
    "visual_description",
    "voiceover",
    "music",
)

EMPTY_VOICEOVER_VALUES = {
    "",
    "无",
    "（无）",
    "(无)",
    "无。",
    "暂无",
    "（无旁白）",
    "(无旁白)",
    "无旁白",
    "n/a",
    "N/A",
    "null",
    "None",
}

EMPTY_MUSIC_VALUES = {
    "",
    "无",
    "无音乐",
    "（无）",
    "(无)",
    "none",
    "None",
    "null",
}

# Shot types that should never be used as a "good" narrative beat.  They are
# used by the synthetic DPO rejected examples and by the validator.
BAD_SHOT_TYPES = {"黑屏转场", "广告插入", "用户访谈"}

TIME_AXIS_RE = re.compile(
    r"^\s*(\d{1,2}):([0-5]\d)\s*-\s*(\d{1,2}):([0-5]\d)\s*$"
)
PUNCTUATION_RE = re.compile(r"[\s，。！？、；：,.!?;:（）()\[\]“”\"'…—\-]")


class ScriptValidationError(ValueError):
    """Raised when a model output cannot be parsed as a script."""


@dataclass
class ScriptValidation:
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


def strip_code_fence(text: str) -> str:
    """Return the first ```json ... ``` block, or the original text."""
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.S | re.I)
    if match:
        return match.group(1).strip()
    return stripped


def extract_json(text: str) -> Any:
    """Parse JSON from plain text, a fenced block, or surrounding prose.

    This is the runtime counterpart to the brittle ``re.search('```json')``
    logic in the old app: it accepts bare JSON (what the SFT model emits) and
    fenced JSON (what the base/DPO model often emits).
    """
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Start at the *first* bracket only.  Scanning every ``{`` would salvage a
    # complete first shot from a truncated array and report it as valid JSON;
    # that is exactly the false positive this metric must avoid.
    stripped = cleaned.strip()
    start = next((i for i, char in enumerate(stripped) if char in "[{"), None)
    if start is None:
        raise ScriptValidationError(
            "no JSON value found in model output: %r" % stripped[:160]
        )
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(stripped[start:])
        return value
    except json.JSONDecodeError as exc:
        raise ScriptValidationError(
            "incomplete JSON (likely truncated): %r" % stripped[:160]
        ) from exc


def as_script_list(value: Any) -> List[Any]:
    """Normalize the common model output shapes to a list of shots."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("shots", "script", "data", "result"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        # A single shot object is accepted for convenience, but will normally
        # fail the minimum-shot check.
        if any(key in value for key in REQUIRED_FIELDS):
            return [value]
    raise ScriptValidationError(
        "expected a JSON list of shots, got %s" % type(value).__name__
    )


def normalize_shot(shot: Any) -> Dict[str, str]:
    if not isinstance(shot, dict):
        raise ScriptValidationError("shot is not an object: %r" % (shot,))
    normalized: Dict[str, str] = {}
    for key in REQUIRED_FIELDS:
        value = shot.get(key)
        if value is None:
            normalized[key] = ""
        elif isinstance(value, (dict, list)):
            normalized[key] = json.dumps(value, ensure_ascii=False)
        else:
            normalized[key] = str(value).strip()
    return normalized


def parse_time_axis(value: str) -> Optional[Tuple[int, int]]:
    """Parse ``MM:SS-MM:SS`` (or ``M:SS``) and return seconds."""
    if not isinstance(value, str):
        return None
    match = TIME_AXIS_RE.match(value)
    if not match:
        return None
    start = int(match.group(1)) * 60 + int(match.group(2))
    end = int(match.group(3)) * 60 + int(match.group(4))
    return start, end


def spoken_char_count(text: str) -> int:
    """Count characters used for the 3-4 chars/second pacing heuristic."""
    return len(PUNCTUATION_RE.sub("", text or ""))


def validate_script(
    value: Any,
    *,
    min_shots: int = 1,
    require_music: bool = True,
    max_chars_per_second: float = 4.0,
    enforce_pacing: bool = True,
    target_duration_sec: Optional[Tuple[int, int]] = None,
) -> ScriptValidation:
    """Validate a parsed script and return machine-readable metrics.

    ``ok`` means "passes the mechanical quality gate".  It does not mean the
    creative content is good.
    """
    errors: List[str] = []
    warnings: List[str] = []
    try:
        raw_shots = as_script_list(value)
    except ScriptValidationError as exc:
        return ScriptValidation(ok=False, errors=[str(exc)])

    shots: List[Dict[str, str]] = []
    for index, raw_shot in enumerate(raw_shots):
        try:
            shots.append(normalize_shot(raw_shot))
        except ScriptValidationError as exc:
            errors.append("shot %d: %s" % (index, exc))

    if len(shots) < min_shots:
        errors.append(
            "has %d shots, requires at least %d" % (len(shots), min_shots)
        )
    if len(shots) < 10:
        warnings.append("fewer than 10 shots (app prompt asks for >= 10)")

    gaps = 0
    overlaps = 0
    zero_duration = 0
    bad_time = 0
    empty_voiceover = 0
    empty_music = 0
    pacing_violations = 0
    pacing_rates: List[float] = []
    total_duration = 0
    last_end: Optional[int] = None
    shot_types = set()
    descriptions: List[str] = []

    for index, shot in enumerate(shots):
        for key in REQUIRED_FIELDS:
            if not shot.get(key):
                errors.append("shot %d: empty field %s" % (index, key))
        if shot.get("voiceover", "").strip() in EMPTY_VOICEOVER_VALUES:
            empty_voiceover += 1
            errors.append("shot %d: empty voiceover" % index)
        if shot.get("music", "").strip() in EMPTY_MUSIC_VALUES:
            empty_music += 1
            if require_music:
                errors.append("shot %d: empty music" % index)
        if shot.get("shot_type") in BAD_SHOT_TYPES:
            warnings.append(
                "shot %d: suspicious shot_type %r" % (index, shot["shot_type"])
            )

        span = parse_time_axis(shot.get("time_axis", ""))
        if span is None:
            bad_time += 1
            errors.append(
                "shot %d: invalid time_axis %r" % (index, shot.get("time_axis"))
            )
        else:
            start, end = span
            duration = end - start
            if duration <= 0:
                zero_duration += 1
                errors.append(
                    "shot %d: non-positive duration %r" % (index, shot["time_axis"])
                )
            else:
                total_duration = max(total_duration, end)
                chars = spoken_char_count(shot.get("voiceover", ""))
                if chars:
                    rate = chars / duration
                    pacing_rates.append(rate)
                    if rate > max_chars_per_second:
                        pacing_violations += 1
                        message = (
                            "shot %d: %.1f chars/sec exceeds %.1f"
                            % (index, rate, max_chars_per_second)
                        )
                        if enforce_pacing:
                            errors.append(message)
                        else:
                            warnings.append(message)
            if last_end is not None:
                if start > last_end:
                    gaps += 1
                    errors.append(
                        "shot %d: time gap %d..%d" % (index, last_end, start)
                    )
                elif start < last_end:
                    overlaps += 1
                    errors.append(
                        "shot %d: time overlap %d..%d" % (index, start, last_end)
                    )
            last_end = end

        shot_types.add(shot.get("shot_type", ""))
        descriptions.append(shot.get("visual_description", ""))

    duplicate_descriptions = len(descriptions) - len(set(descriptions))
    if duplicate_descriptions:
        warnings.append("%d duplicate visual_description(s)" % duplicate_descriptions)

    if target_duration_sec:
        low, high = target_duration_sec
        if total_duration and not (low <= total_duration <= high):
            warnings.append(
                "total duration %ds outside target %d-%ds"
                % (total_duration, low, high)
            )

    metrics = {
        "n_shots": len(shots),
        "total_duration_sec": total_duration,
        "gaps": gaps,
        "overlaps": overlaps,
        "zero_duration": zero_duration,
        "bad_time": bad_time,
        "empty_voiceover": empty_voiceover,
        "empty_music": empty_music,
        "pacing_violations": pacing_violations,
        "pacing_checked": len(pacing_rates),
        "mean_chars_per_second": (
            round(sum(pacing_rates) / len(pacing_rates), 2) if pacing_rates else None
        ),
        "max_chars_per_second": (
            round(max(pacing_rates), 2) if pacing_rates else None
        ),
        "distinct_shot_types": len(shot_types),
        "duplicate_descriptions": duplicate_descriptions,
    }
    return ScriptValidation(
        ok=not errors,
        shots=shots,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def count_hard_defects(script: Any) -> int:
    """Count mechanical defects; used to rank DPO chosen vs rejected."""
    validation = validate_script(
        script,
        min_shots=1,
        require_music=True,
        max_chars_per_second=4.0,
        enforce_pacing=False,
    )
    m = validation.metrics
    bad_types = 0
    try:
        for shot in as_script_list(script):
            if normalize_shot(shot).get("shot_type") in BAD_SHOT_TYPES:
                bad_types += 1
    except ScriptValidationError:
        bad_types += 1
    return bad_types + sum(
        int(m.get(key, 0))
        for key in (
            "gaps",
            "overlaps",
            "zero_duration",
            "bad_time",
            "empty_voiceover",
            "empty_music",
            "pacing_violations",
        )
    ) + len([e for e in validation.errors if "empty field" in e])


def script_to_text(shots: Sequence[Dict[str, str]]) -> str:
    """Compact JSON used as the SFT/DPO target (no code fence)."""
    return json.dumps(list(shots), ensure_ascii=False)


def validation_markdown(validation: ScriptValidation) -> str:
    m = validation.metrics
    status = "✅ 通过" if validation.ok else "❌ 未通过"
    lines = [
        "### 自检：%s" % status,
        "",
        "| 指标 | 值 |",
        "|---|---|",
        "| 镜头数 | %s |" % m.get("n_shots", 0),
        "| 总时长 | %ss |" % m.get("total_duration_sec", 0),
        "| 时间轴断裂 / 重叠 | %s / %s |"
        % (m.get("gaps", 0), m.get("overlaps", 0)),
        "| 0 秒镜头 | %s |" % m.get("zero_duration", 0),
        "| 空口播 | %s |" % m.get("empty_voiceover", 0),
        "| 空音乐 | %s |" % m.get("empty_music", 0),
        "| 语速超标 | %s |" % m.get("pacing_violations", 0),
    ]
    if validation.errors:
        lines += ["", "**错误**："] + ["- %s" % e for e in validation.errors[:8]]
    if validation.warnings:
        lines += ["", "**提示**："] + ["- %s" % w for w in validation.warnings[:8]]
    return "\n".join(lines)
