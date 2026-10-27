"""Shared prompt construction for SFT, DPO, evaluation and the app.

Keeping this in one place avoids the old mismatch where training saw only the
product brief but the app added "时长约 X 秒".
"""
from __future__ import annotations

from typing import Optional, Sequence, Union


def target_hint(
    duration: Union[int, str, Sequence[int]] = 30,
    shots: Optional[Union[int, str]] = None,
) -> str:
    if isinstance(duration, (list, tuple)):
        duration = "%s–%s" % (duration[0], duration[-1])
    text = "目标时长：%s 秒" % duration
    if shots is not None:
        text += "；镜头数：%s 个" % shots
    return text + "。"


def build_user_content(
    product_brief: str,
    *,
    duration: int = 30,
    shots: Optional[Union[int, str]] = None,
) -> str:
    return "%s\n\n产品信息：\n%s" % (
        target_hint(duration, shots),
        product_brief.strip(),
    )


def build_chat_prompt(system_prompt: str, product_brief: str, **kwargs) -> str:
    return "%s\n\n%s" % (system_prompt.strip(), build_user_content(product_brief, **kwargs))
