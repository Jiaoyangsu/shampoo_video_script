"""Deprecated app helpers.

Use :mod:`src.script_schema` for parsing/validation and
:mod:`src.inference.backends` for generation.  This module is kept as a thin
compatibility layer for old imports.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from src.script_schema import extract_json, validate_script, validation_markdown

__all__ = [
    "create_download_file",
    "read_prompt",
    "extract_json",
    "validate_script",
    "validation_markdown",
]


def create_download_file(json_data: Any) -> str:
    json_string = json.dumps(json_data, indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as temp_file:
        temp_file.write(json_string)
        return temp_file.name


def read_prompt(prompt_file: str) -> str:
    return Path(prompt_file).read_text(encoding="utf-8")
