"""Config loading for the Gemma 3 SFT/DPO scripts (stdlib + PyYAML)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "finetune_config.yaml"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "src" / "app" / "system_prompt.md"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for training configs; pip install -r requirements-train.txt"
        ) from exc
    config_path = Path(path) if path else DEFAULT_CONFIG
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping: %s" % config_path)
    return data


def resolve_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path))


def read_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
