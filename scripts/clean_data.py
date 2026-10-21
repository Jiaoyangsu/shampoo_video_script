#!/usr/bin/env python3
"""Clean the SFT data and build leak-free DPO pairs.

Usage:
    python scripts/clean_data.py
    python scripts/clean_data.py --min-shots 10 --out-dir data/clean_strict
    python scripts/clean_data.py --check
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
