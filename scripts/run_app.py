#!/usr/bin/env python3
"""Start the Gradio UI.

Examples
--------
# Local demo with Ollama
OLLAMA_MODEL=gemma_3_12B_q80:latest python scripts/run_app.py

# Server with the actual fine-tuned adapters
SCRIPT_BACKEND=transformers \
SCRIPT_BASE_MODEL=/root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master \
SCRIPT_SFT_ADAPTER=/root/autodl-tmp/outputs/gemma3_sft/final \
SCRIPT_DPO_ADAPTER=/root/autodl-tmp/outputs/gemma3_dpo/final/dpo_adapter \
python scripts/run_app.py --port 9999
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.chat_ui import build_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "9999"))
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=os.environ.get("GRADIO_SHARE", "0") == "1",
    )
    args = parser.parse_args()

    demo = build_app()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
