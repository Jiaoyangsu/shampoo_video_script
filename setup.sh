#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 - <<'PYCHECK'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required (Gradio 5+). Current: %s" % sys.version.split()[0])
PYCHECK
python3 --version
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

echo
echo "Core environment ready."
echo "Training on a GPU server:"
echo "  source .venv/bin/activate"
echo "  pip install -r requirements-train.txt"
echo
echo "Run:"
echo "  python scripts/clean_data.py"
echo "  python scripts/run_eval.py --backend mock --limit 3 --out-dir evaluation/results/mock"
echo "  OLLAMA_MODEL=<model> python scripts/run_app.py"
