# Evaluation outputs

Generated directories are gitignored.  Each run writes:

- `raw.jsonl` — one model output per prompt
- `scored.json` — per-prompt objective metrics
- `summary.json` — aggregate metrics
- `report.md` — human-readable report
- `manifest.json` — git commit, prompt/system-prompt hashes, decoding config

Run the harness self-test without a model:

```bash
python scripts/run_eval.py --backend mock --limit 3 --out-dir evaluation/results/mock
```

Run the real model on AutoDL:

```bash
python scripts/run_eval.py --backend transformers \
  --base-model /root/autodl-tmp/models/LLM-Research--gemma-3-12b-it/snapshots/master \
  --sft-adapter /root/autodl-tmp/outputs/gemma3_sft/final \
  --dpo-adapter /root/autodl-tmp/outputs/gemma3_dpo/final/dpo_adapter \
  --max-new-tokens 700 \
  --out-dir evaluation/results/sft_dpo
```
