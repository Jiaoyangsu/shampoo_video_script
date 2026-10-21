# Legacy evaluation

The 3-prompt comparison in this directory is kept as a historical record.
It had three methodological problems:

1. Only 3 prompts, no held-out test set.
2. The DPO adapter was loaded on the raw base model, not on merged SFT weights.
3. The SFT log came from a different generation wrapper (`SEGMENTS` field is not
   printed by the committed `eval_compare` script).

The supported evaluation is `scripts/run_eval.py` with
`evaluation/prompts_30.json`.
