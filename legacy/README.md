# Legacy / unverified code

These files are kept for history, not as supported entry points.

- `unsloth_trainer.py` / `unsloth_dpo_train.py`: old unsloth pipeline; the
  config/model did not match the Gemma 3 pipeline and the DPO tokenization path
  was unverified.
- `eval_base_3prompt.py` / `eval_compare_3prompt.py`: the old 3-prompt
  comparison.  The DPO adapter was loaded on the raw base, not on merged SFT,
  so the numbers should not be used for new decisions.
- `inference_test_3prompt.py`: old smoke test.

Use `scripts/run_eval.py`, `scripts/run_inference.py`, and
`src/finetune/gemma3/` instead.

- `prepare_dpo_data_legacy.py`: old DPO pair generator.  It used hard-coded
  `/Users/...` paths, a `### Instruction` template that did not match Gemma
  chat, and built DPO val from SFT train.  Use `scripts/clean_data.py`.
