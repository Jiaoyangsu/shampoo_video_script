.PHONY: help test clean-data clean-data-strict rewrite-pilot rewrite-review eval-mock lint

help:
	@echo "make test              # unit tests (stdlib, no GPU)"
	@echo "make clean-data        # build data/clean from data/{train,val}"
	@echo "make clean-data-strict # >=10 shots (95 train / 13 val)"
	@echo "make rewrite-pilot      # rewrite 20 samples to 30s/10-12 shots"
	@echo "make rewrite-review     # build review CSV from rewrite output"
	@echo "make eval-mock         # 30-prompt harness self-test, no model"
	@echo "make lint              # ruff (optional)"

test:
	python3 -m unittest discover -s tests -v

clean-data:
	python3 scripts/clean_data.py

clean-data-strict:
	python3 scripts/clean_data.py --min-shots 10 --out-dir data/clean_strict

rewrite-pilot:
	python3 scripts/rewrite_data.py --limit 20 --out-dir data/rewrite_30s --resume

rewrite-review:
	python3 scripts/summarize_rewrite.py --dir data/rewrite_30s --split train

eval-mock:
	python3 scripts/run_eval.py --backend mock --out-dir evaluation/results/mock_30

lint:
	ruff check src scripts tests
