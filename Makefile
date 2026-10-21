.PHONY: help test clean-data clean-data-strict eval-mock lint

help:
	@echo "make test              # unit tests (stdlib, no GPU)"
	@echo "make clean-data        # build data/clean from data/{train,val}"
	@echo "make clean-data-strict # >=10 shots (95 train / 13 val)"
	@echo "make eval-mock         # 30-prompt harness self-test, no model"
	@echo "make lint              # ruff (optional)"

test:
	python3 -m unittest discover -s tests -v

clean-data:
	python3 scripts/clean_data.py

clean-data-strict:
	python3 scripts/clean_data.py --min-shots 10 --out-dir data/clean_strict

eval-mock:
	python3 scripts/run_eval.py --backend mock --out-dir evaluation/results/mock_30

lint:
	ruff check src scripts tests
