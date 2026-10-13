.PHONY: install run test
install:
	pip install -e ".[dev]"
run:
	python -m tokenizer_tax.run
test:
	pytest -q
