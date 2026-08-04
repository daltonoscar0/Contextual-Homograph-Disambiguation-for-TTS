# First interpreter on PATH that is new enough for the pinned dependencies.
# Override with `make PYTHON=/path/to/python3.12`.
PYTHON ?= $(shell for p in python3.13 python3.12 python3.11 python3; do \
	command -v $$p >/dev/null 2>&1 || continue; \
	$$p -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null \
		&& { command -v $$p; break; }; \
	done)

VENV := .venv
PY := $(VENV)/bin/python

.PHONY: all venv data test baseline probe eval adversarial clean distclean

all: data test eval adversarial

$(VENV)/bin/activate: requirements.txt
	@if [ -z "$(PYTHON)" ]; then \
		echo "no python 3.11+ found on PATH; install one or run 'make PYTHON=...'"; \
		exit 1; \
	fi
	@echo "building venv with $(PYTHON) ($$($(PYTHON) -V))"
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt
	@touch $(VENV)/bin/activate

venv: $(VENV)/bin/activate

data:
	bash scripts/fetch_data.sh

test: venv
	$(PY) -m unittest discover -s tests -v

baseline: venv
	$(PY) -m lede.baseline_pos

probe: venv
	$(PY) -m lede.probe

eval: venv
	$(PY) -m lede.run_all

adversarial: venv
	$(PY) adversarial/build.py
	$(PY) -m lede.adversarial

clean:
	rm -rf cache __pycache__ lede/__pycache__ tests/__pycache__

distclean: clean
	rm -rf $(VENV) data/vendor
