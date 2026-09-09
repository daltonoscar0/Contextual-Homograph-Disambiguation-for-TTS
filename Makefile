# First interpreter on PATH that is new enough for the pinned dependencies.
# Override with `make PYTHON=/path/to/python3.12`.
PYTHON ?= $(shell for p in python3.13 python3.12 python3.11 python3; do \
	command -v $$p >/dev/null 2>&1 || continue; \
	$$p -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null \
		&& { command -v $$p; break; }; \
	done)

VENV := .venv
PY := $(VENV)/bin/python

.PHONY: all venv data test baseline probe probe-large probe-serving eval \
	adversarial weights canonical clean distclean

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
	$(PY) -m pytest tests -q

baseline: venv
	$(PY) -m lede.baseline_pos

probe: venv
	$(PY) -m lede.probe

# Headline result. Slower: roberta-large is a ~1.3GB download and about 40
# minutes of CPU forward passes on first run, then cached like any other.
probe-large: venv
	LEDE_ENCODER=roberta-large $(PY) -m lede.run_all

# The two serving-sized encoders. Both are base-sized downloads and reproduce
# in roughly the time bert-base does.
probe-serving: venv
	LEDE_ENCODER=roberta-base $(PY) -m lede.run_all
	LEDE_ENCODER=distilroberta-base $(PY) -m lede.run_all

eval: venv
	$(PY) -m lede.run_all

adversarial: venv
	$(PY) adversarial/build.py
	$(PY) -m lede.adversarial

# Refit each encoder's probes and write the checked-in serving weights.
# Requires that encoder's embedding cache, i.e. a prior run_all.
weights: venv
	LEDE_ENCODER=distilroberta-base $(PY) -m lede.export_weights
	LEDE_ENCODER=roberta-base $(PY) -m lede.export_weights
	LEDE_ENCODER=roberta-large $(PY) -m lede.export_weights

canonical: venv
	$(PY) -m lede --canonical

clean:
	rm -rf cache __pycache__ lede/__pycache__ tests/__pycache__

distclean: clean
	rm -rf $(VENV) data/vendor
