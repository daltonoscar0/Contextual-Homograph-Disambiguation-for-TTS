PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: all venv data test baseline probe eval clean distclean

all: data test eval

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt

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

clean:
	rm -rf cache __pycache__ lede/__pycache__ tests/__pycache__

distclean: clean
	rm -rf $(VENV) data/vendor
