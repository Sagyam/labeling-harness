# Convenience targets for the analysis notebooks. The backend, frontend and test suite are driven
# by docker compose, npm and pytest as documented in AGENTS.md; nothing here wraps those.

VENV        := notebooks/.venv
PY          := $(VENV)/bin/python
JUPYTER     := $(VENV)/bin/jupyter
STAMP       := $(VENV)/.requirements-stamp
KERNEL_NAME := labeling-harness-eda
NOTEBOOK    := notebooks/01-EDA.ipynb
EXPORT      := exports/analytics/analytics.jsonl
PORT        ?= 8888

# 02-diarization.ipynb needs torch/speechbrain, which the EDA env deliberately does not carry.
DIAR_VENV   := notebooks/.venv-diar
DIAR_PY     := $(DIAR_VENV)/bin/python
DIAR_JUP    := $(DIAR_VENV)/bin/jupyter
DIAR_STAMP  := $(DIAR_VENV)/.requirements-stamp
DIAR_KERNEL := labeling-harness-diar
DIAR_NB     := notebooks/02-diarization.ipynb

.DEFAULT_GOAL := help
.PHONY: help notebook notebook-run diar diar-run diar-env export env clean-env

help:  ## Show the available targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

notebook: env $(EXPORT)  ## Open the EDA notebook in a browser
	$(JUPYTER) lab --port $(PORT) $(NOTEBOOK)

notebook-run: env $(EXPORT)  ## Execute the notebook headlessly, in place
	$(JUPYTER) execute --inplace --kernel_name=$(KERNEL_NAME) $(NOTEBOOK)
	@# Drop per-cell execution timestamps. They change on every run, so leaving them in turns
	@# a re-run of a committed notebook into a 79-line diff that says nothing about the results.
	@$(PY) -c "import nbformat, sys; nb = nbformat.read(sys.argv[1], as_version=4); \
		[c['metadata'].pop('execution', None) for c in nb.cells]; \
		nbformat.write(nb, sys.argv[1])" $(NOTEBOOK)

diar: diar-env $(EXPORT)  ## Open the speaker-diarization notebook in a browser
	$(DIAR_JUP) lab --port $(PORT) $(DIAR_NB)

diar-run: diar-env $(EXPORT)  ## Execute the diarization notebook headlessly, in place
	$(DIAR_JUP) execute --inplace --kernel_name=$(DIAR_KERNEL) $(DIAR_NB)
	@$(DIAR_PY) -c "import nbformat, sys; nb = nbformat.read(sys.argv[1], as_version=4); \
		[c['metadata'].pop('execution', None) for c in nb.cells]; \
		nbformat.write(nb, sys.argv[1])" $(DIAR_NB)

diar-env: $(DIAR_STAMP)  ## Create notebooks/.venv-diar and register its Jupyter kernel

$(DIAR_STAMP): notebooks/requirements-diarization.txt
	@test -x $(DIAR_PY) || uv venv --python 3.13 $(DIAR_VENV)
	@# torch comes from the CPU index; unsafe-best-match lets the rest resolve from PyPI, which
	@# uv otherwise refuses because the first index that carries a name wins.
	VIRTUAL_ENV=$(DIAR_VENV) uv pip install --quiet -r notebooks/requirements-diarization.txt \
		--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
	$(DIAR_PY) -m ipykernel install --user \
		--name $(DIAR_KERNEL) --display-name "labeling-harness (diarization)"
	@touch $@

export:  ## Re-export every dataset kind from Postgres (needs the stack up)
	backend/.venv/bin/python scripts/export_dataset.py --kind all

# Only builds the export when it is missing; `make export` refreshes it deliberately.
$(EXPORT):
	@echo "No analytics export yet — building one (Postgres must be up)."
	@$(MAKE) --no-print-directory export

env: $(STAMP)  ## Create notebooks/.venv and register its Jupyter kernel

$(STAMP): notebooks/requirements.txt
	@test -x $(PY) || uv venv --python 3.13 $(VENV)
	VIRTUAL_ENV=$(VENV) uv pip install --quiet -r notebooks/requirements.txt
	$(PY) -m ipykernel install --user \
		--name $(KERNEL_NAME) --display-name "labeling-harness (EDA)"
	@touch $@

clean-env:  ## Remove the notebook virtualenv and its kernel
	-$(PY) -m jupyter kernelspec remove -f $(KERNEL_NAME)
	rm -rf $(VENV)
