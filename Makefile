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

.DEFAULT_GOAL := help
.PHONY: help notebook notebook-run export env clean-env

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
