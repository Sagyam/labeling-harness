"""The committed notebooks are what their builders write (D105): notebooks/src/build_*.py.

A generated notebook embeds the kits (ftkit.py, evalkit.py, ...) as %%writefile cells, so a kit
changed without a rebuild leaves a notebook that runs the old code in Colab. That happened to the
fine-tuning notebook in September 2026. These tests fail until `python notebooks/src/build_all.py`
has been run. No notebook runs here: a GPU is where that is checked. What can be checked without
one is that every code cell is Python that parses."""

from __future__ import annotations

import ast
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
sys.path.insert(0, str(_NOTEBOOKS / "src"))  # the builders import each other and the kits' text
build_all = importlib.import_module("build_all")
nbkit = importlib.import_module("nbkit")


def _pinned_commit() -> str | None:
    """The harness commit the committed notebooks pin, read back from the one that names it: a
    notebook cannot hold the hash of the commit it is part of, so it is built at the one before."""
    for path in sorted(_NOTEBOOKS.glob("*.ipynb")):
        found = re.search(r'HARNESS_COMMIT = \\"([0-9a-f]{40})\\"', path.read_text("utf-8"))
        if found:
            return found.group(1)
    return None


_BUILT = build_all.notebooks(_pinned_commit())


@pytest.mark.parametrize("name", sorted(_BUILT))
def test_a_committed_notebook_is_what_its_builder_writes(name: str) -> None:
    path = _NOTEBOOKS / name
    assert path.exists(), f"{name} is not committed: run notebooks/src/build_all.py"
    assert path.read_text("utf-8") == nbkit.dumps(_BUILT[name]), (
        f"{name} is stale: run `python notebooks/src/build_all.py`"
    )


def _python(source: str) -> str:
    """A cell's Python: a %%writefile cell is the file it writes, and a line of IPython magic or
    shell is a statement that does nothing."""
    lines = source.split("\n")
    if lines[0].startswith("%%writefile"):
        lines = lines[1:]
    return "\n".join(
        re.sub(r"^(\s*)[%!].*$", r"\1pass", line) if line.lstrip()[:1] in ("%", "!") else line
        for line in lines
    )


@pytest.mark.parametrize("name", sorted(_BUILT))
def test_every_code_cell_parses(name: str) -> None:
    for number, cell in enumerate(_BUILT[name]):
        if cell["cell_type"] != "code":
            continue
        try:
            ast.parse(_python(cell["source"]))
        except SyntaxError as exc:
            pytest.fail(f"{name}, cell {number}, line {exc.lineno}: {exc.msg}\n{exc.text}")


@pytest.mark.parametrize("name", sorted(_BUILT))
def test_no_cell_uses_a_name_the_notebook_never_defines(name: str, tmp_path: Path) -> None:
    """The cells of a notebook share one namespace, top to bottom. Read as one module, a name
    that no earlier cell defines or imports is a NameError waiting on the GPU."""
    cells = [c["source"] for c in _BUILT[name] if c["cell_type"] == "code"]
    module = "\n\n".join(_python(c) for c in cells if not c.startswith("%%writefile")) + "\n"
    assert _undefined_names(module, tmp_path / (Path(name).stem + ".py")) == ""


def _undefined_names(source: str, path: Path) -> str:
    """ruff's report of names `source` uses and never defines; empty when there are none."""
    path.write_text(source, "utf-8")
    command = [sys.executable, "-m", "ruff", "check", "--isolated", "--select", "F821,F823,E9"]
    found = subprocess.run(
        [*command, "--output-format", "concise", str(path)], capture_output=True, text=True
    )
    return "" if found.returncode == 0 else found.stdout


def test_the_omnilingual_script_is_a_whole_program(tmp_path: Path) -> None:
    """06e runs its stages as a script in another Python environment, assembled from the cells
    the other student notebooks run. It gets its Config as JSON, so every name a cell reads has
    to be one the script assigns."""
    script = importlib.import_module("build_students").OMNI_SCRIPT
    ast.parse(script)
    assert _undefined_names(script, tmp_path / "student_omni.py") == ""
    assert "get_ipython" not in script and "\n%" not in script and "\n!" not in script


def test_a_kit_cell_is_the_kit_file() -> None:
    cell = nbkit.kit("sweep")
    text = (_NOTEBOOKS / "src" / "sweep.py").read_text("utf-8")
    assert cell["source"] == ("%%writefile /content/ft/sweep.py\n" + text).strip("\n")


def test_every_notebook_is_generated_or_named_as_hand_written() -> None:
    committed = {p.name for p in _NOTEBOOKS.glob("*.ipynb")}
    assert committed == set(_BUILT) | set(build_all.HAND_WRITTEN)


def test_notebooks_are_numbered_in_the_order_they_run() -> None:
    names = sorted(set(_BUILT) | set(build_all.HAND_WRITTEN))
    assert all(re.match(r"\d\d[a-z]?_", name) for name in names), names
    assert [name[:2] for name in names] == sorted(name[:2] for name in names)
    assert {name[:2] for name in names} == {f"{step:02d}" for step in range(1, 8)}


def test_only_the_report_asks_for_no_gpu() -> None:
    cpu = [
        n for n, cells in _BUILT.items() if "accelerator" not in nbkit.notebook(cells)["metadata"]
    ]
    assert cpu == ["07_Report.ipynb"]


def test_a_built_notebook_is_valid_json_with_one_id_per_cell() -> None:
    name = sorted(_BUILT)[0]
    nb = json.loads(nbkit.dumps(_BUILT[name]))
    ids = [c["id"] for c in nb["cells"]]
    assert nb["nbformat"] == 4 and len(ids) == len(set(ids))


def test_every_student_scores_with_error_mining() -> None:
    """Error mining (D110) needs DuckDB wherever a student is scored: the Colab kernel, or for 06e
    the environment its script runs in. Without it the scores land and the error files silently
    do not, which is what 06e's first smoke run did."""
    students = importlib.import_module("build_students")
    assert re.search(r"uv pip install -q --python \{OMNI_PY\}[^\n]*\bduckdb\b", students.OMNI_ENV)
    for path in sorted(_NOTEBOOKS.glob("06*_Student_*.ipynb")):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        installs = [
            line
            for cell in cells
            for line in "".join(cell["source"]).splitlines()
            if re.match(r"\s*[%!].*pip install", line)
        ]
        assert any(re.search(r"\bduckdb\b", line) for line in installs), path.name
