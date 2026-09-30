"""Rebuild every generated notebook: `python notebooks/src/build_all.py`.

01_EDA and 02_Sociolinguistics are written by hand and have no builder. A builder is a module
with `NOTEBOOKS`, {file name: cells}, or a `notebooks(commit)` function when the notebook pins the
harness at a commit. backend/tests/test_notebook_builds.py checks that the committed notebooks are
what these write."""

import sys
from pathlib import Path

import build_flex
import build_predistill
import build_report
import build_students
import build_teacher
import nbkit

#: Notebooks written by hand, which no builder produces.
HAND_WRITTEN = ("01_EDA.ipynb", "02_Sociolinguistics.ipynb")


def notebooks(commit: str | None = None) -> dict[str, list]:
    """{file name: cells} of every generated notebook. `commit` is the harness commit the
    notebooks that fetch harness code pin; HEAD when not given."""
    return {
        **build_flex.NOTEBOOKS,
        **build_predistill.notebooks(commit),
        **build_teacher.NOTEBOOKS,
        **build_students.NOTEBOOKS,
        **build_report.NOTEBOOKS,
    }


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
    nbkit.write(notebooks(), out_dir)
