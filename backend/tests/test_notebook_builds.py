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


def test_the_omnilingual_llm_script_is_a_whole_program(tmp_path: Path) -> None:
    """07b runs Omnilingual LLM-ASR's stages the way 06e runs the CTC model's (D122): one script
    assembled from the student cells, in fairseq2's environment, handed its Config as JSON."""
    students = importlib.import_module("build_students")
    script = students.OMNI_LLM_SCRIPT
    ast.parse(script)
    assert _undefined_names(script, tmp_path / "student_omni_llm.py") == ""
    assert "get_ipython" not in script and "\n%" not in script and "\n!" not in script
    assert "Seq2SeqBatch" in script and "import llmkit" in script


def test_the_bakeoff_omnilingual_script_is_a_whole_program(tmp_path: Path) -> None:
    """09 runs Omnilingual LLM-ASR as a script in fairseq2's environment (D121): it reads only its
    arguments and ftkit, so every name it uses has to be its own."""
    cell = next(
        c["source"]
        for c in _BUILT["09_SpeechLLM_Bakeoff.ipynb"]
        if c["source"].startswith("%%writefile /content/ft/bakeoff_omni.py")
    )
    script = cell.split("\n", 1)[1]
    ast.parse(script)
    assert _undefined_names(script, tmp_path / "bakeoff_omni.py") == ""
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
    assert {name[:2] for name in names} == {f"{step:02d}" for step in range(1, 10)}


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


def _student_notebooks() -> list[Path]:
    """Every student notebook: the ASR students (06) and the speech-LLM students (07, D122)."""
    found = sorted(_NOTEBOOKS.glob("0[67]*_Student_*.ipynb"))
    assert any(p.name.startswith("07") for p in found), "no speech-LLM student notebook"
    return found


def test_the_gemma_student_runs_on_transformers_5() -> None:
    """Gemma 4 needs transformers 5 (the bake-off ran it on >= 5.13); 06's qwen-asr pins 4.57.6,
    so the speech-LLM student must not inherit that install line."""
    cells = json.loads(
        (_NOTEBOOKS / "07a_Student_LLM_Gemma_E2B.ipynb").read_text(encoding="utf-8")
    )["cells"]
    installs = [
        line
        for cell in cells
        for line in "".join(cell["source"]).splitlines()
        if re.match(r"\s*[%!].*pip install", line)
    ]
    assert any(re.search(r'"transformers>=5\.13', line) for line in installs)
    assert not any("qwen-asr" in line for line in installs)


def test_every_student_scores_with_error_mining() -> None:
    """Error mining (D110) needs DuckDB wherever a student is scored: the Colab kernel, or for 06e
    the environment its script runs in. Without it the scores land and the error files silently
    do not, which is what 06e's first smoke run did."""
    students = importlib.import_module("build_students")
    assert re.search(r"uv pip install -q --python \{OMNI_PY\}[^\n]*\bduckdb\b", students.OMNI_ENV)
    for path in _student_notebooks():
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        installs = [
            line
            for cell in cells
            for line in "".join(cell["source"]).splitlines()
            if re.match(r"\s*[%!].*pip install", line)
        ]
        assert any(re.search(r"\bduckdb\b", line) for line in installs), path.name


def test_the_bakeoff_installs_what_voxtral_tokenizes_with() -> None:
    """Voxtral's tokenizer is mistral-common's: without the package transformers loads one with no
    chat template, and the candidate fails on its first clip (the 2026-10-07 smoke run)."""
    cells = json.loads((_NOTEBOOKS / "09_SpeechLLM_Bakeoff.ipynb").read_text(encoding="utf-8"))[
        "cells"
    ]
    installs = [
        line
        for cell in cells
        for line in "".join(cell["source"]).splitlines()
        if re.match(r"\s*[%!].*pip install", line)
    ]
    assert any(re.search(r"\bmistral-common\[audio\]", line) for line in installs)


def _config(name: str) -> dict:
    """The names a notebook's Config cell (its first code cell) assigns: plain assignments only."""
    cells = json.loads((_NOTEBOOKS / name).read_text(encoding="utf-8"))["cells"]
    namespace: dict = {}
    exec(next("".join(c["source"]) for c in cells if c["cell_type"] == "code"), namespace)
    return namespace


def test_only_the_scratch_conformer_names_its_own_stage3_recipe() -> None:
    """Stage 3 runs the augmentation that won Flex's ablation (D106), except where a student names
    its own (D115): the Conformer from scratch, with 03c's six acoustic stages at their tested
    strengths and crosstalk left out."""
    augment = importlib.import_module("augment")
    for path in _student_notebooks():
        recipe = _config(path.name)["STAGE3_RECIPE"]
        if path.name != "06f_Student_Conformer.ipynb":
            assert recipe is None, path.name
            continue
        assert recipe == {
            "speed": {"p": 0.5},
            "reverb": {"p": 0.3},
            "channel": {"p": 0.3},
            "noise": {"p": 0.5},
            "gain": {"p": 0.5},
            "codec": {"p": 0.3},
        }
        assert augment.AugmentConfig.from_dict(recipe).crosstalk.p == 0


def test_stage3_prefers_the_students_own_recipe_to_flexs_winner() -> None:
    students = importlib.import_module("build_students")
    tree = ast.parse(students.STAGES)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage3_recipe")
    ns: dict = {"winning_recipe": lambda: {"speed": {"p": 0.5}}, "STAGE3_RECIPE": None}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<STAGES>", "exec"), ns)
    assert ns["stage3_recipe"]() == {"speed": {"p": 0.5}}
    ns["STAGE3_RECIPE"] = {"gain": {"p": 0.5}}
    assert ns["stage3_recipe"]() == {"gain": {"p": 0.5}}


def test_only_the_scratch_conformer_takes_smaller_optimizer_steps() -> None:
    """06f sat on all-blank output for 14 epochs at ~12 min of audio per step, about 130 steps an
    epoch (2026-10-05): a model from random weights needs many updates, not big ones (D116). The
    pretrained students keep the recipe they ran with."""
    for path in sorted(_NOTEBOOKS.glob("06*_Student_*.ipynb")):
        effective_s = _config(path.name)["EFFECTIVE_S"]
        assert effective_s == (120.0 if path.name == "06f_Student_Conformer.ipynb" else 720.0)


def test_a_micro_batch_never_holds_more_audio_than_an_optimizer_step() -> None:
    """Steps are built from whole micro-batches, so a probe budget above EFFECTIVE_S made every step
    one micro-batch: on the 96 GB G4, ~1,400 s of audio instead of 720 (06f, 2026-10-05, D116). A
    clip-count budget (inf seconds, the HF students) is left alone."""
    students = importlib.import_module("build_students")
    tree = ast.parse(students.STAGES)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "micro_budget")
    ns: dict = {"EFFECTIVE_S": 120.0}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<STAGES>", "exec"), ns)
    assert ns["micro_budget"](1493.0) == 120.0
    assert ns["micro_budget"](90.0) == 90.0
    assert ns["micro_budget"](float("inf")) == float("inf")


def test_a_fresh_stage_starts_from_an_empty_output_folder(tmp_path: Path) -> None:
    """A stage's best-weights upload sends its whole local folder. On 2026-10-05 the third 06f
    attempt uploaded the first attempt's scores beside its own weights, because the folder on the
    Colab disk was never emptied between attempts."""
    students = importlib.import_module("build_students")
    tree = ast.parse(students.STAGES)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "clear_dir")
    ns: dict = {"Path": Path}  # a notebook global
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<STAGES>", "exec"), ns)
    (tmp_path / "harness" / "errors").mkdir(parents=True)
    (tmp_path / "harness" / "errors" / "val.parquet").write_text("old")
    (tmp_path / "result.json").write_text("old")
    ns["clear_dir"](tmp_path)
    assert tmp_path.is_dir() and not any(tmp_path.iterdir())
    assert "clear_dir(OUT)" in students.STAGES


def test_only_the_scratch_conformer_trains_stage_1_slower_and_without_specaugment() -> None:
    """On 2,000 clips for 2,000 steps (2026-10-05, D116): at lr 1e-3 a fresh 06f never listened;
    at 3e-4 with SpecAugment off its CTC head reached 31 train / 72 val CER, and with SpecAugment
    on it stayed at ~80, CTC alone or hybrid. Stage 2 starts from weights that already listen, so
    it keeps SpecAugment."""
    students = importlib.import_module("build_students")
    for path in sorted(_NOTEBOOKS.glob("06*_Student_*.ipynb")):
        config = _config(path.name)
        scratch = path.name == "06f_Student_Conformer.ipynb"
        assert config["SPECAUG_IN_HUMAN"] is (not scratch), path.name
        if scratch:
            assert config["LR_ENCODER"] == config["LR_HEADS"] == 3e-4
    assert 'stage == "human" and not SPECAUG_IN_HUMAN' in students.STAGES


def _stages_fn(name: str, **globals_: object):
    students = importlib.import_module("build_students")
    tree = ast.parse(students.STAGES)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns: dict = dict(globals_)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<STAGES>", "exec"), ns)
    return ns[name]


def test_the_fit_check_reads_the_same_train_clips_whatever_their_order() -> None:
    """06f, 2026-10-06: its stage 1 train loss fell to 0.33 while val sat at 33, which reads as
    overfitting, but loss is not WER. The fit check decodes train clips with a stage's weights;
    the same clips every time, spread over the train split in segment-id order, so whole episodes
    are not over- or under-drawn."""
    sample = _stages_fn("fit_sample")
    rows = [{"segment_id": f"ep{e}_{i:05d}"} for e in range(4) for i in range(25)]
    picked = sample(rows, 10)
    assert picked == sample(list(reversed(rows)), 10)
    assert len({r["segment_id"] for r in picked}) == 10
    assert [r["segment_id"][:3] for r in picked].count("ep0") in (2, 3)  # a quarter of 10
    assert sample(rows[:7], 10) == sorted(rows[:7], key=lambda r: r["segment_id"])


def test_the_fit_check_comes_after_every_other_cell() -> None:
    """Owners run cells by index (a smoke of stage 3 is a list of them), so the check is appended
    to the end of each student notebook and moves nothing. Omnilingual runs as one script and has
    no check."""
    for path in sorted(_NOTEBOOKS.glob("06*_Student_*.ipynb")):
        cells = json.loads(path.read_text("utf-8"))["cells"]
        sources = ["".join(c["source"]) for c in cells]
        calls = [i for i, s in enumerate(sources) if s.startswith("fit_check(")]
        if path.name == "06e_Student_Omnilingual.ipynb":
            assert not calls
            continue
        assert calls == [len(cells) - 1], path.name
