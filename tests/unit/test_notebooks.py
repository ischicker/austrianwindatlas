"""Static checks on the Colab notebooks under ``notebooks/``.

Does NOT execute the notebooks (the install cell talks to GitHub,
which isn't a CI concern and would be slow).  It only verifies:

* the file parses as nbformat v4;
* it contains an install cell that is idempotent (safe to re-run);
* it contains the "OK." marker that the example asserts on;
* every code cell is valid Python syntax (``compile`` succeeds).

The build script itself is also unit-tested — if it changes, running
``python notebooks/build_notebooks.py`` must still reproduce the
committed ``.ipynb`` byte-for-byte (verified in the last test below).
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks" / "01_synthetic_end_to_end.ipynb"
BUILDER = REPO / "notebooks" / "build_notebooks.py"


def _nb():
    assert NOTEBOOK.is_file(), f"notebook missing: {NOTEBOOK}"
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_parses_as_v4():
    nb = _nb()
    assert nb.nbformat == 4
    assert len(nb.cells) >= 3, "expected at least an intro + install + code cell"


def test_notebook_has_idempotent_install_cell():
    nb = _nb()
    install_cells = [
        c for c in nb.cells
        if c.cell_type == "code" and "pip install" in c.source and "austrianwindatlas" in c.source
    ]
    assert len(install_cells) == 1, "expected exactly one install cell"
    src = install_cells[0].source
    # Must short-circuit when the library is already importable.
    assert "import austrianwindatlas" in src
    assert "except ImportError" in src


def test_notebook_mentions_ok_marker():
    nb = _nb()
    sources = "\n".join(c.source for c in nb.cells)
    # The driver cell asserts rc == 0 which is set by the "OK." branch
    # in the example.  Regression guard that we didn't drop the main()
    # entry point from the script → notebook port.
    assert "main()" in sources
    assert "rc == 0" in sources or "assert rc == 0" in sources


def test_all_code_cells_are_syntactically_valid():
    nb = _nb()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        # Replace jupyter line-magics (``%pip``, ``!cmd``) with ``pass``
        # so empty indented blocks (e.g. the only line in an ``except:``)
        # still parse.
        cleaned_lines = []
        for line in cell.source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%", "!")):
                indent = line[: len(line) - len(stripped)]
                cleaned_lines.append(f"{indent}pass")
            else:
                cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines)
        try:
            ast.parse(cleaned, filename=f"cell-{i}")
        except SyntaxError as e:  # pragma: no cover — surfaces on breakage
            pytest.fail(f"code cell {i} has a SyntaxError: {e}")


def test_build_script_is_idempotent(tmp_path):
    """Running the builder must regenerate the committed notebook byte-for-byte.

    Protects against people hand-editing the ``.ipynb`` — all changes
    must come through the ``.py`` source and the builder.
    """
    committed = NOTEBOOK.read_bytes()
    result = subprocess.run(
        [sys.executable, str(BUILDER)],
        capture_output=True, text=True, timeout=30,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    rebuilt = NOTEBOOK.read_bytes()
    assert rebuilt == committed, (
        "`notebooks/build_notebooks.py` would change the committed notebook; "
        "re-run the builder and commit the result."
    )
