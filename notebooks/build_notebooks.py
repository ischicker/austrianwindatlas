#!/usr/bin/env python3
"""(Re)generate the notebooks under this directory from the examples/*.py scripts.

Running this with no args rebuilds `notebooks/01_synthetic_end_to_end.ipynb`
from `examples/01_synthetic_end_to_end.py`.

The notebook is derived — never hand-edit the `.ipynb` file; edit the `.py`
and rebuild.  We keep the `.ipynb` in git because Colab needs it at a
GitHub URL (no post-clone build step).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "01_synthetic_end_to_end.py"
NOTEBOOK = REPO / "notebooks" / "01_synthetic_end_to_end.ipynb"

# Install cell: Colab has numpy + xarray out of the box, so we only need
# to pip-install the library itself.  Uses `pip install git+...` so the
# notebook auto-tracks main after the repo is live.  No network hit when
# running locally with the library already installed (pip detects it).
INSTALL_CELL_SOURCE = (
    "# Install the library on Colab.  No-op locally (or anywhere else\n"
    "# the package is already importable) so the notebook is safe to\n"
    "# 'Run all' from a dev checkout.\n"
    "try:\n"
    "    import austrianwindatlas  # noqa: F401\n"
    "except ImportError:\n"
    "    %pip install -q git+https://github.com/ischicker/austrianwindatlas.git"
)

INTRO_MARKDOWN = (
    "# Austrian Windatlas — synthetic end-to-end demo\n"
    "\n"
    "Exercises the Stage-1 pipeline\n"
    "(`core.rpca` → `stage1.rpca_eof` → `stage1.reconstruct`) on fully\n"
    "synthetic data — no external files are read.  Runs in ~2 seconds\n"
    "on a free-tier Colab CPU.\n"
    "\n"
    "**Pipeline**\n"
    "\n"
    "1. Build a synthetic station matrix (truth = climatology + low-rank signal).\n"
    "2. Add sparse column-storm contamination (1 % of entries).\n"
    "3. Run Robust PCA to split L (low-rank) + S (sparse).\n"
    "4. Extract EOFs from L via SVD.\n"
    "5. Interpolate station loadings to a grid with IDW.\n"
    "6. Project held observations onto the trained EOFs.\n"
    "7. Reconstruct wind(t, y, x) on the grid.\n"
    "8. Compare against the known truth — expects r ≈ 1 and RMSE ≈ 0.07 m/s.\n"
    "\n"
    "Open in Colab: click the badge in the top-level README."
)


def _strip_shebang_and_module_docstring(src: str) -> str:
    """Return everything after the first ``\"\"\"...\"\"\"`` block in ``src``.

    The module docstring is already rendered into the intro markdown, so
    dropping it from the first code cell keeps the notebook clean.
    """
    src = re.sub(r"^#!.*\n", "", src, count=1)  # shebang
    # Find and drop the first triple-quoted block.
    match = re.search(r'^"""', src, flags=re.MULTILINE)
    if match is None:
        return src.lstrip()
    start = match.start()
    end = src.find('"""', start + 3)
    if end == -1:
        return src.lstrip()
    return src[end + 3:].lstrip()


def _split_pipeline(body: str) -> list[tuple[str, str]]:
    """Carve ``body`` into labelled chunks at the big banner comments.

    Returns a list of ``(markdown_heading, code)`` pairs.  The first chunk
    (everything before the first banner) is returned with an empty
    heading so the caller can emit it as a plain code cell.
    """
    sep = re.compile(r"^# [=\-]{5,}\n(# [^\n]+)\n# [=\-]{5,}\n", flags=re.MULTILINE)
    cuts: list[tuple[int, int, str]] = [
        (m.start(), m.end(), m.group(1).lstrip("# ").strip())
        for m in sep.finditer(body)
    ]
    if not cuts:
        return [("", body.strip() + "\n")]
    chunks: list[tuple[str, str]] = []
    first_end = cuts[0][0]
    pre = body[:first_end].strip()
    if pre:
        chunks.append(("", pre + "\n"))
    for i, (_, end_after, heading) in enumerate(cuts):
        next_start = cuts[i + 1][0] if i + 1 < len(cuts) else len(body)
        block = body[end_after:next_start].strip()
        if block:
            chunks.append((heading, block + "\n"))
    return chunks


def build() -> Path:
    src = EXAMPLE.read_text(encoding="utf-8")
    body = _strip_shebang_and_module_docstring(src)
    chunks = _split_pipeline(body)

    nb = nbformat.v4.new_notebook()
    cells = [
        nbformat.v4.new_markdown_cell(INTRO_MARKDOWN),
        nbformat.v4.new_code_cell(INSTALL_CELL_SOURCE),
    ]
    for heading, code in chunks:
        if heading:
            cells.append(nbformat.v4.new_markdown_cell(f"## {heading}"))
        cells.append(nbformat.v4.new_code_cell(code.rstrip()))
    # Final driver cell so the notebook can be "Run all" in Colab.
    cells.append(nbformat.v4.new_markdown_cell("## Run the pipeline"))
    cells.append(nbformat.v4.new_code_cell("rc = main()\nassert rc == 0, f'pipeline failed with rc={rc}'"))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }

    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    # Pretty-print + strip the random per-cell IDs so repeat builds
    # are byte-identical (verified by tests/unit/test_notebooks.py).
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for i, cell in enumerate(data.get("cells", [])):
        cell["id"] = f"cell-{i}"
    NOTEBOOK.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return NOTEBOOK


if __name__ == "__main__":
    out = build()
    print(f"wrote {out.relative_to(REPO)}")
