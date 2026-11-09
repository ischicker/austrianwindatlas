"""Consistency checks across the three citation-metadata files.

The library ships three files that redundantly record who to cite
and under what DOI:

* ``CITATION.cff``   — the canonical source, read by GitHub's 'Cite
  this repository' button, Zotero, mkdocs-citation, etc.
* ``.zenodo.json``   — override picked up by Zenodo's GitHub
  integration when minting a release DOI.
* ``README.md``      — Zenodo badge URL + a ready-to-paste BibTeX
  snippet in the 'Citing' section.
* ``NOTICE``         — a 'Citing this software' footer.

The release flow in ``docs/RELEASING.md`` requires all four to agree
on the DOI and the author list.  These tests catch silent drift.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
CFF = REPO / "CITATION.cff"
ZENODO = REPO / ".zenodo.json"
README = REPO / "README.md"
NOTICE_FILE = REPO / "NOTICE"

#: Pre-release placeholder; replaced in Phase C of docs/RELEASING.md.
#: The tests below treat this string specially so they don't fail
#: while a release is in flight.
PENDING_SENTINEL = "10.5281/zenodo.PENDING"


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cff() -> dict:
    return yaml.safe_load(CFF.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def zen() -> dict:
    return json.loads(ZENODO.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def notice() -> str:
    return NOTICE_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# File existence + basic shape
# ---------------------------------------------------------------------------
def test_citation_cff_parses_as_cff_v1_2(cff):
    assert cff.get("cff-version") == "1.2.0"
    assert cff.get("type") == "software"
    assert cff.get("title") == "Austrian Windatlas"


def test_zenodo_json_has_required_fields(zen):
    for key in ("title", "description", "upload_type", "creators", "license"):
        assert key in zen, f"missing .zenodo.json key: {key}"
    assert zen["upload_type"] == "software"
    assert zen["license"] == "MIT"


# ---------------------------------------------------------------------------
# Author-list consistency across CFF and Zenodo
# ---------------------------------------------------------------------------
def _cff_author_name(author: dict) -> str:
    """CFF stores 'family-names' + 'given-names'; Zenodo uses 'Family, Given'."""
    family = author.get("family-names", "")
    given = author.get("given-names", "")
    return f"{family}, {given}".strip(", ")


def test_citation_and_zenodo_author_lists_match(cff, zen):
    cff_names = [_cff_author_name(a) for a in cff["authors"]]
    zen_names = [a["name"] for a in zen["creators"]]
    assert cff_names == zen_names, (
        f"CITATION.cff and .zenodo.json disagree on the author list.\n"
        f"CFF   : {cff_names!r}\n"
        f"Zenodo: {zen_names!r}"
    )


def test_citation_and_zenodo_lead_author_has_orcid(cff, zen):
    """Irene Schicker is the corresponding author — ORCID must land in
    both files so Zenodo's 'author identifiers' panel is populated."""
    cff_schicker = next(
        a for a in cff["authors"] if a.get("family-names") == "Schicker"
    )
    zen_schicker = next(
        a for a in zen["creators"] if a["name"].startswith("Schicker")
    )
    assert cff_schicker["orcid"].endswith("0000-0001-6401-2412")
    assert zen_schicker["orcid"] == "0000-0001-6401-2412"


# ---------------------------------------------------------------------------
# License consistency
# ---------------------------------------------------------------------------
def test_license_is_mit_everywhere(cff, zen):
    assert cff["license"] == "MIT"
    assert zen["license"] == "MIT"


# ---------------------------------------------------------------------------
# DOI consistency
# ---------------------------------------------------------------------------
def _collect_dois(cff: dict, readme: str, notice: str) -> set[str]:
    dois: set[str] = set()
    if "doi" in cff and cff["doi"]:
        dois.add(str(cff["doi"]))
    for ident in cff.get("identifiers", []):
        if ident.get("type") == "doi":
            dois.add(str(ident["value"]))
    # README mentions the DOI in a badge URL + a BibTeX block.  The
    # badge path reads ".../zenodo.<ID>.svg" — drop the ".svg" that
    # isn't part of the DOI itself.
    doi_pattern = re.compile(r"10\.5281/zenodo\.[A-Za-z0-9\-]+")
    for m in doi_pattern.finditer(readme):
        dois.add(m.group(0))
    for m in doi_pattern.finditer(notice):
        dois.add(m.group(0))
    return dois


def test_doi_is_consistent_across_all_files(cff, readme, notice):
    """CITATION.cff, README badges/BibTeX and NOTICE must all agree.

    During a release in flight both the PENDING sentinel and the real
    DOI can coexist in the tree for the span of ONE commit — the
    Phase-C commit is the one that flips them.  This test still fails
    in that intermediate state, which is the point: it's a release
    checklist item.
    """
    dois = _collect_dois(cff, readme, notice)
    assert dois, "no DOIs found in CITATION.cff / README / NOTICE"
    assert len(dois) == 1, (
        f"DOIs disagree across files (forgot to replace a PENDING "
        f"sentinel?): {sorted(dois)!r}"
    )


def test_pending_sentinel_stays_in_sync_or_is_fully_replaced(cff, readme, notice):
    """Either every DOI-bearing file still shows the pending sentinel
    (pre-mint state), or none of them do (post-mint state).  A mix is
    a bug (Phase-C commit left one file behind)."""
    has_sentinel = {
        "CITATION.cff": str(cff.get("doi", "")) == PENDING_SENTINEL,
        "README.md": PENDING_SENTINEL in readme,
        "NOTICE": PENDING_SENTINEL in notice,
    }
    unique = set(has_sentinel.values())
    assert len(unique) == 1, (
        f"PENDING sentinel is in some files but not others: {has_sentinel!r}. "
        "docs/RELEASING.md Phase C updates all three at once."
    )


# ---------------------------------------------------------------------------
# Repository-URL consistency
# ---------------------------------------------------------------------------
def test_repository_url_matches_readme_url(cff, readme):
    cff_url = cff.get("repository-code", "")
    assert "github.com/ischicker/austrianwindatlas" in cff_url
    assert "github.com/ischicker/austrianwindatlas" in readme


# ---------------------------------------------------------------------------
# Version consistency between CITATION.cff and pyproject.toml
# ---------------------------------------------------------------------------
def _project_version() -> str:
    """Parse pyproject.toml's [project].version without tomllib."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    # Minimal parser — only the top-level [project] table's version line.
    in_project = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            in_project = (s == "[project]")
            continue
        if in_project and s.startswith("version"):
            m = re.match(r'version\s*=\s*"([^"]+)"', s)
            if m:
                return m.group(1)
    raise AssertionError("[project].version not found in pyproject.toml")


def test_version_in_cff_matches_pyproject(cff):
    pp_version = _project_version()
    assert cff["version"] == pp_version, (
        f"CITATION.cff version {cff['version']!r} != pyproject.toml "
        f"[project].version {pp_version!r}"
    )


# ---------------------------------------------------------------------------
# release checklist discoverable
# ---------------------------------------------------------------------------
def test_releasing_doc_exists():
    doc = REPO / "docs" / "RELEASING.md"
    assert doc.is_file(), f"missing release doc: {doc}"
    text = doc.read_text(encoding="utf-8")
    assert "zenodo" in text.lower()
    assert "CITATION.cff" in text
