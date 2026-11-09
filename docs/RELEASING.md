# Releasing a new version

Release numbering follows SemVer: ``MAJOR.MINOR.PATCH``.  Pre-1.0
releases bump ``MINOR`` for features / breaking changes and ``PATCH``
for bug-fix-only drops.

## One-time setup — Zenodo integration

1. Open <https://zenodo.org/account/settings/github/>.
2. Toggle **ischicker/austrianwindatlas** on.  Zenodo will then
   watch the repo's GitHub releases and mint a DOI for each one.
3. Confirm **Settings → Webhooks** on the repo shows a Zenodo entry.
   Delivery history should show ``✅ 200`` after each new release.

Zenodo picks up its metadata from, in priority order:
1. ``.zenodo.json`` in the repo root (what we ship — authors + ORCIDs
   + keywords + description).
2. ``CITATION.cff``.
3. The repo description + README on GitHub.

Keep ``.zenodo.json``, ``CITATION.cff`` and ``NOTICE`` in sync.  The
test ``tests/unit/test_citation_metadata.py`` catches drift between
the CFF and ``.zenodo.json`` author lists.

### Community membership (``windatlas_at``)

Each release auto-joins the Zenodo **windatlas_at** community
(<https://zenodo.org/communities/windatlas_at>).  The slug is baked
into ``.zenodo.json`` under the ``communities`` key.

* If the submitter is a **curator** of the community, the record is
  auto-approved at mint time and appears in the community listing
  immediately.
* If the submitter is only a **member**, Zenodo mints the DOI as
  usual (the DOI is final from the start) and notifies curators of
  a pending inclusion request.  The record only shows up in the
  community after a curator clicks **Accept** on the request.

To add or remove communities, edit ``.zenodo.json`` under
``communities`` *and* update the hard-coded slug check in
``tests/unit/test_citation_metadata.py`` in the same commit.

## Per-release checklist

### Phase A — pre-tag preparation (one commit)

- [ ] Make sure ``main`` is clean and CI is green.
- [ ] Bump the version:
    - [ ] ``src/austrianwindatlas/__init__.py`` ``__version__``
    - [ ] ``pyproject.toml`` ``[project]`` ``version``
    - [ ] ``CITATION.cff`` ``version`` + ``date-released``
    - [ ] ``.zenodo.json`` (no version field needed — Zenodo pulls
      it from the GitHub release tag)
- [ ] Update the "Status" blurb in ``README.md`` with the new counts
      (number of tests passing, which methods are ported, etc.).
- [ ] Add a ``CHANGELOG.md`` entry (create the file on first release).
- [ ] If the DOI sentinel ``10.5281/zenodo.PENDING`` is still in the
      repo from a previous release, leave it — it will be replaced
      in Phase C.  If it has been resolved to a real DOI, update
      all three files (README badge, CITATION.cff, NOTICE) to the
      *new* concept DOI (same for every release of the same project).

Commit with: ``release: prepare vX.Y.Z``.

### Phase B — tag + GitHub release (triggers Zenodo mint)

```bash
git tag -a vX.Y.Z -m "Austrian Windatlas vX.Y.Z"
git push origin vX.Y.Z
```

Then create the GitHub release:

1. Open <https://github.com/ischicker/austrianwindatlas/releases/new>.
2. Choose the ``vX.Y.Z`` tag.
3. Title: ``vX.Y.Z — <short tagline>``.
4. Body: the ``CHANGELOG.md`` section for this release, plus the
   "Citing" block from README for easy copy-paste.
5. Click **Publish release**.

Zenodo picks up the release event within a minute or two and mints
the DOI.  Watch the "Deliveries" tab of the Zenodo webhook on
<https://github.com/ischicker/austrianwindatlas/settings/hooks> —
you want a ``✅ 200`` response.

### Phase C — post-DOI fix-up (second commit)

Once Zenodo has minted the DOI (visible on <https://zenodo.org/account/settings/github/>
next to the repo entry):

- [ ] Replace **every** occurrence of ``10.5281/zenodo.PENDING`` with
      the **concept DOI** (the DOI that points at "latest release"):

      ```
      grep -rn "zenodo.PENDING" .
      ```

      Files expected: ``README.md`` (badge + BibTeX), ``CITATION.cff``
      (``doi:`` and ``identifiers:``), ``NOTICE`` (Citing block).

- [ ] Double-check the badge resolves by opening the README on
      GitHub — the Zenodo badge should turn into a real DOI link.

Commit with: ``release: lock Zenodo DOI for vX.Y.Z (<concept-DOI>)``.

## Yanking a bad release

Zenodo DOIs are permanent and cannot be deleted, only *made private*.
If you need to pull a release:

1. On Zenodo, go to the record and **Edit → restricted access**.
2. On GitHub, delete the release *and* the tag:

   ```bash
   git tag -d vX.Y.Z
   git push origin :refs/tags/vX.Y.Z
   ```

3. Prepare a new release with the fix and a bumped PATCH number.

Never reuse a tag name — Zenodo would mint a *second* DOI under the
same version string and the history of record gets confusing.
