"""Python bridge to the GAM / BAMLSS EOF interpolators written in R.

The two R scripts in the ``R/`` directory (``gam_rpca_eof.R`` using
``mgcv::bam``, ``bamlss_rpca_eof.R`` using ``bamlss``) are the production
interpolators for the GAM+ERA5 family that the paper delivers.  They
are kept as R because (a) ``bamlss`` has no first-class Python port,
(b) the ``mgcv::bam`` thin-plate-spline implementation is substantially
more mature in R than any Python re-implementation, and (c) keeping them
verbatim preserves bit-for-bit reproducibility of the paper's numbers.

This module wraps them with a Python-friendly API.

Contract
--------
The R scripts expect CSVs in the standard rPCA export layout and write
a single NetCDF of ``(n_eofs, n_rows, n_cols)`` interpolated EOF grids
plus a 2-D ``spatial_mean`` grid.  The Python wrapper writes the CSVs
to a scratch directory, invokes ``Rscript`` and returns the path to
the NetCDF (or loads it into an :class:`xarray.Dataset` on request).

Two high-level entry points are provided:

:func:`run_bamlss`
    Invoke ``R/bamlss_rpca_eof.R`` on prepared rPCA exports.

:func:`run_gam`
    Invoke ``R/gam_rpca_eof.R`` on prepared rPCA exports.

Both accept an :class:`RExportInputs` struct built from in-memory
arrays (station coordinates, EOF coefficients, spatial mean) via
:func:`prepare_rpca_export`; the struct knows how to materialise the
three CSVs the R scripts expect.

The R scripts and their licences are not shipped by this package in
PyPI (R code lives in the ``R/`` directory of the source tree); the
Docker image documented in the top-level ``Dockerfile`` bundles R, the
needed CRAN packages (``bamlss``, ``mgcv``, ``terra``, ``ncdf4``,
``parallel``) and this Python code together.  ``bamlss`` and ``mgcv``
are GPL; see ``NOTICE``.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "RExportInputs",
    "RScriptResult",
    "prepare_rpca_export",
    "r_script_path",
    "rscript_executable",
    "run_bamlss",
    "run_gam",
    "load_eof_grids",
]

logger = logging.getLogger(__name__)

#: Directory inside the installed package / source tree that holds the
#: R scripts.  Resolved relative to this module so the package stays
#: importable from any working directory.
_R_DIR = Path(__file__).resolve().parents[3] / "R"


# ---------------------------------------------------------------------------
# Path / executable discovery
# ---------------------------------------------------------------------------
def r_script_path(name: Literal["bamlss_rpca_eof", "gam_rpca_eof"]) -> Path:
    """Return the on-disk path to one of the bundled R scripts.

    The scripts live in ``$REPO/R/`` relative to the source tree.  When
    the package is installed as a wheel, the same path is reachable
    via :func:`austrianwindatlas.core.paths.resolve` if the wheel is
    built with ``package_data`` picking them up — the ``pyproject.toml``
    is already configured for that.

    Raises
    ------
    FileNotFoundError
        If the script cannot be located; usually a sign that the
        package was imported from an unexpected layout.
    """
    path = _R_DIR / f"{name}.R"
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected bundled R script at {path!s} but it is missing. "
            "Check that the package R/ directory was shipped with the "
            "wheel; see pyproject.toml [tool.setuptools.package-data]."
        )
    return path


def rscript_executable(override: str | None = None) -> str:
    """Return the path to the ``Rscript`` executable.

    Discovery order:

    1. ``override`` argument if supplied.
    2. ``AWA_RSCRIPT`` environment variable.
    3. ``shutil.which("Rscript")``.

    Returns the plain string ``"Rscript"`` as a last-ditch default so
    an informative error comes from :mod:`subprocess` rather than from
    this module when R is simply absent.
    """
    if override:
        return override
    env = os.environ.get("AWA_RSCRIPT")
    if env:
        return env
    found = shutil.which("Rscript")
    return found or "Rscript"


# ---------------------------------------------------------------------------
# rPCA-export packaging (what the R scripts expect on disk)
# ---------------------------------------------------------------------------
@dataclass
class RExportInputs:
    """In-memory counterpart of the three CSVs the R scripts expect.

    ``eof_coefficients_training.csv`` ← ``station_id``, ``eof_1`` ... ``eof_K``
    ``station_metadata.csv``          ← ``station_id``, ``x``, ``y``
    ``spatial_mean_field.csv``        ← ``station_id``, ``mean_wind``

    Call :meth:`write` to materialise them under a directory whose basename
    is ``year_<YYYY>`` (the R scripts parse the year from that name).
    """

    station_id: NDArray[np.integer] | NDArray[np.str_]
    station_x: NDArray[np.floating]
    station_y: NDArray[np.floating]
    eof_coefficients: NDArray[np.floating]
    """Shape ``(n_stations, K)``; column k is EOF :math:`k+1`."""
    spatial_mean: NDArray[np.floating]
    """Shape ``(n_stations,)``: per-station climatological mean wind."""

    def __post_init__(self) -> None:
        n = len(self.station_id)
        if self.station_x.shape != (n,):
            raise ValueError(f"station_x must have length {n}; got {self.station_x.shape!r}")
        if self.station_y.shape != (n,):
            raise ValueError(f"station_y must have length {n}; got {self.station_y.shape!r}")
        if self.eof_coefficients.ndim != 2 or self.eof_coefficients.shape[0] != n:
            raise ValueError(
                f"eof_coefficients must be 2-D with {n} rows; "
                f"got shape {self.eof_coefficients.shape!r}"
            )
        if self.spatial_mean.shape != (n,):
            raise ValueError(
                f"spatial_mean must have length {n}; got {self.spatial_mean.shape!r}"
            )

    @property
    def n_stations(self) -> int:
        return len(self.station_id)

    @property
    def n_eofs(self) -> int:
        return int(self.eof_coefficients.shape[1])

    def write(self, rpca_dir: str | os.PathLike) -> Path:
        """Write the three CSVs under ``rpca_dir`` and return it.

        ``rpca_dir`` must end in ``year_<YYYY>`` for the R scripts to
        parse the target year correctly from the directory name
        (they do ``as.integer(gsub(".*year_", "", opt$rpca_dir))``).
        """
        out = Path(rpca_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._check_year_dirname(out.name)

        eof_cols = [f"eof_{k + 1}" for k in range(self.n_eofs)]
        eof_df = pd.DataFrame(self.eof_coefficients, columns=eof_cols)
        eof_df.insert(0, "station_id", self.station_id)
        eof_df.to_csv(out / "eof_coefficients_training.csv", index=False)

        meta_df = pd.DataFrame({
            "station_id": self.station_id,
            "x": self.station_x,
            "y": self.station_y,
        })
        meta_df.to_csv(out / "station_metadata.csv", index=False)

        mean_df = pd.DataFrame({
            "station_id": self.station_id,
            "mean_wind": self.spatial_mean,
        })
        mean_df.to_csv(out / "spatial_mean_field.csv", index=False)

        return out

    @staticmethod
    def _check_year_dirname(name: str) -> None:
        # Share the exact parsing logic the R script uses; if that
        # returns a year, the dirname is fine.
        _parse_year_from(name)


def prepare_rpca_export(
    *,
    station_id: NDArray[np.integer] | NDArray[np.str_],
    station_x: NDArray[np.floating],
    station_y: NDArray[np.floating],
    eof_coefficients: NDArray[np.floating],
    spatial_mean: NDArray[np.floating],
) -> RExportInputs:
    """Build an :class:`RExportInputs` from in-memory arrays with the
    full round of shape validation.  Thin convenience wrapper around
    the dataclass constructor.
    """
    return RExportInputs(
        station_id=np.asarray(station_id),
        station_x=np.asarray(station_x, dtype=float),
        station_y=np.asarray(station_y, dtype=float),
        eof_coefficients=np.asarray(eof_coefficients, dtype=float),
        spatial_mean=np.asarray(spatial_mean, dtype=float),
    )


# ---------------------------------------------------------------------------
# Subprocess driver
# ---------------------------------------------------------------------------
@dataclass
class RScriptResult:
    """Return value of :func:`run_bamlss` / :func:`run_gam`.

    ``returncode`` is non-zero on R-side failure.  ``output_nc`` is the
    NetCDF file the R script wrote; ``stats_csv`` is the per-EOF
    fit-statistics CSV the R script writes alongside it.
    """

    command: list[str] = field(default_factory=list)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    output_nc: Path | None = None
    stats_csv: Path | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.output_nc is not None and self.output_nc.is_file()


def _build_argv(
    script: Path,
    *,
    rpca_dir: Path,
    covariate_dir: Path,
    output_dir: Path,
    n_eofs: int,
    cores: int,
    quick: bool,
    method: str | None = None,
    n_iter: int | None = None,
    burnin: int | None = None,
    rscript: str | None = None,
) -> list[str]:
    """Shared command-line builder for both R scripts.

    The two scripts share most options.  ``method`` is ``gam``-only
    (values ``"bam"`` or ``"gam"``); ``n_iter`` / ``burnin`` are
    ``bamlss``-only.
    """
    if n_eofs < 1:
        raise ValueError("n_eofs must be >= 1")
    if cores < 1:
        raise ValueError("cores must be >= 1")

    argv: list[str] = [
        rscript_executable(rscript),
        str(script),
        "--rpca_dir", str(rpca_dir),
        "--covariate_dir", str(covariate_dir),
        "--output_dir", str(output_dir),
        "--n_eofs", str(n_eofs),
        "--cores", str(cores),
    ]
    if quick:
        argv.append("--quick")
    if method is not None:
        if method not in {"bam", "gam"}:
            raise ValueError(f"method must be 'bam' or 'gam'; got {method!r}")
        argv.extend(["--method", method])
    if n_iter is not None:
        if n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        argv.extend(["--n_iter", str(n_iter)])
    if burnin is not None:
        if burnin < 0:
            raise ValueError("burnin must be >= 0")
        argv.extend(["--burnin", str(burnin)])
    return argv


def _invoke(
    argv: list[str],
    *,
    output_dir: Path,
    target_year: int,
    label: str,
    stem: str,
    timeout: float | None,
) -> RScriptResult:
    """Run ``argv`` via :func:`subprocess.run` and package the result.

    ``stem`` is the output-file stem used by the R script
    (``bamlss`` or ``gam``); the script writes
    ``eof_grids_{stem}_{year}.nc`` and ``{stem}_fit_stats_{year}.csv``
    under ``output_dir``.
    """
    logger.info("[%s] invoking: %s", label, " ".join(argv))
    try:
        proc = subprocess.run(  # noqa: S603 — argv is a list, no shell
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find the Rscript executable '{argv[0]}'. "
            "Install R and ensure Rscript is on PATH, or set the "
            "AWA_RSCRIPT environment variable."
        ) from exc

    output_nc = output_dir / f"eof_grids_{stem}_{target_year}.nc"
    stats_csv = output_dir / f"{stem}_fit_stats_{target_year}.csv"
    return RScriptResult(
        command=argv,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        output_nc=output_nc if output_nc.is_file() else None,
        stats_csv=stats_csv if stats_csv.is_file() else None,
    )


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------
def run_bamlss(
    rpca_dir: str | os.PathLike,
    *,
    covariate_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    n_eofs: int = 30,
    n_iter: int = 2000,
    burnin: int = 500,
    cores: int = 4,
    quick: bool = False,
    rscript: str | None = None,
    timeout: float | None = None,
) -> RScriptResult:
    """Run ``R/bamlss_rpca_eof.R`` on a prepared rPCA export.

    Parameters
    ----------
    rpca_dir : path
        Directory containing ``eof_coefficients_training.csv``,
        ``station_metadata.csv``, ``spatial_mean_field.csv``.  The
        basename must include ``year_<YYYY>`` (see
        :meth:`RExportInputs.write`).
    covariate_dir : path
        Directory of terrain GeoTIFFs (EU-DEM elevation, TPI, slope,
        aspect-sin, aspect-cos, CORINE roughness).  Not redistributed
        with this package; see ``docs/data.md``.
    output_dir : path
        Where the R script writes ``eof_grids_bamlss_{year}.nc`` and
        ``bamlss_fit_stats_{year}.csv``.  Created if missing.
    n_eofs, n_iter, burnin, cores : int
        Standard BAMLSS knobs; identical defaults to the R script.
    quick : bool, default False
        If ``True``, the R script forces ``n_iter=500``, ``burnin=100``
        and clamps ``n_eofs`` to 10 — useful for CI integration tests.
    rscript : str, optional
        Path to the ``Rscript`` binary; see :func:`rscript_executable`.
    timeout : float, optional
        Hard subprocess timeout in seconds (recommended in CI).
    """
    rpca_dir = Path(rpca_dir)
    covariate_dir = Path(covariate_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_year = _parse_year_from(rpca_dir.name)
    argv = _build_argv(
        r_script_path("bamlss_rpca_eof"),
        rpca_dir=rpca_dir,
        covariate_dir=covariate_dir,
        output_dir=output_dir,
        n_eofs=n_eofs,
        cores=cores,
        quick=quick,
        n_iter=n_iter,
        burnin=burnin,
        rscript=rscript,
    )
    return _invoke(
        argv,
        output_dir=output_dir,
        target_year=target_year,
        label="BAMLSS",
        stem="bamlss",
        timeout=timeout,
    )


def run_gam(
    rpca_dir: str | os.PathLike,
    *,
    covariate_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    n_eofs: int = 30,
    method: Literal["bam", "gam"] = "bam",
    cores: int = 4,
    quick: bool = False,
    rscript: str | None = None,
    timeout: float | None = None,
) -> RScriptResult:
    """Run ``R/gam_rpca_eof.R`` on a prepared rPCA export.

    Faster alternative to :func:`run_bamlss`: ``mgcv::bam`` fits each
    EOF in seconds rather than minutes, with no posterior.  Use
    :func:`run_bamlss` when you need the Bayesian posterior for the
    inter-method spread; use :func:`run_gam` when you just want the
    mean prediction.
    """
    rpca_dir = Path(rpca_dir)
    covariate_dir = Path(covariate_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_year = _parse_year_from(rpca_dir.name)
    argv = _build_argv(
        r_script_path("gam_rpca_eof"),
        rpca_dir=rpca_dir,
        covariate_dir=covariate_dir,
        output_dir=output_dir,
        n_eofs=n_eofs,
        cores=cores,
        quick=quick,
        method=method,
        rscript=rscript,
    )
    return _invoke(
        argv,
        output_dir=output_dir,
        target_year=target_year,
        label="GAM",
        stem="gam",
        timeout=timeout,
    )


def _parse_year_from(dirname: str) -> int:
    """Extract ``YYYY`` from ``.*year_YYYY.*``; matches the R-side
    regex ``gsub(".*year_", "", opt$rpca_dir)``."""
    if "year_" not in dirname:
        raise ValueError(
            "rpca_dir basename must contain 'year_<YYYY>'; "
            f"got {dirname!r}"
        )
    tail = dirname.split("year_", 1)[1]
    digits = "".join(c for c in tail if c.isdigit())
    if not digits:
        raise ValueError(
            f"Could not parse a year from dirname {dirname!r}"
        )
    return int(digits[:4])


# ---------------------------------------------------------------------------
# Output loader (optional convenience)
# ---------------------------------------------------------------------------
def load_eof_grids(result_or_path: RScriptResult | str | os.PathLike):  # noqa: ANN201
    """Open the R-side output NetCDF as an :class:`xarray.Dataset`.

    Lazy import of :mod:`xarray` so that the base library stays
    importable without the ``[viz]`` extra on hosts that only run
    the pure-numpy interpolators.

    Parameters
    ----------
    result_or_path : RScriptResult or path
        Either a result struct from :func:`run_bamlss` / :func:`run_gam`
        or a direct path to the NetCDF.

    Returns
    -------
    xarray.Dataset
        With variables ``eof_coefficients`` (``(eof, y, x)``) and
        ``spatial_mean`` (``(y, x)``), coordinates ``x``, ``y``, ``eof``.
    """
    import xarray as xr

    if isinstance(result_or_path, RScriptResult):
        if result_or_path.output_nc is None:
            raise RuntimeError(
                f"R script did not produce an output NetCDF "
                f"(returncode={result_or_path.returncode}); check stderr."
            )
        path = result_or_path.output_nc
    else:
        path = Path(result_or_path)
    return xr.open_dataset(path)
