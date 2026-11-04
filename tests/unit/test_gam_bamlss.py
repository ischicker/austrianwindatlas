"""Unit tests for `austrianwindatlas.stage1.gam_bamlss`.

Two suites:

* **Unit tests** (always run): exercise the Python-side contract —
  RExportInputs validation, CSV materialisation, command-line
  construction, Rscript discovery, year parsing, output-loader
  failure modes.  No R is invoked.

* **Integration tests** (``skipif`` when ``Rscript`` + ``bamlss`` +
  ``mgcv`` are not available): build tiny synthetic rPCA exports and
  single-pixel covariate GeoTIFFs, invoke the R scripts in
  ``--quick`` mode and verify that the NetCDF comes back with the
  expected shape.  These never run in the default CI but will run
  inside the Docker image.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from austrianwindatlas.stage1 import gam_bamlss as gb

RNG_SEED = 20261104


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_inputs(n_stations: int = 8, n_eofs: int = 3) -> gb.RExportInputs:
    rng = np.random.default_rng(RNG_SEED)
    return gb.prepare_rpca_export(
        station_id=np.arange(n_stations),
        station_x=np.linspace(100.0, 10_000.0, n_stations),
        station_y=np.linspace(200.0, 8_000.0, n_stations),
        eof_coefficients=rng.standard_normal((n_stations, n_eofs)),
        spatial_mean=np.full(n_stations, 2.5) + rng.standard_normal(n_stations) * 0.1,
    )


# ===========================================================================
# RExportInputs validation
# ===========================================================================
class TestInputsValidation:
    def test_builds_from_consistent_arrays(self):
        inp = _make_inputs()
        assert inp.n_stations == 8
        assert inp.n_eofs == 3

    def test_mismatched_x_rejected(self):
        with pytest.raises(ValueError, match="station_x"):
            gb.prepare_rpca_export(
                station_id=np.arange(5),
                station_x=np.zeros(4),
                station_y=np.zeros(5),
                eof_coefficients=np.zeros((5, 2)),
                spatial_mean=np.zeros(5),
            )

    def test_mismatched_y_rejected(self):
        with pytest.raises(ValueError, match="station_y"):
            gb.prepare_rpca_export(
                station_id=np.arange(5),
                station_x=np.zeros(5),
                station_y=np.zeros(6),
                eof_coefficients=np.zeros((5, 2)),
                spatial_mean=np.zeros(5),
            )

    def test_eof_wrong_rank_rejected(self):
        with pytest.raises(ValueError, match="eof_coefficients"):
            gb.prepare_rpca_export(
                station_id=np.arange(5),
                station_x=np.zeros(5),
                station_y=np.zeros(5),
                eof_coefficients=np.zeros(5),
                spatial_mean=np.zeros(5),
            )

    def test_mean_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="spatial_mean"):
            gb.prepare_rpca_export(
                station_id=np.arange(5),
                station_x=np.zeros(5),
                station_y=np.zeros(5),
                eof_coefficients=np.zeros((5, 2)),
                spatial_mean=np.zeros(4),
            )


# ===========================================================================
# CSV materialisation (write())
# ===========================================================================
class TestWrite:
    def test_writes_the_three_csvs(self, tmp_path):
        inp = _make_inputs()
        rpca_dir = tmp_path / "year_2020"
        out = inp.write(rpca_dir)
        assert out == rpca_dir
        assert (out / "eof_coefficients_training.csv").is_file()
        assert (out / "station_metadata.csv").is_file()
        assert (out / "spatial_mean_field.csv").is_file()

    def test_csv_columns_match_contract(self, tmp_path):
        inp = _make_inputs(n_stations=6, n_eofs=4)
        rpca_dir = tmp_path / "year_2018"
        inp.write(rpca_dir)

        eof_df = pd.read_csv(rpca_dir / "eof_coefficients_training.csv")
        assert list(eof_df.columns) == ["station_id", "eof_1", "eof_2", "eof_3", "eof_4"]
        assert len(eof_df) == 6

        meta_df = pd.read_csv(rpca_dir / "station_metadata.csv")
        assert list(meta_df.columns) == ["station_id", "x", "y"]
        assert len(meta_df) == 6

        mean_df = pd.read_csv(rpca_dir / "spatial_mean_field.csv")
        assert list(mean_df.columns) == ["station_id", "mean_wind"]
        assert len(mean_df) == 6

    def test_year_dirname_check_rejects_bad_name(self, tmp_path):
        inp = _make_inputs()
        with pytest.raises(ValueError, match="year_"):
            inp.write(tmp_path / "not_a_year_dir")


# ===========================================================================
# Script path discovery
# ===========================================================================
class TestScriptDiscovery:
    @pytest.mark.parametrize("name", ["bamlss_rpca_eof", "gam_rpca_eof"])
    def test_r_script_path_resolves(self, name):
        p = gb.r_script_path(name)
        assert p.is_file()
        assert p.name == f"{name}.R"

    def test_r_script_path_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            # Literal typing is only hinting; runtime still allows
            # unexpected values and we want to see the FileNotFoundError.
            gb.r_script_path("does_not_exist")  # type: ignore[arg-type]


class TestRScriptExecutable:
    def test_override_wins(self):
        assert gb.rscript_executable(override="/opt/R/bin/Rscript") == "/opt/R/bin/Rscript"

    def test_env_var_consulted(self, monkeypatch):
        monkeypatch.delenv("AWA_RSCRIPT", raising=False)
        monkeypatch.setenv("AWA_RSCRIPT", "/custom/Rscript")
        assert gb.rscript_executable() == "/custom/Rscript"

    def test_which_fallback(self, monkeypatch):
        monkeypatch.delenv("AWA_RSCRIPT", raising=False)
        with patch("austrianwindatlas.stage1.gam_bamlss.shutil.which",
                   return_value="/usr/bin/Rscript"):
            assert gb.rscript_executable() == "/usr/bin/Rscript"

    def test_last_ditch_default_is_bare_name(self, monkeypatch):
        monkeypatch.delenv("AWA_RSCRIPT", raising=False)
        with patch("austrianwindatlas.stage1.gam_bamlss.shutil.which",
                   return_value=None):
            assert gb.rscript_executable() == "Rscript"


# ===========================================================================
# Command-line construction (via the public wrappers under a subprocess mock)
# ===========================================================================
@pytest.fixture
def prepared_export(tmp_path) -> tuple[Path, Path, Path]:
    """Return (rpca_dir, covariate_dir, output_dir) with CSVs already
    written.  The covariate_dir is empty — the R scripts would fail,
    but the subprocess is mocked in this suite so that doesn't matter.
    """
    _make_inputs().write(tmp_path / "year_2019")
    (tmp_path / "covariates").mkdir()
    (tmp_path / "out").mkdir()
    return tmp_path / "year_2019", tmp_path / "covariates", tmp_path / "out"


def _mock_run_factory(returncode: int = 0, write_outputs: tuple[Path, Path] | None = None):
    """Build a subprocess.run replacement that optionally creates the
    expected output files so RScriptResult.ok passes.
    """
    def _run(argv, check=False, capture_output=False, text=False, timeout=None):
        del check, capture_output, text, timeout
        if write_outputs is not None:
            nc, csv = write_outputs
            nc.parent.mkdir(parents=True, exist_ok=True)
            nc.write_bytes(b"")  # placeholder so is_file() is True
            csv.parent.mkdir(parents=True, exist_ok=True)
            csv.write_text("eof,rmse,cor,time_sec\n1,0.1,0.9,1.0\n")

        class _Proc:
            def __init__(self, rc):
                self.returncode = rc
                self.stdout = ""
                self.stderr = ""
        return _Proc(returncode)
    return _run


class TestRunBamlss:
    def test_builds_expected_argv(self, prepared_export):
        rpca_dir, cov_dir, out_dir = prepared_export
        captured = {}

        def _run(argv, **kwargs):  # noqa: ANN001
            captured["argv"] = argv
            class _P:
                returncode = 0
                stdout = ""
                stderr = ""
            return _P()

        with patch("austrianwindatlas.stage1.gam_bamlss.subprocess.run", _run):
            gb.run_bamlss(
                rpca_dir,
                covariate_dir=cov_dir,
                output_dir=out_dir,
                n_eofs=10,
                n_iter=500,
                burnin=100,
                cores=2,
                quick=True,
            )

        argv = captured["argv"]
        assert argv[0] == gb.rscript_executable()
        assert argv[1].endswith("bamlss_rpca_eof.R")
        assert "--rpca_dir" in argv
        assert str(rpca_dir) in argv
        assert "--covariate_dir" in argv
        assert str(cov_dir) in argv
        assert "--output_dir" in argv
        assert str(out_dir) in argv
        assert "--n_eofs" in argv and "10" in argv
        assert "--n_iter" in argv and "500" in argv
        assert "--burnin" in argv and "100" in argv
        assert "--cores" in argv and "2" in argv
        assert "--quick" in argv
        # method flag is for GAM only; must not appear here.
        assert "--method" not in argv

    def test_result_has_output_paths_when_produced(self, prepared_export):
        rpca_dir, cov_dir, out_dir = prepared_export
        nc = out_dir / "eof_grids_bamlss_2019.nc"
        csv = out_dir / "bamlss_fit_stats_2019.csv"
        with patch("austrianwindatlas.stage1.gam_bamlss.subprocess.run",
                   _mock_run_factory(0, (nc, csv))):
            res = gb.run_bamlss(rpca_dir, covariate_dir=cov_dir, output_dir=out_dir)
        assert res.ok is True
        assert res.output_nc == nc
        assert res.stats_csv == csv

    def test_result_not_ok_when_outputs_missing(self, prepared_export):
        rpca_dir, cov_dir, out_dir = prepared_export
        with patch("austrianwindatlas.stage1.gam_bamlss.subprocess.run",
                   _mock_run_factory(1)):
            res = gb.run_bamlss(rpca_dir, covariate_dir=cov_dir, output_dir=out_dir)
        assert res.ok is False
        assert res.output_nc is None
        assert res.returncode == 1

    def test_argv_validation(self, prepared_export):
        rpca_dir, cov_dir, out_dir = prepared_export
        with pytest.raises(ValueError, match="n_eofs"):
            gb.run_bamlss(rpca_dir, covariate_dir=cov_dir, output_dir=out_dir, n_eofs=0)
        with pytest.raises(ValueError, match="cores"):
            gb.run_bamlss(rpca_dir, covariate_dir=cov_dir, output_dir=out_dir, cores=0)
        with pytest.raises(ValueError, match="n_iter"):
            gb.run_bamlss(rpca_dir, covariate_dir=cov_dir, output_dir=out_dir, n_iter=0)
        with pytest.raises(ValueError, match="burnin"):
            gb.run_bamlss(rpca_dir, covariate_dir=cov_dir, output_dir=out_dir, burnin=-1)


class TestRunGam:
    def test_builds_expected_argv(self, prepared_export):
        rpca_dir, cov_dir, out_dir = prepared_export
        captured = {}

        def _run(argv, **kwargs):  # noqa: ANN001
            captured["argv"] = argv
            class _P:
                returncode = 0
                stdout = ""
                stderr = ""
            return _P()

        with patch("austrianwindatlas.stage1.gam_bamlss.subprocess.run", _run):
            gb.run_gam(
                rpca_dir,
                covariate_dir=cov_dir,
                output_dir=out_dir,
                n_eofs=20,
                method="bam",
                cores=4,
            )

        argv = captured["argv"]
        assert argv[1].endswith("gam_rpca_eof.R")
        assert "--method" in argv and "bam" in argv
        # n_iter / burnin are bamlss-only; must not appear.
        assert "--n_iter" not in argv
        assert "--burnin" not in argv

    def test_invalid_method_rejected(self, prepared_export):
        rpca_dir, cov_dir, out_dir = prepared_export
        with pytest.raises(ValueError, match="method"):
            # The Literal typing is only hinting; runtime must still reject.
            gb.run_gam(rpca_dir, covariate_dir=cov_dir, output_dir=out_dir,
                       method="xgboost")  # type: ignore[arg-type]


class TestInvokeMissingRscript:
    def test_raises_informative_error_when_executable_absent(self, prepared_export):
        rpca_dir, cov_dir, out_dir = prepared_export

        def _run(argv, **kwargs):  # noqa: ANN001
            raise FileNotFoundError(2, "No such file or directory", argv[0])

        with (
            patch("austrianwindatlas.stage1.gam_bamlss.subprocess.run", _run),
            pytest.raises(FileNotFoundError, match="Could not find the Rscript"),
        ):
            gb.run_bamlss(rpca_dir, covariate_dir=cov_dir, output_dir=out_dir)


# ===========================================================================
# Year parsing from rpca_dir basename
# ===========================================================================
class TestYearParsing:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("year_2020", 2020),
            ("year_2018_test", 2018),
            ("something_year_1991_x", 1991),
        ],
    )
    def test_parses(self, name, expected):
        assert gb._parse_year_from(name) == expected

    def test_rejects_name_without_year_prefix(self):
        with pytest.raises(ValueError, match="year_"):
            gb._parse_year_from("rpca_2020")

    def test_rejects_name_with_no_digits_after_year(self):
        with pytest.raises(ValueError, match="parse a year"):
            gb._parse_year_from("year_tbd")


# ===========================================================================
# load_eof_grids failure modes
# ===========================================================================
class TestLoadEofGrids:
    def test_raises_when_result_has_no_output(self):
        res = gb.RScriptResult(returncode=1, output_nc=None)
        with pytest.raises(RuntimeError, match="did not produce"):
            gb.load_eof_grids(res)


# ===========================================================================
# Integration (skipped unless Rscript + bamlss + mgcv are available)
# ===========================================================================
def _r_and_pkgs_available(packages: tuple[str, ...]) -> bool:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    cmd = [rscript, "-e",
           " && ".join(
               f"suppressPackageStartupMessages(library({p}))" for p in packages
           )]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


BAMLSS_AVAILABLE = _r_and_pkgs_available(("bamlss", "mgcv", "terra", "ncdf4"))
GAM_AVAILABLE = _r_and_pkgs_available(("mgcv", "terra", "ncdf4"))


@pytest.mark.integration
@pytest.mark.skipif(not GAM_AVAILABLE,
                    reason="Rscript + mgcv + terra + ncdf4 not available")
def test_integration_gam_quick(tmp_path):  # pragma: no cover
    """End-to-end smoke test for run_gam in --quick mode."""
    pytest.importorskip("rasterio")
    import rasterio
    from rasterio.transform import from_origin

    # Minimal rPCA export
    _make_inputs(n_stations=30, n_eofs=5).write(tmp_path / "year_2020")

    # Minimal covariate GeoTIFFs over the same spatial extent
    cov_dir = tmp_path / "covariates"
    cov_dir.mkdir()
    for name, filename in [
        ("EUDEM_AUSTRIA_1km.tif", "EUDEM_AUSTRIA_1km.tif"),
        ("TPI.tif", "TPI.tif"),
        ("SLOPE.tif", "SLOPE.tif"),
        ("ASPECT_SIN.tif", "ASPECT_SIN.tif"),
        ("ASPECT_COS.tif", "ASPECT_COS.tif"),
        ("ROUGHNESS.tif", "ROUGHNESS.tif"),
    ]:
        rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
        data = rng.standard_normal((50, 50)).astype(np.float32)
        with rasterio.open(
            cov_dir / filename, "w", driver="GTiff",
            height=50, width=50, count=1, dtype=data.dtype,
            crs="EPSG:31287",
            transform=from_origin(0.0, 400_000.0, 200.0, 200.0),
        ) as dst:
            dst.write(data, 1)

    out_dir = tmp_path / "out"
    res = gb.run_gam(
        tmp_path / "year_2020",
        covariate_dir=cov_dir,
        output_dir=out_dir,
        n_eofs=5,
        quick=True,
        timeout=600,
    )
    assert res.ok, f"R exited {res.returncode}; stderr tail:\n{res.stderr[-2000:]}"
    import xarray as xr
    ds = xr.open_dataset(res.output_nc)
    try:
        assert "eof_coefficients" in ds.data_vars
        assert "spatial_mean" in ds.data_vars
    finally:
        ds.close()
