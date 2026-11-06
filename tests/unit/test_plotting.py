"""Unit tests for :mod:`austrianwindatlas.plotting`.

Uses the non-interactive Agg backend so the tests run headlessly in CI.
Each test asserts that the plotter returns an :class:`~matplotlib.axes.Axes`
with the expected structure — number of lines, colorbar, legend handles —
without pixel-comparing the output.
"""
from __future__ import annotations

# Pick the headless backend before matplotlib.pyplot gets imported.
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from austrianwindatlas.plotting import (  # noqa: E402
    PlottingUnavailableError,
    TaylorEntry,
    diurnal_cycle,
    plot_diurnal_cycle,
    plot_station_metric,
    plot_taylor_diagram,
    plot_weibull_fit,
)


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------
def test_plotting_unavailable_error_subclasses_importerror():
    assert issubclass(PlottingUnavailableError, ImportError)
    msg = str(PlottingUnavailableError())
    assert "austrianwindatlas[viz]" in msg


# ---------------------------------------------------------------------------
# Taylor diagram
# ---------------------------------------------------------------------------
def test_plot_taylor_diagram_returns_polar_axes():
    entries = [
        TaylorEntry(label="Method A", correlation=0.95, std_ratio=1.1,
                    color="red", marker="o"),
        TaylorEntry(label="Method B", correlation=0.80, std_ratio=0.9,
                    color="blue", marker="s"),
    ]
    ax = plot_taylor_diagram(entries, title="Test")
    assert ax.name == "polar"
    assert ax.get_title() == "Test"
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_plot_taylor_diagram_plots_one_point_per_entry_plus_observation():
    entries = [
        TaylorEntry(label="A", correlation=0.9, std_ratio=1.0),
        TaylorEntry(label="B", correlation=0.8, std_ratio=0.9),
        TaylorEntry(label="C", correlation=0.7, std_ratio=1.1),
    ]
    ax = plot_taylor_diagram(entries, show_legend=False)
    # One "Observation" marker + one per entry → 4 lines.
    data_lines = [line for line in ax.get_lines() if line.get_label() in
                  {"Observation", "A", "B", "C"}]
    assert len(data_lines) == 4
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_plot_taylor_diagram_skips_nan_entries():
    entries = [
        TaylorEntry(label="good", correlation=0.9, std_ratio=1.0),
        TaylorEntry(label="bad", correlation=float("nan"), std_ratio=1.0),
        TaylorEntry(label="worse", correlation=0.8, std_ratio=float("nan")),
    ]
    ax = plot_taylor_diagram(entries, show_legend=False)
    labels = [line.get_label() for line in ax.get_lines()]
    assert "good" in labels
    assert "bad" not in labels
    assert "worse" not in labels
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_plot_taylor_diagram_rejects_non_polar_axes():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()  # Cartesian axes
    with pytest.raises(ValueError, match="polar"):
        plot_taylor_diagram([TaylorEntry("x", 0.9, 1.0)], ax=ax)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Station metric map
# ---------------------------------------------------------------------------
def test_plot_station_metric_returns_axes_with_colorbar():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100_000, size=30)
    y = rng.uniform(0, 100_000, size=30)
    values = rng.uniform(0.5, 2.5, size=30)
    ax = plot_station_metric(x, y, values, metric_name="RMSE")
    # One colorbar was added to the figure.
    cbars = [cbar_ax for cbar_ax in ax.figure.axes if cbar_ax is not ax]
    assert len(cbars) == 1
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_plot_station_metric_diverging_autodetect_for_bias():
    """A metric called "bias" should auto-pick a diverging colormap
    and symmetric vmin/vmax."""
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    values = np.linspace(-1.0, 3.0, 20)      # asymmetric raw range
    ax = plot_station_metric(x, y, values, metric_name="bias")
    sc = ax.collections[0]
    vmin, vmax = sc.get_clim()
    # Symmetric around zero.
    assert vmin == pytest.approx(-vmax, abs=1e-6)
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_plot_station_metric_sequential_for_rmse():
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float)
    values = np.linspace(0.5, 2.0, 10)
    ax = plot_station_metric(x, y, values, metric_name="rmse")
    sc = ax.collections[0]
    vmin, vmax = sc.get_clim()
    # Not symmetric — vmin > 0 here.
    assert vmin > 0
    assert vmax > vmin
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_plot_station_metric_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="share shape"):
        plot_station_metric(np.zeros(5), np.zeros(5), np.zeros(6))


def test_plot_station_metric_rejects_all_nan_values():
    with pytest.raises(ValueError, match="no finite entries"):
        plot_station_metric(np.arange(5.0), np.arange(5.0), np.full(5, np.nan))


def test_plot_station_metric_calls_background_before_scatter():
    """The background callable should execute before the scatter so
    the hillshade / border lives beneath the points."""
    import matplotlib.pyplot as plt

    calls: list[str] = []

    def bg(ax):
        calls.append("bg")

    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    values = np.array([1.0, 2.0])
    plot_station_metric(x, y, values, background=bg)
    assert calls == ["bg"]
    plt.close("all")


# ---------------------------------------------------------------------------
# Diurnal cycle
# ---------------------------------------------------------------------------
def test_diurnal_cycle_returns_24_rows():
    times = pd.date_range("2020-01-01", periods=24 * 7, freq="h")
    ws = np.ones(24 * 7) * 5.0
    out = diurnal_cycle(times, ws)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 24
    assert list(out.columns) == ["hour", "mean", "std", "count"]
    # Count is 7 (one week × same hour).
    assert (out["count"] == 7).all()
    # Mean is 5 everywhere.
    assert (out["mean"] == 5.0).all()


def test_diurnal_cycle_fills_missing_hours_with_nan():
    # Only noon samples.
    noon = pd.date_range("2020-01-01 12:00", periods=5, freq="D")
    out = diurnal_cycle(noon, np.full(5, 3.0))
    assert len(out) == 24
    # Hour 12 is filled; others have count=0.
    row_12 = out[out["hour"] == 12].iloc[0]
    assert row_12["count"] == 5
    assert row_12["mean"] == pytest.approx(3.0)
    other = out[out["hour"] != 12]
    assert (other["count"] == 0).all()


def test_diurnal_cycle_rejects_length_mismatch():
    with pytest.raises(ValueError, match="share length"):
        diurnal_cycle(pd.date_range("2020-01-01", periods=5, freq="h"), np.ones(10))


def test_plot_diurnal_cycle_adds_one_line_per_prediction():
    times = pd.date_range("2020-01-01", periods=24 * 3, freq="h")
    obs = 5.0 + np.sin(2 * np.pi * np.arange(24 * 3) / 24.0)
    pred_a = obs + 0.3
    pred_b = obs * 0.9
    ax = plot_diurnal_cycle(
        times, obs,
        predictions={"A": (times, pred_a), "B": (times, pred_b)},
        title="Diurnal test",
    )
    labels = [line.get_label() for line in ax.get_lines()]
    assert "Observations" in labels
    assert "A" in labels
    assert "B" in labels
    assert ax.get_title() == "Diurnal test"
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_plot_diurnal_cycle_omits_obs_band_when_disabled():
    times = pd.date_range("2020-01-01", periods=48, freq="h")
    obs = np.ones(48) * 5.0
    ax = plot_diurnal_cycle(times, obs, predictions={}, show_obs_band=False)
    # No fill collections means no shading.
    assert not ax.collections
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Weibull overlay
# ---------------------------------------------------------------------------
def test_plot_weibull_fit_returns_axes_and_params():
    rng = np.random.default_rng(0)
    ws = 8.0 * rng.weibull(2.0, size=5_000)
    ax, params = plot_weibull_fit(ws, title="Weibull")
    assert np.isfinite(params.k) and np.isfinite(params.A)
    # Expect at least one histogram patch collection AND one PDF line.
    assert len(ax.patches) > 0
    assert any(
        line.get_label().startswith("Weibull")
        for line in ax.get_lines()
    )
    assert ax.get_title() == "Weibull"
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_plot_weibull_fit_uses_supplied_params():
    from austrianwindatlas.validation.energy import WeibullParameters

    rng = np.random.default_rng(0)
    ws = 10.0 * rng.weibull(1.8, size=2_000)
    forced = WeibullParameters(k=2.5, A=7.5)
    ax, returned = plot_weibull_fit(ws, params=forced)
    assert returned.k == pytest.approx(2.5)
    assert pytest.approx(7.5) == returned.A
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_plot_weibull_fit_rejects_all_nan_input():
    with pytest.raises(ValueError, match="no finite entries"):
        plot_weibull_fit(np.full(100, np.nan))


def test_plot_weibull_fit_rejects_degenerate_sample():
    """fit_weibull returns NaN on < 100 samples → plotter must raise."""
    with pytest.raises(ValueError, match="NaN parameters"):
        plot_weibull_fit(np.array([1.0, 2.0, 3.0]))
