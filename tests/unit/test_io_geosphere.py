"""Unit tests for :mod:`austrianwindatlas.io.geosphere`.

No live HTTP in these tests — the downloader is exercised with a
monkey-patched ``_http_get`` so CI is deterministic and offline-safe.
A separate integration test (``test_fetch_klima_v2_1h_live``) is
marked ``@pytest.mark.integration`` so the live-API call is opt-in.
"""
from __future__ import annotations

import pandas as pd
import pytest

from austrianwindatlas.io.geosphere import (
    KLIMA_V2_1H_URL,
    KlimaQuery,
    cache_dir,
    fetch_klima_v2_1h,
)

SAMPLE_CSV = b"""time,station,ff,substation
2022-02-01T00:00+00:00,105,5.2,5904
2022-02-01T00:00+00:00,39,0.5,11803
2022-02-01T01:00+00:00,105,6.2,5904
2022-02-01T01:00+00:00,39,0.6,11803
"""


# ---------------------------------------------------------------------------
# KlimaQuery contract
# ---------------------------------------------------------------------------
def test_query_as_params_builds_expected_fields():
    q = KlimaQuery(
        station_ids=(39, 105),
        parameters=("ff",),
        start="2022-02-01T00:00",
        end="2022-02-28T23:00",
    )
    params = q.as_params()
    assert params["parameters"] == "ff"
    assert params["start"] == "2022-02-01T00:00"
    assert params["end"] == "2022-02-28T23:00"
    assert params["station_ids"] == "39,105"
    assert params["output_format"] == "csv"


def test_query_as_params_merges_extra():
    q = KlimaQuery(
        station_ids=(105,), parameters=("ff",),
        start="2022-02-01T00:00", end="2022-02-01T23:00",
        extra={"foo": "bar"},
    )
    assert q.as_params()["foo"] == "bar"


def test_query_cache_key_is_deterministic_and_order_insensitive():
    q1 = KlimaQuery(station_ids=(105, 39), parameters=("ff",),
                    start="2022-02-01T00:00", end="2022-02-28T23:00")
    q2 = KlimaQuery(station_ids=(105, 39), parameters=("ff",),
                    start="2022-02-01T00:00", end="2022-02-28T23:00")
    assert q1.cache_key() == q2.cache_key()


def test_query_cache_key_changes_with_params():
    base = {
        "parameters": ("ff",),
        "start": "2022-02-01T00:00",
        "end": "2022-02-28T23:00",
    }
    q1 = KlimaQuery(station_ids=(105,), **base)
    q2 = KlimaQuery(station_ids=(39,), **base)
    q3 = KlimaQuery(station_ids=(105,), parameters=("ff", "dd"),
                    start=base["start"], end=base["end"])
    q4 = KlimaQuery(station_ids=(105,), parameters=("ff",),
                    start=base["start"], end="2022-03-31T23:00")
    keys = {q1.cache_key(), q2.cache_key(), q3.cache_key(), q4.cache_key()}
    # All four produce distinct keys.
    assert len(keys) == 4


def test_query_cache_path_contains_sha_prefix(tmp_path):
    q = KlimaQuery(station_ids=(105,), parameters=("ff",),
                   start="2022-02-01T00:00", end="2022-02-01T23:00")
    path = q.cache_path(tmp_path)
    assert path.parent == tmp_path
    assert path.name.startswith("klima_v2_1h_")
    assert path.name.endswith(".csv")


# ---------------------------------------------------------------------------
# cache_dir honours XDG_CACHE_HOME
# ---------------------------------------------------------------------------
def test_cache_dir_respects_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    d = cache_dir()
    assert d == tmp_path / "austrianwindatlas"
    assert d.is_dir()


def test_cache_dir_fallback_to_home(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    d = cache_dir()
    assert d == tmp_path / ".cache" / "austrianwindatlas"


# ---------------------------------------------------------------------------
# fetch_klima_v2_1h
# ---------------------------------------------------------------------------
def test_fetch_uses_cache_without_network(tmp_path):
    """Pre-populate the cache and assert fetch doesn't try HTTP."""
    q = KlimaQuery(station_ids=(39, 105), parameters=("ff",),
                   start="2022-02-01T00:00", end="2022-02-28T23:00")
    cache_path = q.cache_path(tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(SAMPLE_CSV)

    frame = fetch_klima_v2_1h(q, cache_root=tmp_path, allow_download=False)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["time", "station", "ff", "substation"]
    assert len(frame) == 4
    # Timestamps parsed as tz-naive UTC.
    assert frame["time"].dt.tz is None
    assert frame["time"].iloc[0] == pd.Timestamp("2022-02-01T00:00")


def test_fetch_no_download_raises_on_cold_cache(tmp_path):
    q = KlimaQuery(station_ids=(39,), parameters=("ff",),
                   start="2022-02-01T00:00", end="2022-02-01T23:00")
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        fetch_klima_v2_1h(q, cache_root=tmp_path, allow_download=False)


def test_fetch_downloads_and_writes_cache(tmp_path, monkeypatch):
    """Monkey-patch _http_get so no network call fires."""
    captured: dict[str, str] = {}

    def fake_http_get(url, *, timeout, max_retries):
        captured["url"] = url
        captured["timeout"] = timeout
        return SAMPLE_CSV

    monkeypatch.setattr(
        "austrianwindatlas.io.geosphere._http_get", fake_http_get,
    )

    q = KlimaQuery(station_ids=(39, 105), parameters=("ff",),
                   start="2022-02-01T00:00", end="2022-02-28T23:00")
    frame = fetch_klima_v2_1h(q, cache_root=tmp_path)
    assert len(frame) == 4
    assert captured["url"].startswith(KLIMA_V2_1H_URL)
    assert "station_ids=39%2C105" in captured["url"]
    assert "parameters=ff" in captured["url"]
    # Cache file landed.
    assert q.cache_path(tmp_path).is_file()


def test_fetch_second_call_hits_cache(tmp_path, monkeypatch):
    """Two calls → one HTTP fire."""
    n_calls = {"n": 0}

    def fake_http_get(url, *, timeout, max_retries):
        n_calls["n"] += 1
        return SAMPLE_CSV

    monkeypatch.setattr(
        "austrianwindatlas.io.geosphere._http_get", fake_http_get,
    )

    q = KlimaQuery(station_ids=(39, 105), parameters=("ff",),
                   start="2022-02-01T00:00", end="2022-02-28T23:00")
    fetch_klima_v2_1h(q, cache_root=tmp_path)
    fetch_klima_v2_1h(q, cache_root=tmp_path)
    assert n_calls["n"] == 1


def test_fetch_normalises_datetime_inputs(tmp_path, monkeypatch):
    """Pass ``pd.Timestamp`` and plain-string start/end — same cache key."""
    monkeypatch.setattr(
        "austrianwindatlas.io.geosphere._http_get",
        lambda url, timeout, max_retries: SAMPLE_CSV,
    )
    q_str = KlimaQuery(station_ids=(105,), parameters=("ff",),
                       start="2022-02-01T00:00", end="2022-02-28T23:00")
    q_ts = KlimaQuery(station_ids=(105,), parameters=("ff",),
                      start=pd.Timestamp("2022-02-01T00:00"),
                      end=pd.Timestamp("2022-02-28T23:00"))
    fetch_klima_v2_1h(q_str, cache_root=tmp_path)
    # Second fetch should land in the SAME cache file as the first.
    frame = fetch_klima_v2_1h(q_ts, cache_root=tmp_path)
    assert len(frame) == 4
    # Only one file in the cache.
    files = list((tmp_path / "klima_v2_1h_*").parent.glob("klima_v2_1h_*.csv"))
    assert len(files) == 1


# ---------------------------------------------------------------------------
# Live-API integration test (opt-in via pytest -m integration)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_fetch_klima_v2_1h_live(tmp_path):
    """Hits the real klima-v2-1h endpoint.  Skipped by default; run with

        pytest -m integration

    to exercise the live URL + the retry / timeout plumbing.
    """
    pytest.importorskip("requests")
    q = KlimaQuery(
        station_ids=(105,),
        parameters=("ff",),
        start="2022-02-01T00:00",
        end="2022-02-01T06:00",
    )
    frame = fetch_klima_v2_1h(q, cache_root=tmp_path)
    assert len(frame) >= 1
    assert "ff" in frame.columns
    assert frame["station"].iloc[0] == 105
