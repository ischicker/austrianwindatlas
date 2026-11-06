"""Download helper for GeoSphere Austria's klima-v2-1h open-data API.

Pulls per-station time series from the public endpoint

    https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-1h

caches the result under ``$XDG_CACHE_HOME/austrianwindatlas/`` (or
``~/.cache/austrianwindatlas/`` fallback), and returns the parsed
CSV as a long-form :class:`pandas.DataFrame`.

Design goals
------------

* **Offline-safe tests.** The cache key is a SHA-256 over the query
  parameters, so running the same query twice never hits the network.
  CI jobs that have a warm cache work without network at all.
* **Rate-limit friendly.** The hourly endpoint is capped at 240
  requests/hour / 5 requests/second (per the response headers at
  probe time).  We ask for all stations in one request where possible
  and fall back on a 60-second retry-after-429 policy.
* **Narrow surface.** One public function, :func:`fetch_klima_v2_1h`.
  A :class:`KlimaQuery` dataclass parameterises the request.  No
  authentication, no session state.

The API returns both ``geojson`` and ``csv`` — we pick ``csv``
because long-form (time, station, parameter) is the natural shape
for :class:`pandas.DataFrame` and avoids a nested-JSON parse step.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "KLIMA_V2_1H_URL",
    "DEFAULT_CACHE_DIR",
    "KlimaQuery",
    "cache_dir",
    "fetch_klima_v2_1h",
]

logger = logging.getLogger(__name__)

#: Historical-hourly endpoint.  The ``v1`` prefix has been stable
#: since the 2024 API rework; expect breakage if that changes.
KLIMA_V2_1H_URL: str = (
    "https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-1h"
)


def _default_cache_dir() -> Path:
    """Return the user-level cache directory (``$XDG_CACHE_HOME`` honoured)."""
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "austrianwindatlas"
    return Path.home() / ".cache" / "austrianwindatlas"


#: Default location for cached CSV payloads.  Lazily evaluated on
#: first access so test suites can monkey-patch ``XDG_CACHE_HOME``.
DEFAULT_CACHE_DIR: Path = _default_cache_dir()


def cache_dir() -> Path:
    """Return (and create) the on-disk cache directory."""
    d = _default_cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class KlimaQuery:
    """Parameters for one klima-v2-1h request.

    ``start`` / ``end`` are ISO-8601 strings; the API wants them without
    seconds (``YYYY-MM-DDTHH:MM``).  :func:`fetch_klima_v2_1h`
    canonicalises either ``datetime`` objects or strings.
    """

    station_ids: tuple[int, ...]
    parameters: tuple[str, ...]
    start: str
    end: str
    endpoint: str = KLIMA_V2_1H_URL
    output_format: str = "csv"
    extra: dict[str, str] = field(default_factory=dict)

    def as_params(self) -> dict[str, str]:
        """Return the URL-query mapping matching the API's schema."""
        params = {
            "parameters": ",".join(self.parameters),
            "start": self.start,
            "end": self.end,
            "station_ids": ",".join(str(s) for s in self.station_ids),
            "output_format": self.output_format,
        }
        params.update(self.extra)
        return params

    def cache_key(self) -> str:
        """SHA-256 over the URL + query params — stable regardless of dict order."""
        hasher = hashlib.sha256()
        hasher.update(self.endpoint.encode("utf-8"))
        for k in sorted(self.as_params()):
            hasher.update(f"{k}={self.as_params()[k]}&".encode())
        return hasher.hexdigest()[:16]

    def cache_path(self, root: Path | None = None) -> Path:
        """Return the deterministic cache-file path for this query."""
        root = root if root is not None else cache_dir()
        return root / f"klima_v2_1h_{self.cache_key()}.csv"


def _normalise_timestamp(value: str | pd.Timestamp) -> str:
    """Convert a timestamp to the ``YYYY-MM-DDTHH:MM`` form the API wants."""
    ts = pd.Timestamp(value)
    return ts.strftime("%Y-%m-%dT%H:%M")


def _http_get(url: str, *, timeout: float, max_retries: int) -> bytes:
    """Fetch ``url`` with simple retry-on-429 / 503 logic.  Raises on failure."""
    import requests

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("klima-v2-1h request failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp.content
        if resp.status_code in (429, 503):
            retry_after = float(resp.headers.get("Retry-After", 60))
            logger.warning(
                "klima-v2-1h rate-limited (HTTP %d); sleeping %.0fs",
                resp.status_code, retry_after,
            )
            time.sleep(retry_after)
            continue
        # Hard failure; surface the body for debugging.
        raise RuntimeError(
            f"klima-v2-1h returned HTTP {resp.status_code}: {resp.text[:400]!r}"
        )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"klima-v2-1h: exhausted {max_retries} retries")


def fetch_klima_v2_1h(
    query: KlimaQuery,
    *,
    cache_root: Path | None = None,
    allow_download: bool = True,
    timeout: float = 90.0,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch a klima-v2-1h CSV, cached on disk.

    Parameters
    ----------
    query
        What to request.  Immutable; see :class:`KlimaQuery`.
    cache_root
        Overrides :func:`cache_dir`.  Useful in tests.
    allow_download
        If :data:`False` and the cache is cold, raises
        :class:`FileNotFoundError` instead of hitting the network.
        Lets CI runs bind the test to the cache state only.
    timeout
        HTTP timeout per attempt, in seconds.
    max_retries
        Retry count for transient failures / rate limits.

    Returns
    -------
    :class:`pandas.DataFrame`
        Long-form with columns ``time`` (``pandas.DatetimeIndex``),
        ``station`` (int), ``substation`` (int, nullable), and one
        column per requested parameter.
    """
    normalised = KlimaQuery(
        station_ids=tuple(sorted(query.station_ids)),
        parameters=tuple(query.parameters),
        start=_normalise_timestamp(query.start),
        end=_normalise_timestamp(query.end),
        endpoint=query.endpoint,
        output_format=query.output_format,
        extra=dict(query.extra),
    )

    cache_path = normalised.cache_path(cache_root)
    if cache_path.exists():
        logger.info("klima-v2-1h cache hit: %s", cache_path)
        raw = cache_path.read_bytes()
    else:
        if not allow_download:
            raise FileNotFoundError(
                f"klima-v2-1h cache miss and allow_download=False; expected {cache_path}"
            )
        url = f"{normalised.endpoint}?{urllib.parse.urlencode(normalised.as_params())}"
        logger.info("klima-v2-1h fetching: %s", url)
        raw = _http_get(url, timeout=timeout, max_retries=max_retries)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
        logger.info("klima-v2-1h cached to %s (%d bytes)", cache_path, len(raw))

    import io
    frame = pd.read_csv(io.BytesIO(raw))
    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert(None)
    return frame
