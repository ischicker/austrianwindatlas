"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.temporal_cnn.train`."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from austrianwindatlas.stage1.qn_gnn.temporal_cnn.train import (  # noqa: E402
    TrainConfig,
    TrainResult,
    load_cnn,
    save_cnn,
    train_temporal_cnn,
)


def _toy(n_times=120, n_stations=8, n_eofs=3, seed=0):
    rng = np.random.default_rng(seed)
    wind = rng.uniform(1, 10, size=(n_times, n_stations)).astype(np.float32)
    U_res = rng.standard_normal((n_times, n_eofs)).astype(np.float32) * 0.5
    return wind, U_res


def _small_cfg(**over):
    kw = dict(
        n_eofs=3, window=8, epochs=4, batch_size=16, patience=50,
        dropout=0.0, proj_dim=16, verbose_every=0, seed=0,
        cosine_t0=4,
    )
    kw.update(over)
    return TrainConfig(**kw)


# ---------------------------------------------------------------------------
# train_temporal_cnn
# ---------------------------------------------------------------------------
def test_train_temporal_cnn_runs_and_returns_trainresult():
    wind, U_res = _toy()
    cfg = _small_cfg()
    res = train_temporal_cnn(wind, U_res, config=cfg)
    assert isinstance(res, TrainResult)
    assert res.n_stations == wind.shape[1]
    assert len(res.history) == cfg.epochs
    assert np.isfinite(res.best_val_loss)
    assert res.best_epoch >= 1


def test_train_temporal_cnn_best_state_matches_best_val():
    wind, U_res = _toy()
    cfg = _small_cfg(epochs=6)
    res = train_temporal_cnn(wind, U_res, config=cfg)
    # best_val_loss must be the minimum in history.
    min_val = min(e.val_loss for e in res.history if np.isfinite(e.val_loss))
    assert res.best_val_loss == pytest.approx(min_val, abs=1e-6)


def test_train_temporal_cnn_respects_holdout_mask():
    """When a holdout is given, the model must still run — and the
    held-out stations contribute neither observations nor availability.
    We can't easily inspect per-station contributions after the fact,
    but we can confirm the trainer accepts the arg and finishes.
    """
    wind, U_res = _toy(n_stations=10)
    holdout = np.zeros(10, dtype=bool)
    holdout[[2, 7]] = True
    res = train_temporal_cnn(wind, U_res, holdout_mask=holdout, config=_small_cfg())
    assert np.isfinite(res.best_val_loss)


def test_train_rejects_eof_mismatch():
    wind, U_res = _toy()
    with pytest.raises(ValueError, match="cfg.n_eofs"):
        train_temporal_cnn(wind, U_res, config=_small_cfg(n_eofs=99))


def test_train_rejects_val_frac_leaves_no_data():
    # n_times only just bigger than window → the window+1 floor bumps
    # split past n_times → validator trips.
    wind, U_res = _toy(n_times=9)
    with pytest.raises(ValueError, match="no training data"):
        train_temporal_cnn(wind, U_res, config=_small_cfg(val_frac=0.99))


def test_train_rejects_mismatched_shapes():
    wind, U_res = _toy()
    with pytest.raises(ValueError, match="share n_times"):
        train_temporal_cnn(wind, U_res[:-5], config=_small_cfg())


# ---------------------------------------------------------------------------
# save_cnn / load_cnn
# ---------------------------------------------------------------------------
def test_save_and_load_cnn_round_trip(tmp_path):
    wind, U_res = _toy()
    cfg = _small_cfg()
    res = train_temporal_cnn(wind, U_res, config=cfg)

    out = save_cnn(res, tmp_path / "ckpt", config=cfg, extra_meta={"notes": "test"})
    assert (out / "temporal_cnn.pt").is_file()
    assert (out / "temporal_cnn_meta.json").is_file()

    model_loaded, meta = load_cnn(out)
    # Metadata matches.
    assert meta["n_stations"] == wind.shape[1]
    assert meta["n_eofs"] == cfg.n_eofs
    assert meta["window"] == cfg.window
    assert meta["proj_dim"] == cfg.proj_dim
    assert meta["notes"] == "test"

    # Model gives identical predictions after reload.
    obs = torch.randn(2, cfg.window, wind.shape[1])
    avail = torch.ones_like(obs)
    res.model.eval()
    with torch.no_grad():
        y1 = res.model(obs, avail)
        y2 = model_loaded(obs, avail)
    assert torch.allclose(y1, y2)


def test_load_cnn_missing_metadata(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing metadata"):
        load_cnn(tmp_path / "does-not-exist")


def test_load_cnn_incomplete_metadata(tmp_path):
    import json
    (tmp_path / "temporal_cnn.pt").write_bytes(b"\x00")   # placeholder
    (tmp_path / "temporal_cnn_meta.json").write_text(json.dumps({"n_stations": 10}))
    with pytest.raises(ValueError, match="missing keys"):
        load_cnn(tmp_path)
