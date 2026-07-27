"""The GPU beam search must agree with the notebook's numpy `beam_search`.

The reference here is a verbatim copy of `beam_search` / `run_beam_ensemble` from
`notebook/rogii-another-approch-2nd.ipynb` (cell 12), including its scipy
`savgol_filter` call. Rust unit tests pin the kernel against a Rust reference;
this is the cross-language check that the Rust reference is the *right* algorithm.

Run with any backend:

    python -m pytest python/tests/test_numpy_parity.py
    ROG2_BACKEND=cuda python -m pytest python/tests/test_numpy_parity.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.signal import savgol_filter

import rog2_pf

BACKEND = os.environ.get("ROG2_BACKEND", "auto")


# ---------------------------------------------------------------------------
# Verbatim from the notebook
# ---------------------------------------------------------------------------


def nb_beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))

    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC = mc * np.array([2.0, 1.0, 0.0, 1.0, 2.0])

    bidx = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.0
    bn = 1

    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)

        gr_e = (gv - tw_gr[ci]) ** 2 / es
        tot = bcost[:bn, None] + gr_e + MC[None, :]
        tot = np.where(valid, tot, np.inf)

        ni_f = ni.flatten()
        tot_f = tot.flatten()
        vf = valid.flatten()
        ni_f = ni_f[vf]
        tot_f = tot_f[vf]

        order = np.argsort(tot_f)
        ni_s = ni_f[order]
        tot_s = tot_f[order]

        _, first = np.unique(ni_s, return_index=True)
        ni_u = ni_s[first]
        tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        top = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top = top[np.argsort(tot_u[top])]

        bidx[:kept] = ni_u[top]
        bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:] = bidx[kept - 1]
            bcost[kept:] = np.inf
        bn = kept

        result[step] = tw_tvt[bidx[0]]

    return result


# ---------------------------------------------------------------------------
# Synthetic wells with the same signal structure as the competition data
# ---------------------------------------------------------------------------


#: Adjacent type-well samples are at least 0.2 ft apart, so two trajectories
#: within this distance are sitting on the *same* sample and differ only in how
#: its TVT was rounded — the kernel works in f32, the notebook in f64.
SAME_SAMPLE_FT = 1e-2


def same_sample(a, b):
    """Fraction of rows where both trajectories chose the same type-well sample."""
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b)) < SAME_SAMPLE_FT))


def make_well(seed, n_rows, nt=800):
    rng = np.random.default_rng(seed)

    # An aperiodic GR log: a smoothed random walk, so the search can localise.
    raw = rng.standard_normal(nt)
    kernel = np.ones(25) / 25.0
    tw_gr = np.clip(70.0 + 90.0 * np.convolve(raw, kernel, mode="same"), 5.0, 160.0)
    # Ascending but not uniformly spaced, like a real type-well log.
    tw_tvt = np.cumsum(0.2 + 0.02 * rng.random(nt)) - 300.0

    # True path: a slow random walk through the type well.
    rate = 0.0
    tvt, truth = -40.0, []
    for _ in range(n_rows):
        rate = 0.995 * rate + 0.006 * rng.standard_normal()
        tvt += rate
        truth.append(tvt)
    truth = np.array(truth)

    hgr = np.interp(truth, tw_tvt, tw_gr) + 4.0 * rng.standard_normal(n_rows)
    return hgr, tw_tvt, tw_gr, float(truth[0]), truth


CASES = [
    # (beam_size, move_cost, err_scale, radius) — a spread of the notebook's set,
    # covering an unsmoothed config, the identity radius, and a wide beam.
    (10, 20.0, 144.0, 2),
    (8, 35.0, 220.0, 1),
    (20, 4.0, 36.0, 3),
    (10, 50.0, 400.0, 0),
    (12, 18.0, 120.0, 5),
]


@pytest.mark.parametrize("cfg", CASES)
def test_single_config_matches_numpy(cfg):
    hgr, tw_tvt, tw_gr, last, _ = make_well(seed=3, n_rows=250)
    bs, mc, es, r = cfg

    want = nb_beam_search(hgr, tw_tvt, tw_gr, last, bs, mc, es, r)
    got = rog2_pf.beam_search(
        hgr, tw_tvt, tw_gr, last, bs=bs, mc=mc, es=es, r=r, backend=BACKEND
    )

    assert got.shape == want.shape
    # f32-vs-f64 cost accumulation can flip a near-tie, and the beam then
    # re-converges within a few steps, so a handful of rows are allowed to
    # disagree. In practice this is 100%.
    agree = same_sample(got, want)
    assert agree > 0.97, f"only {agree:.3%} of rows chose the same sample"
    assert np.max(np.abs(got - want)) < 5.0


@pytest.mark.parametrize("r", [1, 2, 3, 4, 5])
def test_savgol_matches_scipy(r):
    """Smoothing in the kernel must equal scipy's ``savgol_filter``.

    Comparing the smoothed series directly is not possible — it never leaves the
    device. Instead the same search is run two ways: once letting the kernel
    smooth, and once on a series scipy already smoothed with ``radius = 0``. Any
    difference is then attributable to the smoother alone, edge rows included.
    """
    hgr, tw_tvt, tw_gr, last, _ = make_well(seed=11, n_rows=250)

    common = dict(bs=10, mc=20.0, es=144.0, backend=BACKEND)
    want = rog2_pf.beam_search(
        savgol_filter(hgr, 2 * r + 1, 2), tw_tvt, tw_gr, last, r=0, **common
    )
    got = rog2_pf.beam_search(hgr, tw_tvt, tw_gr, last, r=r, **common)

    agree = same_sample(got, want)
    assert agree > 0.99, f"radius {r}: only {agree:.3%} of rows agree"


def test_ensemble_matches_the_notebook():
    hgr, tw_tvt, tw_gr, last, _ = make_well(seed=8, n_rows=200)

    want = np.stack(
        [nb_beam_search(hgr, tw_tvt, tw_gr, last, *c) for c in rog2_pf.BEAM_CONFIGS], 0
    ).mean(0)

    res = rog2_pf.run_beam_batch(
        [
            {
                "gr": hgr.astype(np.float32),
                "tw_tvt": tw_tvt.astype(np.float32),
                "tw_gr": tw_gr.astype(np.float32),
                "last_tvt": last,
            }
        ],
        backend=BACKEND,
    )
    got = np.asarray(res["beam_mean"][0], dtype=float)

    rmse = float(np.sqrt(np.mean((got - want) ** 2)))
    assert rmse < 1.0, f"ensemble RMSE {rmse} ft vs the notebook"


def test_batching_matches_well_at_a_time():
    wells = [make_well(seed=s, n_rows=120) for s in (1, 2, 3)]
    dicts = [
        {
            "gr": w[0].astype(np.float32),
            "tw_tvt": w[1].astype(np.float32),
            "tw_gr": w[2].astype(np.float32),
            "last_tvt": w[3],
        }
        for w in wells
    ]

    together = rog2_pf.run_beam_batch(dicts, backend=BACKEND)["beam_mean"]
    for i, d in enumerate(dicts):
        alone = rog2_pf.run_beam_batch([d], backend=BACKEND)["beam_mean"][0]
        np.testing.assert_array_equal(together[i], alone)


def test_empty_evaluation_zone_is_dropped():
    hgr, tw_tvt, tw_gr, last, _ = make_well(seed=4, n_rows=50)
    empty = {
        "gr": np.zeros(0, dtype=np.float32),
        "tw_tvt": tw_tvt.astype(np.float32),
        "tw_gr": tw_gr.astype(np.float32),
        "last_tvt": last,
    }
    live = {
        "gr": hgr.astype(np.float32),
        "tw_tvt": tw_tvt.astype(np.float32),
        "tw_gr": tw_gr.astype(np.float32),
        "last_tvt": last,
    }
    res = rog2_pf.run_beam_batch([empty, live], backend=BACKEND)
    assert list(res["kept"]) == [1]
    assert len(res["beam_mean"]) == 1
