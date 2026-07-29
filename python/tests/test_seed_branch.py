"""Per-well seeding and the selector's PF seed-branch statistics.

The notebook's ``run_pf_lik_ensemble_scales`` seeds each well from its id and
can fill a ``branch_stats`` dict used by cell 53's bimodal branch hedge. The GPU
drop-in used to accept neither: ``wid`` was not a parameter at all, so every
well was filtered from ``seed_base = 0``, and ``branch_stats`` was rejected.

Seeding is not cosmetic here. The branch statistics are a description of the
128-seed ensemble's *spread*, so seeding every well identically makes them a
property of the seed rather than of the well — on the well these tests use, it
is the difference between a hedge shift that saturates the +/-2 ft cap and one
that lands at a third of it.

    python -m pytest python/tests/test_seed_branch.py
    ROG2_BACKEND=cuda python -m pytest python/tests/test_seed_branch.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import rog2_pf

BACKEND = os.environ.get("ROG2_BACKEND", "auto")

ROWS = int(os.environ.get("ROG2_ROWS", "150"))
N_PARTICLES = int(os.environ.get("ROG2_PARTICLES", "128"))
N_SEEDS = int(os.environ.get("ROG2_SEEDS", "32"))


def make_well(seed, n_rows=ROWS, nt=400):
    """A well with a known TVT prefix and an evaluation tail, like the real ones."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(seed)

    raw = rng.standard_normal(nt)
    tw_gr = np.clip(70.0 + 90.0 * np.convolve(raw, np.ones(25) / 25.0, mode="same"), 5.0, 160.0)
    tw_tvt = np.cumsum(0.2 + 0.02 * rng.random(nt)) - 300.0

    rate, tvt, truth = 0.0, -40.0, []
    for _ in range(n_rows):
        rate = 0.995 * rate + 0.006 * rng.standard_normal()
        tvt += rate
        truth.append(tvt)
    truth = np.array(truth)

    gr = np.interp(truth, tw_tvt, tw_gr) + 4.0 * rng.standard_normal(n_rows)
    n_known = n_rows // 3
    tvt_input = np.full(n_rows, np.nan)
    tvt_input[:n_known] = truth[:n_known]

    hw = pd.DataFrame(
        {
            "MD": np.arange(n_rows, dtype=float) * 5.0,
            # `prepare_well` reads TVT as `TVT_input + Z`, so Z carries the
            # vertical offset of the wellbore from the type-well datum.
            "Z": np.linspace(0.0, 12.0, n_rows),
            "GR": gr,
            "TVT_input": tvt_input,
        }
    )
    tw = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})
    return hw, tw


COMMON = dict(n_particles=N_PARTICLES, n_seeds=N_SEEDS, backend=BACKEND)


def test_seed_base_matches_the_notebook_derivation():
    """`md5(str(wid)) % 2**31`, verbatim from cell 38."""
    import hashlib

    for wid in ("00e12e8b", "2b06ad65", 17, None):
        want = (
            0
            if wid is None
            else int(hashlib.md5(str(wid).encode("utf-8")).hexdigest(), 16) % (2**31)
        )
        assert rog2_pf.seed_base_for_well(wid) == want


def test_wid_actually_changes_the_ensemble():
    """The regression: `wid` used to be silently dropped, seeding every well from 0."""
    hw, tw = make_well(3)
    a = rog2_pf.run_pf_lik_ensemble_scales(hw, tw, wid=None, **COMMON)["pf_mean"]
    b = rog2_pf.run_pf_lik_ensemble_scales(hw, tw, wid="00e12e8b", **COMMON)["pf_mean"]
    ev = np.isnan(hw["TVT_input"].values)
    assert not np.allclose(a[ev], b[ev]), "wid did not reach the kernel's seed base"


def test_same_wid_is_reproducible():
    hw, tw = make_well(4)
    a = rog2_pf.run_pf_lik_ensemble_scales(hw, tw, wid="abc123", **COMMON)["pf_mean"]
    b = rog2_pf.run_pf_lik_ensemble_scales(hw, tw, wid="abc123", **COMMON)["pf_mean"]
    np.testing.assert_array_equal(a, b)


def test_wids_and_seed_bases_agree_in_the_batch_path():
    hw, tw = make_well(5)
    by_wid = rog2_pf.lik_pf_batch([(hw, tw)], wids=["00e12e8b"], **COMMON)[0][0]
    by_base = rog2_pf.lik_pf_batch(
        [(hw, tw)], seed_bases=[rog2_pf.seed_base_for_well("00e12e8b")], **COMMON
    )[0][0]
    np.testing.assert_array_equal(by_wid["pf_mean"], by_base["pf_mean"])


def test_seed_paths_are_the_paths_pf_mean_was_blended_from():
    """Proves the readback is aligned, not transposed or shifted by a well."""
    hw, tw = make_well(6)
    out, idx, _ = rog2_pf.lik_pf_batch(
        [(hw, tw)], wids=["00e12e8b"], with_seed_paths=True, **COMMON
    )[0]
    paths = out["pf_seed_paths"]
    assert paths.shape == (N_SEEDS, len(idx))
    assert out["pf_seed_liks"].shape == (N_SEEDS,)
    np.testing.assert_allclose(paths.mean(0), out["pf_mean"], rtol=0, atol=1e-3)


def test_branch_stats_are_filled_and_self_consistent():
    hw, tw = make_well(7)
    branch = {}
    rog2_pf.run_pf_lik_ensemble_scales(hw, tw, branch_stats=branch, wid="00e12e8b", **COMMON)

    if "center_low" not in branch:
        pytest.skip(f"well is not splittable on this backend: {branch}")

    assert branch["center_low"] <= branch["center_high"]
    assert 0.0 < branch["mass_low"] < 1.0
    assert branch["mass_low"] + branch["mass_high"] == pytest.approx(1.0, abs=1e-9)
    assert branch["center_low"] <= branch["weighted_center"] <= branch["center_high"]
    assert branch["seed_count"] <= N_SEEDS
    # cell 53 filters the submission by these row ids
    assert list(branch["eval_rows"]) == list(np.flatnonzero(np.isnan(hw["TVT_input"].values)))


def test_branch_stats_match_the_helper_on_the_same_paths():
    """`run_pf_lik_ensemble_scales` must apply exactly `pf_seed_branch_stats`."""
    hw, tw = make_well(8)
    branch = {}
    rog2_pf.run_pf_lik_ensemble_scales(hw, tw, branch_stats=branch, wid="w1", **COMMON)
    out, idx, _ = rog2_pf.lik_pf_batch(
        [(hw, tw)], wids=["w1"], with_seed_paths=True, **COMMON
    )[0]
    direct = rog2_pf.pf_seed_branch_stats(
        out["pf_seed_paths"], out["pf_seed_liks"], eval_rows=idx
    )
    if not direct:
        pytest.skip("well is not splittable on this backend")
    for k, v in direct.items():
        assert branch[k] == v


def test_branch_stats_are_optional_and_free_when_unused():
    """Not asking for them must not change the answer or raise."""
    hw, tw = make_well(9)
    plain = rog2_pf.run_pf_lik_ensemble_scales(hw, tw, wid="w2", **COMMON)
    branch = {}
    withb = rog2_pf.run_pf_lik_ensemble_scales(hw, tw, branch_stats=branch, wid="w2", **COMMON)
    for k in plain:
        np.testing.assert_array_equal(plain[k], withb[k])


def test_helper_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        rog2_pf.pf_seed_branch_stats(np.zeros((4, 10)), np.zeros(5))


def test_helper_returns_empty_when_too_few_seeds():
    assert rog2_pf.pf_seed_branch_stats(np.zeros((3, 10)), np.zeros(3)) == {}
