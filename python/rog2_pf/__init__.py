"""GPU trackers for the ROGII wellbore-geology competition.

This package wraps the CubeCL/CUDA port of the two trajectory estimators in
``notebook/rogii-another-approch-2nd.ipynb``:

* the 128-seed likelihood-weighted particle filter (``_pf_lik_allseeds`` /
  ``lik_pf``), and
* the 14-config beam search (``beam_search`` / ``run_beam_ensemble``).

The native entry points are :func:`run_batch` and :func:`run_beam_batch`, which
take *all* wells at once so one launch fills the GPU. :func:`lik_pf`,
:func:`lik_pf_batch`, :func:`run_beam_ensemble` and
:func:`run_beam_ensemble_batch` are drop-in replacements for the notebook's
helpers and take the same pandas frames.

Backends are selected via the ``backend`` parameter (``"cuda"``, ``"wgpu"``,
``"hip"`` / ``"rocm"``, ``"cpu"``, or ``"auto"``). The shipped wheel bundles
CUDA + wgpu + CPU for NVIDIA/macOS/CPU systems. For AMD ROCm GPUs, build from
source::

    pip install rog2-algorithm --no-binary :all:
    # or build manually:
    maturin build --release --features pyo3/extension-module,python,hip

The main wheel works on Kaggle T4s, local Vulkan/Metal GPUs, and CPU-only
machines.

    from rog2_pf import lik_pf_batch, run_beam_ensemble_batch
    results = lik_pf_batch([(hw, tw) for hw, tw in wells])
    out, ev_index, quality = results[0]
    out["pf_scale_3"]  # np.ndarray over the evaluation rows

    tvt_beam = run_beam_ensemble_batch([(hw, tw) for hw, tw in wells])[0]
"""

from __future__ import annotations

import numpy as np

from ._rog2_pf import (
    __version__,
    available_backends,
    make_grid,
    notebook_beam_configs,
    run_batch,
    run_beam_batch,
)

__all__ = [
    "__version__",
    "available_backends",
    "make_grid",
    "run_batch",
    "run_beam_batch",
    "prepare_well",
    "prepare_beam_well",
    "lik_pf",
    "lik_pf_batch",
    "beam_search",
    "run_beam_ensemble",
    "run_beam_ensemble_batch",
    "DEFAULT_SCALES",
    "BEAM_CONFIGS",
]

DEFAULT_SCALES = (3.0, 5.0, 8.0, 12.0)

#: The notebook's ``BEAM_CONFIGS``, as ``(bs, mc, es, r)`` tuples.
BEAM_CONFIGS = tuple(notebook_beam_configs())

# Matches the notebook's `_pf_lik_allseeds` call site.
_PF_MOM = 0.998
_PF_VN = 0.002
_PF_PN = 0.005
_PF_ROUGH_P = 0.1
_PF_ROUGH_R = 0.001
_PF_RESAMP = 0.5


def _grid(tw_tvt, tw_gr, step=0.2):
    """Type-well GR on a uniform TVT grid — the notebook's ``_grid``."""
    tmin = float(np.min(tw_tvt))
    tmax = float(np.max(tw_tvt))
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float32), np.float32(tmin), np.float32(step)


def prepare_well(hw, tw, init_spr=4.5, seed_base=0):
    """Turn one ``(horizontal_well, typewell)`` pair into a kernel input dict.

    This reproduces the setup block of the notebook's ``lik_pf``: the GR sigma
    from the known prefix, the initial state from the last known row, and the
    initial rate from the median of ``(dTVT + dZ) / dMD`` over the last 30 known
    rows.

    Returns ``(well_dict, ev_index)``, or ``(None, empty_index)`` when the well
    has no evaluation rows.
    """
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s.TVT.values.astype(np.float64)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(np.float64)

    kn = hw[hw.TVT_input.notna()]
    ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) == 0:
        return None, ev.index.values

    last = kn.iloc[-1]
    ls = float(last.TVT_input) + float(last.Z)

    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10.0, 60.0))

    tail = kn.tail(30)
    dt = np.diff(tail.TVT_input.values)
    dz = np.diff(tail.Z.values)
    dm = np.diff(tail.MD.values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    grid, vmin, step = _grid(tw_tvt, tw_gr)
    gr_v = (
        hw.GR.interpolate(limit_direction="both")
        .fillna(tw_gr.mean())
        .values.astype(np.float32)[ev.index]
    )

    well = {
        "md": ev.MD.values.astype(np.float32),
        "z": ev.Z.values.astype(np.float32),
        "gr": gr_v,
        "grid": grid,
        "vmin": float(vmin),
        "step": float(step),
        "gs": gs,
        "ls": ls,
        "ir": ir,
        "init_spr": float(init_spr),
        "seed_base": int(seed_base) & 0xFFFFFFFF,
    }
    return well, ev.index.values


def _quality(liks_row, n_rows, gs, pt_std):
    return {
        "pf_best_ll": float(liks_row.max()) / max(n_rows, 1),
        "pf_ll_spread": float(liks_row.std()),
        "pf_pt_std": pt_std,
        "pf_gr_sig": gs,
    }


def lik_pf_batch(
    pairs,
    n_particles=500,
    n_seeds=128,
    scales=DEFAULT_SCALES,
    init_spr=4.5,
    seed_bases=None,
    backend="auto",
    with_quality=False,
    cube_dim=256,
    **kwargs,
):
    """Batched drop-in for the notebook's ``lik_pf``.

    ``pairs`` is a sequence of ``(hw, tw)`` DataFrames. Returns one
    ``(out, ev_index, quality)`` tuple per input pair, in input order; wells with
    no evaluation rows yield ``({}, empty_index, {})`` exactly as ``lik_pf`` does.

    Batching is the whole point: 128 seeds x N wells become one grid, so a T4
    stays saturated instead of running one well at a time.
    """
    scales = tuple(float(s) for s in scales)
    prepared, indices, gsigs = [], [], []
    for i, (hw, tw) in enumerate(pairs):
        base = 0 if seed_bases is None else int(seed_bases[i])
        well, ev_index = prepare_well(hw, tw, init_spr=init_spr, seed_base=base)
        prepared.append(well)
        indices.append(ev_index)
        gsigs.append(None if well is None else well["gs"])

    live = [w for w in prepared if w is not None]
    results = [({}, idx, {}) for idx in indices]
    if not live:
        return results

    res = run_batch(
        live,
        scales=list(scales),
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        cube_dim=int(cube_dim),
        backend=backend,
        mom=_PF_MOM,
        vn=_PF_VN,
        pn=_PF_PN,
        rough_p=_PF_ROUGH_P,
        rough_r=_PF_ROUGH_R,
        resamp=_PF_RESAMP,
        with_std=bool(with_quality),
        **kwargs,
    )

    channels = [c for c in res["channels"] if c != "pf_pt_std"]
    liks = res["liks"]
    live_positions = [i for i, w in enumerate(prepared) if w is not None]

    for slot, i in enumerate(live_positions):
        out = {c: res[c][slot] for c in channels}
        q = {}
        if with_quality:
            q = _quality(
                liks[slot],
                len(indices[i]),
                gsigs[i],
                res["pf_pt_std"][slot],
            )
        results[i] = (out, indices[i], q)
    return results


def lik_pf(hw, tw, **kwargs):
    """Single-well drop-in for the notebook's ``lik_pf``.

    Prefer :func:`lik_pf_batch`: one well cannot fill a GPU, so per-well calls
    pay full launch overhead for a fraction of the throughput.
    """
    seed_base = kwargs.pop("seed_base", 0)
    return lik_pf_batch([(hw, tw)], seed_bases=[seed_base], **kwargs)[0]


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


def prepare_beam_well(hw, tw):
    """Turn one ``(horizontal_well, typewell)`` pair into a beam-kernel input dict.

    This reproduces the setup block of the notebook's ``run_beam_ensemble``.
    Returns ``(well_dict, ev_index)``, or ``(None, empty_index)`` when the well
    has no evaluation rows.
    """
    ev = hw[hw.TVT_input.isna()]
    kn = hw[hw.TVT_input.notna()]
    if len(ev) == 0 or len(kn) == 0:
        return None, ev.index.values

    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s.TVT.values.astype(np.float64)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(np.float64)

    gr_all = (
        hw.GR.interpolate(limit_direction="both")
        .fillna(tw_gr.mean())
        .values.astype(np.float64)
    )

    well = {
        "gr": gr_all[ev.index].astype(np.float32),
        "tw_tvt": tw_tvt.astype(np.float32),
        "tw_gr": tw_gr.astype(np.float32),
        "last_tvt": float(kn.iloc[-1].TVT_input),
    }
    return well, ev.index.values


def run_beam_ensemble_batch(pairs, configs=None, backend="auto", cube_dim=64, **kwargs):
    """Batched drop-in for the notebook's ``run_beam_ensemble``.

    ``pairs`` is a sequence of ``(hw, tw)`` DataFrames. Returns one array per
    input pair, laid out like ``hw.TVT_input``: known rows keep their input TVT
    and evaluation rows carry the ensemble mean, exactly as the notebook's
    per-well function returns.

    Batching is the whole point: 14 configs x N wells become one grid, so a T4
    stays saturated instead of running one config at a time in Python.
    """
    prepared, indices, outs = [], [], []
    for hw, tw in pairs:
        well, ev_index = prepare_beam_well(hw, tw)
        prepared.append(well)
        indices.append(ev_index)
        outs.append(hw.TVT_input.values.astype(float).copy())

    live = [w for w in prepared if w is not None]
    if not live:
        return outs

    res = run_beam_batch(
        live,
        configs=None if configs is None else [tuple(c) for c in configs],
        backend=backend,
        cube_dim=int(cube_dim),
        **kwargs,
    )

    live_positions = [i for i, w in enumerate(prepared) if w is not None]
    for slot, i in enumerate(live_positions):
        outs[i][list(indices[i])] = res["beam_mean"][slot]
    return outs


def run_beam_ensemble(hw, tw, **kwargs):
    """Single-well drop-in for the notebook's ``run_beam_ensemble``.

    Prefer :func:`run_beam_ensemble_batch`: one well's 14 configs cannot fill a
    GPU, so per-well calls pay full launch overhead for a fraction of the
    throughput.
    """
    return run_beam_ensemble_batch([(hw, tw)], **kwargs)[0]


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2, **kwargs):
    """Single-config drop-in for the notebook's ``beam_search``.

    Signature and defaults match the original. Returns the tracked TVT for every
    row of ``hgr``.

    One difference: the type-well log is sorted here rather than assumed sorted.
    The kernel locates the starting sample with a binary search, so an unsorted
    ``tw_tvt`` would silently give a wrong answer; the notebook's callers all
    pass ``tw.sort_values('TVT')`` anyway, so this is a no-op for them.
    """
    hgr = np.asarray(hgr, dtype=np.float32)
    if len(hgr) == 0:
        return np.array([float(last_tvt)])

    order = np.argsort(np.asarray(tw_tvt, dtype=np.float64), kind="stable")
    well = {
        "gr": hgr,
        "tw_tvt": np.asarray(tw_tvt, dtype=np.float32)[order],
        "tw_gr": np.asarray(tw_gr, dtype=np.float32)[order],
        "last_tvt": float(last_tvt),
    }
    res = run_beam_batch(
        [well], configs=[(int(bs), float(mc), float(es), int(r))], **kwargs
    )
    return np.asarray(res["beam_mean"][0], dtype=float)
