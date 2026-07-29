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
helpers and take the same pandas frames. :func:`run_particle_filter` and
:func:`run_pf_lik_ensemble_scales` are name-compatible aliases for the
notebook's *own* pure-Python functions of the same name — same call
signature, same return contract — for call sites that want to swap the
notebook's estimator for this GPU one without renaming anything.

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

import hashlib

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
    "run_particle_filter",
    "run_pf_lik_ensemble_scales",
    "seed_base_for_well",
    "pf_seed_branch_stats",
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


def seed_base_for_well(wid):
    """The notebook's per-well RNG seed base: ``md5(str(wid)) % 2**31``.

    Reproduced bit for bit from ``run_pf_lik_ensemble_scales`` in cell 38, so a
    well filtered here draws from the same stream index it would there. `None`
    maps to 0, matching the notebook's own default.

    Seeding per well is not cosmetic: the 128 seed trajectories *are* the
    ensemble, and anything computed across them — the softmax blend, and above
    all the selector's bimodal branch statistics — describes that particular
    draw. Seeding every well identically makes those numbers a property of the
    seed rather than of the well.
    """
    if wid is None:
        return 0
    return int(hashlib.md5(str(wid).encode("utf-8")).hexdigest(), 16) % (2**31)


def pf_seed_branch_stats(seed_paths, liks, eval_rows=None, scale=5.0):
    """The selector's PF seed-branch split, in the notebook's arithmetic.

    ``seed_paths`` is ``[seed, eval_row]`` (what ``lik_pf_batch`` returns under
    ``with_seed_paths=True``) and ``liks`` the matching per-seed total
    log-likelihoods. Each seed is reduced to its median TVT over the evaluation
    rows, those levels are weighted by a softmax of the log-likelihoods at
    ``scale``, and a 1-D two-cluster split minimising within-cluster weighted SSE
    gives the low/high centres and masses that cell 53's hedge qualifies on.

    Returns the dict cell 53 expects, or ``{}`` when the well cannot be split
    (fewer than 4 usable seeds, or no cut leaving 5% of the mass on both sides).
    """
    paths = np.asarray(seed_paths, dtype=float)
    liks = np.asarray(liks, dtype=float)
    if paths.ndim != 2 or paths.shape[0] != liks.shape[0]:
        raise ValueError("seed_paths must be [seed, row] and match liks")

    liks_n = liks - liks.max()
    seed_weight = np.exp(liks_n / float(scale))
    seed_weight = seed_weight / max(float(seed_weight.sum()), 1e-12)
    level = np.nanmedian(paths, axis=1)

    valid = np.isfinite(level) & np.isfinite(seed_weight) & (seed_weight > 0)
    level = level[valid]
    seed_weight = seed_weight[valid]
    seed_weight = seed_weight / max(float(seed_weight.sum()), 1e-12)
    if len(level) < 4:
        return {}

    order = np.argsort(level)
    x = level[order]
    w = seed_weight[order]
    cw = np.cumsum(w)
    cx = np.cumsum(w * x)
    cx2 = np.cumsum(w * x * x)
    total_w, total_x, total_x2 = float(cw[-1]), float(cx[-1]), float(cx2[-1])

    best = None
    for cut in range(1, len(x)):
        wl = float(cw[cut - 1])
        wr = total_w - wl
        if wl < 0.05 or wr < 0.05:
            continue
        xl = float(cx[cut - 1])
        xr = total_x - xl
        ssel = float(cx2[cut - 1] - xl * xl / wl)
        sser = float(total_x2 - cx2[cut - 1] - xr * xr / wr)
        score = max(0.0, ssel) + max(0.0, sser)
        if best is None or score < best[0]:
            best = (score, wl, wr, xl / wl, xr / wr)
    if best is None:
        return {}

    _, mass_low, mass_high, center_low, center_high = best
    stats = dict(
        center_low=float(center_low),
        center_high=float(center_high),
        mass_low=float(mass_low),
        mass_high=float(mass_high),
        weighted_center=float(np.sum(seed_weight * level)),
        seed_count=int(len(level)),
    )
    if eval_rows is not None:
        stats["eval_rows"] = np.asarray(eval_rows, dtype=int).tolist()
    return stats


def lik_pf_batch(
    pairs,
    n_particles=500,
    n_seeds=128,
    scales=DEFAULT_SCALES,
    init_spr=4.5,
    seed_bases=None,
    wids=None,
    backend="auto",
    with_quality=False,
    with_seed_paths=False,
    cube_dim=256,
    mom=None,
    vn=None,
    pn=None,
    rough_p=None,
    rough_r=None,
    resamp=None,
    lik_floor=None,
    **kwargs,
):
    """Batched drop-in for the notebook's ``lik_pf``.

    ``pairs`` is a sequence of ``(hw, tw)`` DataFrames. Returns one
    ``(out, ev_index, quality)`` tuple per input pair, in input order; wells with
    no evaluation rows yield ``({}, empty_index, {})`` exactly as ``lik_pf`` does.

    Batching is the whole point: 128 seeds x N wells become one grid, so a T4
    stays saturated instead of running one well at a time.

    Pass ``wids`` (one well id per pair) to seed each well from its id the way
    the notebook does — see :func:`seed_base_for_well`. ``seed_bases`` still
    takes explicit integers and wins if both are given.

    ``with_seed_paths=True`` adds ``out["pf_seed_paths"]``, a ``[seed, row]``
    array of every seed's own trajectory over the evaluation rows, and
    ``out["pf_seed_liks"]``, their log-likelihoods — the two inputs
    :func:`pf_seed_branch_stats` needs. It multiplies the readback by
    ``n_seeds``, so it is off by default.

    ``mom``/``vn``/``pn``/``rough_p``/``rough_r``/``resamp``/``lik_floor``
    override the process model's own constants (the notebook's values, same as
    :func:`run_batch`'s own defaults); leave any of them ``None`` to keep the
    notebook-matching default. These are process-*model* parameters, not the
    ensemble's size (``n_particles``/``n_seeds``) or its output blend
    (``scales``) — see ``rog2-algorithm/README.md``'s ``run_batch`` section for
    what each one does physically.
    """
    scales = tuple(float(s) for s in scales)
    prepared, indices, gsigs = [], [], []
    for i, (hw, tw) in enumerate(pairs):
        if seed_bases is not None:
            base = int(seed_bases[i])
        elif wids is not None:
            base = seed_base_for_well(wids[i])
        else:
            base = 0
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
        mom=_PF_MOM if mom is None else float(mom),
        vn=_PF_VN if vn is None else float(vn),
        pn=_PF_PN if pn is None else float(pn),
        rough_p=_PF_ROUGH_P if rough_p is None else float(rough_p),
        rough_r=_PF_ROUGH_R if rough_r is None else float(rough_r),
        resamp=_PF_RESAMP if resamp is None else float(resamp),
        **({"lik_floor": float(lik_floor)} if lik_floor is not None else {}),
        with_std=bool(with_quality),
        with_seed_paths=bool(with_seed_paths),
        **kwargs,
    )

    channels = [c for c in res["channels"] if c != "pf_pt_std"]
    liks = res["liks"]
    live_positions = [i for i, w in enumerate(prepared) if w is not None]

    for slot, i in enumerate(live_positions):
        out = {c: res[c][slot] for c in channels}
        if with_seed_paths:
            # The branch split needs the paths and their log-likelihoods
            # together, so they travel together.
            out["pf_seed_paths"] = res["seed_paths"][slot]
            out["pf_seed_liks"] = liks[slot]
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


def run_particle_filter(hw, tw, n_particles=500, seed=42, backend="auto", **kwargs):
    """Name-compatible drop-in for the notebook's own ``run_particle_filter``.

    The notebook version runs one seed's raw particle path in pure Python.
    This wraps :func:`lik_pf_batch` with ``n_seeds=1``: the GPU kernel doesn't
    expose each seed's trajectory separately, only the seed-ensembled
    channels, but with a single seed the softmax ensemble over one element is
    the identity, so ``pf_mean`` *is* that seed's own path. Returns
    ``(pred, log_lik)`` laid out like the notebook's version: known rows keep
    their input TVT, evaluation rows carry the tracked path.

    Prefer :func:`run_pf_lik_ensemble_scales` (or :func:`lik_pf_batch`
    directly): a single seed cannot fill a GPU, so this call exists for
    notebook-compatible call sites and one-off/debug use, not throughput.
    """
    out_vals = hw["TVT_input"].values.astype(float).copy()
    ev_index = hw.index[hw["TVT_input"].isna()].values
    if len(ev_index) == 0:
        return out_vals, 0.0

    out, idx, quality = lik_pf_batch(
        [(hw, tw)],
        n_particles=n_particles,
        n_seeds=1,
        seed_bases=[int(seed)],
        backend=backend,
        with_quality=True,
        **kwargs,
    )[0]
    out_vals[idx] = out["pf_mean"]
    log_lik = quality["pf_best_ll"] * len(idx)
    return out_vals, float(log_lik)


def run_pf_lik_ensemble_scales(
    hw,
    tw,
    scales=DEFAULT_SCALES,
    n_particles=500,
    n_seeds=128,
    branch_stats=None,
    wid=None,
    backend="auto",
    **kwargs,
):
    """Name-compatible drop-in for the notebook's own ``run_pf_lik_ensemble_scales``.

    Same signature and same return contract: one full-row array per requested
    scale (keyed ``f"pf_scale_{s:g}"``) plus ``"pf_mean"``, with known rows
    preserved from ``hw["TVT_input"]`` and evaluation rows carrying the
    seed-ensemble estimate.

    ``wid`` seeds the well from its id exactly as the notebook does (see
    :func:`seed_base_for_well`). Leaving it ``None`` seeds every well from 0,
    which makes the whole 128-seed ensemble — and anything derived from its
    spread, ``branch_stats`` above all — a property of the seed rather than of
    the well. Pass it.

    ``branch_stats``, if given a dict, is filled in place with the selector's
    bimodal branch statistics (``center_low`` / ``center_high`` / ``mass_low`` /
    ``mass_high`` / ``weighted_center`` / ``eval_rows`` / ``seed_count``), the
    keys cell 53's hedge reads. It requires the per-seed trajectories, so it
    turns on ``with_seed_paths`` and costs ``n_seeds`` times the readback; it is
    only computed when the well has at least 10 evaluation rows, matching the
    notebook's own guard. Exceptions are trapped into ``branch_stats['error']``
    the way the notebook traps them, so a hedge failure cannot take down a run.

    Note that these statistics describe *this* estimator's seed ensemble. The
    GPU filter's RNG is counter-based rather than numba's Mersenne Twister (see
    the kernel's `Determinism` section), so the split is the same *analysis* but
    not the same numbers the notebook's own PF produces — a threshold pinned
    against a numba run will not transfer unchanged.

    ``mom``/``vn``/``pn``/``rough_p``/``rough_r``/``resamp``/``lik_floor``, if
    passed as keywords, flow through ``**kwargs`` to :func:`lik_pf_batch`'s
    same-named overrides — see its docstring for what each one does. Leave
    them unset for the notebook-matching defaults.

    Prefer calling :func:`lik_pf_batch` directly across all your wells at once
    — one well's seed ensemble cannot fill a GPU either.
    """
    base = hw["TVT_input"].values.astype(float)
    ev_index = hw.index[hw["TVT_input"].isna()].values
    scale_keys = [f"pf_scale_{s:g}" for s in scales]
    if len(ev_index) == 0:
        return {k: base.copy() for k in (*scale_keys, "pf_mean")}

    want_branch = branch_stats is not None
    res = lik_pf_batch(
        [(hw, tw)],
        n_particles=n_particles,
        n_seeds=n_seeds,
        scales=tuple(float(s) for s in scales),
        seed_bases=[seed_base_for_well(wid)],
        backend=backend,
        with_seed_paths=want_branch,
        **kwargs,
    )
    out, idx, _ = res[0]

    result = {}
    for k in (*scale_keys, "pf_mean"):
        arr = base.copy()
        arr[idx] = out[k]
        result[k] = arr

    if want_branch:
        try:
            if len(idx) >= 10:
                stats = pf_seed_branch_stats(
                    out["pf_seed_paths"], out["pf_seed_liks"], eval_rows=idx
                )
                if stats:
                    branch_stats.update(stats)
        except Exception as exc:  # matches the notebook's own trap
            branch_stats["error"] = repr(exc)
    return result


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
        # Kept at f64: the beam search's dynamic program makes strict-inequality
        # cost comparisons over thousands of sequential steps, and the notebook's
        # numpy reference runs entirely in f64 (see beam_host's launch macros).
        "gr": gr_all[ev.index],
        "tw_tvt": tw_tvt,
        "tw_gr": tw_gr,
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

    Pass ``smoothing="savitzky_golay"`` to use a curvature-preserving quadratic
    fit instead of the notebook-matching default (``"rolling_mean"``, a plain
    centred moving average) for the GR smoothing pass — a configurable
    alternative for testing whether it tracks better in practice.
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
    hgr = np.asarray(hgr, dtype=np.float64)
    if len(hgr) == 0:
        return np.array([float(last_tvt)])

    order = np.argsort(np.asarray(tw_tvt, dtype=np.float64), kind="stable")
    well = {
        "gr": hgr,
        "tw_tvt": np.asarray(tw_tvt, dtype=np.float64)[order],
        "tw_gr": np.asarray(tw_gr, dtype=np.float64)[order],
        "last_tvt": float(last_tvt),
    }
    res = run_beam_batch(
        [well], configs=[(int(bs), float(mc), float(es), int(r))], **kwargs
    )
    return np.asarray(res["beam_mean"][0], dtype=float)
