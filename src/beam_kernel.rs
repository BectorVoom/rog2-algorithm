//! CubeCL implementation of the ROGII beam-search stratigraphic tracker.
//!
//! GPU port of `beam_search` / `run_beam_ensemble` from
//! `notebook/rogii-another-approch-2nd.ipynb` — specifically cell 37's numba
//! `_beam_jit`/`_smooth`/`_nn`, which shadow cell 12's plain-Python versions of
//! the same names and are what `run_beam_ensemble` actually resolves at call
//! time (Python's late name binding; cell 12's versions never run once cell 37
//! has executed). The search walks a horizontal well's evaluation rows one at a
//! time, keeping the `bs` cheapest
//! hypotheses about which type-well sample the current row sits on. Each step a
//! hypothesis may move `-2..=2` grid cells, paying `mc * |move|` plus a GR
//! mismatch term `(gr - tw_gr)^2 / es`.
//!
//! # Parallel decomposition
//!
//! The CPU version is `for config: for step: <vector op over 5 * bs candidates>`.
//! The step loop is a dynamic program and is inherently sequential, so:
//!
//! | CPU loop        | GPU mapping                                       |
//! |-----------------|---------------------------------------------------|
//! | `config`, well  | one **cube** (CUDA block) per `(well, config)`     |
//! | `5 * bs` candidates | **units** within the cube                     |
//! | `np.unique` dedup   | one pairwise "is anyone cheaper on my node" pass |
//! | `argsort` + top-`bs`| one pairwise rank pass; rank *is* the new slot |
//!
//! With 14 configs and, say, 700 wells that is ~10k independent cubes from a
//! single launch.
//!
//! Both collective passes are `O(C^2)` in the candidate count `C = 5 * bs <= 160`,
//! which beats sorting at these sizes: no shared-memory sort network, no
//! multi-pass scan, and every unit does the same branch-free scan. The rank pass
//! doubles as the scatter — a surviving candidate's rank is exactly the beam slot
//! it belongs in — so the new beam is written without a compaction step.
//!
//! # Shared-memory budget
//!
//! Six allocations: the beam (`b_idx`, `b_cost`), the candidate list (`c_idx`,
//! `c_cost`, `c_parent`) and the dedup flags (`c_dead`). At the default
//! `max_beam = 32` that is under 3 KB per cube, so occupancy is bounded by
//! cubes-per-SM rather than by shared memory. It also stays inside the CubeCL
//! CPU runtime's limit of eight shared allocations per kernel.
//!
//! # Global-memory history buffer
//!
//! `_beam_jit` does not emit the per-step argmin — see the tie-breaking bullet
//! below for why not — so this kernel cannot either. Instead every step scatters
//! its survivors' indices and parent slots into `hist_idx`/`hist_parent`
//! (`[n_configs, total_rows, max_beam]`, sized like `out` but `max_beam` wider),
//! and only after the whole sequence has run does one unit per cube walk that
//! history backward from the cheapest final slot to actually fill `out`. This
//! is `max_beam` times larger than the shared-memory buffers above, so
//! `run_beam`'s chunk sizing accounts for it explicitly rather than through the
//! `elem_size` used for `sgr`/`out`.
//!
//! # Deliberate deviations from the notebook
//!
//! * **Cost renormalisation.** Path costs are monotonically increasing sums over
//!   thousands of steps and reach ~1e6 on real wells, where an f32 mantissa can no
//!   longer resolve a single step's increment. After each step the cube subtracts
//!   the best cost from the whole beam. Subtracting a constant from every path is
//!   order-preserving, so the argmin and the surviving set are unchanged, and the
//!   costs stay `O(1)`. numpy runs in f64 and does not strictly need this, but it
//!   is still order-preserving in f64, so there is no reason to remove it now
//!   that this kernel is f64 too.
//! * **Tie-breaking.** `_beam_jit` is a hand-rolled selection sort/dedup over an
//!   array built in `(beam slot, move)` generation order, so on an exact cost
//!   tie it deterministically keeps whichever candidate was recorded *first* in
//!   that order. Both dedup and ranking here tie-break by candidate slot
//!   `u = b*5 + m`, which enumerates in that same order, to match.

use cubecl::prelude::*;

/// Stand-in for `np.inf` in the cost array. A large *finite* value, so that
/// `cost - best` on a dead slot cannot produce a NaN.
pub const BEAM_INF: f32 = 1e30;
/// Anything at or above this is a dead slot. Halfway to [`BEAM_INF`] so that
/// adding a plausible step cost to a dead entry cannot bring it back to life.
const BEAM_DEAD: f32 = 5e29;

// ---------------------------------------------------------------------------
// GR smoothing (centred rolling mean)
// ---------------------------------------------------------------------------

/// Centred rolling-mean smoother, one output plane per distinct radius.
///
/// Reproduces the notebook's *actually active* `_smooth` (cell 37 shadows cell
/// 12's version, and it is cell 37's that `run_beam_ensemble` resolves at call
/// time — see this module's doc comment):
/// `pd.Series(vals).rolling(2*r+1, center=True, min_periods=1).mean()`. That is
/// a plain centred moving average, not a Savitzky-Golay fit: `min_periods=1`
/// means the edge rows just average whatever part of the window exists inside
/// the series rather than fitting an edge polynomial, so the window is simply
/// `[max(0, i-r), min(ev_len, i+r+1))`.
///
/// `radii[k]` is the half-window of output plane `k`; `0` copies the input,
/// matching `_smooth`'s own `if r > 0 else s` branch.
#[cube(launch, launch_unchecked)]
pub fn sg_smooth_kernel<F: Float + CubeElement>(
    gr: &Array<F>,
    row_well: &Array<u32>,
    meta_u: &Array<u32>,
    radii: &Array<u32>,
    out: &mut Array<F>,
    total_rows: u32,
    n_planes: u32,
) {
    let idx = ABSOLUTE_POS;
    let rows = total_rows as usize;
    if idx < rows * n_planes as usize {
        // Consecutive units share `k` and walk consecutive rows, so both the
        // `gr` gather and the `out` store stay coalesced.
        let k = idx / rows;
        let r = idx % rows;

        let well = row_well[r] as usize;
        let ev_off = meta_u[well * 4] as usize;
        let ev_len = meta_u[well * 4 + 1] as usize;
        let i = r - ev_off;

        let m = radii[k] as usize;
        let mut v = gr[ev_off + i];

        if m > 0 {
            let mut lo = 0usize;
            if i >= m {
                lo = i - m;
            }
            let mut hi = i + m + 1;
            if hi > ev_len {
                hi = ev_len;
            }

            let mut acc = F::new(0.0f32);
            let mut cnt = F::new(0.0f32);
            let mut j = lo;
            while j < hi {
                acc += gr[ev_off + j];
                cnt += F::new(1.0f32);
                j += 1;
            }
            v = acc / cnt;
        }

        out[k * rows + r] = v;
    }
}

// ---------------------------------------------------------------------------
// Beam search
// ---------------------------------------------------------------------------

/// Index of the type-well sample nearest `v`, ties going to the lower index.
///
/// `tw` is sorted ascending over `[off, off + len)`, so the notebook's
/// `int(np.argmin(np.abs(tw_tvt - last_tvt)))` is a binary search plus one
/// neighbour comparison instead of a linear scan. Every unit runs it and gets the
/// same answer, so no shared memory or barrier is involved.
#[cube]
fn nearest_index<F: Float>(tw: &Array<F>, off: usize, len: usize, v: F) -> usize {
    let mut lo = 0usize;
    let mut hi = len;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if tw[off + mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // `lo` is the first sample at or above `v`; the nearest is `lo` or `lo - 1`.
    // `lo` is already the first index holding its value, so only the `lo - 1`
    // branch has to walk back over a run of equal TVTs to stay first-minimum.
    let mut si = lo;
    if lo >= len {
        si = len - 1;
    }
    if lo > 0 {
        let d_prev = v - tw[off + lo - 1];
        let mut d_cur = F::new(BEAM_INF);
        if lo < len {
            d_cur = tw[off + lo] - v;
        }
        // `<=`, not `<`: np.argmin returns the *first* minimum.
        if d_prev <= d_cur {
            // A second lower_bound, this time on the value itself: if several
            // samples log the identical TVT, argmin names the first of them, and
            // `lo - 1` is the last. Searching beats walking back one at a time,
            // and unlike a `while si > 0 && tw[si - 1] == dup` walk it cannot
            // form an out-of-range index — CubeCL's `&&` evaluates both sides.
            let dup = tw[off + lo - 1];
            let mut a = 0usize;
            let mut b = lo - 1;
            while a < b {
                let mid = (a + b) / 2;
                if tw[off + mid] < dup {
                    a = mid + 1;
                } else {
                    b = mid;
                }
            }
            si = a;
        }
    }
    si
}

/// Multi-well, multi-config beam search over the type-well index grid.
///
/// Layout of the per-well metadata (`well` indexes both arrays):
///
/// | `meta_u[well * 4 + k]` | meaning                                     |
/// |------------------------|---------------------------------------------|
/// | 0                      | offset of this well's eval rows in `sgr`     |
/// | 1                      | number of eval rows                         |
/// | 2                      | offset of this well's log in `tw_tvt`/`tw_gr`|
/// | 3                      | length of the type-well log                 |
///
/// `meta_f[well]` is `last_tvt`, the TVT the search starts from.
///
/// Per config: `cfg_u[c * 2]` is the beam size, `cfg_u[c * 2 + 1]` the smoothing
/// plane of `sgr` to read; `cfg_f[c * 2]` is the move cost and `cfg_f[c * 2 + 1]`
/// the GR error scale.
///
/// `sgr` is `[n_planes, total_rows]` and the output `[n_configs, total_rows]`,
/// both indexed by the flattened row, so a well's rows stay contiguous.
///
/// `cube_stride` is the total number of cubes launched: the grid-stride step.
/// (`CUBE_COUNT` is not available on every backend, so it is passed explicitly.)
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn beam_search_kernel<F: Float + CubeElement>(
    sgr: &Array<F>,
    tw_tvt: &Array<F>,
    tw_gr: &Array<F>,
    meta_u: &Array<u32>,
    meta_f: &Array<F>,
    cfg_u: &Array<u32>,
    cfg_f: &Array<F>,
    out: &mut Array<F>,
    hist_idx: &mut Array<u32>,
    hist_parent: &mut Array<u32>,
    total_rows: u32,
    n_configs: u32,
    n_tasks: u32,
    cube_stride: u32,
    #[comptime] max_beam: usize,
) {
    let mut b_idx = SharedMemory::<u32>::new(max_beam);
    let mut b_cost = SharedMemory::<F>::new(max_beam);
    let mut c_idx = SharedMemory::<u32>::new(5 * max_beam);
    let mut c_cost = SharedMemory::<F>::new(5 * max_beam);
    let mut c_parent = SharedMemory::<u32>::new(5 * max_beam);
    let mut c_dead = SharedMemory::<u32>::new(5 * max_beam);

    let t = UNIT_POS as usize;
    let cd = CUBE_DIM as usize;
    let rows = total_rows as usize;
    let inf = F::new(BEAM_INF);
    let dead_at = F::new(BEAM_DEAD);

    let mut task = u32::cast_from(CUBE_POS);
    while task < n_tasks {
        let well = (task / n_configs) as usize;
        let cfg = (task % n_configs) as usize;

        let ev_off = meta_u[well * 4] as usize;
        let ev_len = meta_u[well * 4 + 1] as usize;
        let tw_off = meta_u[well * 4 + 2] as usize;
        let tw_len = meta_u[well * 4 + 3] as usize;
        let last = meta_f[well];

        let bs = cfg_u[cfg * 2] as usize;
        let plane = cfg_u[cfg * 2 + 1] as usize;
        let mc = cfg_f[cfg * 2];
        let es = cfg_f[cfg * 2 + 1];

        let n_cand = bs * 5;
        let si = nearest_index::<F>(tw_tvt, tw_off, tw_len, last);

        // ---- seed the beam -------------------------------------------------
        // Only slot 0 is live; the rest are parked at BEAM_INF. The notebook
        // tracks a separate `bn` for this, but a dead cost is the same signal and
        // costs no extra state: candidates spawned from a dead slot are dead too.
        sync_cube();
        let mut j = t;
        while j < bs {
            b_idx[j] = si as u32;
            let mut c = inf;
            if j == 0 {
                c = F::new(0.0f32);
            }
            b_cost[j] = c;
            j += cd;
        }
        sync_cube();

        let mut step = 0usize;
        while step < ev_len {
            let gv = sgr[plane * rows + ev_off + step];

            // ---- expand: every live slot x every move ----------------------
            let mut u = t;
            while u < n_cand {
                let b = u / 5;
                let m = u % 5;
                let base = b_idx[b] as usize;
                let bc = b_cost[b];

                // Moves are -2..=2 and `m` walks them in that order. Splitting
                // into a down and an up component keeps the index arithmetic
                // unsigned: exactly one of the two is ever non-zero.
                let mut down = 0usize;
                let mut up = 0usize;
                if m < 2 {
                    down = 2 - m;
                } else {
                    up = m - 2;
                }

                let mut ni = base + up;
                let mut ok = bc < dead_at;
                if base >= down {
                    ni -= down;
                } else {
                    ok = false;
                }
                if ni >= tw_len {
                    ok = false;
                }
                let away = down + up;

                let mut cost = inf;
                if ok {
                    let d = gv - tw_gr[tw_off + ni];
                    cost = bc + d * d / es + mc * F::cast_from(away);
                }
                c_idx[u] = ni as u32;
                c_cost[u] = cost;
                c_parent[u] = b as u32;
                u += cd;
            }
            sync_cube();

            // ---- dedup: `np.unique` on the cost-sorted candidate list -------
            // A candidate dies if another candidate landed on the same type-well
            // sample more cheaply. Ties go to the lower slot, which is what
            // np.unique's first-occurrence rule does on a stably sorted array.
            let mut u2 = t;
            while u2 < n_cand {
                let cu = c_cost[u2];
                let iu = c_idx[u2];
                let mut dead = 0u32;
                if cu >= dead_at {
                    dead = 1u32;
                } else {
                    let mut v = 0usize;
                    while v < n_cand {
                        let cv = c_cost[v];
                        if cv < dead_at && c_idx[v] == iu && (cv < cu || (cv == cu && v < u2)) {
                            dead = 1u32;
                        }
                        v += 1;
                    }
                }
                c_dead[u2] = dead;
                u2 += cd;
            }
            sync_cube();

            // The beam has already been copied into the candidate list, so it can
            // be reset before the survivors scatter back into it.
            let mut j2 = t;
            while j2 < bs {
                b_cost[j2] = inf;
                j2 += cd;
            }
            sync_cube();

            // ---- rank: the top `bs` survivors, cheapest first ---------------
            // Tie-break by candidate slot `u = b*5 + m`, not type-well index: the
            // reference's `_beam_jit` is a hand-rolled selection sort over an
            // array built in `(beam slot, move)` generation order, so on an exact
            // cost tie it keeps whichever candidate was recorded *first* in that
            // order — not the numerically lower type-well index. `u = b*5 + m`
            // enumerates candidates in that same order (dedup already keeps only
            // the lowest slot per surviving index, matching `_beam_jit`'s
            // first-occurrence-wins dedup), so comparing `v < u3` here reproduces
            // the reference's tie-break exactly.
            let mut u3 = t;
            while u3 < n_cand {
                if c_dead[u3] == 0u32 {
                    let cu = c_cost[u3];
                    let iu = c_idx[u3];
                    let mut rank = 0usize;
                    let mut v = 0usize;
                    while v < n_cand {
                        if c_dead[v] == 0u32 {
                            let cv = c_cost[v];
                            if cv < cu || (cv == cu && v < u3) {
                                rank += 1;
                            }
                        }
                        v += 1;
                    }
                    if rank < bs {
                        b_idx[rank] = iu;
                        b_cost[rank] = cu;
                        // Record this step's survivors (every slot, not just the
                        // current argmin) and which pre-step slot each descended
                        // from, for the end-of-search backtrack below — see the
                        // module doc's note on why this cannot emit greedily.
                        let hist_off = (cfg * rows + ev_off + step) * max_beam + rank;
                        hist_idx[hist_off] = iu;
                        hist_parent[hist_off] = c_parent[u3];
                    }
                }
                u3 += cd;
            }
            sync_cube();

            // ---- renormalise ------------------------------------------------
            let best = b_cost[0];
            let mut j3 = t;
            while j3 < bs {
                let c = b_cost[j3];
                if c < dead_at {
                    b_cost[j3] = c - best;
                }
                j3 += cd;
            }
            sync_cube();

            step += 1;
        }

        // ---- backtrack: find the cheapest final slot, then walk the history
        // backward, emitting each step's survivor on the way. Inherently
        // sequential (each step's slot is only known from the next step's
        // parent pointer), so only one unit does it; every unit still executes
        // the same number of `sync_cube()` calls, via the barrier at the top of
        // the next task iteration (or falls straight through at the last task,
        // where no further shared memory is touched by anyone).
        if t == 0 && ev_len > 0 {
            let mut best_slot = 0usize;
            let mut s = 1usize;
            while s < bs {
                if b_cost[s] < b_cost[best_slot] {
                    best_slot = s;
                }
                s += 1;
            }
            let mut slot = best_slot;
            let mut step2 = ev_len;
            while step2 > 0 {
                step2 -= 1;
                let hist_off = (cfg * rows + ev_off + step2) * max_beam + slot;
                out[cfg * rows + ev_off + step2] = tw_tvt[tw_off + hist_idx[hist_off] as usize];
                slot = hist_parent[hist_off] as usize;
            }
        }

        task += cube_stride;
    }
}

// ---------------------------------------------------------------------------
// Ensemble mean
// ---------------------------------------------------------------------------

/// Collapses `[n_configs, rows]` to the ensemble mean: `run_beam_ensemble`'s
/// `np.stack(beam_results, 0).mean(0)`.
///
/// Done on the device so a host that only wants the mean never pays to read back
/// `n_configs` times the output size.
#[cube(launch, launch_unchecked)]
pub fn beam_mean_kernel<F: Float + CubeElement>(
    per_config: &Array<F>,
    out: &mut Array<F>,
    total_rows: u32,
    n_configs: u32,
) {
    let r = ABSOLUTE_POS;
    let rows = total_rows as usize;
    if r < rows {
        let mut acc = F::new(0.0f32);
        let mut c = 0usize;
        while c < n_configs as usize {
            acc += per_config[c * rows + r];
            c += 1;
        }
        out[r] = acc / F::cast_from(n_configs);
    }
}
