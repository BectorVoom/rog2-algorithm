//! CubeCL implementation of the ROGII beam-search stratigraphic tracker.
//!
//! GPU port of `beam_search` / `run_beam_ensemble` from
//! `notebook/rogii-another-approch-2nd.ipynb` (cell 12). The search walks a
//! horizontal well's evaluation rows one at a time, keeping the `bs` cheapest
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
//! Five allocations: the beam (`b_idx`, `b_cost`), the candidate list (`c_idx`,
//! `c_cost`) and the dedup flags (`c_dead`). At the default `max_beam = 32` that
//! is under 2.5 KB per cube, so occupancy is bounded by cubes-per-SM rather than
//! by shared memory. It also stays inside the CubeCL CPU runtime's limit of eight
//! shared allocations per kernel.
//!
//! # Deliberate deviations from the notebook
//!
//! * **Cost renormalisation.** Path costs are monotonically increasing sums over
//!   thousands of steps and reach ~1e6 on real wells, where an f32 mantissa can no
//!   longer resolve a single step's increment. After each step the cube subtracts
//!   the best cost from the whole beam. Subtracting a constant from every path is
//!   order-preserving, so the argmin and the surviving set are unchanged, and the
//!   costs stay `O(1)`. numpy runs in f64 and does not need this.
//! * **Tie-breaking.** `np.argsort`/`np.unique` leave the order of exactly equal
//!   costs unspecified. Here dedup keeps the lowest candidate slot and ranking
//!   breaks ties by type-well index, both of which are deterministic and
//!   independent of the thread schedule.

use cubecl::prelude::*;

/// Stand-in for `np.inf` in the cost array. A large *finite* value, so that
/// `cost - best` on a dead slot cannot produce a NaN.
pub const BEAM_INF: f32 = 1e30;
/// Anything at or above this is a dead slot. Halfway to [`BEAM_INF`] so that
/// adding a plausible step cost to a dead entry cannot bring it back to life.
const BEAM_DEAD: f32 = 5e29;

// ---------------------------------------------------------------------------
// Savitzky-Golay smoothing
// ---------------------------------------------------------------------------

/// Quadratic Savitzky-Golay smoother, one output plane per distinct radius.
///
/// Reproduces `savgol_filter(hgr, 2 * r + 1, 2)` with scipy's default
/// `mode='interp'`: interior points are the least-squares quadratic through the
/// centred window evaluated at its centre, and the `r` points at each end come
/// from the single edge window's fit evaluated off-centre.
///
/// Rather than convolving with tabulated coefficients inside and switching to a
/// polynomial fit at the edges, this fits the quadratic directly in *centred*
/// window coordinates and evaluates it at an offset. The two cases collapse into
/// one code path (the interior is just offset zero), and centring kills the odd
/// moments, so the 3x3 normal equations decouple into a 2x2 solve plus a
/// division — well conditioned enough to stay in f32 where the uncentred
/// Vandermonde system would not be.
///
/// `radii[k]` is the half-window of output plane `k`; `0` copies the input, as
/// does any well too short for the notebook's `n > max(3, 2 * r + 1)` guard.
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

        // `ev_len > 2 * m + 1` and `ev_len > 3` are the notebook's guard; they
        // also guarantee the window fits and that `s2`/`det` below are non-zero.
        if m > 0 && ev_len > 3 && ev_len > 2 * m + 1 {
            let w = 2 * m + 1;

            // Window start, and where in centred coordinates this row sits.
            let mut s = i - m;
            if i < m {
                s = 0;
            } else if i + m >= ev_len {
                s = ev_len - w;
            }
            let xe = F::cast_from(i) - F::cast_from(s) - F::cast_from(m);

            let mut s0 = F::new(0.0f32);
            let mut s2 = F::new(0.0f32);
            let mut s4 = F::new(0.0f32);
            let mut t0 = F::new(0.0f32);
            let mut t1 = F::new(0.0f32);
            let mut t2 = F::new(0.0f32);

            let mut j = 0usize;
            while j < w {
                let x = F::cast_from(j) - F::cast_from(m);
                let x2 = x * x;
                let y = gr[ev_off + s + j];
                s0 += F::new(1.0f32);
                s2 += x2;
                s4 += x2 * x2;
                t0 += y;
                t1 += x * y;
                t2 += x2 * y;
                j += 1;
            }

            let det = s0 * s4 - s2 * s2;
            let a0 = (t0 * s4 - t2 * s2) / det;
            let a1 = t1 / s2;
            let a2 = (t2 * s0 - t0 * s2) / det;
            v = a0 + a1 * xe + a2 * xe * xe;
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
            si = lo - 1;
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
            // Survivors have distinct type-well indices, so `(cost, index)` is a
            // total order and every rank is unique — the scatter cannot collide.
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
                            if cv < cu || (cv == cu && c_idx[v] < iu) {
                                rank += 1;
                            }
                        }
                        v += 1;
                    }
                    if rank < bs {
                        b_idx[rank] = iu;
                        b_cost[rank] = cu;
                    }
                }
                u3 += cd;
            }
            sync_cube();

            // ---- emit, then renormalise ------------------------------------
            // Slot 0 is the cheapest surviving path. Read `best` before anyone
            // rewrites it.
            let best = b_cost[0];
            if t == 0 {
                out[cfg * rows + ev_off + step] = tw_tvt[tw_off + b_idx[0] as usize];
            }
            sync_cube();

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
