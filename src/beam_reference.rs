//! Scalar CPU reference for the beam search.
//!
//! Same maths, same tie-breaks and the same cost renormalisation as
//! [`crate::beam_kernel`], written as a straight loop. It exists so
//! `tests/beam_parity.rs` can pin the kernel against something readable, and so a
//! machine with no GPU can still run the algorithm.
//!
//! It is *not* the notebook's numpy code: that runs in f64 and leaves equal-cost
//! ordering to `np.argsort`. See the kernel's module docs for the two deliberate
//! deviations.

use crate::beam::{
    BeamConfig, BeamOptions, BeamOutput, BeamWellInput, FlatBeamBatch, SmoothingKind,
    smoothing_planes,
};
use crate::PfError;

// Independent f64 copies of the kernel's sentinels (beam_kernel's BEAM_INF/
// BEAM_DEAD stay f32 — they only ever feed CubeCL's `F::new()`, which requires
// an f32 argument regardless of the instantiated float type).
const BEAM_INF: f64 = 1e30;
const BEAM_DEAD: f64 = 5e29;

/// GR smoothing of one well, matching [`crate::beam_kernel::sg_smooth_kernel`]
/// (which see for both algorithms and why `RollingMean`, not `SavitzkyGolay`,
/// is the notebook-matching default).
pub fn sg_smooth(gr: &[f64], radius: u32, out: &mut [f64], kind: SmoothingKind) {
    match kind {
        SmoothingKind::RollingMean => sg_smooth_rolling_mean(gr, radius, out),
        SmoothingKind::SavitzkyGolay => sg_smooth_savgol(gr, radius, out),
    }
}

fn sg_smooth_rolling_mean(gr: &[f64], radius: u32, out: &mut [f64]) {
    let n = gr.len();
    let m = radius as usize;
    if m == 0 {
        out.copy_from_slice(gr);
        return;
    }
    for (i, o) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(m);
        let hi = (i + m + 1).min(n);
        let acc: f64 = gr[lo..hi].iter().sum();
        *o = acc / (hi - lo) as f64;
    }
}

/// Quadratic Savitzky-Golay smoothing, offered as a configurable alternative
/// to the notebook's rolling mean (see `sg_smooth_kernel`'s doc comment).
fn sg_smooth_savgol(gr: &[f64], radius: u32, out: &mut [f64]) {
    let n = gr.len();
    let m = radius as usize;
    if m == 0 || n <= 3 || n <= 2 * m + 1 {
        out.copy_from_slice(gr);
        return;
    }
    let w = 2 * m + 1;

    for (i, o) in out.iter_mut().enumerate() {
        let s = if i < m {
            0
        } else if i + m >= n {
            n - w
        } else {
            i - m
        };
        let xe = i as f64 - s as f64 - m as f64;

        let (mut s0, mut s2, mut s4) = (0.0f64, 0.0f64, 0.0f64);
        let (mut t0, mut t1, mut t2) = (0.0f64, 0.0f64, 0.0f64);
        for j in 0..w {
            let x = j as f64 - m as f64;
            let x2 = x * x;
            let y = gr[s + j];
            s0 += 1.0;
            s2 += x2;
            s4 += x2 * x2;
            t0 += y;
            t1 += x * y;
            t2 += x2 * y;
        }
        let det = s0 * s4 - s2 * s2;
        let a0 = (t0 * s4 - t2 * s2) / det;
        let a1 = t1 / s2;
        let a2 = (t2 * s0 - t0 * s2) / det;
        *o = a0 + a1 * xe + a2 * xe * xe;
    }
}

/// Index of the type-well sample nearest `v`, ties going to the lower index.
fn nearest_index(tw: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = tw.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if tw[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let mut si = lo.min(tw.len() - 1);
    if lo > 0 {
        let d_prev = v - tw[lo - 1];
        let d_cur = if lo < tw.len() {
            tw[lo] - v
        } else {
            BEAM_INF
        };
        if d_prev <= d_cur {
            si = lo - 1;
            // np.argmin returns the first minimum, so back up over a run of
            // samples logging the identical TVT.
            while si > 0 && tw[si - 1] == tw[si] {
                si -= 1;
            }
        }
    }
    si
}

/// One well, one config. `sgr` is the already-smoothed evaluation-row GR.
pub fn beam_search_one(
    sgr: &[f64],
    tw_tvt: &[f64],
    tw_gr: &[f64],
    last_tvt: f64,
    cfg: &BeamConfig,
    out: &mut [f64],
) {
    let bs = cfg.beam_size as usize;
    let n_cand = bs * 5;
    let nt = tw_tvt.len();
    let n = sgr.len();

    let si = nearest_index(tw_tvt, last_tvt);
    let mut b_idx = vec![si as u32; bs];
    let mut b_cost = vec![BEAM_INF; bs];
    b_cost[0] = 0.0;

    // `_beam_jit` does not emit the per-step argmin: it keeps every step's
    // survivor indices *and* which pre-step slot each descended from
    // (`hI`/`hP`), then — only after the whole sequence has run — finds the
    // single cheapest FINAL path and walks `hP` backward to recover it. A
    // candidate that is not the cheapest at step `t` can still end up on the
    // globally-best path, so emitting the step-wise argmin greedily (the
    // previous version of this function) is a different algorithm, not just a
    // different tie-break.
    let mut hist_idx = vec![0u32; n * bs];
    let mut hist_parent = vec![0u32; n * bs];

    let mut c_idx = vec![0u32; n_cand];
    let mut c_cost = vec![BEAM_INF; n_cand];
    let mut c_parent = vec![0u32; n_cand];
    let mut c_dead = vec![true; n_cand];

    for step in 0..n {
        let gv = sgr[step];

        for u in 0..n_cand {
            let b = u / 5;
            let m = u % 5;
            let base = b_idx[b] as usize;
            let bc = b_cost[b];

            let (down, up) = if m < 2 { (2 - m, 0) } else { (0, m - 2) };

            let mut ni = base + up;
            let mut ok = bc < BEAM_DEAD;
            if base >= down {
                ni -= down;
            } else {
                ok = false;
            }
            if ni >= nt {
                ok = false;
            }

            c_idx[u] = ni as u32;
            c_parent[u] = b as u32;
            c_cost[u] = if ok {
                let d = gv - tw_gr[ni];
                bc + d * d / cfg.err_scale + cfg.move_cost * (down + up) as f64
            } else {
                BEAM_INF
            };
        }

        for u in 0..n_cand {
            let (cu, iu) = (c_cost[u], c_idx[u]);
            c_dead[u] = cu >= BEAM_DEAD
                || (0..n_cand).any(|v| {
                    let cv = c_cost[v];
                    cv < BEAM_DEAD && c_idx[v] == iu && (cv < cu || (cv == cu && v < u))
                });
        }

        b_cost.fill(BEAM_INF);
        for u in 0..n_cand {
            if c_dead[u] {
                continue;
            }
            let (cu, iu) = (c_cost[u], c_idx[u]);
            // Tie-break by candidate slot, not type-well index — see the
            // matching comment in `beam_kernel::beam_search_kernel`.
            let rank = (0..n_cand)
                .filter(|v| {
                    if c_dead[*v] {
                        return false;
                    }
                    let cv = c_cost[*v];
                    cv < cu || (cv == cu && *v < u)
                })
                .count();
            if rank < bs {
                b_idx[rank] = iu;
                b_cost[rank] = cu;
                hist_idx[step * bs + rank] = iu;
                hist_parent[step * bs + rank] = c_parent[u];
            }
        }

        let best = b_cost[0];
        for c in b_cost.iter_mut() {
            if *c < BEAM_DEAD {
                *c -= best;
            }
        }
    }

    // ---- backtrack: find the cheapest final slot, then walk `hist_parent`
    // backward, emitting each step's survivor on the way.
    let mut best_slot = 0usize;
    for s in 1..bs {
        if b_cost[s] < b_cost[best_slot] {
            best_slot = s;
        }
    }
    let mut slot = best_slot;
    for step in (0..n).rev() {
        out[step] = tw_tvt[hist_idx[step * bs + slot] as usize];
        slot = hist_parent[step * bs + slot] as usize;
    }
}

/// Scalar equivalent of [`crate::beam_host::run_beam`].
pub fn run_beam_reference(
    wells: &[BeamWellInput],
    configs: &[BeamConfig],
    opts: &BeamOptions,
) -> Result<BeamOutput, PfError> {
    opts.validate(configs)?;
    let (batch, kept) = FlatBeamBatch::build(wells)?;
    let n_configs = configs.len();
    let total_rows = batch.total_rows();

    let mut out = BeamOutput {
        mean: vec![0.0; total_rows],
        per_config: Some(vec![0.0; n_configs * total_rows]),
        ev_offsets: batch.ev_offsets.clone(),
        ev_lens: batch.ev_lens.clone(),
        kept,
    };
    if batch.n_wells() == 0 {
        out.per_config = if opts.with_per_config {
            out.per_config
        } else {
            None
        };
        return Ok(out);
    }

    let (radii, plane_of) = smoothing_planes(configs);
    let per_config = out.per_config.as_mut().expect("allocated above");

    for w in 0..batch.n_wells() {
        let ev_off = batch.ev_offsets[w] as usize;
        let ev_len = batch.ev_lens[w] as usize;
        let tw_off = batch.meta_u[w * 4 + 2] as usize;
        let tw_len = batch.meta_u[w * 4 + 3] as usize;
        let gr = &batch.gr[ev_off..ev_off + ev_len];
        let tw_tvt = &batch.tw_tvt[tw_off..tw_off + tw_len];
        let tw_gr = &batch.tw_gr[tw_off..tw_off + tw_len];
        let last = batch.meta_f[w];

        let planes: Vec<Vec<f64>> = radii
            .iter()
            .map(|r| {
                let mut s = vec![0.0f64; ev_len];
                sg_smooth(gr, *r, &mut s, opts.smoothing);
                s
            })
            .collect();

        for (c, cfg) in configs.iter().enumerate() {
            let dst = c * total_rows + ev_off;
            beam_search_one(
                &planes[plane_of[c] as usize],
                tw_tvt,
                tw_gr,
                last,
                cfg,
                &mut per_config[dst..dst + ev_len],
            );
        }
    }

    for r in 0..total_rows {
        let mut acc = 0.0f64;
        for c in 0..n_configs {
            acc += per_config[c * total_rows + r];
        }
        // Division, not multiplication by a reciprocal: the kernel divides, and
        // the parity test compares these bit-for-bit on small batches.
        out.mean[r] = acc / n_configs as f64;
    }

    if !opts.with_per_config {
        out.per_config = None;
    }
    Ok(out)
}
