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
    BeamConfig, BeamOptions, BeamOutput, BeamWellInput, FlatBeamBatch, smoothing_planes,
};
use crate::beam_kernel::BEAM_INF;
use crate::PfError;

const BEAM_DEAD: f32 = 5e29;

/// Quadratic Savitzky-Golay smoothing of one well's GR, matching
/// [`crate::beam_kernel::sg_smooth_kernel`].
pub fn sg_smooth(gr: &[f32], radius: u32, out: &mut [f32]) {
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
        let xe = i as f32 - s as f32 - m as f32;

        let (mut s0, mut s2, mut s4) = (0.0f32, 0.0f32, 0.0f32);
        let (mut t0, mut t1, mut t2) = (0.0f32, 0.0f32, 0.0f32);
        for j in 0..w {
            let x = j as f32 - m as f32;
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
fn nearest_index(tw: &[f32], v: f32) -> usize {
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
    sgr: &[f32],
    tw_tvt: &[f32],
    tw_gr: &[f32],
    last_tvt: f32,
    cfg: &BeamConfig,
    out: &mut [f32],
) {
    let bs = cfg.beam_size as usize;
    let n_cand = bs * 5;
    let nt = tw_tvt.len();

    let si = nearest_index(tw_tvt, last_tvt);
    let mut b_idx = vec![si as u32; bs];
    let mut b_cost = vec![BEAM_INF; bs];
    b_cost[0] = 0.0;

    let mut c_idx = vec![0u32; n_cand];
    let mut c_cost = vec![BEAM_INF; n_cand];
    let mut c_dead = vec![true; n_cand];

    for (step, o) in out.iter_mut().enumerate() {
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
            c_cost[u] = if ok {
                let d = gv - tw_gr[ni];
                bc + d * d / cfg.err_scale + cfg.move_cost * (down + up) as f32
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
            let rank = (0..n_cand)
                .filter(|v| {
                    if c_dead[*v] {
                        return false;
                    }
                    let cv = c_cost[*v];
                    cv < cu || (cv == cu && c_idx[*v] < iu)
                })
                .count();
            if rank < bs {
                b_idx[rank] = iu;
                b_cost[rank] = cu;
            }
        }

        *o = tw_tvt[b_idx[0] as usize];

        let best = b_cost[0];
        for c in b_cost.iter_mut() {
            if *c < BEAM_DEAD {
                *c -= best;
            }
        }
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

        let planes: Vec<Vec<f32>> = radii
            .iter()
            .map(|r| {
                let mut s = vec![0.0f32; ev_len];
                sg_smooth(gr, *r, &mut s);
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
        let mut acc = 0.0f32;
        for c in 0..n_configs {
            acc += per_config[c * total_rows + r];
        }
        // Division, not multiplication by a reciprocal: the kernel divides, and
        // the parity test compares these bit-for-bit on small batches.
        out.mean[r] = acc / n_configs as f32;
    }

    if !opts.with_per_config {
        out.per_config = None;
    }
    Ok(out)
}
