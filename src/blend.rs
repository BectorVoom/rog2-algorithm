//! Seed-blending kernel.
//!
//! The CPU notebook does this on the host:
//!
//! ```python
//! ln = liks - liks.max()
//! for sc in scales:
//!     wts = np.exp(ln / sc); wts /= wts.sum()
//!     out[f"pf_scale_{sc}"] = (wts[:, None] * preds).sum(0)
//! out["pf_mean"] = preds.mean(0)
//! ```
//!
//! Materialising `preds` on the host would cost `n_seeds` (128) times the output
//! size, so the seed axis is reduced on the device instead: only the per-seed
//! log-likelihoods come back, the host turns them into softmax weights, and this
//! kernel collapses `[n_seeds, rows]` down to `[n_out, rows]`.

use cubecl::prelude::*;

/// Weighted reduction of the per-seed trajectories.
///
/// `wts[(well * n_out + k) * n_seeds + s]` is the weight of seed `s` in output
/// channel `k` of that well. Output channel `k` of row `r` lands in
/// `out[k * total_rows + r]`, so each channel is a contiguous row-aligned slice.
///
/// Channel `std_channel` (pass `u32::MAX` to disable) instead receives the
/// unweighted standard deviation across seeds.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn pf_blend_kernel<F: Float + CubeElement>(
    preds: &Array<F>,
    wts: &Array<F>,
    meta_u: &Array<u32>,
    row_well: &Array<u32>,
    out: &mut Array<F>,
    n_seeds: u32,
    n_out: u32,
    total_rows: u32,
    std_channel: u32,
) {
    let idx = ABSOLUTE_POS;
    let rows = total_rows as usize;
    if idx < rows * n_out as usize {
        // Consecutive units share `k` and walk consecutive rows, so both the
        // `preds` gather and the `out` store stay coalesced.
        let k = idx / rows;
        let r = idx % rows;

        let well = row_well[r] as usize;
        let ev_off = meta_u[well * 6] as usize;
        let ev_len = meta_u[well * 6 + 1] as usize;
        let p_off = meta_u[well * 6 + 4] as usize;
        let i = r - ev_off;

        let mut acc = F::new(0.0f32);
        if k == std_channel as usize {
            // Unweighted spread across seeds: the notebook's `pf_pt_std` quality
            // feature. Computed here because `preds` never reaches the host.
            let mut m1 = F::new(0.0f32);
            let mut m2 = F::new(0.0f32);
            let mut s = 0usize;
            while s < n_seeds as usize {
                let v = preds[p_off + s * ev_len + i];
                m1 += v;
                m2 += v * v;
                s += 1;
            }
            let inv = F::new(1.0f32) / F::cast_from(n_seeds);
            m1 *= inv;
            m2 *= inv;
            let mut var = m2 - m1 * m1;
            if var < F::new(0.0f32) {
                var = F::new(0.0f32);
            }
            acc = var.sqrt();
        } else {
            let wbase = (well * n_out as usize + k) * n_seeds as usize;
            let mut s = 0usize;
            while s < n_seeds as usize {
                acc += wts[wbase + s] * preds[p_off + s * ev_len + i];
                s += 1;
            }
        }
        out[k * rows + r] = acc;
    }
}
