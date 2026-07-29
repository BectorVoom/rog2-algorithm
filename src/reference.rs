//! Scalar CPU reference for [`crate::kernel::pf_lik_kernel`].
//!
//! This mirrors the kernel statement for statement, including the RNG streams and
//! the chunked tree-reduction order, so that differences between it and a device
//! run are attributable to the backend (FMA contraction, fast-math) rather than to
//! the algorithm. It is used by the parity tests and as a no-GPU fallback.

use crate::batch::{PfConfig, WellInput, seed_weights};
use crate::{PfError, PfOutput};

const TWO_PI: f32 = 6.283_185_5;
const INV_2P24: f32 = 5.960_464_5e-8;
const DD_MAX: f32 = 600.0;
const LOG2_E: f32 = core::f32::consts::LOG2_E;
const LN2: f32 = core::f32::consts::LN_2;
const TWO_OVER_PI: f32 = core::f32::consts::FRAC_2_PI;
const PI_OVER_2: f32 = core::f32::consts::FRAC_PI_2;

/// Scalar twin of [`crate::kernel::exp_precise`] — see that function for why the
/// filter cannot use the platform `exp`.
#[inline]
fn exp_precise(x: f32) -> f32 {
    let k = (x * LOG2_E).round();
    let r = (-k).mul_add(LN2, x);

    let mut p = 1.0f32 / 5040.0;
    p = p.mul_add(r, 1.0 / 720.0);
    p = p.mul_add(r, 1.0 / 120.0);
    p = p.mul_add(r, 1.0 / 24.0);
    p = p.mul_add(r, 1.0 / 6.0);
    p = p.mul_add(r, 0.5);
    p = p.mul_add(r, 1.0);
    p = p.mul_add(r, 1.0);

    p * 2.0f32.powf(k)
}

/// Scalar twins of [`crate::kernel::sin_precise`] / [`crate::kernel::cos_precise`],
/// valid for the Box-Muller angle range `[0, 2*pi]`.
#[inline]
fn sin_cos_precise(x: f32) -> (f32, f32) {
    let q = (x * TWO_OVER_PI).round();
    let r = (-q).mul_add(PI_OVER_2, x);
    let r2 = r * r;

    let mut s = 1.0f32 / 362_880.0;
    s = s.mul_add(r2, -1.0 / 5040.0);
    s = s.mul_add(r2, 1.0 / 120.0);
    s = s.mul_add(r2, -1.0 / 6.0);
    s = s.mul_add(r2, 1.0);
    let sr = s * r;

    let mut c = 1.0f32 / 40320.0;
    c = c.mul_add(r2, -1.0 / 720.0);
    c = c.mul_add(r2, 1.0 / 24.0);
    c = c.mul_add(r2, -0.5);
    c = c.mul_add(r2, 1.0);

    match (q as u32) & 3 {
        1 => (c, -sr),
        2 => (-sr, -c),
        3 => (-c, sr),
        _ => (sr, c),
    }
}

#[inline]
fn hash32(a: u32, b: u32) -> u32 {
    let mut h = a
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(b.wrapping_mul(0x85EB_CA6B))
        .wrapping_add(0x1656_67B1);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}

/// The `(radius, sin, cos)` of one Box-Muller pair, matching the kernel's draw
/// order. The two normals are `rad * cos` and `rad * sin`; callers combine them
/// with the surrounding multiply-add explicitly, so the rounding matches the
/// kernel's `fma` exactly (see `kernel::pf_lik_kernel`).
#[inline]
fn normals(ctr: u32, k0: u32, k1: u32) -> (f32, f32, f32) {
    let rad = (-2.0 * unit_f(hash32(ctr, k0)).ln()).sqrt();
    let ang = TWO_PI * unit_f(hash32(ctr, k1));
    let (sn, cs) = sin_cos_precise(ang);
    (rad, sn, cs)
}

#[inline]
fn unit_f(x: u32) -> f32 {
    ((x >> 8) + 1) as f32 * INV_2P24
}

#[inline]
fn interp1(grid: &[f32], v: f32, vmin: f32, inv_step: f32) -> f32 {
    let fi = (v - vmin) * inv_step;
    if fi < 0.0 {
        return grid[0];
    }
    let iu = fi as u32 as usize;
    if iu >= grid.len() - 1 {
        return grid[grid.len() - 1];
    }
    let t = fi - iu as f32;
    grid[iu].mul_add(1.0 - t, grid[iu + 1] * t)
}

/// Reproduces the kernel's fused three-way tree reduction over `cube_dim` lanes.
fn tree_sum3(r0: &mut [f32], r1: &mut [f32], r2: &mut [f32]) -> (f32, f32, f32) {
    let mut s = r0.len() / 2;
    while s > 0 {
        for t in 0..s {
            r0[t] += r0[t + s];
            r1[t] += r1[t + s];
            r2[t] += r2[t + s];
        }
        s /= 2;
    }
    (r0[0], r1[0], r2[0])
}

/// Reproduces the kernel's two-level prefix scan.
fn two_level_scan(w: &[f32], cum: &mut [f32], red: &mut [f32], chunk: usize, n: usize) {
    for (t, red_t) in red.iter_mut().enumerate() {
        let start = t * chunk;
        let end = (start + chunk).min(n);
        let mut acc = 0.0f32;
        for j in start..end {
            acc += w[j];
            cum[j] = acc;
        }
        *red_t = acc;
    }
    let mut off = 1usize;
    while off < red.len() {
        let snapshot: Vec<f32> = red.to_vec();
        for t in off..red.len() {
            red[t] += snapshot[t - off];
        }
        off *= 2;
    }
    for t in 0..red.len() {
        let base = if t > 0 { red[t - 1] } else { 0.0 };
        let start = t * chunk;
        let end = (start + chunk).min(n);
        for c in cum.iter_mut().take(end).skip(start) {
            *c += base;
        }
    }
}

#[inline]
fn lower_bound(cum: &[f32], n: usize, u: f32) -> usize {
    let mut lo = 0usize;
    let mut hi = n - 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cum[mid] < u {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Runs one `(well, seed)` filter, writing `ev_len` estimates into `preds` and
/// returning the total log-likelihood.
///
/// The index-based loops mirror the kernel's slot addressing one for one; that
/// correspondence is the point of this function, so they are not rewritten as
/// iterator chains.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn run_single(wl: &WellInput, cfg: &PfConfig, seed: u32, preds: &mut [f32]) -> f32 {
    let n = cfg.n_particles as usize;
    let cube_dim = cfg.cube_dim as usize;
    let chunk = n.div_ceil(cube_dim);
    let n_f = n as f32;
    let inv_n = 1.0 / n_f;

    let mut pos = vec![0.0f32; n];
    let mut rate = vec![0.0f32; n];
    let mut w = vec![0.0f32; n];
    let mut cum = vec![0.0f32; n];
    let mut r0 = vec![0.0f32; cube_dim];
    let mut r1 = vec![0.0f32; cube_dim];
    let mut r2 = vec![0.0f32; cube_dim];

    let tmax = (wl.grid.len() as f32).mul_add(wl.step, wl.vmin);
    // Mirrors the reciprocals `FlatBatch::build` uploads; see `batch.rs`.
    let inv_step = 1.0 / wl.step;
    let inv_gs = 1.0 / wl.gs;
    let lo_clamp = wl.vmin - 100.0;
    let hi_clamp = tmax + 100.0;

    let stream = wl.seed_base.wrapping_add(seed);
    for j in 0..n {
        let base = hash32(stream, j as u32);
        let (rad, sn, cs) = normals(base, 0, 1);
        pos[j] = (wl.init_spr * rad).mul_add(cs, wl.ls);
        rate[j] = (0.01 * rad).mul_add(sn, wl.ir);
        w[j] = inv_n;
    }

    let mut log_lik = 0.0f32;

    for i in 0..wl.md.len() {
        let zi = wl.z[i];
        let gri = wl.gr[i];
        let dm = if i > 0 {
            (wl.md[i] - wl.md[i - 1]).max(1.0)
        } else {
            1.0
        };
        let step_key = i as u32;

        for t in 0..cube_dim {
            let start = t * chunk;
            let end = (start + chunk).min(n);
            let (mut s_w, mut s_w2, mut s_wt) = (0.0f32, 0.0f32, 0.0f32);
            for p in start..end {
                let ctr = hash32(hash32(stream, p as u32), step_key);
                let (rad, sn, cs) = normals(ctr, 0, 1);

                let rj = (cfg.vn * rad).mul_add(cs, cfg.mom * rate[p]);
                let tvt = ((cfg.pn * rad).mul_add(sn, rj.mul_add(dm, pos[p])) - zi)
                    .clamp(lo_clamp, hi_clamp);
                rate[p] = rj;
                pos[p] = tvt + zi;

                let eg = interp1(&wl.grid, tvt, wl.vmin, inv_step);
                let d = (gri - eg) * inv_gs;
                let dd = (d * d).min(DD_MAX);
                let lk = exp_precise(-0.5 * dd).max(cfg.lik_floor);

                let wn = w[p] * lk;
                w[p] = wn;
                s_w += wn;
                s_w2 = wn.mul_add(wn, s_w2);
                s_wt = wn.mul_add(tvt, s_wt);
            }
            r0[t] = s_w;
            r1[t] = s_w2;
            r2[t] = s_wt;
        }

        let (ws, ws2, wt) = tree_sum3(&mut r0, &mut r1, &mut r2);
        log_lik += ws.ln();
        let mut est = wt / ws;
        let neff = if ws2 > 0.0 { ws * ws / ws2 } else { n_f };

        for wj in w.iter_mut() {
            *wj /= ws;
        }

        if neff < cfg.resamp * n_f {
            two_level_scan(&w, &mut cum, &mut r0, chunk, n);
            let u0 = unit_f(hash32(hash32(stream, 0xA5A5_5A5A), step_key)) * inv_n;

            for a in 0..n {
                let u = (a as f32).mul_add(inv_n, u0);
                w[a] = lower_bound(&cum, n, u) as u32 as f32;
            }
            for b in 0..n {
                let ci = w[b] as u32 as usize;
                let (rad, _, cs) = normals(hash32(hash32(stream, b as u32), step_key), 2, 3);
                cum[b] = (cfg.rough_p * rad).mul_add(cs, pos[ci]);
            }
            for t in 0..cube_dim {
                let start = t * chunk;
                let end = (start + chunk).min(n);
                let mut acc = 0.0f32;
                for c in start..end {
                    pos[c] = cum[c];
                    acc += cum[c] - zi;
                }
                r0[t] = acc;
                r1[t] = 0.0;
                r2[t] = 0.0;
            }
            let src: Vec<usize> = w.iter().map(|v| *v as u32 as usize).collect();
            for e in 0..n {
                let (rad, sn, _) = normals(hash32(hash32(stream, e as u32), step_key), 2, 3);
                cum[e] = (cfg.rough_r * rad).mul_add(sn, rate[src[e]]);
            }
            for g in 0..n {
                rate[g] = cum[g];
                w[g] = inv_n;
            }
            let (acc, _, _) = tree_sum3(&mut r0, &mut r1, &mut r2);
            est = acc * inv_n;
        }

        preds[i] = est;
    }

    log_lik
}

/// CPU equivalent of [`crate::run_pf`], with the same output layout.
pub fn run_reference(
    wells: &[WellInput],
    cfg: &PfConfig,
    scales: &[f32],
) -> Result<PfOutput, PfError> {
    cfg.validate()?;
    let seeds = cfg.n_seeds as usize;
    let n_out = scales.len() + 1 + usize::from(cfg.with_std);

    let kept: Vec<usize> = wells
        .iter()
        .enumerate()
        .filter(|(_, w)| !w.md.is_empty())
        .map(|(i, _)| i)
        .collect();

    let mut ev_offsets = Vec::with_capacity(kept.len());
    let mut ev_lens = Vec::with_capacity(kept.len());
    let mut total_rows = 0usize;
    for &i in &kept {
        ev_offsets.push(total_rows as u32);
        ev_lens.push(wells[i].md.len() as u32);
        total_rows += wells[i].md.len();
    }

    let mut channels: Vec<String> = scales
        .iter()
        .map(|s| {
            if *s == s.trunc() {
                format!("pf_scale_{}", *s as i64)
            } else {
                format!("pf_scale_{s}")
            }
        })
        .collect();
    channels.push("pf_mean".to_string());
    if cfg.with_std {
        channels.push("pf_pt_std".to_string());
    }

    let mut values = vec![0.0f32; n_out * total_rows];
    let mut all_liks = vec![0.0f32; kept.len() * seeds];
    let mut pred_offsets = Vec::with_capacity(kept.len());
    let mut pred_len = 0usize;
    for &l in &ev_lens {
        pred_offsets.push(pred_len as u32);
        pred_len += l as usize * seeds;
    }
    let mut seed_paths = if cfg.with_seed_paths {
        Some(vec![0.0f32; pred_len])
    } else {
        None
    };

    for (w, &wi) in kept.iter().enumerate() {
        let wl = &wells[wi];
        let len = wl.md.len();
        let off = ev_offsets[w] as usize;

        let mut preds = vec![0.0f32; seeds * len];
        for s in 0..seeds {
            let lik = run_single(wl, cfg, s as u32, &mut preds[s * len..(s + 1) * len]);
            all_liks[w * seeds + s] = lik;
        }

        let liks = &all_liks[w * seeds..(w + 1) * seeds];
        let mut wts = vec![0.0f32; seeds];
        for (k, sc) in scales.iter().enumerate() {
            seed_weights(liks, *sc, &mut wts);
            blend_into(&preds, &wts, len, &mut values[k * total_rows + off..][..len]);
        }
        wts.fill(1.0 / seeds as f32);
        blend_into(
            &preds,
            &wts,
            len,
            &mut values[scales.len() * total_rows + off..][..len],
        );
        if let Some(paths) = seed_paths.as_mut() {
            let dst = pred_offsets[w] as usize;
            paths[dst..dst + preds.len()].copy_from_slice(&preds);
        }
        if cfg.with_std {
            let dst = &mut values[(scales.len() + 1) * total_rows + off..][..len];
            for (i, d) in dst.iter_mut().enumerate() {
                let mut m1 = 0.0f32;
                let mut m2 = 0.0f32;
                for s in 0..seeds {
                    let v = preds[s * len + i];
                    m1 += v;
                    m2 = v.mul_add(v, m2);
                }
                let inv = 1.0 / seeds as f32;
                m1 *= inv;
                m2 *= inv;
                *d = (-m1).mul_add(m1, m2).max(0.0).sqrt();
            }
        }
    }

    Ok(PfOutput {
        channels,
        values,
        liks: all_liks,
        n_seeds: cfg.n_seeds,
        seed_paths,
        pred_offsets,
        ev_offsets,
        ev_lens,
        kept,
    })
}

fn blend_into(preds: &[f32], wts: &[f32], len: usize, out: &mut [f32]) {
    out.fill(0.0);
    for (s, wt) in wts.iter().enumerate() {
        let row = &preds[s * len..(s + 1) * len];
        for (o, p) in out.iter_mut().zip(row) {
            *o = wt.mul_add(*p, *o);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::exp_precise;

    /// `exp_precise` has to stay comfortably inside 2 ULP over the range the
    /// likelihood can reach — `-0.5 * dd` down to `ln(lik_floor)`, about -27.6.
    /// The point is not to beat libm (it does not) but to be the *same*
    /// expression on the host and in the kernel, so that no backend's `exp`
    /// gets a vote. wgpu's own `exp` is ~14 ULP over this range.
    #[test]
    fn exp_precise_is_accurate_over_the_likelihood_range() {
        let mut worst_ulp = 0.0f64;
        let mut worst_at = 0.0f32;
        for i in 0..200_001u32 {
            let x = -27.6 * (i as f32) / 200_000.0;
            let want = (x as f64).exp();
            let ulp = ((exp_precise(x) as f64) - want).abs() / (want * 1.192_092_9e-7);
            if ulp > worst_ulp {
                worst_ulp = ulp;
                worst_at = x;
            }
        }
        assert!(
            worst_ulp < 2.0,
            "exp_precise is {worst_ulp:.2} ULP off at x = {worst_at}"
        );
    }

    /// Below the likelihood floor the result is clamped anyway, but it still has
    /// to decay rather than blow up or go NaN.
    #[test]
    fn exp_precise_decays_past_the_floor() {
        for i in 0..600u32 {
            let x = -(i as f32);
            let v = exp_precise(x);
            assert!(v.is_finite() && v >= 0.0, "exp_precise({x}) = {v}");
            assert!(v <= 1.000_001, "exp_precise({x}) = {v} is not <= 1");
        }
    }
}
