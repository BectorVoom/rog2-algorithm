//! CubeCL implementation of the ROGII multi-seed likelihood-weighted particle filter.
//!
//! This is a GPU port of `_pf_lik_allseeds` from
//! `notebook/rogii-another-approch-2nd.ipynb` (cell 38) — the "workhorse" tracker
//! that produces the `pf_scale_*` trajectories consumed by the selector / ridge blend.
//!
//! # Parallel decomposition
//!
//! The CPU version is a triple loop `seed -> md step -> particle`. Only the middle
//! loop is sequential (it is a filter), so the natural mapping is:
//!
//! * one **cube** (CUDA block) per `(well, seed)` pair — fully independent,
//! * one **unit** (thread) per contiguous slice of the particle cloud,
//! * shared memory holds the whole cloud (`pos`, `rate`, `w`),
//! * the three per-step particle reductions (weight sum, weight sum of squares,
//!   weighted position) are fused into a single tree reduction,
//! * systematic resampling uses a two-level in-cube prefix scan plus a per-slot
//!   binary search, replacing the CPU version's serial `while ci < N-1` walk.
//!
//! Wells are batched: all `(well, seed)` pairs are flattened into one task list and
//! the grid is sized to the device with a grid-stride loop over tasks
//! (see `10_grid_stride_occupancy.md`).
//!
//! # Shared-memory budget
//!
//! Only five allocations are used: `pos`, `rate`, `w`, `cum` and one packed
//! reduction scratch. Resampling would normally want two more buffers to hold the
//! permuted cloud; instead it round-trips through `cum` in separate passes, using
//! `w` (dead until it is reset to `1/N`) to carry the source indices. That keeps
//! occupancy up on a T4 and stays well inside the CubeCL CPU runtime's limit of
//! eight shared allocations per kernel.
//!
//! # Determinism
//!
//! The CPU version draws from numba's Mersenne Twister, seeded per seed index. That
//! stream cannot be reproduced bit-for-bit on a GPU where particles advance in
//! parallel. Instead the RNG is *counter-based*: every draw is a pure function of
//! `(seed_base + seed, particle slot, step index, draw index)`, so results are
//! bitwise reproducible across runs, devices and thread schedules for a fixed
//! `(seed_base, n_particles)`, but are not bit-identical to the numba
//! implementation. Nothing carries RNG state, so resampling — which permutes
//! particles between slots — cannot duplicate a stream.

use cubecl::prelude::*;

/// 2*pi, for Box-Muller.
const TWO_PI: f32 = 6.283_185_5;
/// Reciprocal of 2^24, for turning 24 random bits into a unit float.
const INV_2P24: f32 = 5.960_464_5e-8;
/// Matches the CPU version's `if dd > 600.: dd = 600.`.
const DD_MAX: f32 = 600.0;

/// Number of `u32` metadata slots per well.
pub const META_U_STRIDE: usize = 6;
/// Number of float metadata slots per well.
pub const META_F_STRIDE: usize = 6;

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

/// Murmur3-style 32-bit finalizer. Used both to key streams and to draw from
/// them, so no RNG state has to be stored anywhere.
#[cube]
pub fn hash32(a: u32, b: u32) -> u32 {
    let mut h = a * 0x9E37_79B9u32 + b * 0x85EB_CA6Bu32 + 0x1656_67B1u32;
    h ^= h >> 16u32;
    h *= 0x7FEB_352Du32;
    h ^= h >> 15u32;
    h *= 0x846C_A68Bu32;
    h ^= h >> 16u32;
    h
}

/// Map a random `u32` to a float in `(0, 1]` (never 0, so `ln` is finite).
#[cube]
pub fn unit_f<F: Float>(x: u32) -> F {
    F::cast_from((x >> 8u32) + 1u32) * F::new(INV_2P24)
}

// ---------------------------------------------------------------------------
// Type-well GR grid lookup
// ---------------------------------------------------------------------------

/// Linear interpolation on the uniform TVT grid of type-well GR values.
///
/// Mirrors `_interp1` from the notebook. The only deviation: for `v` in
/// `(vmin - step, vmin)` the CPU version linearly extrapolates below the grid,
/// while this clamps to `grid[0]`. Particle TVT is clamped to `vmin - 100` before
/// the lookup, so that window is unreachable in practice.
#[cube]
fn interp1<F: Float>(grid: &Array<F>, off: usize, len: usize, v: F, vmin: F, step: F) -> F {
    let fi = (v - vmin) / step;
    let mut out = grid[off];
    if fi >= F::new(0.0f32) {
        let iu = u32::cast_from(fi) as usize;
        if iu >= len - 1 {
            out = grid[off + len - 1];
        } else {
            let t = fi - F::cast_from(iu);
            let a = grid[off + iu];
            let b = grid[off + iu + 1];
            out = a * (F::new(1.0f32) - t) + b * t;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// In-cube collectives
// ---------------------------------------------------------------------------

/// Fused tree reduction of three per-unit partial sums.
///
/// The three lanes are packed into one shared allocation of `3 * cd` elements; on
/// return `red[0]`, `red[cd]` and `red[2 * cd]` hold the cube-wide sums and every
/// unit may read them. Requires `cd` to be a power of two.
///
/// The leading barrier is what makes it safe to call repeatedly: it guarantees
/// every unit has finished reading the *previous* result before any unit
/// overwrites the scratch.
#[cube]
fn cube_reduce3<F: Float>(red: &mut SharedMemory<F>, v0: F, v1: F, v2: F, #[comptime] cd: usize) {
    let t = UNIT_POS as usize;
    sync_cube();
    red[t] = v0;
    red[cd + t] = v1;
    red[2 * cd + t] = v2;
    sync_cube();

    // Bound comes from the runtime `CUBE_DIM`: a comptime loop counter cannot be
    // mutated (`Can't have a mutable operation on a const variable`).
    let mut s = CUBE_DIM as usize / 2;
    while s > 0 {
        if t < s {
            let a0 = red[t + s];
            let a1 = red[cd + t + s];
            let a2 = red[2 * cd + t + s];
            red[t] += a0;
            red[cd + t] += a1;
            red[2 * cd + t] += a2;
        }
        sync_cube();
        s /= 2;
    }
}

/// Two-level inclusive prefix scan of `w[0..n]` into `cum[0..n]`.
///
/// Unit `t` owns the contiguous slice `[start, end)`. It scans its own slice
/// serially, then a Hillis-Steele scan over the per-unit totals in the first lane
/// of `red` supplies the exclusive offset that is folded back into the slice.
#[cube]
fn cube_scan<F: Float>(
    w: &mut SharedMemory<F>,
    cum: &mut SharedMemory<F>,
    red: &mut SharedMemory<F>,
    start: usize,
    end: usize,
) {
    let t = UNIT_POS as usize;

    let mut acc = F::new(0.0f32);
    let mut j = start;
    while j < end {
        acc += w[j];
        cum[j] = acc;
        j += 1;
    }

    sync_cube();
    red[t] = acc;
    sync_cube();

    // `1usize` would be a comptime constant, which cannot be mutated inside a
    // loop; `runtime()` forces it into a device register.
    let mut off = 1usize.runtime();
    while off < CUBE_DIM as usize {
        // Read into a register, barrier, then write: avoids the in-place race.
        let mut v = F::new(0.0f32);
        if t >= off {
            v = red[t - off];
        }
        sync_cube();
        if t >= off {
            red[t] += v;
        }
        sync_cube();
        off *= 2;
    }

    let mut base = F::new(0.0f32);
    if t > 0 {
        base = red[t - 1];
    }
    let mut k = start;
    while k < end {
        cum[k] += base;
        k += 1;
    }
    sync_cube();
}

/// Smallest index `i` with `cum[i] >= u`, capped at `n - 1`.
///
/// Equivalent to the CPU version's forward walk, in `O(log n)` instead of `O(n)`.
#[cube]
fn lower_bound<F: Float>(cum: &mut SharedMemory<F>, n: usize, u: F) -> usize {
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

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// Multi-well, multi-seed likelihood-weighted particle filter.
///
/// Layout of the per-well metadata (`well` indexes both arrays):
///
/// | `meta_u[well * 6 + k]` | meaning                                        |
/// |------------------------|------------------------------------------------|
/// | 0                      | offset of this well's eval rows in `md/z/gr`    |
/// | 1                      | number of eval rows                            |
/// | 2                      | offset of this well's GR grid in `grid`         |
/// | 3                      | length of the GR grid                          |
/// | 4                      | offset of this well's block in `preds`          |
/// | 5                      | per-well RNG seed base                         |
///
/// | `meta_f[well * 6 + k]` | meaning                                        |
/// |------------------------|------------------------------------------------|
/// | 0                      | `vmin`: TVT of the first GR grid sample         |
/// | 1                      | `step`: GR grid spacing in TVT                  |
/// | 2                      | `gs`: GR mismatch sigma                         |
/// | 3                      | `ls`: last-known `TVT_input + Z` (initial pos)  |
/// | 4                      | `ir`: initial along-hole TVT rate               |
/// | 5                      | `init_spr`: initial position spread             |
///
/// `preds[pred_off + seed * ev_len + i]` receives the filtered TVT estimate and
/// `liks[well * n_seeds + seed]` the total log-likelihood of the seed.
///
/// `cube_stride` is the total number of cubes launched: the grid-stride step.
/// (`CUBE_COUNT` is not available on every backend, so it is passed explicitly.)
///
/// `lik_floor` is the per-observation likelihood floor (the CPU version hardcodes
/// `1e-300` in f64); the host picks a value the element type can represent
/// throughout the weight recursion.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn pf_lik_kernel<F: Float + CubeElement>(
    md: &Array<F>,
    z: &Array<F>,
    gr: &Array<F>,
    grid: &Array<F>,
    meta_u: &Array<u32>,
    meta_f: &Array<F>,
    preds: &mut Array<F>,
    liks: &mut Array<F>,
    mom: F,
    vn: F,
    pn: F,
    rough_p: F,
    rough_r: F,
    resamp: F,
    lik_floor: F,
    n_particles: u32,
    n_seeds: u32,
    n_tasks: u32,
    cube_stride: u32,
    #[comptime] smem_particles: usize,
    #[comptime] cube_dim: usize,
) {
    let mut pos = SharedMemory::<F>::new(smem_particles);
    let mut rate = SharedMemory::<F>::new(smem_particles);
    let mut w = SharedMemory::<F>::new(smem_particles);
    let mut cum = SharedMemory::<F>::new(smem_particles);
    let mut red = SharedMemory::<F>::new(3 * cube_dim);

    let t = UNIT_POS as usize;
    let n = n_particles as usize;
    let n_f = F::cast_from(n_particles);
    let inv_n = F::new(1.0f32) / n_f;
    // `usize::div_ceil` is not a CubeCL operation, so the ceiling is spelled out.
    #[allow(clippy::manual_div_ceil)]
    let chunk = (n + cube_dim - 1) / cube_dim;
    let start = t * chunk;
    let mut end = start + chunk;
    if end > n {
        end = n;
    }

    let mut task = u32::cast_from(CUBE_POS);
    while task < n_tasks {
        let well = (task / n_seeds) as usize;
        let seed = task % n_seeds;

        let ev_off = meta_u[well * 6] as usize;
        let ev_len = meta_u[well * 6 + 1] as usize;
        let g_off = meta_u[well * 6 + 2] as usize;
        let g_len = meta_u[well * 6 + 3] as usize;
        let p_off = meta_u[well * 6 + 4] as usize;
        let well_seed = meta_u[well * 6 + 5];

        let vmin = meta_f[well * 6];
        let step = meta_f[well * 6 + 1];
        let gs = meta_f[well * 6 + 2];
        let ls = meta_f[well * 6 + 3];
        let ir = meta_f[well * 6 + 4];
        let spr = meta_f[well * 6 + 5];

        let tmax = vmin + F::cast_from(g_len) * step;
        let lo_clamp = vmin - F::new(100.0f32);
        let hi_clamp = tmax + F::new(100.0f32);

        // ---- initialise the cloud -----------------------------------------
        let stream = well_seed + seed;
        let mut j = start;
        while j < end {
            let base = hash32(stream, j as u32);
            let rad = (F::new(-2.0f32) * unit_f::<F>(hash32(base, 0u32)).ln()).sqrt();
            let ang = F::new(TWO_PI) * unit_f::<F>(hash32(base, 1u32));
            pos[j] = ls + spr * rad * ang.cos();
            rate[j] = ir + F::new(0.01f32) * rad * ang.sin();
            w[j] = inv_n;
            j += 1;
        }
        sync_cube();

        let mut log_lik = F::new(0.0f32);

        let mut i = 0usize;
        while i < ev_len {
            let row = ev_off + i;
            let zi = z[row];
            let gri = gr[row];
            // Equivalent to the CPU version's `prev_md` walk: it starts from
            // `md[0] - 1`, so the first step always sees `dm = 1`.
            let mut dm = F::new(1.0f32);
            if i > 0 {
                dm = md[row] - md[row - 1];
                if dm < F::new(1.0f32) {
                    dm = F::new(1.0f32);
                }
            }
            let step_key = i as u32;

            // ---- propagate + weight ---------------------------------------
            let mut s_w = F::new(0.0f32);
            let mut s_w2 = F::new(0.0f32);
            let mut s_wt = F::new(0.0f32);

            let mut p = start;
            while p < end {
                let ctr = hash32(hash32(stream, p as u32), step_key);
                let u1 = unit_f::<F>(hash32(ctr, 0u32));
                let u2 = unit_f::<F>(hash32(ctr, 1u32));
                let rad = (F::new(-2.0f32) * u1.ln()).sqrt();
                let ang = F::new(TWO_PI) * u2;

                let rj = mom * rate[p] + vn * rad * ang.cos();
                let mut tvt = pos[p] + rj * dm + pn * rad * ang.sin() - zi;
                if tvt < lo_clamp {
                    tvt = lo_clamp;
                }
                if tvt > hi_clamp {
                    tvt = hi_clamp;
                }
                rate[p] = rj;
                pos[p] = tvt + zi;

                let eg = interp1::<F>(grid, g_off, g_len, tvt, vmin, step);
                let d = (gri - eg) / gs;
                let mut dd = d * d;
                if dd > F::new(DD_MAX) {
                    dd = F::new(DD_MAX);
                }
                let mut lk = (F::new(-0.5f32) * dd).exp();
                if lk < lik_floor {
                    lk = lik_floor;
                }

                let wn = w[p] * lk;
                w[p] = wn;
                s_w += wn;
                s_w2 += wn * wn;
                s_wt += wn * tvt;
                p += 1;
            }

            cube_reduce3::<F>(&mut red, s_w, s_w2, s_wt, cube_dim);
            let ws = red[0];
            let ws2 = red[cube_dim];
            let wt = red[2 * cube_dim];

            log_lik += ws.ln();
            let mut est = wt / ws;

            // Effective sample size of the *normalised* weights:
            // sum((w/ws)^2) = ws2 / ws^2, so neff = ws^2 / ws2.
            let mut neff = n_f;
            if ws2 > F::new(0.0f32) {
                neff = ws * ws / ws2;
            }

            // ---- normalise ------------------------------------------------
            let mut q = start;
            while q < end {
                w[q] = w[q] / ws;
                q += 1;
            }

            // ---- systematic resampling (cube-uniform branch) --------------
            if neff < resamp * n_f {
                cube_scan::<F>(&mut w, &mut cum, &mut red, start, end);

                let u0 = unit_f::<F>(hash32(hash32(stream, 0xA5A5_5A5Au32), step_key)) * inv_n;

                // Pass 1: resolve source slots. `w` is dead until it is reset to
                // 1/N below, so it can carry the indices without an extra buffer.
                let mut a = start;
                while a < end {
                    let u = u0 + F::cast_from(a) * inv_n;
                    w[a] = F::cast_from(lower_bound::<F>(&mut cum, n, u) as u32);
                    a += 1;
                }
                sync_cube();

                // Pass 2: permuted, roughened positions staged through `cum`.
                let mut b = start;
                while b < end {
                    let ci = u32::cast_from(w[b]) as usize;
                    let rctr = hash32(hash32(stream, b as u32), step_key);
                    let rad = (F::new(-2.0f32) * unit_f::<F>(hash32(rctr, 2u32)).ln()).sqrt();
                    let ang = F::new(TWO_PI) * unit_f::<F>(hash32(rctr, 3u32));
                    cum[b] = pos[ci] + rough_p * rad * ang.cos();
                    b += 1;
                }
                sync_cube();

                let mut acc = F::new(0.0f32);
                let mut c = start;
                while c < end {
                    let v = cum[c];
                    pos[c] = v;
                    acc += v - zi;
                    c += 1;
                }
                sync_cube();

                // Pass 3: same for the rates, reusing `cum` again.
                let mut e = start;
                while e < end {
                    let ci = u32::cast_from(w[e]) as usize;
                    let rctr = hash32(hash32(stream, e as u32), step_key);
                    let rad = (F::new(-2.0f32) * unit_f::<F>(hash32(rctr, 2u32)).ln()).sqrt();
                    let ang = F::new(TWO_PI) * unit_f::<F>(hash32(rctr, 3u32));
                    cum[e] = rate[ci] + rough_r * rad * ang.sin();
                    e += 1;
                }
                sync_cube();

                let mut g = start;
                while g < end {
                    rate[g] = cum[g];
                    w[g] = inv_n;
                    g += 1;
                }

                cube_reduce3::<F>(&mut red, acc, F::new(0.0f32), F::new(0.0f32), cube_dim);
                est = red[0] * inv_n;
            }

            if t == 0 {
                preds[p_off + seed as usize * ev_len + i] = est;
            }
            i += 1;
        }

        if t == 0 {
            liks[well * n_seeds as usize + seed as usize] = log_lik;
        }
        sync_cube();
        task += cube_stride;
    }
}
