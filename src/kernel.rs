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
//!
//! # Backend-independent arithmetic
//!
//! Determinism across *backends* needs more than a counter-based RNG, because
//! systematic resampling turns rounding into a discrete choice: the normalised
//! weights are prefix-summed and `lower_bound` picks which particle each
//! resampling slot copies, so a single ULP can hand a slot to its neighbour and
//! the trajectory never comes back. Three things had to be pinned down, all of
//! them places where a shader compiler is allowed to differ from `rustc` and
//! from `nvcc`:
//!
//! * **`exp`.** wgpu's is the raw hardware `exp2(x * log2 e)`, ~14 ULP over the
//!   likelihood's range. [`exp_precise`] replaces it.
//! * **`sin` / `cos`.** AMD's are ~4e-7 absolute, enough to seed two backends
//!   with different particle clouds. [`sin_precise`] / [`cos_precise`] replace
//!   them over the Box-Muller angle range.
//! * **Multiply-add contraction and division.** wgpu contracts every
//!   `a * b + c` into an FMA and `rustc` contracts none, so every multiply-add
//!   in the weight path is written as an explicit [`fma`]. SPIR-V division is
//!   not correctly rounded either, so `1 / step` and `1 / gs` are computed on
//!   the host (see [`crate::batch::FlatBatch::build`]) and multiplied here.
//! * **`ln` / `sqrt` in the Box-Muller radius, and the runtime `/`s in the
//!   weight path.** These used to be the last disagreement wgpu/ROCm had with
//!   the CPU reference (13/192 seeds in `tests/wgpu_parity.rs`'s sweep, an
//!   identical 13/192 on ROCm). [`ln_precise`] replaces the hardware `ln` with
//!   an exact power-of-two range reduction (comparisons and multiplies by
//!   powers of two are exact in IEEE 754, not approximations) plus a Taylor
//!   series, so it never calls the hardware transcendental at all.
//!   [`sqrt_precise`] and [`recip_precise`] instead seed a Newton-Raphson
//!   refinement from one hardware `sqrt` / `/`; the refinement itself only
//!   multiplies and `fma`s (confirmed bit-identical on every backend here —
//!   see `src/bin/probe3.rs`), and each iteration squares the seed's
//!   backend-specific error, driving it far below one ULP of `f32` for the
//!   large majority of inputs regardless of which backend produced the seed.
//!   [`recip_precise`] closes the same gap for `w / ws`, `wt / ws` and
//!   `ws^2 / ws2` in [`pf_lik_kernel`] — runtime, per-step values a host
//!   precompute can't reach, and `ws^2 / ws2` in particular gates *whether*
//!   resampling runs at all this step, a binary decision at least as sensitive
//!   to a rounding wobble as `lower_bound` itself.
//!
//!   None of this reaches zero disagreement. wgpu's `/`, measured directly
//!   (`src/bin/probe2.rs`), is occasionally off by more than the "a few ULP"
//!   SPIR-V nominally allows, and on the rare input where the true quotient
//!   sits close to a rounding boundary, that larger seed error survives
//!   Newton refinement as a same-magnitude 1-ULP disagreement in the
//!   converged result — running a third iteration changes nothing, confirming
//!   it is not a convergence-depth problem. Measured on the same sweep: wgpu's
//!   13/192 fell to 9/192, ROCm's 13/192 fell to 6/192. Eliminating that
//!   residual for good would mean seeding from a bit-manipulation reciprocal
//!   (no hardware `/` anywhere, at the cost of a generic `FloatBits` bound)
//!   rather than refining one; not done here because the return no longer
//!   justifies that bound spreading through every caller in this file.

use cubecl::prelude::*;

/// 2*pi, for Box-Muller.
const TWO_PI: f32 = 6.283_185_5;
/// Reciprocal of 2^24, for turning 24 random bits into a unit float.
const INV_2P24: f32 = 5.960_464_5e-8;
/// Matches the CPU version's `if dd > 600.: dd = 600.`.
const DD_MAX: f32 = 600.0;

/// `log2(e)` and `ln 2`, for [`exp_precise`].
const LOG2_E: f32 = core::f32::consts::LOG2_E;
const LN2: f32 = core::f32::consts::LN_2;
/// `2/pi` and `pi/2`, for [`sin_precise`] / [`cos_precise`].
const TWO_OVER_PI: f32 = core::f32::consts::FRAC_2_PI;
const PI_OVER_2: f32 = core::f32::consts::FRAC_PI_2;

/// Number of `u32` metadata slots per well.
pub const META_U_STRIDE: usize = 6;
/// Number of float metadata slots per well.
pub const META_F_STRIDE: usize = 8;

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
// Likelihood exponential
// ---------------------------------------------------------------------------

/// `exp(x)` within ~1.2 ULP, and — the actual point — bit-identical on every
/// backend and in the scalar reference.
///
/// `Float::exp` is not equally accurate everywhere. CUDA and the CPU runtime
/// call an argument-reduced `expf` (~1 ULP), but wgpu lowers it to the raw
/// hardware `exp2(x * log2 e)`, and rounding that product in f32 costs ~14 ULP —
/// 1.7e-6 of relative error — at the tail of the range the observation
/// likelihood uses (`x` reaches `ln(lik_floor)`, about -27.6). That is far too
/// coarse for this filter: the per-step weights feed a prefix scan whose
/// `lower_bound` boundaries decide *which particle* systematic resampling
/// duplicates, so a 1e-6 wobble flips whole particle selections, and after a few
/// dozen steps the trajectory has drifted feet away from the CUDA/CPU answer.
/// Measured on a Radeon 840M: `pf_mean` diverged by up to 0.46 ft at 128
/// particles, where the CPU runtime tracked the scalar reference to 4e-6 ft.
///
/// The argument reduction here moves that error into a polynomial instead: `x`
/// becomes `k * ln 2 + r` with `|r| <= ln2/2`, `exp(r)` comes from a degree-7
/// Taylor series (relative error < 5e-9 over that interval), and `2^k` is exact.
/// The residual is `k * (ln 2 - f32(ln 2))`, under 8e-8 wherever the result
/// clears `lik_floor`.
///
/// Sized for f32, which is what the particle filter launches; an f64
/// instantiation is still correct but no more accurate than about 1e-7.
#[cube]
pub fn exp_precise<F: Float>(x: F) -> F {
    let k = (x * F::new(LOG2_E)).round();
    // A single `fma`, not a two-word Cody-Waite split of `ln 2`: the split's two
    // constants sum back to `LN2` exactly in f32, so wgpu's shader compiler folds
    // them together and fuses the result anyway (verified: it emits exactly this
    // expression), while rustc leaves the split alone. Spelling the folded form
    // is what makes device and host agree bit for bit, and it costs nothing —
    // one rounded `fma` keeps `|r| <= ln2/2` and the residual `k * (ln 2 - LN2)`
    // stays under 8e-8 across the range that clears `lik_floor`.
    let r = fma(-k, F::new(LN2), x);

    // Explicit `fma`, not `p * r + c`: whether a backend contracts a multiply-add
    // is its own choice (SPIR-V may, rustc does not), and an unmatched
    // contraction is exactly the 1-ULP wobble this function exists to remove.
    // The scalar twin in `reference.rs` spells the same chain with `mul_add`.
    let mut p = F::new(1.0f32 / 5040.0);
    p = fma(p, r, F::new(1.0f32 / 720.0));
    p = fma(p, r, F::new(1.0f32 / 120.0));
    p = fma(p, r, F::new(1.0f32 / 24.0));
    p = fma(p, r, F::new(1.0f32 / 6.0));
    p = fma(p, r, F::new(0.5f32));
    p = fma(p, r, F::new(1.0f32));
    p = fma(p, r, F::new(1.0f32));

    // `2^k` for integral `k`. CubeCL's `Float` bound has no `exp2`, but `powf`
    // with a base of two lowers to `exp2(k * log2 2)` — an exact scaling — on
    // every backend here.
    p * F::powf(F::new(2.0f32), k)
}

/// `ln(x)` for `x` in `(0, 1]` — the only range [`unit_f`] can produce — bit-
/// identical on every backend because it never calls the hardware `ln`.
///
/// Range reduction writes `x = m * 2^-k` with `m` in `[0.5, 1]`, found by a
/// cascade of "double `m` and count it in `k`" steps gated on exact powers of
/// two. Doubling by an exact power of two only shifts the exponent field, so
/// unlike [`exp_precise`]'s argument reduction this step carries zero
/// rounding error, not just a bounded one, and needs no compensation term.
///
/// `ln(m)` then comes from the identity `ln(m) = 2*atanh(s)`, `s = (m-1)/(m+1)`:
/// over `m` in `[0.5, 1]`, `s` stays in `[-1/3, 0]`, so the series
/// `s + s^3/3 + s^5/5 + s^7/7 + s^9/9` (truncation error under 2e-9) converges
/// fast; the reciprocal that produces `s` is itself [`recip_precise`], so nothing
/// here calls the hardware `/` either. `ln(x) = ln(m) - k*ln 2`, folded through
/// one final `fma` for the same reason [`exp_precise`] uses one. End to end this
/// is within ~11 ULP of the true `ln`, worse than the series' own truncation
/// bound because the reciprocal refinement and the range reduction both add
/// their own rounding — the goal, as with [`exp_precise`], is the same answer
/// on every backend, not a better one than libm.
#[cube]
pub fn ln_precise<F: Float>(x: F) -> F {
    let mut m = x;
    let mut k = F::new(0.0f32);
    if m < F::new(1.0f32 / 65536.0) {
        m *= F::new(65536.0f32);
        k += F::new(16.0f32);
    }
    if m < F::new(1.0f32 / 256.0) {
        m *= F::new(256.0f32);
        k += F::new(8.0f32);
    }
    if m < F::new(1.0f32 / 16.0) {
        m *= F::new(16.0f32);
        k += F::new(4.0f32);
    }
    if m < F::new(0.25f32) {
        m *= F::new(4.0f32);
        k += F::new(2.0f32);
    }
    if m < F::new(0.5f32) {
        m *= F::new(2.0f32);
        k += F::new(1.0f32);
    }

    // `s = (m - 1) / (m + 1)` without a device division: wgpu's `/` is not
    // correctly rounded (same reason `sqrt_precise` seeds from one instead of
    // using it directly), so the reciprocal of `m + 1` is refined by the same
    // division-free Newton iteration, `z <- z * (2 - d*z)`, which halves the
    // error's *exponent* every step and so washes out the seed's rounding
    // before it can reach `s`.
    let d = m + F::new(1.0f32);
    let z0 = F::new(1.0f32) / d;
    let z1 = z0 * fma(-d, z0, F::new(2.0f32));
    let z2 = z1 * fma(-d, z1, F::new(2.0f32));
    let s = (m - F::new(1.0f32)) * z2;
    let u = s * s;
    let mut p = F::new(1.0f32 / 9.0);
    p = fma(p, u, F::new(1.0f32 / 7.0));
    p = fma(p, u, F::new(1.0f32 / 5.0));
    p = fma(p, u, F::new(1.0f32 / 3.0));
    p = fma(p, u, F::new(1.0f32));
    let ln_m = F::new(2.0f32) * (s * p);

    fma(-k, F::new(LN2), ln_m)
}

/// `sqrt(x)` for `x >= 0`, bit-identical on every backend regardless of how
/// the hardware `sqrt` and `/` behind the seed round.
///
/// Seeds a reciprocal-square-root estimate from the hardware `sqrt` and one
/// division (both merely close, not identical, across backends: `sqrt` is
/// ~0.7 ULP, wgpu's `/` is not correctly rounded), then runs the classic
/// division-free Newton iteration `z <- z * (1.5 - 0.5*x*z^2)` twice. That
/// iteration only multiplies and `fma`s — both correctly rounded everywhere —
/// and squares the *relative* error every step: a seed wrong by a few ULP
/// (~1e-6) lands the first iterate wrong by ~1e-12, already below `f32`'s
/// rounding quantum, so the result stops depending on which backend produced
/// the seed. The second iteration is margin. `x * z` recovers `sqrt(x)`.
/// `1/a`, bit-identical on every backend regardless of how the hardware `/`
/// that seeds it rounds.
///
/// Same division-free Newton iteration as the reciprocal inside
/// [`ln_precise`], pulled out because the weight-normalisation path
/// (`w / ws`, `wt / ws`, `ws*ws / ws2`) needs it too: those are runtime,
/// per-step values a host precompute can't reach, and `ws^2/ws2` in
/// particular gates *whether* systematic resampling runs at all this step —
/// a binary decision at least as sensitive to a rounding wobble as the
/// `lower_bound` search resampling itself does.
#[cube]
pub fn recip_precise<F: Float>(a: F) -> F {
    let z0 = F::new(1.0f32) / a;
    let z1 = z0 * fma(-a, z0, F::new(2.0f32));
    z1 * fma(-a, z1, F::new(2.0f32))
}

#[cube]
pub fn sqrt_precise<F: Float>(x: F) -> F {
    let mut out = F::new(0.0f32);
    if x > F::new(0.0f32) {
        let z0 = F::new(1.0f32) / F::sqrt(x);
        let xz0 = x * z0;
        let z1 = z0 * fma(F::new(-0.5f32), xz0 * z0, F::new(1.5f32));
        let xz1 = x * z1;
        let z2 = z1 * fma(F::new(-0.5f32), xz1 * z1, F::new(1.5f32));
        out = x * z2;
    }
    out
}

// ---------------------------------------------------------------------------
// Box-Muller trigonometry
// ---------------------------------------------------------------------------

/// Quadrant reduction shared by [`sin_precise`] and [`cos_precise`].
///
/// Returns `x - q * (pi/2)` with `|r| <= pi/4`; the caller re-derives `q` for the
/// quadrant select. A single `fma`, so no backend gets to choose an association
/// (see [`exp_precise`] for what happens when one does).
#[cube]
fn quadrant_r<F: Float>(x: F, q: F) -> F {
    fma(-q, F::new(PI_OVER_2), x)
}

/// `sin(x)` for the Box-Muller angle `x` in `[0, 2*pi]`, identical on every
/// backend.
///
/// AMD's `v_sin_f32` (what wgpu's `sin` becomes) is good to about 4e-7 absolute,
/// roughly 3.5 ULP, and CUDA's `sinf` rounds differently again. That error is
/// multiplied by `init_spr` when the cloud is seeded, which puts it above one
/// ULP of a particle position — so the two backends start from different clouds
/// and the resampler compounds the difference from there. A quadrant reduction
/// plus a degree-9 Taylor series over `|r| <= pi/4` is both tighter (~1.5 ULP,
/// bounded by `q * (pi/2 - f32(pi/2))`) and, more to the point, the same
/// expression everywhere.
///
/// Only valid for `x` in `[0, 2*pi]`, which is all `TWO_PI * unit_f(..)` can
/// produce: there is no Payne-Hanek fallback for large arguments.
#[cube]
pub fn sin_precise<F: Float>(x: F) -> F {
    let q = (x * F::new(TWO_OVER_PI)).round();
    let r = quadrant_r::<F>(x, q);
    let r2 = r * r;

    let mut s = F::new(1.0f32 / 362_880.0);
    s = fma(s, r2, F::new(-1.0f32 / 5040.0));
    s = fma(s, r2, F::new(1.0f32 / 120.0));
    s = fma(s, r2, F::new(-1.0f32 / 6.0));
    s = fma(s, r2, F::new(1.0f32));
    let sr = s * r;

    let mut c = F::new(1.0f32 / 40320.0);
    c = fma(c, r2, F::new(-1.0f32 / 720.0));
    c = fma(c, r2, F::new(1.0f32 / 24.0));
    c = fma(c, r2, F::new(-0.5f32));
    c = fma(c, r2, F::new(1.0f32));

    let qi = u32::cast_from(q) & 3u32;
    let mut out = sr;
    if qi == 1u32 {
        out = c;
    } else if qi == 2u32 {
        out = -sr;
    } else if qi == 3u32 {
        out = -c;
    }
    out
}

/// `cos(x)` for the Box-Muller angle `x` in `[0, 2*pi]` — see [`sin_precise`].
#[cube]
pub fn cos_precise<F: Float>(x: F) -> F {
    let q = (x * F::new(TWO_OVER_PI)).round();
    let r = quadrant_r::<F>(x, q);
    let r2 = r * r;

    let mut s = F::new(1.0f32 / 362_880.0);
    s = fma(s, r2, F::new(-1.0f32 / 5040.0));
    s = fma(s, r2, F::new(1.0f32 / 120.0));
    s = fma(s, r2, F::new(-1.0f32 / 6.0));
    s = fma(s, r2, F::new(1.0f32));
    let sr = s * r;

    let mut c = F::new(1.0f32 / 40320.0);
    c = fma(c, r2, F::new(-1.0f32 / 720.0));
    c = fma(c, r2, F::new(1.0f32 / 24.0));
    c = fma(c, r2, F::new(-0.5f32));
    c = fma(c, r2, F::new(1.0f32));

    let qi = u32::cast_from(q) & 3u32;
    let mut out = c;
    if qi == 1u32 {
        out = -sr;
    } else if qi == 2u32 {
        out = -c;
    } else if qi == 3u32 {
        out = sr;
    }
    out
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
fn interp1<F: Float>(grid: &Array<F>, off: usize, len: usize, v: F, vmin: F, inv_step: F) -> F {
    let fi = (v - vmin) * inv_step;
    let mut out = grid[off];
    if fi >= F::new(0.0f32) {
        let iu = u32::cast_from(fi) as usize;
        if iu >= len - 1 {
            out = grid[off + len - 1];
        } else {
            let t = fi - F::cast_from(iu);
            let a = grid[off + iu];
            let b = grid[off + iu + 1];
            // Explicit `fma` for the same reason as in `exp_precise`: left as
            // `a * (1 - t) + b * t` the backends are free to disagree on whether
            // to contract, and a 1-ULP wobble in the expected GR is multiplied by
            // `0.5 * dd` (up to 300) once it reaches the likelihood.
            out = fma(a, F::new(1.0f32) - t, b * t);
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
/// | `meta_f[well * 8 + k]` | meaning                                        |
/// |------------------------|------------------------------------------------|
/// | 0                      | `vmin`: TVT of the first GR grid sample         |
/// | 1                      | `step`: GR grid spacing in TVT                  |
/// | 2                      | `gs`: GR mismatch sigma                         |
/// | 3                      | `ls`: last-known `TVT_input + Z` (initial pos)  |
/// | 4                      | `ir`: initial along-hole TVT rate               |
/// | 5                      | `init_spr`: initial position spread             |
/// | 6                      | `1 / step`, precomputed on the host             |
/// | 7                      | `1 / gs`, precomputed on the host               |
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

        let vmin = meta_f[well * 8];
        let step = meta_f[well * 8 + 1];
        let ls = meta_f[well * 8 + 3];
        let ir = meta_f[well * 8 + 4];
        let spr = meta_f[well * 8 + 5];
        let inv_step = meta_f[well * 8 + 6];
        let inv_gs = meta_f[well * 8 + 7];

        let tmax = fma(F::cast_from(g_len), step, vmin);
        let lo_clamp = vmin - F::new(100.0f32);
        let hi_clamp = tmax + F::new(100.0f32);

        // ---- initialise the cloud -----------------------------------------
        let stream = well_seed + seed;
        let mut j = start;
        while j < end {
            let base = hash32(stream, j as u32);
            let rad = sqrt_precise::<F>(F::new(-2.0f32) * ln_precise::<F>(unit_f::<F>(hash32(base, 0u32))));
            let ang = F::new(TWO_PI) * unit_f::<F>(hash32(base, 1u32));
            pos[j] = fma(spr * rad, cos_precise::<F>(ang), ls);
            rate[j] = fma(F::new(0.01f32) * rad, sin_precise::<F>(ang), ir);
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
                let rad = sqrt_precise::<F>(F::new(-2.0f32) * ln_precise::<F>(u1));
                let ang = F::new(TWO_PI) * u2;

                let rj = fma(vn * rad, cos_precise::<F>(ang), mom * rate[p]);
                let mut tvt = fma(pn * rad, sin_precise::<F>(ang), fma(rj, dm, pos[p])) - zi;
                if tvt < lo_clamp {
                    tvt = lo_clamp;
                }
                if tvt > hi_clamp {
                    tvt = hi_clamp;
                }
                rate[p] = rj;
                pos[p] = tvt + zi;

                let eg = interp1::<F>(grid, g_off, g_len, tvt, vmin, inv_step);
                let d = (gri - eg) * inv_gs;
                let mut dd = d * d;
                if dd > F::new(DD_MAX) {
                    dd = F::new(DD_MAX);
                }
                let mut lk = exp_precise::<F>(F::new(-0.5f32) * dd);
                if lk < lik_floor {
                    lk = lik_floor;
                }

                let wn = w[p] * lk;
                w[p] = wn;
                s_w += wn;
                s_w2 = fma(wn, wn, s_w2);
                s_wt = fma(wn, tvt, s_wt);
                p += 1;
            }

            cube_reduce3::<F>(&mut red, s_w, s_w2, s_wt, cube_dim);
            let ws = red[0];
            let ws2 = red[cube_dim];
            let wt = red[2 * cube_dim];

            log_lik += ws.ln();
            let inv_ws = recip_precise::<F>(ws);
            let mut est = wt * inv_ws;

            // Effective sample size of the *normalised* weights:
            // sum((w/ws)^2) = ws2 / ws^2, so neff = ws^2 / ws2.
            let mut neff = n_f;
            if ws2 > F::new(0.0f32) {
                neff = ws * ws * recip_precise::<F>(ws2);
            }

            // ---- normalise ------------------------------------------------
            let mut q = start;
            while q < end {
                w[q] = w[q] * inv_ws;
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
                    let u = fma(F::cast_from(a), inv_n, u0);
                    w[a] = F::cast_from(lower_bound::<F>(&mut cum, n, u) as u32);
                    a += 1;
                }
                sync_cube();

                // Pass 2: permuted, roughened positions staged through `cum`.
                let mut b = start;
                while b < end {
                    let ci = u32::cast_from(w[b]) as usize;
                    let rctr = hash32(hash32(stream, b as u32), step_key);
                    let rad = sqrt_precise::<F>(F::new(-2.0f32) * ln_precise::<F>(unit_f::<F>(hash32(rctr, 2u32))));
                    let ang = F::new(TWO_PI) * unit_f::<F>(hash32(rctr, 3u32));
                    cum[b] = fma(rough_p * rad, cos_precise::<F>(ang), pos[ci]);
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
                    let rad = sqrt_precise::<F>(F::new(-2.0f32) * ln_precise::<F>(unit_f::<F>(hash32(rctr, 2u32))));
                    let ang = F::new(TWO_PI) * unit_f::<F>(hash32(rctr, 3u32));
                    cum[e] = fma(rough_r * rad, sin_precise::<F>(ang), rate[ci]);
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
