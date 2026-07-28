//! Synthetic wells with the same shape as the competition data.
//!
//! Used by the tests and the benchmark binary so they can run without the ROGII
//! dataset. The generator is deterministic given a seed.

use crate::batch::WellInput;
use crate::beam::BeamWellInput;

/// Deterministic 32-bit PRNG, so generated wells are reproducible everywhere.
struct Lcg(u32);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / 16_777_216.0
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Builds one synthetic well with `n_rows` evaluation samples.
///
/// The type-well GR log is a sum of three sinusoids plus noise, the true TVT path
/// is a slow random walk, and the horizontal GR is that path sampled through the
/// type well with additive noise — the same signal structure the real filter sees.
pub fn synthetic_well(seed: u32, n_rows: usize) -> (WellInput, Vec<f32>) {
    let mut rng = Lcg(seed.wrapping_mul(2_654_435_761).wrapping_add(12345) | 1);

    let vmin = -300.0f32;
    let step = 0.2f32;
    // A real GR log is aperiodic, which is what lets a particle filter localise.
    // A smoothed random walk reproduces that; sinusoids would be ambiguous by
    // construction and the filter could only ever lock onto an arbitrary branch.
    let g_len = 4000usize; // 800 ft of TVT at 0.2 ft
    let raw: Vec<f32> = (0..g_len).map(|_| rng.normal()).collect();
    let half = 12usize; // ~5 ft smoothing window
    let grid: Vec<f32> = (0..g_len)
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(g_len);
            let m: f32 = raw[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
            (70.0 + 90.0 * m).clamp(5.0, 160.0)
        })
        .collect();

    // True stratigraphic path: slow random walk in TVT.
    let mut truth = Vec::with_capacity(n_rows);
    let mut tvt = -40.0f32;
    let mut rate = 0.0f32;
    let mut md = Vec::with_capacity(n_rows);
    let mut z = Vec::with_capacity(n_rows);
    let mut gr = Vec::with_capacity(n_rows);

    for i in 0..n_rows {
        rate = 0.995 * rate + 0.006 * rng.normal();
        tvt += rate * 1.0;
        truth.push(tvt);

        md.push(9000.0 + i as f32);
        z.push(12.0 * (i as f32 / 700.0).sin());

        let fi = (tvt - vmin) / step;
        let k = (fi as usize).min(g_len - 2);
        let t = fi - k as f32;
        let clean = grid[k] * (1.0 - t) + grid[k + 1] * t;
        gr.push(clean + 4.0 * rng.normal());
    }

    // The real pipeline seeds `ls` from the last known `TVT_input + Z` and `ir`
    // from the median along-hole rate over the tail of the known prefix. Mimic
    // both, otherwise the filter starts with a state velocity that cannot keep up
    // with the wellbore geometry.
    let ls = truth[0] + z[0];
    let ir = ((truth[1] + z[1]) - (truth[0] + z[0])) / (md[1] - md[0]);

    let well = WellInput {
        md,
        z,
        gr,
        grid,
        vmin,
        step,
        gs: 6.0,
        ls,
        ir,
        init_spr: 4.5,
        seed_base: seed,
    };
    (well, truth)
}

/// The same synthetic well shaped for the beam search.
///
/// The beam search walks the type-well log directly rather than a uniformly
/// resampled grid, so the `(TVT, GR)` pairs are handed over as-is — with a little
/// jitter on the TVT spacing, since the kernel must not assume a uniform axis.
/// Returns the well and the true TVT path.
pub fn synthetic_beam_well(seed: u32, n_rows: usize) -> (BeamWellInput, Vec<f64>) {
    let (w, truth) = synthetic_well(seed, n_rows);
    let tw_tvt: Vec<f64> = (0..w.grid.len())
        .map(|i| {
            // Stays strictly ascending: the jitter is well under one step.
            (w.vmin + i as f32 * w.step + 0.05 * w.step * (i as f32 * 0.7).sin()) as f64
        })
        .collect();

    let well = BeamWellInput {
        gr: w.gr.iter().map(|&x| x as f64).collect(),
        tw_tvt,
        tw_gr: w.grid.iter().map(|&x| x as f64).collect(),
        last_tvt: truth[0] as f64,
    };
    (well, truth.iter().map(|&x| x as f64).collect())
}

/// Root-mean-square error between two equal-length series.
pub fn rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let s: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (*x - *y) as f64;
            d * d
        })
        .sum();
    (s / a.len() as f64).sqrt() as f32
}

/// [`rmse`], for the beam search's f64 trajectories.
pub fn rmse64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    (s / a.len() as f64).sqrt()
}
