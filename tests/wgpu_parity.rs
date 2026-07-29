//! Parity tests against the scalar reference on the wgpu backend.
//!
//!     cargo test --release --features wgpu --test wgpu_parity
//!
//! These need a working Vulkan/Metal/DX12 device and compile away in builds
//! without the `wgpu` feature.
//!
//! wgpu is the backend most likely to drift from the reference, because its
//! shader compiler is the least constrained: it contracts every `a * b + c` into
//! an FMA, its `exp` is the raw hardware `exp2(x * log2 e)`, and its `/` is not
//! correctly rounded. The particle filter turns any of those into a *discrete*
//! error — the per-step weights feed a prefix scan whose `lower_bound`
//! boundaries pick which particle systematic resampling duplicates — so a 1e-6
//! wobble becomes feet of trajectory drift. `kernel::exp_precise`, the
//! host-precomputed `1/step` and `1/gs` in `batch::FlatBatch::build` and the
//! explicit `fma`s through the weight path exist to close that gap; these tests
//! are what pin it shut.

#![cfg(feature = "wgpu")]

use rog2_pf::{
    Backend, PfConfig, PfOutput, WellInput, reference::run_reference, run_on,
    synthetic::synthetic_well,
};

const SCALES: [f32; 2] = [3.0, 8.0];

fn wells(n: usize, rows: usize) -> Vec<WellInput> {
    (0..n)
        .map(|s| synthetic_well(s as u32 + 1, rows + 10 * s).0)
        .collect()
}

fn worst_channel_dev(a: &PfOutput, b: &PfOutput) -> f32 {
    a.channels
        .iter()
        .map(|ch| {
            let (x, y) = (a.channel(ch).unwrap(), b.channel(ch).unwrap());
            x.iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
        })
        .fold(0.0f32, f32::max)
}

#[test]
fn wgpu_matches_reference() {
    let cfg = PfConfig {
        n_particles: 32,
        n_seeds: 3,
        cube_dim: 16,
        ..PfConfig::default()
    };
    let w = wells(2, 40);

    let gpu = run_on(Backend::Wgpu, &w, &cfg, &SCALES).expect("wgpu run");
    let cpu = run_reference(&w, &cfg, &SCALES).expect("reference run");

    assert_eq!(gpu.channels, cpu.channels);
    assert_eq!(gpu.ev_lens, cpu.ev_lens);
    for (i, (a, b)) in gpu.liks.iter().zip(&cpu.liks).enumerate() {
        let tol = 1e-3 * a.abs().max(1.0);
        assert!((a - b).abs() <= tol, "log-lik {i}: wgpu {a} vs reference {b}");
    }
    let worst = worst_channel_dev(&gpu, &cpu);
    assert!(worst <= 1e-4, "worst channel deviation {worst} exceeds 1e-4");
}

/// The regression this file exists for.
///
/// A resample decision is discrete: once a rounding difference flips one
/// `lower_bound` boundary, that seed's trajectory separates from the reference
/// and stays separated, so the useful measure is not how far a diverged seed
/// drifts but *how often* a seed diverges at all. Sweeping 8 wells x 3 cloud
/// sizes x 8 seeds, 95 of 192 seeds diverged before the weight path was made
/// backend-independent (`kernel::exp_precise`, `sin_precise`, `cos_precise`, the
/// host-precomputed reciprocals and the explicit `fma`s); 13 do now, all of them
/// traceable to `ln`/`sqrt` in the Box-Muller radius, which are still the
/// hardware's. The bound here sits between the two.
#[test]
fn wgpu_rarely_diverges_from_reference_across_seeds() {
    let mut diverged = 0usize;
    let mut total = 0usize;
    for well_seed in 1..=8u32 {
        let w = vec![synthetic_well(well_seed, 60).0];
        for n_particles in [128u32, 256, 500] {
            let cfg = PfConfig {
                n_particles,
                n_seeds: 8,
                cube_dim: 64,
                ..PfConfig::default()
            };
            let gpu = run_on(Backend::Wgpu, &w, &cfg, &SCALES).expect("wgpu run");
            let cpu = run_reference(&w, &cfg, &SCALES).expect("reference run");
            for (a, b) in gpu.liks.iter().zip(&cpu.liks) {
                total += 1;
                if (a - b).abs() / a.abs().max(1.0) > 1e-4 {
                    diverged += 1;
                }
            }
        }
    }
    assert!(
        diverged * 5 <= total,
        "{diverged}/{total} seeds diverged from the reference (pre-fix: 95/192, fixed: 13/192)"
    );
}

/// Seeds that do *not* trip a resample flip have to agree to f32 noise, which is
/// what pins the elementwise weight path: `exp`, the GR interpolation and the
/// `(gr - eg) / gs` scaling all used to leak 1e-6 of backend-specific rounding
/// in here.
#[test]
fn wgpu_agrees_to_f32_noise_when_no_resample_flips() {
    let cfg = PfConfig {
        n_particles: 500,
        n_seeds: 8,
        cube_dim: 64,
        resamp: 0.0, // never resample: no discrete decision to flip
        ..PfConfig::default()
    };
    let w = wells(2, 60);

    let gpu = run_on(Backend::Wgpu, &w, &cfg, &SCALES).expect("wgpu run");
    let cpu = run_reference(&w, &cfg, &SCALES).expect("reference run");

    for (i, (a, b)) in gpu.liks.iter().zip(&cpu.liks).enumerate() {
        let rel = (a - b).abs() / a.abs().max(1.0);
        assert!(rel <= 1e-5, "log-lik {i}: wgpu {a} vs reference {b} (rel {rel:.2e})");
    }
    let worst = worst_channel_dev(&gpu, &cpu);
    assert!(worst <= 1e-3, "worst channel deviation {worst} exceeds 1e-3");
}

/// The filter must not depend on how the cloud is sliced across units.
#[test]
fn wgpu_is_stable_across_cube_dims() {
    let w = wells(1, 60);
    let base = PfConfig {
        n_particles: 128,
        n_seeds: 2,
        cube_dim: 64,
        ..PfConfig::default()
    };
    let cpu = run_reference(&w, &base, &SCALES).expect("reference run");

    for cube_dim in [16u32, 32, 64, 128] {
        let cfg = PfConfig { cube_dim, ..base };
        let gpu = run_on(Backend::Wgpu, &w, &cfg, &SCALES).expect("wgpu run");
        let worst = worst_channel_dev(&gpu, &cpu);
        assert!(worst <= 1e-3, "cube_dim {cube_dim}: worst deviation {worst}");
    }
}

#[test]
fn wgpu_reruns_are_bit_identical() {
    let cfg = PfConfig {
        n_particles: 128,
        n_seeds: 4,
        cube_dim: 64,
        ..PfConfig::default()
    };
    let w = wells(2, 40);

    let a = run_on(Backend::Wgpu, &w, &cfg, &SCALES).expect("first wgpu run");
    let b = run_on(Backend::Wgpu, &w, &cfg, &SCALES).expect("second wgpu run");
    assert_eq!(a.values, b.values, "output values differ between runs");
    assert_eq!(a.liks, b.liks, "log-likelihoods differ between runs");
}

/// The beam search is deterministic — no RNG, no resampling — so it has to
/// reproduce the reference exactly, at the notebook's full 14-config ensemble.
#[test]
fn wgpu_beam_ensemble_matches_reference() {
    use rog2_pf::beam::{BeamOptions, NOTEBOOK_BEAM_CONFIGS};
    use rog2_pf::beam_reference::run_beam_reference;
    use rog2_pf::run_beam_on;
    use rog2_pf::synthetic::synthetic_beam_well;

    let o = BeamOptions {
        with_per_config: true,
        ..BeamOptions::default()
    };
    let w: Vec<_> = (0..4)
        .map(|s| synthetic_beam_well(s + 1, 300 + 37 * s as usize).0)
        .collect();

    let gpu = run_beam_on(Backend::Wgpu, &w, &NOTEBOOK_BEAM_CONFIGS, &o).expect("wgpu run");
    let cpu = run_beam_reference(&w, &NOTEBOOK_BEAM_CONFIGS, &o).expect("reference run");

    assert_eq!(gpu.ev_lens, cpu.ev_lens);
    assert_eq!(gpu.kept, cpu.kept);
    for c in 0..NOTEBOOK_BEAM_CONFIGS.len() {
        assert_eq!(
            gpu.config_rows(c).unwrap(),
            cpu.config_rows(c).unwrap(),
            "config {c} trajectory differs"
        );
    }
    // The per-config trajectories are bit-identical, but the ensemble mean is
    // `acc / n_configs` in f64 and AMD implements f64 division by reciprocal
    // refinement, so the last bit is the device's to choose.
    let worst = gpu
        .mean
        .iter()
        .zip(&cpu.mean)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(worst <= 1e-12, "ensemble mean deviates by {worst}");
}
