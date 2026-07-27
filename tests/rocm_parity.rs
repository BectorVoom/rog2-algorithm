//! Parity tests against the scalar reference on the ROCm/HIP backend.
//!
//!     cargo test --release --no-default-features --features hip --test rocm_parity
//!
//! These require a functional ROCm installation and an AMD GPU.

use rog2_pf::{Backend, PfConfig, reference::run_reference, run_on, synthetic::synthetic_well};

fn cfg() -> PfConfig {
    PfConfig {
        n_particles: 32,
        n_seeds: 3,
        cube_dim: 16,
        ..PfConfig::default()
    }
}

const SCALES: [f32; 2] = [3.0, 8.0];

#[cfg(feature = "hip")]
#[test]
fn hip_matches_reference() {
    let cfg = cfg();
    let wells: Vec<_> = (0..2)
        .map(|s| synthetic_well(s + 1, 40 + 10 * s as usize).0)
        .collect();

    let gpu = run_on(Backend::Hip, &wells, &cfg, &SCALES).expect("hip run");
    let cpu = run_reference(&wells, &cfg, &SCALES).expect("reference run");

    assert_eq!(gpu.channels, cpu.channels);
    assert_eq!(gpu.ev_lens, cpu.ev_lens);

    for (i, (a, b)) in gpu.liks.iter().zip(&cpu.liks).enumerate() {
        let tol = 1e-3 * a.abs().max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "log-lik {i}: hip {a} vs reference {b}"
        );
    }

    for ch in &gpu.channels {
        let a = gpu.channel(ch).unwrap();
        let b = cpu.channel(ch).unwrap();
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-4,
            "channel {ch}: worst deviation {worst} exceeds 1e-4"
        );
    }
}

#[cfg(feature = "hip")]
#[test]
fn hip_reruns_are_bit_identical() {
    use rog2_pf::synthetic::synthetic_well;

    let cfg = cfg();
    let wells: Vec<_> = (0..2)
        .map(|s| synthetic_well(s + 1, 40 + 10 * s as usize).0)
        .collect();

    let a = run_on(Backend::Hip, &wells, &cfg, &SCALES).expect("first hip run");
    let b = run_on(Backend::Hip, &wells, &cfg, &SCALES).expect("second hip run");

    assert_eq!(a.values, b.values, "output values differ between runs");
    assert_eq!(a.liks, b.liks, "log-likelihoods differ between runs");
}
