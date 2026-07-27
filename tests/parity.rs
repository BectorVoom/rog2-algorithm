//! The CubeCL kernel must reproduce the scalar reference implementation.
//!
//! These run on CubeCL's CPU runtime, which interprets the kernel and is orders of
//! magnitude slower than a GPU, so the configurations here are deliberately tiny.
//! Real-scale verification against the notebook's numba `lik_pf` happens on Kaggle
//! (see `kaggle/rog2-pf-cubecl-t4.ipynb`).

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

#[test]
fn kernel_matches_reference_on_cpu_runtime() {
    let cfg = cfg();
    let wells: Vec<_> = (0..2)
        .map(|s| synthetic_well(s + 1, 40 + 10 * s as usize).0)
        .collect();

    let gpu = run_on(Backend::Cpu, &wells, &cfg, &SCALES).expect("cubecl run");
    let cpu = run_reference(&wells, &cfg, &SCALES).expect("reference run");

    assert_eq!(gpu.channels, cpu.channels);
    assert_eq!(gpu.ev_lens, cpu.ev_lens);

    // Log-likelihoods accumulate one term per step, so compare relatively.
    for (i, (a, b)) in gpu.liks.iter().zip(&cpu.liks).enumerate() {
        let tol = 1e-3 * a.abs().max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "log-lik {i}: kernel {a} vs reference {b}"
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
        assert!(worst < 1e-2, "channel {ch}: max abs diff {worst}");
    }
}

#[test]
fn reruns_are_bit_identical() {
    let cfg = cfg();
    let wells = vec![synthetic_well(7, 40).0];
    let a = run_on(Backend::Cpu, &wells, &cfg, &SCALES).unwrap();
    let b = run_on(Backend::Cpu, &wells, &cfg, &SCALES).unwrap();
    assert_eq!(a.values, b.values);
    assert_eq!(a.liks, b.liks);
}

/// Changing `cube_dim` changes the reduction tree and the chunking, so results
/// shift slightly — but the trajectory must stay the same to within noise.
#[test]
fn cube_dim_does_not_change_the_answer_materially() {
    let base = PfConfig {
        n_particles: 32,
        n_seeds: 3,
        cube_dim: 8,
        ..PfConfig::default()
    };
    let wide = PfConfig {
        cube_dim: 32,
        ..base
    };
    let wells = vec![synthetic_well(5, 60).0];

    let a = run_on(Backend::Cpu, &wells, &base, &SCALES).unwrap();
    let b = run_on(Backend::Cpu, &wells, &wide, &SCALES).unwrap();

    let x = a.channel("pf_scale_3").unwrap();
    let y = b.channel("pf_scale_3").unwrap();
    let d = rog2_pf::synthetic::rmse(x, y);
    assert!(d < 3.0, "cube_dim changed pf_scale_3 by RMSE {d}");
}

#[test]
fn empty_wells_are_dropped_not_fatal() {
    let cfg = cfg();
    let mut wells = vec![synthetic_well(3, 30).0];
    let mut empty = synthetic_well(4, 30).0;
    empty.md.clear();
    empty.z.clear();
    empty.gr.clear();
    wells.insert(0, empty);

    let out = run_on(Backend::Cpu, &wells, &cfg, &SCALES).unwrap();
    assert_eq!(out.kept, vec![1]);
    assert_eq!(out.ev_lens, vec![30]);
}

#[test]
fn rejects_a_non_power_of_two_cube_dim() {
    let cfg = PfConfig {
        cube_dim: 100,
        ..PfConfig::default()
    };
    let wells = vec![synthetic_well(1, 10).0];
    assert!(run_on(Backend::Cpu, &wells, &cfg, &SCALES).is_err());
}

/// The filter must actually track: with enough particles it should beat the
/// last-known-value baseline on a drifting synthetic well. This exercises the
/// reference implementation (native code) rather than the interpreted runtime.
#[test]
fn filter_beats_the_last_known_baseline() {
    let cfg = PfConfig {
        n_particles: 256,
        n_seeds: 16,
        cube_dim: 64,
        ..PfConfig::default()
    };
    let (well, truth) = synthetic_well(11, 600);
    let out = run_reference(std::slice::from_ref(&well), &cfg, &SCALES).unwrap();

    let pred = out.well_channel("pf_scale_3", 0).unwrap();
    let baseline = vec![well.ls - well.z[0]; truth.len()];

    let pf = rog2_pf::synthetic::rmse(pred, &truth);
    let base = rog2_pf::synthetic::rmse(&baseline, &truth);
    assert!(
        pf < base,
        "particle filter RMSE {pf} should beat last-known baseline {base}"
    );
}
