//! The CubeCL beam-search kernel must reproduce the scalar reference implementation.
//!
//! These run on CubeCL's CPU runtime, which interprets the kernel and is orders of
//! magnitude slower than a GPU, so the configurations here are deliberately tiny.
//! Real-scale verification against the notebook's numpy `beam_search` happens on
//! Kaggle (see `kaggle/rog2-beam-cubecl-t4.ipynb`).

use rog2_pf::beam::{BeamConfig, BeamOptions, max_beam_capacity, smoothing_planes};
use rog2_pf::beam_reference::{beam_search_one, run_beam_reference, sg_smooth};
use rog2_pf::synthetic::{rmse, synthetic_beam_well};
use rog2_pf::{Backend, run_beam_on};

/// Small enough for the interpreted CPU runtime, but still covering a smoothed
/// config, an unsmoothed one, and three different beam widths.
const CONFIGS: [BeamConfig; 3] = [
    BeamConfig::new(4, 20.0, 144.0, 2),
    BeamConfig::new(6, 8.0, 64.0, 0),
    BeamConfig::new(5, 12.0, 100.0, 3),
];

fn opts() -> BeamOptions {
    BeamOptions {
        cube_dim: 16,
        with_per_config: true,
        ..BeamOptions::default()
    }
}

/// The trajectory is a table lookup into `tw_tvt`, so agreement is not a matter
/// of tolerance: either the kernel picked the same type-well samples as the
/// reference or it did not.
#[test]
fn kernel_matches_reference_on_cpu_runtime() {
    let o = opts();
    let wells: Vec<_> = (0..2)
        .map(|s| synthetic_beam_well(s + 1, 40 + 10 * s as usize).0)
        .collect();

    let gpu = run_beam_on(Backend::Cpu, &wells, &CONFIGS, &o).expect("cubecl run");
    let cpu = run_beam_reference(&wells, &CONFIGS, &o).expect("reference run");

    assert_eq!(gpu.ev_lens, cpu.ev_lens);
    assert_eq!(gpu.kept, cpu.kept);
    for c in 0..CONFIGS.len() {
        assert_eq!(
            gpu.config_rows(c).unwrap(),
            cpu.config_rows(c).unwrap(),
            "config {c} trajectory differs"
        );
    }
    assert_eq!(gpu.mean, cpu.mean);
}

#[test]
fn reruns_are_bit_identical() {
    let o = opts();
    let wells = vec![synthetic_beam_well(7, 40).0];
    let a = run_beam_on(Backend::Cpu, &wells, &CONFIGS, &o).unwrap();
    let b = run_beam_on(Backend::Cpu, &wells, &CONFIGS, &o).unwrap();
    assert_eq!(a.mean, b.mean);
    assert_eq!(a.per_config, b.per_config);
}

/// Unlike the particle filter, nothing in the beam search depends on how the work
/// is split across units: dedup and ranking are defined over the whole candidate
/// list. Changing `cube_dim` must therefore change nothing at all.
#[test]
fn cube_dim_does_not_change_the_answer() {
    let wells = vec![synthetic_beam_well(5, 60).0];
    let narrow = BeamOptions {
        cube_dim: 8,
        with_per_config: true,
        ..BeamOptions::default()
    };
    let wide = BeamOptions {
        cube_dim: 32,
        ..narrow
    };

    let a = run_beam_on(Backend::Cpu, &wells, &CONFIGS, &narrow).unwrap();
    let b = run_beam_on(Backend::Cpu, &wells, &CONFIGS, &wide).unwrap();
    assert_eq!(a.per_config, b.per_config);
    assert_eq!(a.mean, b.mean);
}

#[test]
fn empty_wells_are_dropped_not_fatal() {
    let o = opts();
    let mut wells = vec![synthetic_beam_well(3, 30).0];
    let mut empty = synthetic_beam_well(4, 30).0;
    empty.gr.clear();
    wells.insert(0, empty);

    let out = run_beam_on(Backend::Cpu, &wells, &CONFIGS, &o).unwrap();
    assert_eq!(out.kept, vec![1]);
    assert_eq!(out.ev_lens, vec![30]);
}

#[test]
fn rejects_a_non_power_of_two_cube_dim() {
    let o = BeamOptions {
        cube_dim: 100,
        ..BeamOptions::default()
    };
    let wells = vec![synthetic_beam_well(1, 10).0];
    assert!(run_beam_on(Backend::Cpu, &wells, &CONFIGS, &o).is_err());
}

#[test]
fn rejects_a_degenerate_config() {
    let wells = vec![synthetic_beam_well(1, 10).0];
    let o = BeamOptions::default();
    for bad in [
        BeamConfig::new(0, 20.0, 144.0, 2),
        BeamConfig::new(4, 20.0, 0.0, 2),
        BeamConfig::new(4, -1.0, 144.0, 2),
        BeamConfig::new(4, 20.0, 144.0, 10_000),
    ] {
        assert!(
            run_beam_on(Backend::Cpu, &wells, &[bad], &o).is_err(),
            "{bad:?} should have been rejected"
        );
    }
    assert!(run_beam_on(Backend::Cpu, &wells, &[], &o).is_err());
}

/// The search must actually track: on a drifting synthetic well it should beat
/// holding the last known TVT. Uses the reference (native code) rather than the
/// interpreted runtime so the well can be long enough for drift to matter.
#[test]
fn beam_beats_the_last_known_baseline() {
    let (well, truth) = synthetic_beam_well(11, 600);
    let out = run_beam_reference(
        std::slice::from_ref(&well),
        &rog2_pf::NOTEBOOK_BEAM_CONFIGS,
        &BeamOptions::default(),
    )
    .unwrap();

    let pred = out.well_mean(0).unwrap();
    let baseline = vec![well.last_tvt; truth.len()];

    let beam = rmse(pred, &truth);
    let base = rmse(&baseline, &truth);
    assert!(
        beam < base,
        "beam search RMSE {beam} should beat last-known baseline {base}"
    );
}

// ---------------------------------------------------------------------------
// Savitzky-Golay
// ---------------------------------------------------------------------------

/// A quadratic smoother reproduces quadratics exactly — including in the `r` edge
/// rows, where scipy's `mode='interp'` fits the end window instead of padding.
/// This pins both branches of the smoother without a scipy dependency.
#[test]
fn sg_smooth_is_exact_on_quadratics() {
    let n = 40;
    let f = |i: usize| {
        let x = i as f32;
        3.0 - 0.5 * x + 0.02 * x * x
    };
    let gr: Vec<f32> = (0..n).map(f).collect();

    for r in 1..=5u32 {
        let mut out = vec![0.0f32; n];
        sg_smooth(&gr, r, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert!(
                (v - f(i)).abs() < 2e-3,
                "radius {r} row {i}: {v} vs exact {}",
                f(i)
            );
        }
    }
}

/// In the interior the smoother must equal the textbook quadratic Savitzky-Golay
/// convolution — for a 5-point window, `[-3, 12, 17, 12, -3] / 35`.
#[test]
fn sg_smooth_matches_the_textbook_five_point_kernel() {
    let gr: Vec<f32> = (0..30).map(|i| ((i * 37) % 23) as f32).collect();
    let mut out = vec![0.0f32; gr.len()];
    sg_smooth(&gr, 2, &mut out);

    let coef = [-3.0, 12.0, 17.0, 12.0, -3.0];
    for i in 2..gr.len() - 2 {
        let expect: f32 = (0..5).map(|j| coef[j] * gr[i - 2 + j]).sum::<f32>() / 35.0;
        assert!(
            (out[i] - expect).abs() < 1e-3,
            "row {i}: {} vs {expect}",
            out[i]
        );
    }
}

#[test]
fn sg_smooth_passes_short_series_through_untouched() {
    // The notebook's guard: no smoothing unless `n > max(3, 2 * r + 1)`, so with
    // five samples only radius 1 is wide enough to run at all.
    let gr = vec![5.0, 9.0, 2.0, 7.0, 1.0];
    for r in [0u32, 2, 3, 4] {
        let mut out = vec![0.0f32; gr.len()];
        sg_smooth(&gr, r, &mut out);
        assert_eq!(out, gr, "radius {r} should have been skipped");
    }
}

/// A quadratic through three points is interpolation, not regression, so
/// `radius = 1` is the identity — as it is in scipy. Two of the notebook's 14
/// configs specify it, and they are really running on raw GR.
#[test]
fn sg_smooth_radius_one_is_the_identity() {
    let gr: Vec<f32> = (0..20).map(|i| ((i * 41) % 17) as f32).collect();
    let mut out = vec![0.0f32; gr.len()];
    sg_smooth(&gr, 1, &mut out);
    for (a, b) in out.iter().zip(&gr) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

// ---------------------------------------------------------------------------
// Algorithmic properties of the search itself
// ---------------------------------------------------------------------------

/// With no move penalty the search is free to jump, so it should sit on the
/// type-well sample whose GR is closest to the observation.
#[test]
fn zero_move_cost_tracks_the_nearest_gr_sample() {
    let tw_tvt: Vec<f32> = (0..200).map(|i| i as f32 * 0.5).collect();
    let tw_gr: Vec<f32> = (0..200).map(|i| i as f32).collect();

    // Ramp slowly enough that a 2-cell-per-step beam can keep up.
    let sgr: Vec<f32> = (0..80).map(|i| 100.0 + i as f32).collect();
    let mut out = vec![0.0f32; sgr.len()];
    beam_search_one(
        &sgr,
        &tw_tvt,
        &tw_gr,
        50.0,
        &BeamConfig::new(8, 0.0, 1.0, 0),
        &mut out,
    );

    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, tw_tvt[100 + i], "row {i} left the matching sample");
    }
}

/// A huge move penalty should pin the search to wherever it started.
#[test]
fn prohibitive_move_cost_freezes_the_path() {
    let tw_tvt: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    let tw_gr: Vec<f32> = (0..100).map(|i| (i % 7) as f32 * 10.0).collect();
    let sgr = vec![65.0f32; 50];

    let mut out = vec![0.0f32; sgr.len()];
    beam_search_one(
        &sgr,
        &tw_tvt,
        &tw_gr,
        10.0,
        &BeamConfig::new(6, 1e6, 1.0, 0),
        &mut out,
    );
    assert!(out.iter().all(|v| *v == 10.0), "path drifted: {out:?}");
}

/// The start index is `np.argmin(|tw_tvt - last_tvt|)`, which returns the *first*
/// minimum. A type well with repeated TVT samples must therefore start on the
/// first of the run, not the last — a binary search alone gets this wrong.
#[test]
fn duplicate_type_well_depths_start_on_the_first_sample() {
    // Samples 2..=4 all sit at TVT 5.0 but log different GR. Starting at 5.4,
    // the nearest TVT is 5.0, and the notebook picks index 2.
    let tw_tvt = vec![1.0f32, 3.0, 5.0, 5.0, 5.0, 9.0];
    let tw_gr = vec![0.0f32, 0.0, 100.0, 50.0, 20.0, 0.0];
    // Freeze the path so the output is purely a function of the start index.
    let sgr = vec![0.0f32; 6];

    let mut out = vec![0.0f32; sgr.len()];
    beam_search_one(
        &sgr,
        &tw_tvt,
        &tw_gr,
        5.4,
        &BeamConfig::new(1, 1e9, 1.0, 0),
        &mut out,
    );
    // A beam of one with a prohibitive move cost stays put, and index 2 is the
    // only one of the three whose GR keeps it there at cost 100^2.
    assert!(out.iter().all(|v| *v == 5.0));

    // The same start, reached through the kernel, must agree with the reference.
    let well = rog2_pf::BeamWellInput {
        gr: sgr.clone(),
        tw_tvt,
        tw_gr,
        last_tvt: 5.4,
    };
    let cfgs = [BeamConfig::new(1, 1e9, 1.0, 0)];
    let o = BeamOptions {
        cube_dim: 8,
        with_per_config: true,
        ..BeamOptions::default()
    };
    let gpu = run_beam_on(Backend::Cpu, std::slice::from_ref(&well), &cfgs, &o).unwrap();
    let cpu = run_beam_reference(std::slice::from_ref(&well), &cfgs, &o).unwrap();
    assert_eq!(gpu.config_rows(0).unwrap(), cpu.config_rows(0).unwrap());
    assert_eq!(gpu.config_rows(0).unwrap(), &out[..]);
}

/// The search may not run off either end of the type-well log.
#[test]
fn moves_are_clamped_to_the_type_well() {
    let tw_tvt: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let tw_gr = vec![0.0f32; 12];
    let sgr = vec![1000.0f32; 40];

    for start in [0.0f32, 11.0] {
        let mut out = vec![0.0f32; sgr.len()];
        beam_search_one(
            &sgr,
            &tw_tvt,
            &tw_gr,
            start,
            &BeamConfig::new(5, 0.0, 1.0, 0),
            &mut out,
        );
        assert!(
            out.iter().all(|v| *v >= 0.0 && *v <= 11.0),
            "escaped the log from {start}: {out:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Batch plumbing
// ---------------------------------------------------------------------------

#[test]
fn smoothing_planes_are_shared_between_configs() {
    let (radii, plane) = smoothing_planes(&rog2_pf::NOTEBOOK_BEAM_CONFIGS);
    // The notebook's 14 configs use only six distinct radii.
    assert_eq!(radii.len(), 6);
    assert_eq!(plane.len(), 14);
    for (c, p) in rog2_pf::NOTEBOOK_BEAM_CONFIGS.iter().zip(&plane) {
        assert_eq!(radii[*p as usize], c.radius);
    }
}

#[test]
fn beam_capacity_covers_the_widest_beam() {
    let cap = max_beam_capacity(&rog2_pf::NOTEBOOK_BEAM_CONFIGS);
    assert!(cap.is_power_of_two());
    for c in rog2_pf::NOTEBOOK_BEAM_CONFIGS {
        assert!(c.beam_size as usize <= cap);
    }
}
