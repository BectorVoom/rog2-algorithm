//! Benchmark and self-check for the CubeCL beam search.
//!
//! ```text
//! beam-bench --backend cuda --wells 64 --rows 3000
//! beam-bench --backend cpu --wells 2 --rows 100 --verify
//! ```
//!
//! `--verify` additionally runs the scalar reference and reports how many rows
//! landed on a different type-well sample, which is the check that catches a
//! broken kernel. The trajectory is a table lookup, so that count should be zero.

use std::time::Instant;

use rog2_pf::beam::{BeamConfig, BeamOptions, NOTEBOOK_BEAM_CONFIGS};
use rog2_pf::beam_reference::run_beam_reference;
use rog2_pf::synthetic::synthetic_beam_well;
use rog2_pf::{Backend, run_beam_on};

struct Args {
    backend: Backend,
    wells: usize,
    rows: usize,
    cube_dim: u32,
    beam_size: Option<u32>,
    verify: bool,
    repeat: usize,
}

fn parse() -> Args {
    let mut a = Args {
        backend: Backend::Auto,
        wells: 16,
        rows: 2000,
        cube_dim: 64,
        beam_size: None,
        verify: false,
        repeat: 1,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let val = |i: usize| -> String {
            argv.get(i + 1)
                .unwrap_or_else(|| panic!("{} needs a value", argv[i]))
                .clone()
        };
        match argv[i].as_str() {
            "--backend" => {
                a.backend = val(i).parse().expect("backend");
                i += 2;
            }
            "--wells" => {
                a.wells = val(i).parse().unwrap();
                i += 2;
            }
            "--rows" => {
                a.rows = val(i).parse().unwrap();
                i += 2;
            }
            "--cube-dim" => {
                a.cube_dim = val(i).parse().unwrap();
                i += 2;
            }
            "--beam-size" => {
                a.beam_size = Some(val(i).parse().unwrap());
                i += 2;
            }
            "--repeat" => {
                a.repeat = val(i).parse().unwrap();
                i += 2;
            }
            "--verify" => {
                a.verify = true;
                i += 1;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    a
}

fn main() {
    let args = parse();

    // `--beam-size` overrides every config's width, so the cost model can be
    // probed independently of the notebook's mixture.
    let configs: Vec<BeamConfig> = NOTEBOOK_BEAM_CONFIGS
        .iter()
        .map(|c| match args.beam_size {
            Some(bs) => BeamConfig { beam_size: bs, ..*c },
            None => *c,
        })
        .collect();

    let opts = BeamOptions {
        cube_dim: args.cube_dim,
        with_per_config: args.verify,
        ..BeamOptions::default()
    };

    let wells: Vec<_> = (0..args.wells)
        .map(|w| synthetic_beam_well(w as u32 + 1, args.rows).0)
        .collect();

    println!(
        "backends available: {:?}\nwells {} x {} rows, {} configs, cube_dim {}",
        rog2_pf::available_backends(),
        args.wells,
        args.rows,
        configs.len(),
        args.cube_dim
    );

    // Candidate-steps is the honest unit of work: every step of every config
    // expands `5 * beam_size` candidates, and both collective passes are
    // quadratic in that count.
    let candidate_steps: f64 = configs
        .iter()
        .map(|c| args.wells as f64 * args.rows as f64 * 5.0 * c.beam_size as f64)
        .sum();

    let mut best = f64::INFINITY;
    let mut out = None;
    for r in 0..args.repeat {
        let t0 = Instant::now();
        let o = run_beam_on(args.backend, &wells, &configs, &opts).expect("run");
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "  run {}: {:.3} s  ({:.2} G candidate-steps/s)",
            r,
            dt,
            candidate_steps / dt / 1e9
        );
        best = best.min(dt);
        out = Some(o);
    }
    let out = out.unwrap();
    println!(
        "best: {:.3} s  ({:.2} G candidate-steps/s)",
        best,
        candidate_steps / best / 1e9
    );

    let finite = out.mean.iter().filter(|v| v.is_finite()).count();
    println!(
        "beam_mean: {} rows, {} finite, first {:?}",
        out.mean.len(),
        finite,
        &out.mean[..out.mean.len().min(4)]
    );
    assert_eq!(finite, out.mean.len(), "kernel produced non-finite output");

    if args.verify {
        let t0 = Instant::now();
        let refr = run_beam_reference(&wells, &configs, &opts).expect("reference");
        println!("reference: {:.3} s", t0.elapsed().as_secs_f64());

        let mut total_diff = 0usize;
        for c in 0..configs.len() {
            let a = out.config_rows(c).unwrap();
            let b = refr.config_rows(c).unwrap();
            let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
            total_diff += diff;
            if diff > 0 {
                println!("  config {c}: {diff} / {} rows differ", a.len());
            }
        }
        println!(
            "verify: {total_diff} differing rows across {} configs ({} expected)",
            configs.len(),
            0
        );
        assert_eq!(total_diff, 0, "kernel disagrees with the reference");
    }
}
