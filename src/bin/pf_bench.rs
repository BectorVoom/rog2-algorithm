//! Benchmark and self-check for the CubeCL particle filter.
//!
//! ```text
//! pf-bench --backend cuda --wells 64 --rows 3000 --particles 500 --seeds 128
//! pf-bench --backend cpu --wells 2 --rows 100 --verify
//! ```
//!
//! `--verify` additionally runs the scalar reference and reports the worst
//! per-row deviation, which is the check that catches a broken kernel.

use std::time::Instant;

use rog2_pf::{
    Backend, PfConfig, reference::run_reference, run_on, synthetic::rmse, synthetic::synthetic_well,
};

struct Args {
    backend: Backend,
    wells: usize,
    rows: usize,
    particles: u32,
    seeds: u32,
    cube_dim: u32,
    verify: bool,
    repeat: usize,
}

fn parse() -> Args {
    let mut a = Args {
        backend: Backend::Auto,
        wells: 16,
        rows: 2000,
        particles: 500,
        seeds: 128,
        cube_dim: 256,
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
            "--particles" => {
                a.particles = val(i).parse().unwrap();
                i += 2;
            }
            "--seeds" => {
                a.seeds = val(i).parse().unwrap();
                i += 2;
            }
            "--cube-dim" => {
                a.cube_dim = val(i).parse().unwrap();
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
    let cfg = PfConfig {
        n_particles: args.particles,
        n_seeds: args.seeds,
        cube_dim: args.cube_dim,
        ..PfConfig::default()
    };
    let scales = [3.0f32, 5.0, 8.0, 12.0];

    let wells: Vec<_> = (0..args.wells)
        .map(|w| synthetic_well(w as u32 + 1, args.rows).0)
        .collect();

    println!(
        "backends available: {:?}\nwells {} x {} rows, {} particles, {} seeds, cube_dim {}",
        rog2_pf::available_backends(),
        args.wells,
        args.rows,
        args.particles,
        args.seeds,
        args.cube_dim
    );

    // Particle-steps is the honest unit of work: it is what both the CPU and the
    // GPU implementation have to get through.
    let particle_steps =
        args.wells as f64 * args.rows as f64 * args.particles as f64 * args.seeds as f64;

    let mut best = f64::INFINITY;
    let mut out = None;
    for r in 0..args.repeat {
        let t0 = Instant::now();
        let o = run_on(args.backend, &wells, &cfg, &scales).expect("run");
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "  run {}: {:.3} s  ({:.2} G particle-steps/s)",
            r,
            dt,
            particle_steps / dt / 1e9
        );
        best = best.min(dt);
        out = Some(o);
    }
    let out = out.unwrap();
    println!(
        "best: {:.3} s  ({:.2} G particle-steps/s)",
        best,
        particle_steps / best / 1e9
    );

    let ch = out.channel("pf_scale_3").unwrap();
    let finite = ch.iter().filter(|v| v.is_finite()).count();
    println!(
        "pf_scale_3: {} rows, {} finite, first {:?}",
        ch.len(),
        finite,
        &ch[..ch.len().min(4)]
    );
    assert_eq!(finite, ch.len(), "kernel produced non-finite output");

    if args.verify {
        let t0 = Instant::now();
        let refr = run_reference(&wells, &cfg, &scales).expect("reference");
        println!("reference: {:.3} s", t0.elapsed().as_secs_f64());

        for name in &out.channels {
            let a = out.channel(name).unwrap();
            let b = refr.channel(name).unwrap();
            let worst = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            println!("  {name}: rmse {:.3e}  max |diff| {:.3e}", rmse(a, b), worst);
        }
    }
}
