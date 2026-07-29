//! Diagnostic: are `ln_precise`/`sqrt_precise` themselves bit-identical across
//! backends, isolated from the particle filter's chaotic resampling?
//!
//!     cargo run --release --features wgpu --bin probe -- wgpu
//!     cargo run --release --features hip  --bin probe -- hip

use cubecl::prelude::*;
use rog2_pf::kernel::{ln_precise, sqrt_precise};

#[cube(launch, launch_unchecked)]
fn probe_kernel<F: Float + CubeElement>(xs: &Array<F>, ln_out: &mut Array<F>, sqrt_out: &mut Array<F>) {
    let i = ABSOLUTE_POS as usize;
    if i < xs.len() as usize {
        let x = xs[i];
        ln_out[i] = ln_precise::<F>(x);
        let arg = F::new(-2.0f32) * ln_precise::<F>(x);
        sqrt_out[i] = sqrt_precise::<F>(arg);
    }
}

fn run<R: Runtime>(client: &ComputeClient<R>, xs: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = xs.len();
    let h_xs = client.create_from_slice(bytemuck::cast_slice(xs));
    let h_ln = client.empty(n * 4);
    let h_sqrt = client.empty(n * 4);
    let cube_dim = CubeDim { x: 64, y: 1, z: 1 };
    let cube_count = CubeCount::Static(n.div_ceil(64) as u32, 1, 1);
    unsafe {
        probe_kernel::launch_unchecked::<f32, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(h_xs.clone(), n),
            ArrayArg::from_raw_parts(h_ln.clone(), n),
            ArrayArg::from_raw_parts(h_sqrt.clone(), n),
        );
    }
    let ln_bytes = client.read_one(h_ln).unwrap();
    let sqrt_bytes = client.read_one(h_sqrt).unwrap();
    (
        bytemuck::cast_slice(&ln_bytes).to_vec(),
        bytemuck::cast_slice(&sqrt_bytes).to_vec(),
    )
}

fn main() {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "cpu".to_string());

    // Same domain the kernel calls it over: (0, 1].
    let mut xs = Vec::new();
    let mut state = 0x1234_5678u32;
    for _ in 0..100_000 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let u = ((state >> 8) + 1) as f32 * 5.960_464_5e-8;
        xs.push(u);
    }

    let (ln_a, sqrt_a) = match backend.as_str() {
        #[cfg(feature = "wgpu")]
        "wgpu" => {
            let device = cubecl::wgpu::WgpuDevice::default();
            let client = <cubecl::wgpu::WgpuRuntime as Runtime>::client(&device);
            run(&client, &xs)
        }
        #[cfg(feature = "hip")]
        "hip" => {
            let device = cubecl::hip::AmdDevice::default();
            let client = <cubecl::hip::HipRuntime as Runtime>::client(&device);
            run(&client, &xs)
        }
        #[cfg(feature = "cpu")]
        "cpu" => {
            let device = cubecl::cpu::CpuDevice;
            let client = <cubecl::cpu::CpuRuntime as Runtime>::client(&device);
            run(&client, &xs)
        }
        other => panic!("unknown/unbuilt backend {other}"),
    };

    // Ground truth: plain f64 ln/sqrt, rounded to f32 at the end.
    let mut worst_ln = 0.0f64;
    let mut worst_sqrt = 0.0f64;
    let mut worst_ln_ulp = 0.0f64;
    let mut worst_sqrt_ulp = 0.0f64;
    for (i, &x) in xs.iter().enumerate() {
        let want_ln = (x as f64).ln();
        let want_sqrt = (-2.0 * want_ln).sqrt();
        let got_ln = ln_a[i] as f64;
        let got_sqrt = sqrt_a[i] as f64;
        let rel_ln = (got_ln - want_ln).abs() / want_ln.abs().max(1e-12);
        let rel_sqrt = (got_sqrt - want_sqrt).abs() / want_sqrt.abs().max(1e-12);
        worst_ln = worst_ln.max(rel_ln);
        worst_sqrt = worst_sqrt.max(rel_sqrt);
        let ulp_ln = (got_ln - want_ln).abs() / (want_ln.abs().max(1.0) * 1.192_092_9e-7);
        let ulp_sqrt = (got_sqrt - want_sqrt).abs() / (want_sqrt.abs().max(1.0) * 1.192_092_9e-7);
        worst_ln_ulp = worst_ln_ulp.max(ulp_ln);
        worst_sqrt_ulp = worst_sqrt_ulp.max(ulp_sqrt);
    }
    println!(
        "{backend}: worst ln rel={worst_ln:e} ({worst_ln_ulp:.2} ULP), worst sqrt rel={worst_sqrt:e} ({worst_sqrt_ulp:.2} ULP)"
    );

    // Also dump raw bits for the first few entries so two backends' outputs
    // can be diffed byte-for-byte by hand.
    for i in 0..8 {
        println!(
            "  x={:.9} ln={:08x} sqrt={:08x}",
            xs[i],
            ln_a[i].to_bits(),
            sqrt_a[i].to_bits()
        );
    }

    if let Some(path) = std::env::args().nth(2) {
        let mut bits: Vec<u32> = ln_a.iter().map(|v| v.to_bits()).collect();
        bits.extend(sqrt_a.iter().map(|v| v.to_bits()));
        std::fs::write(path, bytemuck::cast_slice(&bits)).unwrap();
    }
}
