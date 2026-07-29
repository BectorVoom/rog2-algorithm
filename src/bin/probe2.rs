//! Diagnostic: is `recip_precise` bit-identical across backends over the wide
//! magnitude range `ws`/`ws2` actually take in the filter (unlike the
//! domain-restricted Box-Muller inputs)?
//!
//!     cargo run --release --features wgpu --bin probe2 -- wgpu
//!     cargo run --release --features hip  --bin probe2 -- hip

use cubecl::prelude::*;
use rog2_pf::kernel::recip_precise;

#[cube(launch, launch_unchecked)]
fn probe2_kernel<F: Float + CubeElement>(xs: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS as usize;
    if i < xs.len() as usize {
        out[i] = recip_precise::<F>(xs[i]);
    }
}

fn run<R: Runtime>(client: &ComputeClient<R>, xs: &[f32]) -> Vec<f32> {
    let n = xs.len();
    let h_xs = client.create_from_slice(bytemuck::cast_slice(xs));
    let h_out = client.empty(n * 4);
    let cube_dim = CubeDim { x: 64, y: 1, z: 1 };
    let cube_count = CubeCount::Static(n.div_ceil(64) as u32, 1, 1);
    unsafe {
        probe2_kernel::launch_unchecked::<f32, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(h_xs.clone(), n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
        );
    }
    let bytes = client.read_one(h_out).unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

fn main() {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "cpu".to_string());

    // Sweep magnitudes from near-subnormal to large, both scales that ws/ws2
    // can plausibly take across many resampling steps.
    let mut xs = Vec::new();
    let mut state = 0x1234_5678u32;
    for _ in 0..200_000 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let mantissa = ((state >> 8) + 1) as f32 * 5.960_464_5e-8; // (0,1]
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let exp = (state % 80) as i32 - 40; // 2^-40 .. 2^39
        xs.push(mantissa * 2f32.powi(exp));
    }

    let out = match backend.as_str() {
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

    let mut worst_rel = 0.0f64;
    let mut worst_ulp = 0.0f64;
    let mut n_nonfinite = 0usize;
    for (i, &x) in xs.iter().enumerate() {
        let want = 1.0 / (x as f64);
        let got = out[i] as f64;
        if !got.is_finite() || !want.is_finite() {
            n_nonfinite += 1;
            continue;
        }
        let rel = (got - want).abs() / want.abs().max(1e-300);
        worst_rel = worst_rel.max(rel);
        let ulp = (got - want).abs() / (want.abs().max(1e-300) * 1.192_092_9e-7);
        worst_ulp = worst_ulp.max(ulp);
    }
    println!(
        "{backend}: worst rel={worst_rel:e} ({worst_ulp:.2} ULP), non-finite={n_nonfinite}, bits_of_first_8={:?}",
        &out[..8].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    if let Some(path) = std::env::args().nth(2) {
        let bits: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
        std::fs::write(path, bytemuck::cast_slice(&bits)).unwrap();
    }
}
