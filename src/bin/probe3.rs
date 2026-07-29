//! Diagnostic: is the raw `fma` builtin itself bit-identical across backends
//! for arbitrary inputs (the foundational assumption behind exp_precise /
//! sin_precise / ln_precise / sqrt_precise / recip_precise)?
//!
//!     cargo run --release --features wgpu --bin probe3 -- wgpu

use cubecl::prelude::*;

#[cube(launch, launch_unchecked)]
fn fma_probe_kernel<F: Float + CubeElement>(a: &Array<F>, b: &Array<F>, c: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS as usize;
    if i < a.len() as usize {
        out[i] = fma(a[i], b[i], c[i]);
    }
}

fn run<R: Runtime>(client: &ComputeClient<R>, a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
    let n = a.len();
    let h_a = client.create_from_slice(bytemuck::cast_slice(a));
    let h_b = client.create_from_slice(bytemuck::cast_slice(b));
    let h_c = client.create_from_slice(bytemuck::cast_slice(c));
    let h_out = client.empty(n * 4);
    let cube_dim = CubeDim { x: 64, y: 1, z: 1 };
    let cube_count = CubeCount::Static(n.div_ceil(64) as u32, 1, 1);
    unsafe {
        fma_probe_kernel::launch_unchecked::<f32, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(h_a.clone(), n),
            ArrayArg::from_raw_parts(h_b.clone(), n),
            ArrayArg::from_raw_parts(h_c.clone(), n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
        );
    }
    let bytes = client.read_one(h_out).unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

fn main() {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "cpu".to_string());

    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    let mut state = 0xDEAD_BEEFu32;
    for _ in 0..100_000 {
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let mantissa = ((state >> 8) + 1) as f32 * 5.960_464_5e-8;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let exp = (state % 40) as i32 - 20;
            let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
            sign * mantissa * 2f32.powi(exp)
        };
        a.push(next());
        b.push(next());
        c.push(next());
    }

    let out = match backend.as_str() {
        #[cfg(feature = "wgpu")]
        "wgpu" => {
            let device = cubecl::wgpu::WgpuDevice::default();
            let client = <cubecl::wgpu::WgpuRuntime as Runtime>::client(&device);
            run(&client, &a, &b, &c)
        }
        #[cfg(feature = "hip")]
        "hip" => {
            let device = cubecl::hip::AmdDevice::default();
            let client = <cubecl::hip::HipRuntime as Runtime>::client(&device);
            run(&client, &a, &b, &c)
        }
        #[cfg(feature = "cpu")]
        "cpu" => {
            let device = cubecl::cpu::CpuDevice;
            let client = <cubecl::cpu::CpuRuntime as Runtime>::client(&device);
            run(&client, &a, &b, &c)
        }
        other => panic!("unknown/unbuilt backend {other}"),
    };

    // Ground truth: f64 fma-equivalent (exact product, then add, then round).
    let mut diffs = 0usize;
    let mut max_bit_diff = 0i64;
    for i in 0..a.len() {
        let want = (a[i] as f64).mul_add(b[i] as f64, c[i] as f64) as f32;
        if want.to_bits() != out[i].to_bits() {
            diffs += 1;
            max_bit_diff = max_bit_diff.max((want.to_bits() as i64 - out[i].to_bits() as i64).abs());
        }
    }
    println!(
        "{backend}: {diffs}/{} fma results differ from f64-rounded truth (max bit diff {max_bit_diff})",
        a.len()
    );

    if let Some(path) = std::env::args().nth(2) {
        let bits: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
        std::fs::write(path, bytemuck::cast_slice(&bits)).unwrap();
    }
}
