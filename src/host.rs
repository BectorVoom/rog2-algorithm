//! Host-side driver: uploads a batch, launches the filter and blend kernels, and
//! brings back only the seed-reduced trajectories.

use cubecl::prelude::*;
use cubecl::client::ComputeClient;
use cubek_std::cube_count::cube_count_spread_with_total;

use crate::batch::{FlatBatch, PfConfig, WellInput, seed_weights};
use crate::blend::pf_blend_kernel;
use crate::kernel::{META_F_STRIDE, pf_lik_kernel};
use crate::{PfError, PfOutput};

/// Bounds-checked launches are used when the `checked-launch` feature is on;
/// they cost a little throughput but turn indexing mistakes into errors instead
/// of undefined behaviour.
#[cfg(feature = "checked-launch")]
macro_rules! launch_pf {
    ($($a:tt)*) => { pf_lik_kernel::launch::<f32, R>($($a)*) };
}
#[cfg(not(feature = "checked-launch"))]
macro_rules! launch_pf {
    ($($a:tt)*) => { pf_lik_kernel::launch_unchecked::<f32, R>($($a)*) };
}
#[cfg(feature = "checked-launch")]
macro_rules! launch_blend {
    ($($a:tt)*) => { pf_blend_kernel::launch::<f32, R>($($a)*) };
}
#[cfg(not(feature = "checked-launch"))]
macro_rules! launch_blend {
    ($($a:tt)*) => { pf_blend_kernel::launch_unchecked::<f32, R>($($a)*) };
}

const F32: usize = std::mem::size_of::<f32>();
const U32: usize = std::mem::size_of::<u32>();

impl FlatBatch {
    /// Rebases wells `[w0, w1)` into a standalone batch.
    ///
    /// Returns the sub-batch and the index of its first row in the parent.
    fn sub_batch(&self, w0: usize, w1: usize, n_seeds: u32) -> (FlatBatch, usize) {
        let row0 = self.ev_offsets[w0] as usize;
        let row1 = if w1 < self.n_wells() {
            self.ev_offsets[w1] as usize
        } else {
            self.total_rows()
        };
        let g0 = self.meta_u[w0 * 6 + 2] as usize;
        let g1 = if w1 < self.n_wells() {
            self.meta_u[w1 * 6 + 2] as usize
        } else {
            self.grid.len()
        };

        let mut out = FlatBatch {
            md: self.md[row0..row1].to_vec(),
            z: self.z[row0..row1].to_vec(),
            gr: self.gr[row0..row1].to_vec(),
            grid: self.grid[g0..g1].to_vec(),
            meta_f: self.meta_f[w0 * META_F_STRIDE..w1 * META_F_STRIDE].to_vec(),
            ..Default::default()
        };

        let mut pred_len = 0usize;
        for w in w0..w1 {
            let ev_len = self.ev_lens[w];
            let ev_off = self.ev_offsets[w] as usize - row0;
            let g_off = self.meta_u[w * 6 + 2] as usize - g0;
            let g_len = self.meta_u[w * 6 + 3];
            out.meta_u.extend_from_slice(&[
                ev_off as u32,
                ev_len,
                g_off as u32,
                g_len,
                pred_len as u32,
                self.meta_u[w * 6 + 5],
            ]);
            out.ev_offsets.push(ev_off as u32);
            out.ev_lens.push(ev_len);
            out.pred_offsets.push(pred_len as u32);
            out.row_well
                .extend(std::iter::repeat_n((w - w0) as u32, ev_len as usize));
            pred_len += ev_len as usize * n_seeds as usize;
        }
        out.pred_len = pred_len;
        (out, row0)
    }
}

/// Runs the multi-seed particle filter over `wells` and returns, per output
/// channel, one trajectory value per evaluation row.
///
/// Output channels are `pf_scale_{s}` for each `s` in `scales`, followed by
/// `pf_mean`. Values are laid out `[channel][row]` with rows in the same order as
/// the flattened, non-empty wells.
pub fn run_pf<R: Runtime>(
    client: &ComputeClient<R>,
    wells: &[WellInput],
    cfg: &PfConfig,
    scales: &[f32],
) -> Result<PfOutput, PfError> {
    cfg.validate()?;
    let n_seeds = cfg.n_seeds;
    let (batch, kept) = FlatBatch::build(wells, n_seeds)?;

    let n_out = scales.len() + 1 + usize::from(cfg.with_std);
    let std_channel = if cfg.with_std {
        (n_out - 1) as u32
    } else {
        u32::MAX
    };
    let total_rows = batch.total_rows();
    let n_wells = batch.n_wells();

    let mut channels: Vec<String> = scales.iter().map(|s| format!("pf_scale_{}", fmt_g(*s))).collect();
    channels.push("pf_mean".to_string());
    if cfg.with_std {
        channels.push("pf_pt_std".to_string());
    }

    let mut out = PfOutput {
        channels,
        values: vec![0.0; n_out * total_rows],
        liks: vec![0.0; n_wells * n_seeds as usize],
        n_seeds,
        seed_paths: if cfg.with_seed_paths {
            Some(vec![0.0; total_rows * n_seeds as usize])
        } else {
            None
        },
        pred_offsets: batch.pred_offsets.clone(),
        ev_offsets: batch.ev_offsets.clone(),
        ev_lens: batch.ev_lens.clone(),
        kept,
    };
    if n_wells == 0 {
        return Ok(out);
    }

    let smem = cfg.smem_particles();
    if cfg.n_particles as usize > smem {
        return Err(PfError::Config(format!(
            "n_particles {} exceeds shared-memory capacity {}",
            cfg.n_particles, smem
        )));
    }

    for (w0, w1) in batch.chunks(n_seeds, cfg.pred_budget_bytes, F32) {
        let (sub, row0) = batch.sub_batch(w0, w1, n_seeds);
        let sub_wells = sub.n_wells();
        let sub_rows = sub.total_rows();
        let n_tasks = (sub_wells * n_seeds as usize) as u32;

        // ---- upload -------------------------------------------------------
        let h_md = client.create_from_slice(bytemuck::cast_slice(&sub.md));
        let h_z = client.create_from_slice(bytemuck::cast_slice(&sub.z));
        let h_gr = client.create_from_slice(bytemuck::cast_slice(&sub.gr));
        let h_grid = client.create_from_slice(bytemuck::cast_slice(&sub.grid));
        let h_meta_u = client.create_from_slice(bytemuck::cast_slice(&sub.meta_u));
        let h_meta_f = client.create_from_slice(bytemuck::cast_slice(&sub.meta_f));
        let h_preds = client.empty(sub.pred_len * F32);
        let h_liks = client.empty(sub_wells * n_seeds as usize * F32);

        let (cube_count, total_cubes) = cube_count_spread_with_total::<R>(client, n_tasks as usize);
        let cube_dim = CubeDim {
            x: cfg.cube_dim,
            y: 1,
            z: 1,
        };

        // SAFETY: every index the kernel forms is derived from `meta_u`, which is
        // built here from the same lengths used to size the buffers above.
        #[allow(unused_unsafe)]
        unsafe {
            launch_pf!(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(h_md.clone(), sub.md.len()),
                ArrayArg::from_raw_parts(h_z.clone(), sub.z.len()),
                ArrayArg::from_raw_parts(h_gr.clone(), sub.gr.len()),
                ArrayArg::from_raw_parts(h_grid.clone(), sub.grid.len()),
                ArrayArg::from_raw_parts(h_meta_u.clone(), sub.meta_u.len()),
                ArrayArg::from_raw_parts(h_meta_f.clone(), sub.meta_f.len()),
                ArrayArg::from_raw_parts(h_preds.clone(), sub.pred_len),
                ArrayArg::from_raw_parts(h_liks.clone(), sub_wells * n_seeds as usize),
                cfg.mom,
                cfg.vn,
                cfg.pn,
                cfg.rough_p,
                cfg.rough_r,
                cfg.resamp,
                cfg.lik_floor,
                cfg.n_particles,
                n_seeds,
                n_tasks,
                total_cubes as u32,
                smem,
                cfg.cube_dim as usize,
            );
        }

        // ---- per-seed trajectories (opt-in; `n_seeds` times the output size) --
        if let Some(paths) = out.seed_paths.as_mut() {
            let bytes = client
                .read_one(h_preds.clone())
                .map_err(|e| PfError::Device(format!("{e:?}")))?;
            let vals: &[f32] = bytemuck::cast_slice(&bytes);
            // `sub`'s pred offsets are chunk-relative; rebase onto the parent.
            let dst0 = batch.pred_offsets[w0] as usize;
            paths[dst0..dst0 + sub.pred_len].copy_from_slice(&vals[..sub.pred_len]);
        }

        // ---- seed weights (needs the log-likelihoods back) -----------------
        let liks_bytes = client.read_one(h_liks).map_err(|e| PfError::Device(format!("{e:?}")))?;
        let liks: &[f32] = bytemuck::cast_slice(&liks_bytes);
        let seeds = n_seeds as usize;
        out.liks[w0 * seeds..(w0 + sub_wells) * seeds].copy_from_slice(liks);

        let mut wts = vec![0.0f32; sub_wells * n_out * seeds];
        for w in 0..sub_wells {
            let l = &liks[w * seeds..(w + 1) * seeds];
            for (k, sc) in scales.iter().enumerate() {
                let base = (w * n_out + k) * seeds;
                seed_weights(l, *sc, &mut wts[base..base + seeds]);
            }
            let base = (w * n_out + scales.len()) * seeds;
            wts[base..base + seeds].fill(1.0 / seeds as f32);
            // The std channel ignores `wts`, but keep the block initialised.
            if cfg.with_std {
                let base = (w * n_out + scales.len() + 1) * seeds;
                wts[base..base + seeds].fill(0.0);
            }
        }

        // ---- blend --------------------------------------------------------
        let h_wts = client.create_from_slice(bytemuck::cast_slice(&wts));
        let h_row_well = client.create_from_slice(bytemuck::cast_slice(&sub.row_well));
        let h_out = client.empty(n_out * sub_rows * F32);

        let n_threads = n_out * sub_rows;
        let blend_dim = CubeDim { x: 256, y: 1, z: 1 };
        let (blend_count, _) = cube_count_spread_with_total::<R>(client, n_threads.div_ceil(256));

        // SAFETY: the kernel guards on `idx < total_rows * n_out` and every other
        // index is bounded by the metadata used to allocate the buffers.
        #[allow(unused_unsafe)]
        unsafe {
            launch_blend!(
                client,
                blend_count,
                blend_dim,
                ArrayArg::from_raw_parts(h_preds.clone(), sub.pred_len),
                ArrayArg::from_raw_parts(h_wts.clone(), wts.len()),
                ArrayArg::from_raw_parts(h_meta_u.clone(), sub.meta_u.len()),
                ArrayArg::from_raw_parts(h_row_well.clone(), sub.row_well.len()),
                ArrayArg::from_raw_parts(h_out.clone(), n_out * sub_rows),
                n_seeds,
                n_out as u32,
                sub_rows as u32,
                std_channel,
            );
        }

        let out_bytes = client.read_one(h_out).map_err(|e| PfError::Device(format!("{e:?}")))?;
        let vals: &[f32] = bytemuck::cast_slice(&out_bytes);
        for k in 0..n_out {
            let src = &vals[k * sub_rows..(k + 1) * sub_rows];
            let dst = k * total_rows + row0;
            out.values[dst..dst + sub_rows].copy_from_slice(src);
        }

        let _ = (h_md, h_z, h_gr, h_grid, h_meta_f, U32);
    }

    Ok(out)
}

/// Formats a scale the way numpy's `%g` does, so channel names match the
/// notebook's `f"pf_scale_{sc:g}"`.
fn fmt_g(v: f32) -> String {
    if v == v.trunc() && v.abs() < 1e7 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}
