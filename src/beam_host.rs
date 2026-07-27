//! Host-side driver for the beam search: uploads a batch, runs the smoothing,
//! search and mean kernels, and brings back the ensemble trajectory.

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubek_std::cube_count::cube_count_spread_with_total;

use crate::beam::{
    BeamConfig, BeamOptions, BeamOutput, BeamWellInput, FlatBeamBatch, max_beam_capacity,
    smoothing_planes,
};
use crate::beam_kernel::{beam_mean_kernel, beam_search_kernel, sg_smooth_kernel};
use crate::PfError;

#[cfg(feature = "checked-launch")]
macro_rules! launch_sg {
    ($($a:tt)*) => { sg_smooth_kernel::launch::<f32, R>($($a)*) };
}
#[cfg(not(feature = "checked-launch"))]
macro_rules! launch_sg {
    ($($a:tt)*) => { sg_smooth_kernel::launch_unchecked::<f32, R>($($a)*) };
}
#[cfg(feature = "checked-launch")]
macro_rules! launch_beam {
    ($($a:tt)*) => { beam_search_kernel::launch::<f32, R>($($a)*) };
}
#[cfg(not(feature = "checked-launch"))]
macro_rules! launch_beam {
    ($($a:tt)*) => { beam_search_kernel::launch_unchecked::<f32, R>($($a)*) };
}
#[cfg(feature = "checked-launch")]
macro_rules! launch_beam_mean {
    ($($a:tt)*) => { beam_mean_kernel::launch::<f32, R>($($a)*) };
}
#[cfg(not(feature = "checked-launch"))]
macro_rules! launch_beam_mean {
    ($($a:tt)*) => { beam_mean_kernel::launch_unchecked::<f32, R>($($a)*) };
}

const F32: usize = std::mem::size_of::<f32>();

impl FlatBeamBatch {
    /// Rebases wells `[w0, w1)` into a standalone batch.
    ///
    /// Returns the sub-batch and the index of its first row in the parent.
    fn sub_batch(&self, w0: usize, w1: usize) -> (FlatBeamBatch, usize) {
        let row0 = self.ev_offsets[w0] as usize;
        let row1 = if w1 < self.n_wells() {
            self.ev_offsets[w1] as usize
        } else {
            self.total_rows()
        };
        let t0 = self.meta_u[w0 * 4 + 2] as usize;
        let t1 = if w1 < self.n_wells() {
            self.meta_u[w1 * 4 + 2] as usize
        } else {
            self.tw_tvt.len()
        };

        let mut out = FlatBeamBatch {
            gr: self.gr[row0..row1].to_vec(),
            tw_tvt: self.tw_tvt[t0..t1].to_vec(),
            tw_gr: self.tw_gr[t0..t1].to_vec(),
            meta_f: self.meta_f[w0..w1].to_vec(),
            ..Default::default()
        };

        for w in w0..w1 {
            let ev_len = self.ev_lens[w];
            let ev_off = self.ev_offsets[w] as usize - row0;
            let tw_off = self.meta_u[w * 4 + 2] as usize - t0;
            let tw_len = self.meta_u[w * 4 + 3];
            out.meta_u
                .extend_from_slice(&[ev_off as u32, ev_len, tw_off as u32, tw_len]);
            out.ev_offsets.push(ev_off as u32);
            out.ev_lens.push(ev_len);
            out.row_well
                .extend(std::iter::repeat_n((w - w0) as u32, ev_len as usize));
        }
        (out, row0)
    }
}

/// Runs the beam-search ensemble over `wells` and returns, per evaluation row,
/// the mean over `configs` of the tracked TVT.
///
/// This is `run_beam_ensemble` from the notebook, batched: every
/// `(well, config)` pair becomes one cube of a single launch instead of a
/// sequential Python loop over 14 configs per well.
pub fn run_beam<R: Runtime>(
    client: &ComputeClient<R>,
    wells: &[BeamWellInput],
    configs: &[BeamConfig],
    opts: &BeamOptions,
) -> Result<BeamOutput, PfError> {
    opts.validate(configs)?;
    let (batch, kept) = FlatBeamBatch::build(wells)?;

    let n_configs = configs.len();
    let total_rows = batch.total_rows();

    let mut out = BeamOutput {
        mean: vec![0.0; total_rows],
        per_config: if opts.with_per_config {
            Some(vec![0.0; n_configs * total_rows])
        } else {
            None
        },
        ev_offsets: batch.ev_offsets.clone(),
        ev_lens: batch.ev_lens.clone(),
        kept,
    };
    if batch.n_wells() == 0 {
        return Ok(out);
    }

    let max_beam = max_beam_capacity(configs);
    let (radii, plane_of) = smoothing_planes(configs);
    let n_planes = radii.len();

    let cfg_u: Vec<u32> = configs
        .iter()
        .zip(&plane_of)
        .flat_map(|(c, p)| [c.beam_size, *p])
        .collect();
    let cfg_f: Vec<f32> = configs
        .iter()
        .flat_map(|c| [c.move_cost, c.err_scale])
        .collect();

    let h_cfg_u = client.create_from_slice(bytemuck::cast_slice(&cfg_u));
    let h_cfg_f = client.create_from_slice(bytemuck::cast_slice(&cfg_f));
    let h_radii = client.create_from_slice(bytemuck::cast_slice(&radii));

    // The smoothing planes and the per-config trajectories are the two buffers
    // that scale with the batch, so the chunk budget covers both.
    for (w0, w1) in batch.chunks(n_planes + n_configs, opts.budget_bytes, F32) {
        let (sub, row0) = batch.sub_batch(w0, w1);
        let sub_wells = sub.n_wells();
        let sub_rows = sub.total_rows();

        // ---- upload -------------------------------------------------------
        let h_gr = client.create_from_slice(bytemuck::cast_slice(&sub.gr));
        let h_tw_tvt = client.create_from_slice(bytemuck::cast_slice(&sub.tw_tvt));
        let h_tw_gr = client.create_from_slice(bytemuck::cast_slice(&sub.tw_gr));
        let h_meta_u = client.create_from_slice(bytemuck::cast_slice(&sub.meta_u));
        let h_meta_f = client.create_from_slice(bytemuck::cast_slice(&sub.meta_f));
        let h_row_well = client.create_from_slice(bytemuck::cast_slice(&sub.row_well));
        let h_sgr = client.empty(n_planes * sub_rows * F32);
        let h_traj = client.empty(n_configs * sub_rows * F32);
        let h_mean = client.empty(sub_rows * F32);

        // ---- smooth -------------------------------------------------------
        let sg_threads = n_planes * sub_rows;
        let flat_dim = CubeDim { x: 256, y: 1, z: 1 };
        let (sg_count, _) = cube_count_spread_with_total::<R>(client, sg_threads.div_ceil(256));

        // SAFETY: the kernel guards on `idx < total_rows * n_planes`, and every
        // row index it derives comes from `row_well`/`meta_u`, both built here
        // from the same lengths used to size the buffers above.
        #[allow(unused_unsafe)]
        unsafe {
            launch_sg!(
                client,
                sg_count,
                flat_dim,
                ArrayArg::from_raw_parts(h_gr.clone(), sub.gr.len()),
                ArrayArg::from_raw_parts(h_row_well.clone(), sub.row_well.len()),
                ArrayArg::from_raw_parts(h_meta_u.clone(), sub.meta_u.len()),
                ArrayArg::from_raw_parts(h_radii.clone(), radii.len()),
                ArrayArg::from_raw_parts(h_sgr.clone(), n_planes * sub_rows),
                sub_rows as u32,
                n_planes as u32,
            );
        }

        // ---- search -------------------------------------------------------
        let n_tasks = (sub_wells * n_configs) as u32;
        let (beam_count, total_cubes) = cube_count_spread_with_total::<R>(client, n_tasks as usize);
        let beam_dim = CubeDim {
            x: opts.cube_dim,
            y: 1,
            z: 1,
        };

        // SAFETY: `max_beam_capacity` rounds the widest beam in `configs` up to a
        // power of two, so no `bs` can overrun the shared buffers; type-well
        // moves are range-checked in the kernel; every offset comes from the
        // `meta_u` built above.
        #[allow(unused_unsafe)]
        unsafe {
            launch_beam!(
                client,
                beam_count,
                beam_dim,
                ArrayArg::from_raw_parts(h_sgr.clone(), n_planes * sub_rows),
                ArrayArg::from_raw_parts(h_tw_tvt.clone(), sub.tw_tvt.len()),
                ArrayArg::from_raw_parts(h_tw_gr.clone(), sub.tw_gr.len()),
                ArrayArg::from_raw_parts(h_meta_u.clone(), sub.meta_u.len()),
                ArrayArg::from_raw_parts(h_meta_f.clone(), sub.meta_f.len()),
                ArrayArg::from_raw_parts(h_cfg_u.clone(), cfg_u.len()),
                ArrayArg::from_raw_parts(h_cfg_f.clone(), cfg_f.len()),
                ArrayArg::from_raw_parts(h_traj.clone(), n_configs * sub_rows),
                sub_rows as u32,
                n_configs as u32,
                n_tasks,
                total_cubes as u32,
                max_beam,
            );
        }

        // ---- ensemble mean -------------------------------------------------
        let (mean_count, _) = cube_count_spread_with_total::<R>(client, sub_rows.div_ceil(256));

        // SAFETY: the kernel guards on `r < total_rows`.
        #[allow(unused_unsafe)]
        unsafe {
            launch_beam_mean!(
                client,
                mean_count,
                flat_dim,
                ArrayArg::from_raw_parts(h_traj.clone(), n_configs * sub_rows),
                ArrayArg::from_raw_parts(h_mean.clone(), sub_rows),
                sub_rows as u32,
                n_configs as u32,
            );
        }

        let mean_bytes = client
            .read_one(h_mean)
            .map_err(|e| PfError::Device(format!("{e:?}")))?;
        out.mean[row0..row0 + sub_rows].copy_from_slice(bytemuck::cast_slice(&mean_bytes));

        if let Some(pc) = out.per_config.as_mut() {
            let traj_bytes = client
                .read_one(h_traj.clone())
                .map_err(|e| PfError::Device(format!("{e:?}")))?;
            let vals: &[f32] = bytemuck::cast_slice(&traj_bytes);
            for c in 0..n_configs {
                let src = &vals[c * sub_rows..(c + 1) * sub_rows];
                let dst = c * total_rows + row0;
                pc[dst..dst + sub_rows].copy_from_slice(src);
            }
        }

        let _ = (h_gr, h_tw_tvt, h_tw_gr, h_meta_f, h_traj);
    }

    Ok(out)
}
