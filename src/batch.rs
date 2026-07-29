//! Host-side description of a particle-filter batch and its flattened device layout.

use crate::kernel::{META_F_STRIDE, META_U_STRIDE};

/// Filter hyper-parameters. Defaults match `lik_pf` in the ROGII notebook.
#[derive(Debug, Clone, Copy)]
pub struct PfConfig {
    /// Particles per cloud (`CFG.PF_PARTICLES`, 500).
    pub n_particles: u32,
    /// Independent seeds per well (`CFG.PF_SEEDS`, 128).
    pub n_seeds: u32,
    /// Units per cube. Must be a power of two.
    pub cube_dim: u32,
    /// Rate momentum (`MOM`).
    pub mom: f32,
    /// Rate process noise (`VN`).
    pub vn: f32,
    /// Position process noise (`PN`).
    pub pn: f32,
    /// Roughening applied to position on resample (`RP`).
    pub rough_p: f32,
    /// Roughening applied to rate on resample (`RR`).
    pub rough_r: f32,
    /// Resample when `neff < resamp * n_particles` (`RESAMP`).
    pub resamp: f32,
    /// Per-observation likelihood floor.
    pub lik_floor: f32,
    /// Device bytes to spend on the per-seed trajectory buffer before splitting
    /// the batch into chunks.
    pub pred_budget_bytes: usize,
    /// Emit an extra `pf_pt_std` channel: the unweighted spread across seeds,
    /// which the notebook uses as a quality feature.
    pub with_std: bool,
    /// Also bring each seed's own trajectory back from the device, not just the
    /// seed-ensembled channels.
    ///
    /// The notebook's selector-side bimodal branch hedge is built on the raw
    /// per-seed paths — it takes each seed's median TVT over the evaluation
    /// rows and splits those 128 levels into two likelihood-weighted clusters —
    /// so a caller that needs `branch_stats` needs this. It costs `n_seeds`
    /// times the output size in readback, which is why it is off by default;
    /// `pred_budget_bytes` already bounds the per-chunk size.
    pub with_seed_paths: bool,
}

impl Default for PfConfig {
    fn default() -> Self {
        Self {
            n_particles: 500,
            n_seeds: 128,
            cube_dim: 256,
            mom: 0.998,
            vn: 0.002,
            pn: 0.005,
            rough_p: 0.1,
            rough_r: 0.001,
            resamp: 0.5,
            // f32 keeps the weight recursion representable down to ~1e-15;
            // the f64 CPU version uses 1e-300.
            lik_floor: 1e-12,
            pred_budget_bytes: 512 << 20,
            with_std: false,
            with_seed_paths: false,
        }
    }
}

impl PfConfig {
    /// Shared-memory particle capacity the kernel is specialised for. Bucketed so
    /// that varying `n_particles` reuses a small number of compiled kernels.
    pub fn smem_particles(&self) -> usize {
        let n = self.n_particles.max(self.cube_dim) as usize;
        n.next_power_of_two()
    }

    /// Validates the invariants the kernel relies on.
    pub fn validate(&self) -> Result<(), crate::PfError> {
        if !self.cube_dim.is_power_of_two() || self.cube_dim == 0 {
            return Err(crate::PfError::Config(format!(
                "cube_dim must be a non-zero power of two, got {}",
                self.cube_dim
            )));
        }
        if self.n_particles == 0 {
            return Err(crate::PfError::Config("n_particles must be > 0".into()));
        }
        if self.n_seeds == 0 {
            return Err(crate::PfError::Config("n_seeds must be > 0".into()));
        }
        Ok(())
    }
}

/// One well's evaluation zone plus its type-well GR grid.
#[derive(Debug, Clone)]
pub struct WellInput {
    /// Measured depth of every evaluation row, ascending.
    pub md: Vec<f32>,
    /// True vertical depth offset `Z` of every evaluation row.
    pub z: Vec<f32>,
    /// Gamma ray of every evaluation row (already gap-filled).
    pub gr: Vec<f32>,
    /// Type-well GR resampled onto a uniform TVT grid.
    pub grid: Vec<f32>,
    /// TVT of `grid[0]`.
    pub vmin: f32,
    /// TVT spacing of `grid`.
    pub step: f32,
    /// GR mismatch sigma estimated on the known prefix.
    pub gs: f32,
    /// Initial particle position: last known `TVT_input + Z`.
    pub ls: f32,
    /// Initial along-hole TVT rate.
    pub ir: f32,
    /// Initial position spread.
    pub init_spr: f32,
    /// RNG seed base for this well. Per AGENTS.md this should be derived
    /// deterministically from the well id.
    pub seed_base: u32,
}

impl WellInput {
    fn validate(&self, idx: usize) -> Result<(), crate::PfError> {
        let n = self.md.len();
        if self.z.len() != n || self.gr.len() != n {
            return Err(crate::PfError::Config(format!(
                "well {idx}: md/z/gr length mismatch ({}, {}, {})",
                n,
                self.z.len(),
                self.gr.len()
            )));
        }
        if self.grid.len() < 2 {
            return Err(crate::PfError::Config(format!(
                "well {idx}: GR grid needs at least 2 samples, got {}",
                self.grid.len()
            )));
        }
        if !self.step.is_finite() || self.step <= 0.0 {
            return Err(crate::PfError::Config(format!(
                "well {idx}: grid step must be positive, got {}",
                self.step
            )));
        }
        if !self.gs.is_finite() || self.gs <= 0.0 {
            return Err(crate::PfError::Config(format!(
                "well {idx}: gr sigma must be positive, got {}",
                self.gs
            )));
        }
        Ok(())
    }
}

/// Flattened, device-ready view of a set of wells.
#[derive(Debug, Clone, Default)]
pub struct FlatBatch {
    pub md: Vec<f32>,
    pub z: Vec<f32>,
    pub gr: Vec<f32>,
    pub grid: Vec<f32>,
    /// `META_U_STRIDE` entries per well, see [`crate::kernel::pf_lik_kernel`].
    pub meta_u: Vec<u32>,
    /// `META_F_STRIDE` entries per well.
    pub meta_f: Vec<f32>,
    /// Well index of every flattened evaluation row.
    pub row_well: Vec<u32>,
    /// Evaluation row count per well.
    pub ev_lens: Vec<u32>,
    /// Start of each well's rows in `md`/`z`/`gr`.
    pub ev_offsets: Vec<u32>,
    /// Start of each well's `[n_seeds, ev_len]` block in the trajectory buffer.
    pub pred_offsets: Vec<u32>,
    /// Total per-seed trajectory elements: `sum(ev_len) * n_seeds`.
    pub pred_len: usize,
}

impl FlatBatch {
    /// Number of wells in the batch.
    pub fn n_wells(&self) -> usize {
        self.ev_lens.len()
    }

    /// Total evaluation rows across all wells.
    pub fn total_rows(&self) -> usize {
        self.row_well.len()
    }

    /// Flattens `wells`, dropping any with an empty evaluation zone.
    ///
    /// Returns the flattened batch plus the indices of the wells that were kept,
    /// so callers can map results back to their own ordering.
    pub fn build(
        wells: &[WellInput],
        n_seeds: u32,
    ) -> Result<(Self, Vec<usize>), crate::PfError> {
        let mut b = FlatBatch::default();
        let mut kept = Vec::new();

        for (idx, wl) in wells.iter().enumerate() {
            wl.validate(idx)?;
            if wl.md.is_empty() {
                continue;
            }

            let ev_off = b.md.len() as u32;
            let ev_len = wl.md.len() as u32;
            let g_off = b.grid.len() as u32;
            let g_len = wl.grid.len() as u32;
            let p_off = b.pred_len as u32;
            let well_idx = b.ev_lens.len() as u32;

            b.md.extend_from_slice(&wl.md);
            b.z.extend_from_slice(&wl.z);
            b.gr.extend_from_slice(&wl.gr);
            b.grid.extend_from_slice(&wl.grid);
            b.row_well.extend(std::iter::repeat_n(well_idx, ev_len as usize));

            b.meta_u
                .extend_from_slice(&[ev_off, ev_len, g_off, g_len, p_off, wl.seed_base]);
            // `1/step` and `1/gs` are uploaded rather than divided for on the
            // device: SPIR-V division is not correctly rounded, so a device
            // `x / gs` lands on a different f32 than CUDA's or the host's for a
            // quarter of inputs, and the likelihood amplifies that into resample
            // decisions. A multiply by a host-computed reciprocal is IEEE-exact
            // everywhere. See `kernel::pf_lik_kernel`.
            b.meta_f.extend_from_slice(&[
                wl.vmin,
                wl.step,
                wl.gs,
                wl.ls,
                wl.ir,
                wl.init_spr,
                1.0 / wl.step,
                1.0 / wl.gs,
            ]);

            b.ev_offsets.push(ev_off);
            b.ev_lens.push(ev_len);
            b.pred_offsets.push(p_off);
            b.pred_len += ev_len as usize * n_seeds as usize;
            kept.push(idx);
        }

        debug_assert_eq!(b.meta_u.len(), b.n_wells() * META_U_STRIDE);
        debug_assert_eq!(b.meta_f.len(), b.n_wells() * META_F_STRIDE);
        Ok((b, kept))
    }

    /// Splits the batch into contiguous well ranges whose trajectory buffers each
    /// stay under `budget_bytes`. A single well always gets its own chunk even if
    /// it exceeds the budget on its own.
    pub fn chunks(&self, n_seeds: u32, budget_bytes: usize, elem_size: usize) -> Vec<(usize, usize)> {
        let per_row = n_seeds as usize * elem_size;
        let max_rows = (budget_bytes / per_row).max(1);

        let mut out = Vec::new();
        let mut start = 0usize;
        let mut rows = 0usize;
        for w in 0..self.n_wells() {
            let len = self.ev_lens[w] as usize;
            if rows > 0 && rows + len > max_rows {
                out.push((start, w));
                start = w;
                rows = 0;
            }
            rows += len;
        }
        if start < self.n_wells() {
            out.push((start, self.n_wells()));
        }
        out
    }
}

/// Softmax weights over seeds for one output channel of one well.
///
/// Mirrors `wts = np.exp((liks - liks.max()) / sc); wts /= wts.sum()`.
pub fn seed_weights(liks: &[f32], scale: f32, out: &mut [f32]) {
    let max = liks.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for (o, l) in out.iter_mut().zip(liks) {
        let v = ((l - max) / scale).exp();
        *o = v;
        sum += v as f64;
    }
    if sum > 0.0 {
        let inv = (1.0 / sum) as f32;
        for o in out.iter_mut() {
            *o *= inv;
        }
    } else {
        let u = 1.0 / liks.len() as f32;
        out.fill(u);
    }
}
