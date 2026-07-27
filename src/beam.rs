//! Host-side description of a beam-search batch and its flattened device layout.
//!
//! Ports `beam_search` / `run_beam_ensemble` from
//! `notebook/rogii-another-approch-2nd.ipynb` (cell 12) — the Viterbi-style
//! stratigraphic tracker whose 14-config ensemble mean feeds the `*_beam_*`
//! selector variants.

/// The five moves the search may take on the type-well index grid: `-2..=2`.
pub const N_MOVES: usize = 5;

/// Largest accepted Savitzky-Golay half-window. The notebook's widest is 5; the
/// cap exists so a mistyped radius cannot turn into a very long kernel.
pub const MAX_RADIUS: u32 = 64;

/// Number of `u32` metadata slots per well.
pub const BEAM_META_U_STRIDE: usize = 4;
/// Number of float metadata slots per well.
pub const BEAM_META_F_STRIDE: usize = 1;

/// One `(beam size, move cost, error scale, smoothing radius)` tuple — an entry
/// of the notebook's `BEAM_CONFIGS`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamConfig {
    /// Beam width `bs`: how many hypotheses survive each step.
    pub beam_size: u32,
    /// Per-move penalty `mc`; a move of `k` grid cells costs `mc * |k|`.
    pub move_cost: f32,
    /// GR mismatch scale `es`: the observation term is `(gr - tw_gr)^2 / es`.
    pub err_scale: f32,
    /// Savitzky-Golay half-window `r` applied to the horizontal GR. `0` disables
    /// smoothing.
    pub radius: u32,
}

impl BeamConfig {
    /// Convenience constructor matching the notebook's tuple order.
    pub const fn new(beam_size: u32, move_cost: f32, err_scale: f32, radius: u32) -> Self {
        Self {
            beam_size,
            move_cost,
            err_scale,
            radius,
        }
    }
}

/// The notebook's `BEAM_CONFIGS`, in order. `run_beam_ensemble` averages the 14
/// trajectories these produce.
pub const NOTEBOOK_BEAM_CONFIGS: [BeamConfig; 14] = [
    BeamConfig::new(10, 20.0, 144.0, 2),
    BeamConfig::new(10, 8.0, 64.0, 2),
    BeamConfig::new(8, 35.0, 220.0, 1),
    BeamConfig::new(10, 14.0, 90.0, 5),
    BeamConfig::new(20, 4.0, 36.0, 3),
    BeamConfig::new(12, 12.0, 100.0, 3),
    BeamConfig::new(15, 25.0, 180.0, 2),
    BeamConfig::new(20, 30.0, 200.0, 2),
    BeamConfig::new(15, 10.0, 80.0, 4),
    BeamConfig::new(25, 6.0, 50.0, 3),
    BeamConfig::new(10, 40.0, 300.0, 1),
    BeamConfig::new(12, 18.0, 120.0, 5),
    BeamConfig::new(30, 8.0, 70.0, 2),
    BeamConfig::new(10, 50.0, 400.0, 0),
];

/// Launch-shape and output options.
#[derive(Debug, Clone, Copy)]
pub struct BeamOptions {
    /// Units per cube. Must be a power of two. The per-step work is
    /// `5 * beam_size` candidates, so there is nothing to gain from a cube much
    /// wider than the widest beam in the config set.
    pub cube_dim: u32,
    /// Also return the individual per-config trajectories, not just their mean.
    pub with_per_config: bool,
    /// Device bytes to spend on the `[n_configs, rows]` trajectory buffer before
    /// splitting the batch into chunks.
    pub budget_bytes: usize,
}

impl Default for BeamOptions {
    fn default() -> Self {
        Self {
            cube_dim: 64,
            with_per_config: false,
            budget_bytes: 512 << 20,
        }
    }
}

impl BeamOptions {
    /// Validates the invariants the kernel relies on.
    pub fn validate(&self, configs: &[BeamConfig]) -> Result<(), crate::PfError> {
        if !self.cube_dim.is_power_of_two() || self.cube_dim == 0 {
            return Err(crate::PfError::Config(format!(
                "cube_dim must be a non-zero power of two, got {}",
                self.cube_dim
            )));
        }
        if configs.is_empty() {
            return Err(crate::PfError::Config(
                "at least one beam config is required".into(),
            ));
        }
        for (i, c) in configs.iter().enumerate() {
            if c.beam_size == 0 {
                return Err(crate::PfError::Config(format!(
                    "beam config {i}: beam_size must be > 0"
                )));
            }
            if !c.err_scale.is_finite() || c.err_scale <= 0.0 {
                return Err(crate::PfError::Config(format!(
                    "beam config {i}: err_scale must be positive, got {}",
                    c.err_scale
                )));
            }
            if !c.move_cost.is_finite() || c.move_cost < 0.0 {
                return Err(crate::PfError::Config(format!(
                    "beam config {i}: move_cost must be non-negative, got {}",
                    c.move_cost
                )));
            }
            // The smoother's window loop runs `2 * radius + 1` times per row, so
            // an accidental huge radius would silently turn into a very long
            // kernel. The notebook's widest is 5.
            if c.radius > MAX_RADIUS {
                return Err(crate::PfError::Config(format!(
                    "beam config {i}: radius must be <= {MAX_RADIUS}, got {}",
                    c.radius
                )));
            }
        }
        Ok(())
    }
}

/// Shared-memory beam capacity the kernel is specialised for. Bucketed to a power
/// of two so that varying beam sizes reuse a small number of compiled kernels.
pub fn max_beam_capacity(configs: &[BeamConfig]) -> usize {
    configs
        .iter()
        .map(|c| c.beam_size as usize)
        .max()
        .unwrap_or(1)
        .next_power_of_two()
}

/// One well's evaluation-zone GR plus the type-well `(TVT, GR)` log the search
/// walks along.
#[derive(Debug, Clone)]
pub struct BeamWellInput {
    /// Gamma ray of every evaluation row (already gap-filled), in MD order.
    pub gr: Vec<f32>,
    /// Type-well TVT, **ascending** — the notebook's `tw.sort_values('TVT')`.
    pub tw_tvt: Vec<f32>,
    /// Type-well GR, aligned with `tw_tvt`.
    pub tw_gr: Vec<f32>,
    /// `TVT_input` of the last known row: where the search starts.
    pub last_tvt: f32,
}

impl BeamWellInput {
    fn validate(&self, idx: usize) -> Result<(), crate::PfError> {
        if self.tw_tvt.len() != self.tw_gr.len() {
            return Err(crate::PfError::Config(format!(
                "beam well {idx}: tw_tvt/tw_gr length mismatch ({}, {})",
                self.tw_tvt.len(),
                self.tw_gr.len()
            )));
        }
        if self.tw_tvt.is_empty() {
            return Err(crate::PfError::Config(format!(
                "beam well {idx}: the type-well log is empty"
            )));
        }
        if !self.last_tvt.is_finite() {
            return Err(crate::PfError::Config(format!(
                "beam well {idx}: last_tvt must be finite, got {}",
                self.last_tvt
            )));
        }
        Ok(())
    }
}

/// Flattened, device-ready view of a set of wells.
#[derive(Debug, Clone, Default)]
pub struct FlatBeamBatch {
    pub gr: Vec<f32>,
    pub tw_tvt: Vec<f32>,
    pub tw_gr: Vec<f32>,
    /// `BEAM_META_U_STRIDE` entries per well: `[ev_off, ev_len, tw_off, tw_len]`.
    pub meta_u: Vec<u32>,
    /// `BEAM_META_F_STRIDE` entries per well: `[last_tvt]`.
    pub meta_f: Vec<f32>,
    /// Well index of every flattened evaluation row.
    pub row_well: Vec<u32>,
    /// Evaluation row count per well.
    pub ev_lens: Vec<u32>,
    /// Start of each well's rows in `gr`.
    pub ev_offsets: Vec<u32>,
}

impl FlatBeamBatch {
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
    pub fn build(wells: &[BeamWellInput]) -> Result<(Self, Vec<usize>), crate::PfError> {
        let mut b = FlatBeamBatch::default();
        let mut kept = Vec::new();

        for (idx, wl) in wells.iter().enumerate() {
            wl.validate(idx)?;
            if wl.gr.is_empty() {
                continue;
            }

            let ev_off = b.gr.len() as u32;
            let ev_len = wl.gr.len() as u32;
            let tw_off = b.tw_tvt.len() as u32;
            let tw_len = wl.tw_tvt.len() as u32;
            let well_idx = b.ev_lens.len() as u32;

            b.gr.extend_from_slice(&wl.gr);
            b.tw_tvt.extend_from_slice(&wl.tw_tvt);
            b.tw_gr.extend_from_slice(&wl.tw_gr);
            b.row_well.extend(std::iter::repeat_n(well_idx, ev_len as usize));

            b.meta_u.extend_from_slice(&[ev_off, ev_len, tw_off, tw_len]);
            b.meta_f.push(wl.last_tvt);

            b.ev_offsets.push(ev_off);
            b.ev_lens.push(ev_len);
            kept.push(idx);
        }

        debug_assert_eq!(b.meta_u.len(), b.n_wells() * BEAM_META_U_STRIDE);
        debug_assert_eq!(b.meta_f.len(), b.n_wells() * BEAM_META_F_STRIDE);
        Ok((b, kept))
    }

    /// Splits the batch into contiguous well ranges whose `[n_planes, rows]`
    /// device buffers each stay under `budget_bytes`. A single well always gets
    /// its own chunk even if it exceeds the budget on its own.
    pub fn chunks(&self, n_planes: usize, budget_bytes: usize, elem_size: usize) -> Vec<(usize, usize)> {
        let per_row = n_planes * elem_size;
        let max_rows = (budget_bytes / per_row.max(1)).max(1);

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

/// The distinct Savitzky-Golay radii used by a config set, plus, for each config,
/// the index of the smoothing plane it reads.
///
/// Configs routinely share a radius (the notebook's 14 use only six distinct
/// values), and smoothing is a whole-batch pass, so it is computed once per
/// distinct radius rather than once per config.
pub fn smoothing_planes(configs: &[BeamConfig]) -> (Vec<u32>, Vec<u32>) {
    let mut radii: Vec<u32> = Vec::new();
    let mut plane = Vec::with_capacity(configs.len());
    for c in configs {
        let p = match radii.iter().position(|r| *r == c.radius) {
            Some(p) => p,
            None => {
                radii.push(c.radius);
                radii.len() - 1
            }
        };
        plane.push(p as u32);
    }
    (radii, plane)
}

/// Beam-search trajectories for a batch of wells.
#[derive(Debug, Clone)]
pub struct BeamOutput {
    /// Ensemble mean over configs, one value per evaluation row: the notebook's
    /// `beam_mean`.
    pub mean: Vec<f32>,
    /// `[config][row]` trajectories, present only when
    /// [`BeamOptions::with_per_config`] was set.
    pub per_config: Option<Vec<f32>>,
    /// Start of each well's rows within a trajectory.
    pub ev_offsets: Vec<u32>,
    /// Evaluation row count per well.
    pub ev_lens: Vec<u32>,
    /// Indices into the caller's `wells` slice for the wells actually searched
    /// (wells with an empty evaluation zone are dropped).
    pub kept: Vec<usize>,
}

impl BeamOutput {
    /// Total evaluation rows across all wells.
    pub fn total_rows(&self) -> usize {
        self.mean.len()
    }

    /// One well's slice of the ensemble mean.
    pub fn well_mean(&self, well: usize) -> Option<&[f32]> {
        let off = *self.ev_offsets.get(well)? as usize;
        let len = *self.ev_lens.get(well)? as usize;
        Some(&self.mean[off..off + len])
    }

    /// All rows of one config's trajectory, or `None` if per-config output was
    /// not requested.
    pub fn config_rows(&self, config: usize) -> Option<&[f32]> {
        let pc = self.per_config.as_ref()?;
        let n = self.total_rows();
        pc.get(config * n..(config + 1) * n)
    }

    /// One well's slice of one config's trajectory.
    pub fn well_config(&self, config: usize, well: usize) -> Option<&[f32]> {
        let ch = self.config_rows(config)?;
        let off = *self.ev_offsets.get(well)? as usize;
        let len = *self.ev_lens.get(well)? as usize;
        Some(&ch[off..off + len])
    }
}
