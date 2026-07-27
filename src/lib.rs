//! GPU trackers for the ROGII wellbore-geology competition.
//!
//! Ports of the two trajectory estimators from
//! `notebook/rogii-another-approch-2nd.ipynb` to
//! [CubeCL](https://github.com/tracel-ai/cubecl), with CUDA, ROCm/HIP, wgpu and
//! CPU backends and Python bindings:
//!
//! * the 128-seed likelihood-weighted **particle filter** (`_pf_lik_allseeds` /
//!   `lik_pf`), which produces the `pf_scale_*` channels, and
//! * the 14-config **beam search** (`beam_search` / `run_beam_ensemble`), the
//!   Viterbi-style tracker behind the `*_beam_*` selector variants.
//!
//! ```no_run
//! use rog2_pf::{PfConfig, WellInput, run_auto};
//!
//! let wells: Vec<WellInput> = vec![/* one entry per well */];
//! let out = run_auto(&wells, &PfConfig::default(), &[3.0, 5.0, 8.0, 12.0]).unwrap();
//! let pf_scale_3 = out.channel("pf_scale_3").unwrap();
//! ```
//!
//! ```no_run
//! use rog2_pf::{BeamOptions, BeamWellInput, NOTEBOOK_BEAM_CONFIGS, run_beam_auto};
//!
//! let wells: Vec<BeamWellInput> = vec![/* one entry per well */];
//! let out = run_beam_auto(&wells, &NOTEBOOK_BEAM_CONFIGS, &BeamOptions::default()).unwrap();
//! let tvt_beam = out.well_mean(0).unwrap();
//! ```

pub mod batch;
pub mod beam;
pub mod beam_host;
pub mod beam_kernel;
pub mod beam_reference;
pub mod blend;
pub mod host;
pub mod kernel;
pub mod reference;
pub mod synthetic;

pub use batch::{FlatBatch, PfConfig, WellInput, seed_weights};
pub use beam::{
    BeamConfig, BeamOptions, BeamOutput, BeamWellInput, FlatBeamBatch, NOTEBOOK_BEAM_CONFIGS,
};
pub use beam_host::run_beam;
pub use host::run_pf;

#[cfg(any(feature = "python", feature = "python-test"))]
mod python;

/// Errors surfaced by the host driver.
#[derive(Debug, thiserror::Error)]
pub enum PfError {
    /// The batch or configuration violates an invariant the kernel relies on.
    #[error("invalid particle-filter configuration: {0}")]
    Config(String),
    /// A device operation failed.
    #[error("device error: {0}")]
    Device(String),
    /// No CubeCL backend was compiled in, or none could be initialised.
    #[error("no usable backend: {0}")]
    NoBackend(String),
}

/// Seed-reduced trajectories for a batch of wells.
#[derive(Debug, Clone)]
pub struct PfOutput {
    /// Output channel names: `pf_scale_*` in the requested order, then `pf_mean`.
    pub channels: Vec<String>,
    /// `[channel][row]`, row-major, `channels.len() * total_rows` elements.
    pub values: Vec<f32>,
    /// Per-seed total log-likelihood, `[well][seed]`.
    pub liks: Vec<f32>,
    /// Start of each well's rows within a channel.
    pub ev_offsets: Vec<u32>,
    /// Evaluation row count per well.
    pub ev_lens: Vec<u32>,
    /// Indices into the caller's `wells` slice for the wells actually filtered
    /// (wells with an empty evaluation zone are dropped).
    pub kept: Vec<usize>,
}

impl PfOutput {
    /// Total evaluation rows across all wells.
    pub fn total_rows(&self) -> usize {
        self.ev_lens.iter().map(|l| *l as usize).sum()
    }

    /// All rows of one output channel, or `None` if the name is unknown.
    pub fn channel(&self, name: &str) -> Option<&[f32]> {
        let k = self.channels.iter().position(|c| c == name)?;
        let n = self.total_rows();
        Some(&self.values[k * n..(k + 1) * n])
    }

    /// One well's slice of one output channel.
    pub fn well_channel(&self, name: &str, well: usize) -> Option<&[f32]> {
        let ch = self.channel(name)?;
        let off = *self.ev_offsets.get(well)? as usize;
        let len = *self.ev_lens.get(well)? as usize;
        Some(&ch[off..off + len])
    }
}

/// Which CubeCL backend to run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Pick HIP (ROCm), then CUDA, then wgpu, then CPU, depending on what
    /// is compiled in and initialises successfully.
    Auto,
    /// AMD ROCm / HIP.
    Hip,
    /// NVIDIA CUDA (the Kaggle T4 path).
    Cuda,
    /// wgpu (Vulkan/Metal/DX12).
    Wgpu,
    /// CubeCL's CPU runtime — for tests and machines without a GPU.
    Cpu,
}

impl std::str::FromStr for Backend {
    type Err = PfError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Backend::Auto),
            "hip" | "rocm" => Ok(Backend::Hip),
            "cuda" | "gpu" => Ok(Backend::Cuda),
            "wgpu" | "vulkan" | "metal" => Ok(Backend::Wgpu),
            "cpu" => Ok(Backend::Cpu),
            other => Err(PfError::Config(format!("unknown backend {other:?}"))),
        }
    }
}

/// Backends this build can actually use, in `Auto` preference order.
pub fn available_backends() -> Vec<Backend> {
    [
        #[cfg(feature = "hip")]
        Backend::Hip,
        #[cfg(feature = "cuda")]
        Backend::Cuda,
        #[cfg(feature = "wgpu")]
        Backend::Wgpu,
        #[cfg(feature = "cpu")]
        Backend::Cpu,
    ]
    .to_vec()
}

/// Initialises the client for one backend and calls a `run_*` driver on it.
///
/// Each arm is behind its own `cfg`, so a build without a backend still compiles
/// and reports the missing feature at runtime instead of failing to link.
macro_rules! on_backend {
    ($backend:expr, $auto:expr, $call:path, $($arg:expr),* $(,)?) => {
        match $backend {
            Backend::Auto => $auto,
            Backend::Hip => {
                #[cfg(feature = "hip")]
                {
                    let device = cubecl::hip::AmdDevice::default();
                    let client = <cubecl::hip::HipRuntime as cubecl::Runtime>::client(&device);
                    $call(&client, $($arg),*)
                }
                #[cfg(not(feature = "hip"))]
                Err(PfError::NoBackend("built without the `hip` feature".into()))
            }
            Backend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    let device = cubecl::cuda::CudaDevice::default();
                    let client = <cubecl::cuda::CudaRuntime as cubecl::Runtime>::client(&device);
                    $call(&client, $($arg),*)
                }
                #[cfg(not(feature = "cuda"))]
                Err(PfError::NoBackend("built without the `cuda` feature".into()))
            }
            Backend::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    let device = cubecl::wgpu::WgpuDevice::default();
                    let client = <cubecl::wgpu::WgpuRuntime as cubecl::Runtime>::client(&device);
                    $call(&client, $($arg),*)
                }
                #[cfg(not(feature = "wgpu"))]
                Err(PfError::NoBackend("built without the `wgpu` feature".into()))
            }
            Backend::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    let device = cubecl::cpu::CpuDevice;
                    let client = <cubecl::cpu::CpuRuntime as cubecl::Runtime>::client(&device);
                    $call(&client, $($arg),*)
                }
                #[cfg(not(feature = "cpu"))]
                Err(PfError::NoBackend("built without the `cpu` feature".into()))
            }
        }
    };
}

/// Tries every compiled-in backend in `Auto` preference order (HIP → CUDA → wgpu → CPU).
macro_rules! first_working_backend {
    ($run:expr) => {{
        let candidates = available_backends();
        if candidates.is_empty() {
            return Err(PfError::NoBackend(
                "compile with one of the `hip`, `cuda`, `wgpu` or `cpu` features".into(),
            ));
        }
        let mut last = None;
        for b in candidates {
            match $run(b) {
                Ok(out) => return Ok(out),
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap())
    }};
}

/// Runs the particle filter on an explicitly chosen backend.
pub fn run_on(
    backend: Backend,
    wells: &[WellInput],
    cfg: &PfConfig,
    scales: &[f32],
) -> Result<PfOutput, PfError> {
    on_backend!(
        backend,
        run_auto(wells, cfg, scales),
        run_pf,
        wells,
        cfg,
        scales
    )
}

/// Runs the particle filter on the first backend that initialises, preferring CUDA.
pub fn run_auto(wells: &[WellInput], cfg: &PfConfig, scales: &[f32]) -> Result<PfOutput, PfError> {
    first_working_backend!(|b| run_on(b, wells, cfg, scales))
}

/// Runs the beam-search ensemble on an explicitly chosen backend.
pub fn run_beam_on(
    backend: Backend,
    wells: &[BeamWellInput],
    configs: &[BeamConfig],
    opts: &BeamOptions,
) -> Result<BeamOutput, PfError> {
    on_backend!(
        backend,
        run_beam_auto(wells, configs, opts),
        run_beam,
        wells,
        configs,
        opts
    )
}

/// Runs the beam-search ensemble on the first backend that initialises,
/// preferring CUDA.
pub fn run_beam_auto(
    wells: &[BeamWellInput],
    configs: &[BeamConfig],
    opts: &BeamOptions,
) -> Result<BeamOutput, PfError> {
    first_working_backend!(|b| run_beam_on(b, wells, configs, opts))
}
