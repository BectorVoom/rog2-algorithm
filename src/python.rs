//! Python bindings.
//!
//! Built with `maturin build --release --features python,cuda` on Kaggle. The
//! module mirrors the notebook's `lik_pf`, except that it takes every well at once
//! so a single launch fills the GPU.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::batch::{PfConfig, WellInput};
use crate::{Backend, PfError, run_on};

fn err(e: PfError) -> PyErr {
    match e {
        PfError::Config(m) => PyValueError::new_err(m),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Pulls a 1-D float array out of a Python object, accepting f32, f64 or any
/// sequence of floats. Pandas/numpy columns are usually f64, so this avoids
/// forcing every caller to cast first.
fn as_f32_vec(obj: &Bound<'_, PyAny>, field: &str) -> PyResult<Vec<f32>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray1<f32>>() {
        return Ok(a.as_slice()?.to_vec());
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(a.as_slice()?.iter().map(|v| *v as f32).collect());
    }
    obj.extract::<Vec<f64>>()
        .map(|v| v.into_iter().map(|x| x as f32).collect())
        .map_err(|_| PyValueError::new_err(format!("`{field}` must be a 1-D float array")))
}

fn item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("well dict is missing `{key}`")))
}

fn scalar(d: &Bound<'_, PyDict>, key: &str, default: Option<f32>) -> PyResult<f32> {
    match d.get_item(key)? {
        Some(v) => v
            .extract::<f64>()
            .map(|x| x as f32)
            .map_err(|_| PyValueError::new_err(format!("`{key}` must be a float"))),
        None => {
            default.ok_or_else(|| PyValueError::new_err(format!("well dict is missing `{key}`")))
        }
    }
}

fn parse_well(obj: &Bound<'_, PyAny>) -> PyResult<WellInput> {
    let d = obj
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("each well must be a dict"))?;
    Ok(WellInput {
        md: as_f32_vec(&item(d, "md")?, "md")?,
        z: as_f32_vec(&item(d, "z")?, "z")?,
        gr: as_f32_vec(&item(d, "gr")?, "gr")?,
        grid: as_f32_vec(&item(d, "grid")?, "grid")?,
        vmin: scalar(d, "vmin", None)?,
        step: scalar(d, "step", None)?,
        gs: scalar(d, "gs", None)?,
        ls: scalar(d, "ls", None)?,
        ir: scalar(d, "ir", Some(0.0))?,
        init_spr: scalar(d, "init_spr", Some(4.5))?,
        seed_base: match d.get_item("seed_base")? {
            Some(v) => v.extract::<u32>()?,
            None => 0,
        },
    })
}

/// Runs the likelihood-weighted particle filter over a batch of wells.
///
/// `wells` is a list of dicts with keys `md`, `z`, `gr` (the evaluation-zone
/// logs), `grid` (type-well GR resampled on a uniform TVT grid), `vmin`, `step`,
/// `gs`, `ls`, and optionally `ir`, `init_spr` and `seed_base`.
///
/// Returns a dict mapping each output channel (`pf_scale_3`, ..., `pf_mean`) to a
/// list of per-well float32 arrays, plus `liks` (`[n_wells, n_seeds]`) and `kept`
/// (indices of the input wells that had a non-empty evaluation zone).
#[pyfunction]
#[pyo3(signature = (
    wells,
    scales = vec![3.0, 5.0, 8.0, 12.0],
    n_particles = 500,
    n_seeds = 128,
    cube_dim = 256,
    backend = "auto",
    mom = 0.998,
    vn = 0.002,
    pn = 0.005,
    rough_p = 0.1,
    rough_r = 0.001,
    resamp = 0.5,
    lik_floor = 1e-12,
    pred_budget_mb = 512,
    with_std = false,
))]
#[allow(clippy::too_many_arguments)]
fn run_batch<'py>(
    py: Python<'py>,
    wells: &Bound<'py, PyList>,
    scales: Vec<f32>,
    n_particles: u32,
    n_seeds: u32,
    cube_dim: u32,
    backend: &str,
    mom: f32,
    vn: f32,
    pn: f32,
    rough_p: f32,
    rough_r: f32,
    resamp: f32,
    lik_floor: f32,
    pred_budget_mb: usize,
    with_std: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let parsed: Vec<WellInput> = wells
        .iter()
        .map(|w| parse_well(&w))
        .collect::<PyResult<_>>()?;
    let backend: Backend = backend.parse().map_err(err)?;
    let cfg = PfConfig {
        n_particles,
        n_seeds,
        cube_dim,
        mom,
        vn,
        pn,
        rough_p,
        rough_r,
        resamp,
        lik_floor,
        pred_budget_bytes: pred_budget_mb << 20,
        with_std,
    };

    // The kernel does not touch Python state, so other threads can run.
    let out = py
        .detach(|| run_on(backend, &parsed, &cfg, &scales))
        .map_err(err)?;

    let result = PyDict::new(py);
    let total = out.total_rows();
    for (k, name) in out.channels.iter().enumerate() {
        let ch = &out.values[k * total..(k + 1) * total];
        let per_well = PyList::empty(py);
        for (w, len) in out.ev_lens.iter().enumerate() {
            let off = out.ev_offsets[w] as usize;
            let slice = ch[off..off + *len as usize].to_vec();
            per_well.append(slice.into_pyarray(py))?;
        }
        result.set_item(name, per_well)?;
    }

    let n_wells = out.ev_lens.len();
    let liks_rows: Vec<Vec<f32>> = out
        .liks
        .chunks(n_seeds as usize)
        .take(n_wells)
        .map(|c| c.to_vec())
        .collect();
    result.set_item("liks", PyArray2::from_vec2(py, &liks_rows)?)?;
    result.set_item("kept", out.kept)?;
    result.set_item("channels", out.channels)?;
    Ok(result)
}

/// Backends this build can use, most preferred first.
#[pyfunction]
fn available_backends() -> Vec<String> {
    crate::available_backends()
        .into_iter()
        .map(|b| format!("{b:?}").to_lowercase())
        .collect()
}

/// Resamples a type-well `(TVT, GR)` log onto the uniform grid the kernel expects.
///
/// Equivalent to `_grid` in the notebook: returns `(grid, vmin, step)`.
#[pyfunction]
#[pyo3(signature = (tw_tvt, tw_gr, step = 0.2))]
fn make_grid<'py>(
    py: Python<'py>,
    tw_tvt: &Bound<'py, PyAny>,
    tw_gr: &Bound<'py, PyAny>,
    step: f32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, f32, f32)> {
    let tvt = as_f32_vec(tw_tvt, "tw_tvt")?;
    let gr = as_f32_vec(tw_gr, "tw_gr")?;
    if tvt.len() != gr.len() || tvt.len() < 2 {
        return Err(PyValueError::new_err(
            "tw_tvt and tw_gr must have equal length >= 2",
        ));
    }
    let tmin = tvt.iter().copied().fold(f32::INFINITY, f32::min);
    let tmax = tvt.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let n = (((tmax - tmin) / step).floor() as usize) + 1;

    let mut out = Vec::with_capacity(n);
    let mut k = 0usize;
    for i in 0..n {
        let x = tmin + i as f32 * step;
        while k + 2 < tvt.len() && tvt[k + 1] < x {
            k += 1;
        }
        let (x0, x1) = (tvt[k], tvt[k + 1]);
        let t = if (x1 - x0).abs() > 0.0 {
            ((x - x0) / (x1 - x0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        out.push(gr[k] * (1.0 - t) + gr[k + 1] * t);
    }
    Ok((out.into_pyarray(py), tmin, step))
}

#[pymodule]
fn _rog2_pf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(available_backends, m)?)?;
    m.add_function(wrap_pyfunction!(make_grid, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
