# rog2-algorithm

GPU-accelerated particle filter and beam search for the [ROGII wellbore geology
competition](https://kaggle.com/competitions/rogii-wellbore-geology-prediction).

Drop-in replacements for the notebook's GPU-free estimators — 700x faster on a
Kaggle T4, bitwise reproducible, and batching across wells so the GPU stays
saturated.

```python
from rog2_pf import lik_pf_batch, run_beam_ensemble_batch

results = lik_pf_batch([(hw, tw) for hw, tw in wells], with_quality=True)
for (out, ev_index, quality), wid in zip(results, well_ids):
    likpf_map[wid] = (out, ev_index, quality)

tvt_beams = run_beam_ensemble_batch([(hw, tw) for hw, tw in wells])
for wid, tvt_beam in zip(well_ids, tvt_beams):
    beam_map[wid] = tvt_beam
```

## Installation

```bash
pip install rog2-algorithm
```

The shipped wheel bundles CUDA + wgpu + CPU backends — same package works on
Kaggle T4s, local Vulkan/Metal GPUs, and CPU-only machines. Backend is selected
at the call site via the `backend` parameter.

**AMD ROCm / HIP** — build from source:

```bash
pip install rog2-algorithm --no-binary :all:
# or manually:
maturin build --release --features pyo3/extension-module,python,hip
pip install dist/rog2_algorithm-*.whl
```

**Dependencies:** Python 3.9+ and numpy.

## API

All batched functions take *every well at once* — one GPU launch replaces a
Python loop over wells, keeping the GPU saturated.

### Particle filter

#### `lik_pf_batch(pairs, ...)`

Batched drop-in for the notebook's `lik_pf`. Returns one
`(out, ev_index, quality)` tuple per input pair.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pairs` | `list[(DataFrame, DataFrame)]` | — | Sequence of `(horizontal_well, typewell)` pairs |
| `n_particles` | `int` | `500` | Particles per seed |
| `n_seeds` | `int` | `128` | Independent seeds (each gets one GPU block) |
| `scales` | `tuple[float, ...]` | `(3.0, 5.0, 8.0, 12.0)` | GR sigmas for the softmax-blended channels |
| `init_spr` | `float` | `4.5` | Initial spread of the particle cloud (ft) |
| `seed_bases` | `list[int] \| None` | `None` | Per-well RNG seeds — use stable hash of well ID for reproducibility |
| `backend` | `str` | `"auto"` | `"cuda"`, `"wgpu"`, `"hip"` / `"rocm"`, `"cpu"`, or `"auto"` |
| `with_quality` | `bool` | `False` | Compute per-well quality diagnostics (`pf_pt_std`, requires extra GPU pass) |
| `cube_dim` | `int` | `256` | Threads per block — tune for kernel occupancy |

Returns `list[(out, ev_index, quality)]`:

- **`out`** — `dict[str, np.ndarray]` — channels `pf_scale_3/5/8/12` and `pf_mean`, one float32 array per well over its evaluation rows.
- **`ev_index`** — `np.ndarray` — index slice the evaluation rows occupy in the original DataFrame.
- **`quality`** — `dict` — `pf_best_ll`, `pf_ll_spread`, `pf_pt_std`, `pf_gr_sig` (empty when `with_quality=False`).

Wells with no evaluation rows return `({}, empty_index, {})`.

#### `lik_pf(hw, tw, **kwargs)`

Single-well convenience. Prefer the batched form.

### Beam search

#### `run_beam_ensemble_batch(pairs, ...)`

Batched drop-in for the notebook's `run_beam_ensemble`. Returns one array per
input pair matching `hw.TVT_input` layout (known rows preserved, evaluation
rows filled with the 14-config ensemble mean).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pairs` | `list[(DataFrame, DataFrame)]` | — | Sequence of `(horizontal_well, typewell)` pairs |
| `configs` | `list[tuple] \| None` | `None` | Override `BEAM_CONFIGS`; each tuple is `(beam_size, move_cost, err_scale, radius)` |
| `backend` | `str` | `"auto"` | Same backends as particle filter |
| `cube_dim` | `int` | `64` | Threads per block |
| `smoothing` | `str` | `"rolling_mean"` | GR-smoothing algorithm: `"rolling_mean"` matches the notebook's actually-active `_smooth` (a plain centred moving average); `"savitzky_golay"` is a configurable alternative (quadratic curve fit) offered for experimentation. Measured on 155 real wells: pooled RMSE against true hidden-section TVT is within 0.006 ft either way (14.857 vs 14.863 ft) — not a meaningful accuracy difference, so `"rolling_mean"` (notebook fidelity) remains the default. |

Wells with no evaluation rows return a copy of their input `TVT_input` column.

#### `run_beam_ensemble(hw, tw, **kwargs)`

Single-well convenience — prefer the batched form.

#### `beam_search(hgr, tw_tvt, tw_gr, last_tvt, ...)`

Single-config drop-in for the notebook's `beam_search`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hgr` | `np.ndarray` | — | Horizontal-well gamma ray over evaluation rows |
| `tw_tvt` | `np.ndarray` | — | Type-well TVT (does not need to be sorted) |
| `tw_gr` | `np.ndarray` | — | Type-well GR |
| `last_tvt` | `float` | — | Last known TVT before the evaluation zone |
| `bs` | `int` | `10` | Beam size |
| `mc` | `float` | `20.0` | Move cost per type-well step |
| `es` | `float` | `144.0` | Error scale for the GR mismatch term |
| `r` | `int` | `2` | Savitzky-Golay smoother radius |

### Low-level native API

The native `_rog2_pf` module is re-exported through the `rog2_pf` package.
Use these directly when you have pre-built input dicts.

#### `run_batch(wells, scales, ...)`

Low-level particle filter — no pandas, no `prepare_well`. Takes pre-built
well dicts.

| Parameter | Type | Description |
|-----------|------|-------------|
| `wells` | `list[dict]` | Well dicts with keys `md`, `z`, `gr`, `grid`, `vmin`, `step`, `gs`, `ls`, `ir`, `init_spr`, `seed_base` |
| `scales` | `list[float]` | GR sigmas for output channels |
| `n_particles` | `int` | Particles per seed |
| `n_seeds` | `int` | Number of independent seeds |
| `cube_dim` | `int` | Threads per block |
| `backend` | `str` | Backend selection |
| `mom, vn, pn, rough_p, rough_r, resamp, lik_floor` | `float` | Process model parameters |
| `pred_budget_mb` | `int` | Maximum prediction buffer per launch (MB) |
| `with_std` | `bool` | Compute per-well position std |

Returns a dict with keys `pf_scale_*`, `pf_mean` (lists of per-well arrays),
`pf_pt_std` (per-well std when `with_std=True`), `liks` (`[n_wells, n_seeds]`),
`kept` (indices of non-empty wells), and `channels`.

#### `run_beam_batch(wells, ...)`

Low-level beam search.

| Parameter | Type | Description |
|-----------|------|-------------|
| `wells` | `list[dict]` | Well dicts with keys `gr`, `tw_tvt`, `tw_gr`, `last_tvt` |
| `configs` | `list[tuple] \| None` | Beam configs or `None` for `BEAM_CONFIGS` |
| `cube_dim` | `int` | Threads per block |
| `backend` | `str` | Backend selection |
| `with_per_config` | `bool` | Return per-config results |
| `budget_mb` | `int` | Maximum prediction buffer per launch (MB) |

Returns a dict with `beam_mean` (list of per-well arrays), `kept`, and
optionally `per_config` (`[config][well]` arrays).

#### `make_grid(tw_tvt, tw_gr, step)`

Resamples a type-well log onto a uniform TVT grid. Returns
`(grid_array, vmin, step)`.

#### `available_backends()`

Returns `list[str]` — the backends the installed wheel supports (e.g.,
`["cuda", "wgpu", "cpu"]`).

#### `notebook_beam_configs()`

Returns the default 14 `(beam_size, move_cost, err_scale, radius)` tuples
matching the competition notebook.

### Helpers & constants

- **`prepare_well(hw, tw, init_spr, seed_base)`** — Build a particle-filter input dict from a `(hw, tw)` pair. Returns `(well_dict, ev_index)`.
- **`prepare_beam_well(hw, tw)`** — Build a beam-search input dict. Returns `(well_dict, ev_index)`.
- **`DEFAULT_SCALES`** — `(3.0, 5.0, 8.0, 12.0)`.
- **`BEAM_CONFIGS`** — The 14 notebook configs as `(bs, mc, es, r)` tuples.

## Deterministic seeding

Pass `seed_bases=[stable_hash(wid)]` to `lik_pf_batch` for per-well reproducible
seeding. The RNG is counter-based — every draw is a pure function of
`(seed_base + seed, particle slot, step index, draw index)`. Results are
bitwise reproducible across runs, devices and thread schedules for a fixed
`(seed_base, n_particles)`.

The beam search has no RNG and is bit-identical across runs, devices, and
`cube_dim`.

## Backend selection

The `backend` parameter accepts: `"cuda"`, `"wgpu"`, `"hip"` / `"rocm"`,
`"cpu"`, or `"auto"` (tries compiled-in backends in preference order).

```python
from rog2_pf import lik_pf_batch, available_backends

print(available_backends())         # what the installed wheel supports
results = lik_pf_batch(pairs, backend="wgpu")  # explicit backend
```

## Build from source

```bash
# CUDA
maturin build --release --no-default-features \
  --features 'pyo3/extension-module,python,cuda' -o dist

# ROCm / HIP
maturin build --release --no-default-features \
  --features 'pyo3/extension-module,python,hip' -o dist

# wgpu (Vulkan / Metal / DX12)
maturin build --release --no-default-features \
  --features 'pyo3/extension-module,python,wgpu' -o dist

# CPU only
maturin build --release --no-default-features \
  --features 'pyo3/extension-module,python,cpu' -o dist

pip install dist/rog2_algorithm-*.whl
```

Requires Rust toolchain and `maturin` (`pip install maturin`).

## Performance

| | Particle filter | Beam search |
|---|---|---|
| GPU, warm | 0.080 s (6.2 G particle-steps/s) | proportionally faster |
| Speedup vs numba | ~730x (single-thread) / ~100–180x (joblib) | deterministic match |

See `tests/` for the cross-language parity assertions and `kaggle/` for the
T4 benchmark notebooks.
