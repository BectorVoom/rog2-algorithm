# rog2-pf — GPU particle filter for the ROGII competition

A [CubeCL](https://github.com/tracel-ai/cubecl) port of the 128-seed
likelihood-weighted particle filter from
`notebook/rogii-another-approch-2nd.ipynb` (`_pf_lik_allseeds` / `lik_pf`,
cell 38) — the tracker that produces the `pf_scale_*` trajectories feeding the
ridge/selector blend.

Runs on CUDA (the Kaggle T4 target), wgpu, or CubeCL's CPU runtime, and ships
Python bindings that are drop-in for the notebook's `lik_pf`.

## Why it maps well to a GPU

The numba version is `for seed: for md_step: for particle:`. Only the middle loop
is inherently sequential — it is a filter. So:

| CPU loop         | GPU mapping                                              |
|------------------|----------------------------------------------------------|
| `seed`, `well`   | one **cube** (CUDA block) per `(well, seed)` pair         |
| `particle`       | **units** within the cube, over a shared-memory cloud     |
| weight sums      | one fused tree reduction of three lanes per step          |
| resampling walk  | two-level in-cube prefix scan + per-slot binary search    |

With 128 seeds and, say, 700 wells that is ~90k independent cubes from a single
launch — a T4 stays saturated. The per-seed trajectories are also reduced to the
softmax-blended channels *on the device*: materialising them on the host would
cost 128x the output size.

Shared memory holds five allocations (`pos`, `rate`, `w`, `cum`, and one packed
reduction scratch). Resampling would normally want two more buffers for the
permuted cloud; instead it stages through `cum` in separate passes and parks the
source indices in `w`, which is dead until it is reset to `1/N`.

## Determinism

The RNG is **counter-based**: every draw is a pure function of
`(seed_base + seed, particle slot, step index, draw index)`. Results are bitwise
reproducible across runs, devices and thread schedules for a fixed
`(seed_base, n_particles)`.

They are **not** bit-identical to numba. numba draws from one Mersenne Twister
stream consumed in particle order, which cannot be reproduced when particles
advance in parallel. Both are Monte Carlo estimators of the same posterior, so
they agree statistically, not row-by-row — see the Kaggle notebook for the
accuracy comparison that actually matters.

Two other deliberate deviations from the f64 CPU version, both documented at
their definitions:

* `lik_floor` defaults to `1e-12` (numba hardcodes `1e-300`). f32 cannot carry
  the weight recursion below ~1e-15; the floor bites only when the whole cloud is
  more than ~7.4 GR-sigma off, where those particles are already negligible.
* `interp1` clamps to `grid[0]` just below the grid instead of extrapolating.
  Particle TVT is clamped 100 ft below `vmin` first, so the window is unreachable.

## Measured on a Kaggle T4

From `kaggle/rog2-pf-cubecl-t4.ipynb`, 4 wells x 1950 evaluation rows,
500 particles, 128 seeds (0.5 G particle-steps):

| | |
|---|---|
| GPU, warm | **0.080 s** — 6.2 G particle-steps/s |
| GPU, first call | 1.56 s (one-off NVRTC compile) |
| numba, single thread | 58.9 s — 0.0085 G particle-steps/s |
| speedup | **734x** |
| bitwise reproducible across runs | yes |

Caveats worth keeping straight:

* The numba baseline is single-threaded. The competition pipeline runs it under
  `joblib` across ~4–8 cores, so the practical end-to-end gain is roughly 100–180x,
  not 734x.
* Throughput still climbs with batch size (5.3 G/s at one well, 6.3 G/s at four),
  which is why the API takes every well at once. A T4 is nowhere near saturated by
  four wells.
* That run used synthetic wells: the account pushing the notebook had not accepted
  the competition rules, so `competition_sources` was left empty and the notebook
  fell back to its generator. Accept the rules and re-push with
  `WITH_COMPETITION_DATA=1` for the accuracy comparison on real wells.
* Agreement with numba (median 5.9 ft RMSE on `pf_scale_3`) sits inside the
  across-seed spread (9.6 ft), which is what two independent draws from the same
  estimator should look like. Do not read a per-well RMSE difference in either
  direction as a quality gap — with a multimodal GR likelihood a single well swings
  tens of feet on which branch a seed locks onto.

## Layout

```
src/kernel.rs      the particle-filter kernel and its in-cube collectives
src/blend.rs       device-side softmax reduction over seeds (+ pf_pt_std)
src/host.rs        upload, launch, chunking; uses cubek's cube_count_spread
src/batch.rs       PfConfig / WellInput and the flattened device layout
src/reference.rs   scalar CPU reference — same maths, same RNG, same reduction order
src/synthetic.rs   synthetic wells for tests and benchmarks
src/python.rs      pyo3 bindings
python/rog2_pf/    Python package: lik_pf / lik_pf_batch drop-ins
kaggle/            T4 verification + benchmark notebook and its push script
```

## Build

```bash
cargo test  --features cpu           # parity against the scalar reference
cargo clippy --features cpu --all-targets
cargo run   --features cuda --bin pf-bench -- --backend cuda --wells 64 --rows 3000
```

`--verify` on `pf-bench` also runs the reference and reports the worst per-row
deviation (expect ~1e-5 ft — pure f32 rounding).

### Python wheel

```bash
maturin build --release --no-default-features \
  --features 'pyo3/extension-module,python,cuda' -o dist
pip install dist/rog2_pf-*.whl
```

Swap `cuda` for `cpu` on a machine without an NVIDIA toolkit. `cubecl-cuda`'s
build script reads the CUDA version from the toolkit, so `nvcc` must be on
`PATH` when building with the `cuda` feature.

## Using it from the competition notebook

`lik_pf_batch` returns the same `(out, ev_index, quality)` tuples as the original
`lik_pf`, so the per-well loop collapses into one batched call:

```python
from rog2_pf import lik_pf_batch

results = lik_pf_batch([(hw, tw) for hw, tw in wells], with_quality=True)
for (out, ev_index, q), wid in zip(results, well_ids):
    likpf_map[wid] = (out, ev_index, q)   # build_well() is unchanged
```

`out` carries `pf_scale_3/5/8/12` and `pf_mean`; `quality` carries `pf_best_ll`,
`pf_ll_spread`, `pf_pt_std` and `pf_gr_sig`, matching `with_quality=True`.

Pass `seed_bases=[stable_hash(wid) for wid in well_ids]` for the per-well
deterministic seeding `AGENTS.md` asks for; the default `0` reproduces the
notebook's shared-stream behaviour.

## Kaggle

```bash
KAGGLE_USER=<you> ./kaggle/push.sh          # first run: creates the source dataset
KAGGLE_USER=<you> ./kaggle/push.sh --update # later runs
```

The notebook (GPU T4, internet on) builds the wheel, then verifies against the
numba kernel on real train wells and reports pooled RMSE vs true TVT for both,
agreement between them, determinism, and a batch-size scaling table.

## Known backend limitation

CubeCL's **CPU runtime** (`--features cpu`) miscompiles kernels with **more than
eight shared-memory allocations** — the ninth silently corrupts the others and
can trip heap corruption in the host process. This kernel uses five, so it is
unaffected; keep it that way if you add buffers. The CUDA backend has no such
limit. The CPU runtime is also an interpreter and is several orders of magnitude
slower than native code: it is for correctness tests only.
