#!/usr/bin/env python3
"""Generates the Kaggle T4 verification/benchmark notebook for the beam search.

Kept as a generator rather than a hand-edited .ipynb so the cells stay
reviewable in git.  Run `python kaggle/build_beam_notebook.py` after editing.
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------

md(
    r"""
# ROGII beam search on a T4 — CubeCL/CUDA port

This notebook builds `rog2-pf` (the Rust/CubeCL port) against CUDA, then
**verifies** and **benchmarks** its beam-search ensemble against the original
numpy `beam_search` / `run_beam_ensemble` from
`notebook/rogii-another-approch-2nd.ipynb` (cell 12).

Requirements: **GPU T4 accelerator**, **internet ON** (for `rustup` and crates.io),
and the competition data attached.

## What is being compared

Unlike the particle filter, the beam search is **deterministic** — no RNG
anywhere. The GPU port should therefore reproduce numpy essentially exactly, and
this notebook holds it to that standard rather than to a statistical one.

The trajectory a config emits is `tw_tvt[best_index]`: a lookup into the
type-well table. So "did the two implementations agree" is not a tolerance
question — either they selected the same type-well sample on a given row or they
did not. Adjacent samples are ~0.2 ft apart, so any two rows within 0.01 ft are
the same sample rounded differently (the kernel works in f32, numpy in f64).

The checks below are:

1. **Sample agreement** — the fraction of rows where the GPU and numpy chose the
   same type-well sample, per config. Expect 100%, or a handful of rows off where
   two paths were within f32 rounding of a tie.
2. **Ensemble RMSE** — between the two 14-config means, in feet.
3. **Accuracy** — RMSE against the true TVT on train wells, for both.
4. **Determinism** — reruns must be bit-identical, and the answer must not
   depend on `cube_dim`.
5. **Throughput** — wall-clock for the same work, plus a batch-size scaling
   table.

## Where the parallelism comes from

`beam_search` is `for config: for step: <numpy op over 5 x bs candidates>`. The
step loop is a dynamic program and cannot be parallelised, but the config and
well axes are fully independent, so every `(well, config)` pair becomes one cube
(CUDA block). With 14 configs and a few hundred wells that is thousands of
independent cubes from a single launch, and each cube's `5 * bs <= 150`
candidates map onto its units.
"""
)

code(
    r"""
import os, subprocess, sys, time
from pathlib import Path

def sh(cmd, **kw):
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    if r.stdout: print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
    return r

sh("nvidia-smi")
sh("nvcc --version || ls /usr/local | grep -i cuda")
print("python", sys.version)
"""
)

md(
    r"""
## 1. Toolchain

`cubecl-cuda` compiles kernels at runtime through NVRTC, but its build script
reads the CUDA version from the toolkit, so `nvcc` must be on `PATH` at build
time. Kaggle's GPU image ships it under `/usr/local/cuda`.
"""
)

code(
    r"""
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]
os.environ.setdefault("CUDA_ROOT", "/usr/local/cuda")

if subprocess.run("cargo --version", shell=True, capture_output=True).returncode != 0:
    sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
sh("cargo --version && rustc --version")
sh("pip install -q 'maturin>=1.7,<2.0'")
"""
)

md(
    r"""
## 2. Locate the crate and build the wheel

The crate is expected as a Kaggle dataset (`kaggle/push.sh` in the repo uploads
it) or, when running locally, straight from the repo.
"""
)

code(
    r"""
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
BUILD = WORK / "rog2-algorithm"

# The crate is uploaded as a tarball; Kaggle extracts dataset archives on ingest,
# so accept either shape. cargo needs a writable tree, so always work on a copy
# in /kaggle/working.
SRC = None
if Path("/kaggle/input").exists():
    print("inputs:", [str(p) for p in Path("/kaggle/input").iterdir()])
    hits = sorted(Path("/kaggle/input").glob("*/**/Cargo.toml"))
    if hits:
        SRC = hits[0].parent
    else:
        tars = sorted(Path("/kaggle/input").glob("*/**/rog2-src.tar.gz"))
        assert tars, "attach the rog2-pf-src dataset to this notebook"
        sh(f"rm -rf '{BUILD}' && tar -xzf '{tars[0]}' -C '{WORK}'")
if SRC is None and not (BUILD / "Cargo.toml").exists():
    # Running from a checkout of the repo.
    SRC = next((p for p in [Path("rog2-algorithm"), Path(".")] if (p / "Cargo.toml").exists()), None)

if SRC is not None:
    print("crate source:", SRC)
    sh(f"rm -rf '{BUILD}' && cp -r '{SRC}' '{BUILD}'")

assert (BUILD / "Cargo.toml").exists(), f"{BUILD} is not a crate"
print("build dir:", BUILD.resolve())
"""
)

code(
    r"""
t0 = time.time()
r = sh(
    "maturin build --release --no-default-features "
    "--features 'pyo3/extension-module,python,cuda' -o dist",
    cwd=str(BUILD),
)
print(f"build took {time.time()-t0:.0f}s, rc={r.returncode}")
assert r.returncode == 0, "maturin build failed — see stderr above"
"""
)

code(
    r"""
import glob
wheel = sorted(glob.glob(str(BUILD / "dist" / "*.whl")))[-1]
print("wheel:", wheel)
sh(f"pip install --force-reinstall --no-deps -q '{wheel}'")

import importlib
import rog2_pf
importlib.reload(rog2_pf)
print("rog2_pf", rog2_pf.__version__, "backends:", rog2_pf.available_backends())
assert "cuda" in rog2_pf.available_backends(), "CUDA backend missing from this build"
print("configs:", len(rog2_pf.BEAM_CONFIGS))
"""
)

md(
    r"""
## 3. The numpy reference

Verbatim from `notebook/rogii-another-approch-2nd.ipynb` (cell 12): `beam_search`
and the 14-config `BEAM_CONFIGS` ensemble.
"""
)

code(
    r"""
import numpy as np, pandas as pd
from scipy.signal import savgol_filter

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10,  8.0,  64.0, 2), ( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5), (20,  4.0,  36.0, 3), (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2), (15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2), (10, 50.0, 400.0, 0),
]
assert [tuple(c) for c in rog2_pf.BEAM_CONFIGS] == BEAM_CONFIGS, "config set drifted"

def nb_beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n  = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))

    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC    = mc * np.array([2., 1., 0., 1., 2.])

    bidx  = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.
    bn = 1

    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)

        gr_e = (gv - tw_gr[ci])**2 / es
        tot  = bcost[:bn, None] + gr_e + MC[None, :]
        tot  = np.where(valid, tot, np.inf)

        ni_f  = ni.flatten(); tot_f = tot.flatten(); vf = valid.flatten()
        ni_f  = ni_f[vf];     tot_f = tot_f[vf]

        order = np.argsort(tot_f)
        ni_s  = ni_f[order];  tot_s = tot_f[order]

        _, first = np.unique(ni_s, return_index=True)
        ni_u  = ni_s[first];  tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        top  = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top  = top[np.argsort(tot_u[top])]

        bidx[:kept]  = ni_u[top]
        bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:]  = bidx[kept - 1]
            bcost[kept:] = np.inf
        bn = kept

        result[step] = tw_tvt[bidx[0]]

    return result

def nb_beam_all(hgr, tw_tvt, tw_gr, last_tvt):
    return np.stack([nb_beam_search(hgr, tw_tvt, tw_gr, last_tvt, *c)
                     for c in BEAM_CONFIGS], 0)

print("numpy reference ready")
"""
)

md(
    r"""
## 4. Load competition wells

Falls back to synthetic wells when the competition data is not attached, so the
notebook still runs end to end (and still exercises every code path — the search
does not care whether the GR log is real).
"""
)

code(
    r"""
from rog2_pf import prepare_beam_well, run_beam_batch

DATA = None
for c in ["/kaggle/input/rogii-wellbore-geology-prediction",
          "/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
    if Path(c, "train").exists():
        DATA = Path(c); break
print("data:", DATA)

N_WELLS = 32        # wells to verify/benchmark on
N_NUMPY = 4         # wells to run the (slow) numpy baseline over

pairs, truths, names = [], [], []
if DATA is not None:
    files = sorted(DATA.glob("train/*__horizontal_well.csv"))[:N_WELLS]
    for f in files:
        wid = f.name.replace("__horizontal_well.csv", "")
        hw = pd.read_csv(f)
        tw = pd.read_csv(DATA / "train" / f"{wid}__typewell.csv").sort_values("TVT")
        if "TVT_input" not in hw or hw.TVT_input.isna().sum() == 0:
            continue
        pairs.append((hw, tw)); names.append(wid)
        ev = hw[hw.TVT_input.isna()]
        truths.append(ev.TVT.values.astype(float) if "TVT" in hw else None)
else:
    # Synthetic stand-in with the same signal structure: an aperiodic type-well
    # GR log and a slow random walk through it.
    rng = np.random.default_rng(0)
    for w in range(8):
        nt, n_rows = 3000, 2000
        tw_gr = np.clip(70 + 90*np.convolve(rng.standard_normal(nt), np.ones(25)/25, "same"), 5, 160)
        tw_tvt = np.cumsum(0.2 + 0.02*rng.random(nt)) - 300.0
        rate, tvt, truth = 0.0, -40.0, []
        for _ in range(n_rows):
            rate = 0.995*rate + 0.006*rng.standard_normal(); tvt += rate; truth.append(tvt)
        truth = np.array(truth)
        z = 12*np.sin(np.arange(n_rows)/700)
        hw = pd.DataFrame({
            "MD": 9000 + np.arange(n_rows, dtype=float),
            "Z": z, "TVT": truth,
            "GR": np.interp(truth, tw_tvt, tw_gr) + 4*rng.standard_normal(n_rows),
            "TVT_input": np.nan,
        })
        # Give it a known prefix, as the competition wells have.
        hw.loc[:99, "TVT_input"] = truth[:100]
        tw = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})
        pairs.append((hw, tw)); names.append(f"synthetic-{w}")
        truths.append(truth[hw.TVT_input.isna().values])

print(f"{len(pairs)} wells, {sum(int(hw.TVT_input.isna().sum()) for hw, _ in pairs)} evaluation rows")
"""
)

md(
    r"""
## 5. Verify against numpy

Both implementations are run on the same wells. The GPU takes them all in one
launch; numpy runs `N_NUMPY` wells because it is a Python loop over MD steps and
does not scale to the whole batch inside a notebook session.
"""
)

code(
    r"""
SAME_SAMPLE_FT = 1e-2   # adjacent type-well samples are ~0.2 ft apart

prepared, indices = [], []
for hw, tw in pairs:
    well, ev_index = prepare_beam_well(hw, tw)
    prepared.append(well); indices.append(ev_index)
live = [w for w in prepared if w is not None]

t0 = time.time()
gpu = run_beam_batch(live, backend="cuda", with_per_config=True)
t_gpu_cold = time.time() - t0
t0 = time.time()
gpu = run_beam_batch(live, backend="cuda", with_per_config=True)
t_gpu = time.time() - t0
print(f"GPU: {t_gpu_cold:.3f}s cold (NVRTC compile), {t_gpu:.3f}s warm")
"""
)

code(
    r"""
rows_np = 0
t0 = time.time()
np_all = []
for w in live[:N_NUMPY]:
    np_all.append(nb_beam_all(
        w["gr"].astype(np.float64), w["tw_tvt"].astype(np.float64),
        w["tw_gr"].astype(np.float64), float(w["last_tvt"])))
    rows_np += len(w["gr"])
t_np = time.time() - t0
print(f"numpy: {t_np:.1f}s for {N_NUMPY} wells x 14 configs ({rows_np} rows)")
"""
)

code(
    r"""
print(f"{'config':>28}  {'same sample':>12}  {'max |diff| ft':>14}")
worst_cfg = 1.0
for c, cfg in enumerate(BEAM_CONFIGS):
    same, tot, mx = 0, 0, 0.0
    for w in range(len(np_all)):
        a = np.asarray(gpu["per_config"][c][w], dtype=float)
        b = np_all[w][c]
        d = np.abs(a - b)
        same += int((d < SAME_SAMPLE_FT).sum()); tot += len(d); mx = max(mx, float(d.max()))
    frac = same / tot
    worst_cfg = min(worst_cfg, frac)
    print(f"{str(cfg):>28}  {frac:>11.4%}  {mx:>14.3e}")
print(f"\nworst config agreement: {worst_cfg:.4%}")
"""
)

code(
    r"""
# Ensemble mean: the quantity the pipeline actually consumes.
diffs = []
for w in range(len(np_all)):
    a = np.asarray(gpu["beam_mean"][w], dtype=float)
    b = np_all[w].mean(0)
    diffs.append(a - b)
d = np.concatenate(diffs)
rmse_agree = float(np.sqrt(np.mean(d**2)))
print(f"ensemble agreement: RMSE {rmse_agree:.6f} ft, max |diff| {np.abs(d).max():.6f} ft")
"""
)

md(
    r"""
## 6. Accuracy against the true TVT

Agreement with numpy is the correctness check; this is the number that matters
for the competition. Both implementations should land on the same value — a
difference here would mean the port changed the algorithm, not just its
arithmetic.
"""
)

code(
    r"""
def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m])**2))) if m.any() else float("nan")

live_positions = [i for i, w in enumerate(prepared) if w is not None]
rows = []
for slot, i in enumerate(live_positions):
    if truths[i] is None:
        continue
    g = np.asarray(gpu["beam_mean"][slot], dtype=float)
    row = {"well": names[i], "n_eval": len(g),
           "gpu_rmse": rmse(g, truths[i]),
           "hold_last_rmse": rmse(np.full(len(g), live[slot]["last_tvt"]), truths[i])}
    if slot < len(np_all):
        row["numpy_rmse"] = rmse(np_all[slot].mean(0), truths[i])
    rows.append(row)

acc = pd.DataFrame(rows)
display(acc)
print("\npooled:", {k: round(float(acc[k].mean()), 3) for k in acc.columns if k.endswith("rmse")})
"""
)

md(
    r"""
## 7. Determinism

The search has no RNG, and dedup and ranking are defined over the whole candidate
list rather than per unit, so neither a rerun nor a different `cube_dim` may
change a single row.
"""
)

code(
    r"""
a = run_beam_batch(live, backend="cuda")["beam_mean"]
b = run_beam_batch(live, backend="cuda")["beam_mean"]
same_rerun = all(np.array_equal(x, y) for x, y in zip(a, b))
print("rerun bit-identical:", same_rerun)

c = run_beam_batch(live, backend="cuda", cube_dim=128)["beam_mean"]
same_dim = all(np.array_equal(x, y) for x, y in zip(a, c))
print("cube_dim 64 vs 128 bit-identical:", same_dim)
assert same_rerun and same_dim
"""
)

md(
    r"""
## 8. Throughput

The honest unit of work is *candidate-steps*: every MD step of every config
expands `5 * beam_size` candidates, and both of the kernel's collective passes
are quadratic in that count. Summed over the notebook's 14 configs that is 1035
candidates per row.
"""
)

code(
    r"""
CAND_PER_ROW = sum(5 * c[0] for c in BEAM_CONFIGS)
rows_gpu = sum(len(w["gr"]) for w in live)

steps_gpu = rows_gpu * CAND_PER_ROW
steps_np  = rows_np  * CAND_PER_ROW
per_row_np = t_np / rows_np

print(f"GPU   : {rows_gpu:>8} rows in {t_gpu:7.3f} s  -> {steps_gpu/t_gpu/1e9:6.2f} G candidate-steps/s")
print(f"numpy : {rows_np:>8} rows in {t_np:7.3f} s  -> {steps_np/t_np/1e9:6.4f} G candidate-steps/s")
print(f"\nspeedup on equal work: {(per_row_np * rows_gpu) / t_gpu:.0f}x")
print(f"numpy extrapolated to all {rows_gpu} rows: {per_row_np*rows_gpu/60:.1f} min")
"""
)

code(
    r"""
# Throughput still climbs with batch size, which is why the API takes every well
# at once: one well's 14 configs are 14 cubes and cannot fill a T4.
print(f"{'wells':>6}  {'rows':>8}  {'s':>8}  {'G cand-steps/s':>16}")
for k in [1, 2, 4, 8, 16, len(live)]:
    if k > len(live):
        continue
    sub = live[:k]
    run_beam_batch(sub, backend="cuda")           # warm
    t0 = time.time(); run_beam_batch(sub, backend="cuda"); dt = time.time() - t0
    r = sum(len(w["gr"]) for w in sub)
    print(f"{k:>6}  {r:>8}  {dt:>8.3f}  {r*CAND_PER_ROW/dt/1e9:>16.2f}")
"""
)

md(
    r"""
## 9. Summary
"""
)

code(
    r"""
SUMMARY = {
    "wells": len(live),
    "eval_rows": rows_gpu,
    "gpu_warm_s": round(t_gpu, 4),
    "gpu_cold_s": round(t_gpu_cold, 4),
    "numpy_s_for_all_rows_est": round(per_row_np * rows_gpu, 1),
    "speedup_x": round((per_row_np * rows_gpu) / t_gpu, 1),
    "worst_config_sample_agreement": round(worst_cfg, 6),
    "ensemble_agreement_rmse_ft": round(rmse_agree, 6),
    "deterministic": bool(same_rerun and same_dim),
}
if len(acc):
    SUMMARY["gpu_rmse_vs_truth"] = round(float(acc.gpu_rmse.mean()), 3)
    if "numpy_rmse" in acc:
        SUMMARY["numpy_rmse_vs_truth"] = round(float(acc.numpy_rmse.dropna().mean()), 3)
    SUMMARY["hold_last_rmse"] = round(float(acc.hold_last_rmse.mean()), 3)

import json as _json
print(_json.dumps(SUMMARY, indent=2))
Path("/kaggle/working/beam_summary.json").write_text(_json.dumps(SUMMARY, indent=2)) if Path("/kaggle/working").exists() else None
"""
)

md(
    r"""
## Using it from the competition notebook

`run_beam_ensemble_batch` returns the same arrays as the notebook's
`run_beam_ensemble`, so the per-well loop collapses into one batched call:

```python
from rog2_pf import run_beam_ensemble_batch

tvt_beams = run_beam_ensemble_batch([(hw, tw) for hw, tw in wells])
for wid, tvt_beam in zip(well_ids, tvt_beams):
    beam_map[wid] = tvt_beam        # apply_selector_variant() is unchanged
```

The single-well `run_beam_ensemble(hw, tw)` and single-config
`beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)` drop-ins exist too,
but prefer the batched form: one well's 14 configs are 14 cubes, nowhere near
enough to fill a T4.
"""
)

# ---------------------------------------------------------------------------

nb = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": text.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if kind == "code" else {}),
        }
        for kind, text in CELLS
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).with_name("rog2-beam-cubecl-t4.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out} ({len(CELLS)} cells)")
