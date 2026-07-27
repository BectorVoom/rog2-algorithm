#!/usr/bin/env python3
"""Generates the Kaggle T4 verification/benchmark notebook.

Kept as a generator rather than a hand-edited .ipynb so the cells stay
reviewable in git.  Run `python kaggle/build_notebook.py` after editing.
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
# ROGII particle filter on a T4 — CubeCL/CUDA port

This notebook builds `rog2-pf` (the Rust/CubeCL port of the notebook's 128-seed
likelihood-weighted particle filter) against CUDA, then **verifies** and
**benchmarks** it against the original numba `_pf_lik_allseeds`.

Requirements: **GPU T4 accelerator**, **internet ON** (for `rustup` and crates.io),
and the competition data attached.

## What is being compared

The GPU port is not a bit-for-bit clone of the numba kernel — it cannot be. numba
draws from a single Mersenne Twister stream that particles consume *in order*,
which is inherently serial. The port uses a counter-based RNG keyed by
`(seed, particle slot, step)`, so it is reproducible across runs and devices but
draws different numbers.

Two independent Monte Carlo estimators of the same posterior therefore should not
match row-by-row; they should agree *statistically*. So this notebook checks:

1. **Accuracy** — RMSE against the true TVT on train wells. This is the number
   that matters; the GPU path must be as accurate as numba, not identical to it.
2. **Agreement** — RMSE between the two trajectories, which should be small next
   to the spread across seeds (`pf_pt_std`).
3. **Throughput** — wall-clock for the same work.
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
"""
)

md(
    r"""
## 3. The numba reference

Verbatim from `notebook/rogii-another-approch-2nd.ipynb` (cells 37 and 38): the
`_interp1` grid lookup and the `_pf_lik_allseeds` workhorse.
"""
)

code(
    r"""
import numpy as np, pandas as pd
from numba import njit

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True, nogil=True)
def _pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base,
                     MOM, VN, PN, RP, RR, RESAMP, init_spr):
    n = len(md_v); preds = np.empty((n_seeds, n)); liks = np.empty(n_seeds); tmax = vmin + len(gg)*step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
        for j in range(N):
            pos[j] = ls + init_spr*np.random.randn(); rate[j] = ir + 0.01*np.random.randn()
        log_lik = 0.0; prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0: dm = 1.0
            for j in range(N):
                rate[j] = MOM*rate[j] + VN*np.random.randn(); pos[j] += rate[j]*dm + PN*np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin-100.: tvt_j = vmin-100.
                if tvt_j > tmax+100.: tvt_j = tmax+100.
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs; dd = d*d
                if dd > 600.: dd = 600.
                lk = np.exp(-0.5*dd)
                if lk < 1e-300: lk = 1e-300
                avg_lk += w[j]*lk; w[j] = w[j]*lk
            if avg_lk < 1e-300: avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N): ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
            neff = 0.0
            for j in range(N): neff += w[j]*w[j]
            neff = 1.0/neff
            if neff < RESAMP*N:
                cum = np.empty(N); c = 0.0
                for j in range(N): c += w[j]; cum[j] = c
                u0 = np.random.uniform(0., 1./N); newpos = np.empty(N); newrate = np.empty(N); ci = 0
                for j in range(N):
                    u = u0 + j/N
                    while ci < N-1 and cum[ci] < u: ci += 1
                    newpos[j] = pos[ci] + RP*np.random.randn(); newrate[j] = rate[ci] + RR*np.random.randn()
                for j in range(N): pos[j] = newpos[j]; rate[j] = newrate[j]; w[j] = 1./N
            est = 0.0
            for j in range(N): est += w[j]*(pos[j]-z_v[i])
            preds[s, i] = est; prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks

def numba_lik_pf(well, n_particles, n_seeds, scales):
    # Runs the numba kernel on one `prepare_well()` dict, with the notebook's blend.
    preds, liks = _pf_lik_allseeds(
        well["md"].astype(np.float64), well["z"].astype(np.float64), well["gr"].astype(np.float64),
        well["grid"].astype(np.float64), well["vmin"], well["step"], well["gs"],
        well["ls"], well["ir"], n_particles, n_seeds, well.get("seed_base", 0),
        0.998, 0.002, 0.005, 0.1, 0.001, 0.5, well["init_spr"])
    ln = liks - liks.max(); out = {}
    for sc in scales:
        wts = np.exp(ln/float(sc)); wts /= wts.sum()
        out[f"pf_scale_{sc:g}"] = (wts[:, None]*preds).sum(0)
    out["pf_mean"] = preds.mean(0)
    return out, preds, liks

# Warm up the JIT so timings below measure the kernel, not compilation.
_w = np.linspace(1, 50, 20); _g = np.full(20, 50.); _gg = np.linspace(45, 55, 100)
_pf_lik_allseeds(_w, np.zeros(20), _g, _gg, 45., .1, 20., 50., 0., 64, 2, 0,
                 .998, .002, .005, .1, .001, .5, 4.5)
print("numba reference compiled")
"""
)

md(
    r"""
## 4. Load competition wells

Falls back to a synthetic well when the competition data is not attached, so the
notebook still runs end to end.
"""
)

code(
    r"""
from rog2_pf import prepare_well, lik_pf_batch, DEFAULT_SCALES

DATA = None
for c in ["/kaggle/input/rogii-wellbore-geology-prediction",
          "/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
    if Path(c, "train").exists():
        DATA = Path(c); break
print("data:", DATA)

N_WELLS = 32          # wells to verify/benchmark on
N_PARTICLES = 500     # CFG.PF_PARTICLES
N_SEEDS = 128         # CFG.PF_SEEDS
SCALES = list(DEFAULT_SCALES)

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
    # Synthetic stand-in with the same signal structure.
    rng = np.random.default_rng(0)
    for k in range(4):
        n, gl = 2000, 4000
        grid = np.clip(70 + 90*np.convolve(rng.normal(size=gl), np.ones(25)/25, "same"), 5, 160)
        tvt, rate, rows = -40.0, 0.0, []
        for i in range(n):
            rate = 0.995*rate + 0.006*rng.normal(); tvt += rate; rows.append(tvt)
        truth = np.array(rows)
        z = 12*np.sin(np.arange(n)/700)
        gr = np.interp((truth + 300)/0.2, np.arange(gl), grid) + 4*rng.normal(size=n)
        # A known prefix is required: `prepare_well` reads the GR sigma and the
        # initial rate off it, exactly as the competition wells provide.
        prefix = 50
        hw = pd.DataFrame({"MD": 9000+np.arange(n), "Z": z, "GR": gr,
                           "TVT_input": np.nan, "TVT": truth})
        hw.loc[:prefix-1, "TVT_input"] = truth[:prefix]
        tw = pd.DataFrame({"TVT": -300 + 0.2*np.arange(gl), "GR": grid})
        pairs.append((hw, tw)); names.append(f"synthetic-{k}"); truths.append(truth[prefix:])

print(f"{len(pairs)} wells")
prepared = [prepare_well(hw, tw)[0] for hw, tw in pairs]
keep = [i for i, w in enumerate(prepared) if w is not None]
print("eval rows total:", sum(len(prepared[i]["md"]) for i in keep))
"""
)

md(
    r"""
## 5. Run both implementations
"""
)

code(
    r"""
# CubeCL compiles the kernel through NVRTC on first use. That is a one-off cost
# per process, so time it separately instead of charging it to the batch.
t0 = time.time()
lik_pf_batch([pairs[keep[0]]], n_particles=N_PARTICLES, n_seeds=N_SEEDS,
             scales=SCALES, backend="cuda")
jit_time = time.time() - t0
print(f"first call (NVRTC compile + one well): {jit_time:.2f} s")

t0 = time.time()
gpu = lik_pf_batch([pairs[i] for i in keep], n_particles=N_PARTICLES, n_seeds=N_SEEDS,
                   scales=SCALES, backend="cuda", with_quality=True)
gpu_time = time.time() - t0
print(f"GPU (CubeCL/CUDA, warm): {gpu_time:.2f} s for {len(keep)} wells")

t0 = time.time()
cpu = [numba_lik_pf(prepared[i], N_PARTICLES, N_SEEDS, SCALES) for i in keep]
cpu_time = time.time() - t0
print(f"CPU (numba, single-threaded): {cpu_time:.2f} s")
print(f"speedup: {cpu_time/gpu_time:.1f}x")

steps = sum(len(prepared[i]["md"]) for i in keep) * N_PARTICLES * N_SEEDS
print(f"work: {steps/1e9:.2f} G particle-steps -> "
      f"GPU {steps/gpu_time/1e9:.2f} G/s, numba {steps/cpu_time/1e9:.3f} G/s")
"""
)

md(
    r"""
## 6. Verification
"""
)

code(
    r"""
def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a-b)**2)))

rows = []
for slot, i in enumerate(keep):
    g_out, ev_index, q = gpu[slot]
    c_out, _, _ = cpu[slot]
    r = {"well": names[i], "rows": len(ev_index),
         "seed_spread": float(np.mean(q["pf_pt_std"]))}
    for ch in ["pf_scale_3", "pf_scale_12", "pf_mean"]:
        r[f"agree_{ch}"] = rmse(g_out[ch], c_out[ch])
    if truths[i] is not None and len(truths[i]) == len(ev_index):
        r["gpu_vs_truth"] = rmse(g_out["pf_scale_3"], truths[i])
        r["numba_vs_truth"] = rmse(c_out["pf_scale_3"], truths[i])
    rows.append(r)

report = pd.DataFrame(rows)
print(report.to_string(index=False))
print()

SUMMARY = {
    "wells": len(keep),
    "n_particles": N_PARTICLES,
    "n_seeds": N_SEEDS,
    "eval_rows": int(sum(len(prepared[i]["md"]) for i in keep)),
    "gpu_seconds_warm": round(gpu_time, 3),
    "gpu_first_call_seconds": round(jit_time, 3),
    "numba_seconds": round(cpu_time, 3),
    "speedup": round(cpu_time / gpu_time, 2),
    "gpu_gsteps_per_s": round(steps / gpu_time / 1e9, 3),
    "numba_gsteps_per_s": round(steps / cpu_time / 1e9, 4),
    "data": "competition" if DATA is not None else "synthetic",
}

for ch in ["pf_scale_3", "pf_scale_12", "pf_mean"]:
    print(f"{ch}: median agreement RMSE {report[f'agree_{ch}'].median():.3f} ft")
print(f"median across-seed spread (pf_pt_std): {report['seed_spread'].median():.3f} ft")

for ch in ["pf_scale_3", "pf_scale_12", "pf_mean"]:
    SUMMARY[f"median_agreement_{ch}_ft"] = round(float(report[f"agree_{ch}"].median()), 4)
SUMMARY["median_seed_spread_ft"] = round(float(report["seed_spread"].median()), 4)

if "gpu_vs_truth" in report:
    g = np.sqrt((report.gpu_vs_truth**2 * report.rows).sum() / report.rows.sum())
    c = np.sqrt((report.numba_vs_truth**2 * report.rows).sum() / report.rows.sum())
    SUMMARY["pooled_rmse_gpu_ft"] = round(float(g), 4)
    SUMMARY["pooled_rmse_numba_ft"] = round(float(c), 4)
    print(f"\npooled RMSE vs true TVT: GPU {g:.3f} ft, numba {c:.3f} ft "
          f"({100*(g-c)/c:+.1f}%)")
    print("\nRow-by-row equality is NOT expected: different RNG streams. Read the "
          "pooled RMSEs as 'same accuracy to within Monte Carlo noise' — with a "
          "multimodal GR likelihood a single well can swing tens of feet on which "
          "branch a seed happens to lock onto, so per-well differences in either "
          "direction are branch luck, not a quality gap. Only a large well count "
          "makes that comparison meaningful.")
"""
)

md(
    r"""
## 7. Determinism

Re-running the GPU path with the same inputs must reproduce the previous result
bit for bit — the counter-based RNG has no state and the reduction order is fixed
by `cube_dim`.
"""
)

code(
    r"""
again = lik_pf_batch([pairs[i] for i in keep[:4]], n_particles=N_PARTICLES,
                     n_seeds=N_SEEDS, scales=SCALES, backend="cuda")
ok = all(np.array_equal(again[s][0]["pf_scale_3"], gpu[s][0]["pf_scale_3"])
         for s in range(len(again)))
print("bitwise reproducible:", ok)
SUMMARY["bitwise_reproducible"] = bool(ok)
assert ok
"""
)

md(
    r"""
## 8. Scaling

Throughput against batch size — a single well cannot fill a T4, which is why the
API is batched.
"""
)

code(
    r"""
sizes = [b for b in (1, 2, 4, 8, 16, 32) if b <= len(keep)]
scaling = []
for b in sizes:
    sub = [pairs[i] for i in keep[:b]]
    t0 = time.time()
    lik_pf_batch(sub, n_particles=N_PARTICLES, n_seeds=N_SEEDS, scales=SCALES, backend="cuda")
    dt = time.time() - t0
    s = sum(len(prepared[i]["md"]) for i in keep[:b]) * N_PARTICLES * N_SEEDS
    scaling.append({"wells": b, "seconds": round(dt, 3),
                    "G particle-steps/s": round(s/dt/1e9, 2)})
print(pd.DataFrame(scaling).to_string(index=False))
SUMMARY["scaling"] = scaling
"""
)

code(
    r"""
import json, shutil

with open(str(WORK / "summary.json"), "w") as fh:
    json.dump(SUMMARY, fh, indent=2)
print(json.dumps(SUMMARY, indent=2))

# The cargo build tree is hundreds of MB and Kaggle would keep it as notebook
# output; the wheel and the summary are the only artifacts worth retaining.
shutil.copy(wheel, str(WORK / Path(wheel).name))
shutil.rmtree(BUILD, ignore_errors=True)
print("kept:", sorted(p.name for p in WORK.iterdir()))
"""
)

md(
    r"""
## Using it in the competition notebook

`lik_pf_batch` returns the same `(out, ev_index, quality)` tuples as the original
`lik_pf`, so the call site changes from a per-well loop to one batched call:

```python
from rog2_pf import lik_pf_batch

results = lik_pf_batch([(hw, tw) for hw, tw in wells], with_quality=True)
for (out, ev_index, q), wid in zip(results, well_ids):
    likpf_map[wid] = (out, ev_index, q)   # feeds build_well() unchanged
```

Keep `seed_bases=[stable_hash(wid) for wid in well_ids]` if you want the per-well
deterministic seeding that `AGENTS.md` asks for; the default `0` reproduces the
notebook's shared-stream behaviour.
"""
)

# ---------------------------------------------------------------------------

nb = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": (src + "\n").splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        }
        for kind, src in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "rog2-pf-cubecl-t4.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out} ({len(CELLS)} cells)")
