# Defect-Induced Parrondo Drift in Noisy Discrete-Time Quantum Walks

This repository accompanies the manuscript:

**Defect-Induced Parrondo Drift Without Robust Directed Transport in Discrete-Time Quantum Walks Under Coin Dephasing: A Pre-Specified Atlas Study**  

Requires **Python 3.10+** (see `pyproject.toml`).

It provides:

- A compact, unit-tested Python implementation of a 1D single-qubit coined discrete-time quantum walk (DTQW)
  with an **origin phase defect** and **coin phase-damping (dephasing) noise**.
- A pre-specified **Stage-3 confirmatory atlas** over (φ, p) and paper-ready figures/tables.
- Fully reproducible run folders under `outputs/` (configs, manifests, logs, and `atlas.npz`).

---

## Model summary (matches the paper)

- **Lattice:** 1D line, positions x ∈ ℤ, coin basis |0⟩, |1⟩.
- **Shift convention:** |0⟩ → x+1 (right), |1⟩ → x−1 (left).
  - *Guardrail:* shift **does not wrap** around the array (no circular `np.roll`); boundaries are zero-filled.
- **Games (coins):** fixed SU(2) coins A and B using the Jan *et al.* angles (degrees):
  - **A:** (α, β, γ) = (137.2°, 29.4°, 52.1°)
  - **B:** (α, β, γ) = (149.6°, 67.4°, 132.5°)
- **Composite game:** periodic sequence `ABB` (period 3).
- **Defect:** origin phase defect D(φ) multiplying both coin amplitudes at x=0 by exp(iφ).
- **Noise:** coin-only **phase damping** (pure dephasing) with per-step strength p.
  - This project’s PD definition damps coin coherences as `rho_01 -> (1 - p) * rho_01` and leaves
    populations unchanged.
  - *Important:* this is **not** the Pauli‑Z phase-flip channel.
  - Kraus operators (coin-only) are:

    ```text
    K0 = sqrt(1-p) * I
    K1 = sqrt(p)   * |0><0|
    K2 = sqrt(p)   * |1><1|
    ```

- **Per-step operator order (confirmatory):** `coin → defect → coin-noise → shift` (see `mrc.yaml`).

The primary transport metric used in the paper is the late-time drift slope `v_fit`:

- `v_fit` = OLS slope of ⟨x⟩(t) vs t on the late window t ∈ [T0, T]

Localization diagnostics use late-time mass near the origin, e.g. `w_loc(|x| ≤ 3)`.

---

## What is included for reproduction

### Confirmatory Stage-3 run folder

The exact Stage-3 confirmatory run referenced in the manuscript is included at:

- `outputs/run_20260212_090815_stage3_pd_confirmatory/`

### Paper figure layers

Convenience copies of the Stage-3 figure layers referenced by the LaTeX manuscript are in:

- `figures/stage3/S3_*.png`

(These are also present under the Stage-3 run directory at `outputs/.../figures/`.)

### Paper-ready tables

Compact tables and a short insights report are in:

- `reports/` (CSV/MD)

---

## Stage-3 headline results (confirmatory atlas)

The Stage-3 grid is 72×31 (2232 points), with `T=300`, `T0=200`, `T1=250`, and `N_traj=400`
(see `mrc.yaml` and `reports/stage3_insights_report.md`).

Headline outcomes (drift-based strict predicates + localization thresholds):

- Strict eligibility points (both baselines decisively losing): **45** (all at p=0)
- Parrondo-positive directed transport points:
  - **Primary** transport threshold: **0**
  - **Sensitivity** threshold: **1** (isolated point at p=0)
- Maximum drift advantage: **adv_v ≈ 0.159182** at (φ/π, p) = (1/3, 0)
  (large drift reversal, but strongly localized near the defect)

---

## Quickstart

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies and the local package

From the repository root:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install --editable .
```

Optional (not required for the paper pipeline but good to have):

```bash
python -m pip install -r requirements-extra.txt
```

### 3) Run the unit tests

```bash
pytest
```

---

## Reproducing the Stage-3 paper artifacts

### Regenerate figures and paper tables from the included Stage-3 run folder

You can rebuild the figure layers and the `reports/` tables **without rerunning** the atlas simulation:

```bash
python scripts/stage3_postprocess_figures.py --run_dir outputs/run_20260212_090815_stage3_pd_confirmatory
python scripts/stage3_extract_insights.py --run_dir outputs/run_20260212_090815_stage3_pd_confirmatory
```

If you want to refresh the convenience copy used by LaTeX:

```powershell
# Windows PowerShell
Copy-Item outputs\run_20260212_090815_stage3_pd_confirmatory\figures\S3_*.png figures\stage3\
```

### Fully re-run the confirmatory Stage-3 atlas (CPU)

The locked confirmatory settings are in `mrc.yaml`. To recompute the atlas from scratch:

```bash
python scripts/stage3_confirmatory_pd_atlas_cpu.py --config mrc.yaml --jobs 8
python scripts/stage3_postprocess_figures.py --run_dir outputs/run_YYYYMMDD_HHMMSS_stage3_pd_confirmatory
python scripts/stage3_extract_insights.py --run_dir outputs/run_YYYYMMDD_HHMMSS_stage3_pd_confirmatory
```

Notes:

- The CPU script supports `--smoke` (tiny validation run) and `--resume --run_dir <existing>`
  (checkpointed restart). See `--help` for details.
- On Windows, the CPU script is the recommended entry point because it is spawn-safe and caps
  BLAS/OpenMP threads internally.

---

## Stage scripts

- **Stage 0** (Jan-style baseline replication, p=0, φ=0):
  - `scripts/stage0_jan_replication.py`
- **Stage 1** (defect-only φ scan at p=0):
  - `scripts/stage1_defect_phi_scan.py`
- **Stage 2** (exploratory coarse atlas):
  - `scripts/stage2_pd_atlas.py`
  - `scripts/stage2_postprocess_figures.py`
- **Stage 3** (confirmatory atlas):
  - `scripts/stage3_confirmatory_pd_atlas_cpu.py` (confirmatory atlas simulator; paper entry point)
  - `scripts/stage3_postprocess_figures.py`
  - `scripts/stage3_extract_insights.py`

All scripts accept `--help`.

Configuration:

- `dev.yaml` contains small, fast settings used for development stages.
- `mrc.yaml` is the paper-locked minimal reproducible configuration for the confirmatory atlas.

---

## Repository layout

- `src/dtqw/` — core DTQW implementation (coins, defect, shift, noise, metrics, IO)
- `src/tests/` — unit tests (pytest)
- `scripts/` — end-to-end pipelines (Stages 0–3) and postprocessing
- `outputs/` — captured reproducibility runs (configs, logs, manifests, `atlas.npz`)
- `figures/stage3/` — convenience copy of Stage-3 figure layers referenced by the paper
- `reports/` — compact Stage-3 tables + insights report (CSV/MD)
- `notes/` — internal readiness report(s)

---

## Reproducibility notes

Each run directory under `outputs/` contains:

- `config_used.yaml` (verbatim config snapshot)
- `manifest.json` (Python version, platform, package versions)
- `logs.txt`
- run-specific artifacts (`atlas.npz`, `metrics_summary.json`, figures, etc.)

The confirmatory atlas uses deterministic, parameter-indexed seeding (see `mrc.yaml` and the Stage-3
scripts) so that reruns are reproducible across platforms and multiprocessing.

---

## Citing

If you use this repository in academic work, please cite:

- the accompanying manuscript (title at the top of this README), and
- M. M. Jan *et al.*, “Experimental Realization of Parrondo’s Paradox in 1D Quantum Walks,”
  *Advanced Quantum Technologies* **3**, 1900127 (2020), DOI: 10.1002/qute.201900127.

--- 
