#!/usr/bin/env python3
"""
Stage 3 (confirmatory) PD atlas - CPU-only, parallel over phi slices, Windows-safe.

Why this version is safer (prevents runaway spawn / instability on Windows):
  - All multiprocessing is under: if __name__ == "__main__"
  - Calls multiprocessing.freeze_support()
  - Uses spawn (Windows default), no nested pools, no global pool at import time

Why this version is faster (CPU-only):
  - Parallelizes over phi slices with ProcessPoolExecutor
  - Forces BLAS/OpenMP threads to 1 (can also be set in PowerShell)
  - Uses vectorized batched state updates across N_traj trajectories

Outputs (in outputs/run_*_stage3_pd_confirmatory/):
  atlas.npz
  metrics_summary.json
  config_used.yaml
  manifest.json
  logs.txt
  figures/ (created by separate postprocess script)

Run:
  python scripts/stage3_confirmatory_pd_atlas_cpu.py --config mrc.yaml --jobs 8
  python scripts/stage3_confirmatory_pd_atlas_cpu.py --config mrc.yaml --smoke

Notes:
  - Checkpointing: per-phi results are written to run_dir/partials/phi_###.npz.
    If you re-run with --resume --run_dir <existing>, completed phi slices are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from dtqw.io import load_yaml, make_run_dir, write_manifest, write_json, write_text
from dtqw.coins import su2_coin

# Default thread caps (PowerShell also sets these). Must be set before heavy numpy work.
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(k, "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
SM_CONST = np.uint64(0x9E3779B97F4A7C15)
SM_M1 = np.uint64(0xBF58476D1CE4E5B9)
SM_M2 = np.uint64(0x94D049BB133111EB)

def mix64(x: np.ndarray) -> np.ndarray:
    z = (x + SM_CONST) & MASK64
    z = (z ^ (z >> np.uint64(30))) * SM_M1 & MASK64
    z = (z ^ (z >> np.uint64(27))) * SM_M2 & MASK64
    z = (z ^ (z >> np.uint64(31))) & MASK64
    return z

class SplitMixRNG:
    def __init__(self, state_u64: np.ndarray):
        self.state = state_u64.astype(np.uint64, copy=True)

    def next_u64(self) -> np.ndarray:
        self.state = (self.state + SM_CONST) & MASK64
        z = self.state
        z = (z ^ (z >> np.uint64(30))) * SM_M1 & MASK64
        z = (z ^ (z >> np.uint64(27))) * SM_M2 & MASK64
        z = (z ^ (z >> np.uint64(31))) & MASK64
        return z

    def random(self) -> np.ndarray:
        r = self.next_u64()
        return ((r >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53)))

def seed_u64(base_seed: int, channel_id: int, seq_id: int, i: int, j: int, traj_ids: np.ndarray) -> np.ndarray:
    # Deterministic "hash-like" seed, consistent across processes and platforms.
    # uint64 overflow is intentional here (wraparound); suppress only that warning locally.
    with np.errstate(over="ignore"):
        x = np.uint64(base_seed)
        x = x ^ (np.uint64(channel_id) * np.uint64(0xD2B74407B1CE6E93))
        x = x ^ (np.uint64(seq_id) * np.uint64(0xCA5A826395121157))
        x = x ^ (np.uint64(i) * np.uint64(0x9E3779B97F4A7C15))
        x = x ^ (np.uint64(j) * np.uint64(0xBF58476D1CE4E5B9))
        x = x ^ (traj_ids.astype(np.uint64) * np.uint64(0x94D049BB133111EB))
    return mix64(x)

class PDWorkspace:
    def __init__(self, n_traj: int, npos: int):
        self.n_traj = n_traj
        self.npos = npos
        self.psi = np.zeros((n_traj, 2, npos), dtype=np.complex128)
        self.mid = np.zeros_like(self.psi)
        self.prob = np.zeros((n_traj, npos), dtype=np.float64)
        self.tmp = np.zeros((n_traj, npos), dtype=np.float64)
        self.P0 = np.zeros((n_traj,), dtype=np.float64)

def init_state(ws: PDWorkspace, idx0: int):
    ws.psi.fill(0.0)
    a = 1.0 / math.sqrt(2.0)
    ws.psi[:, 0, idx0] = a
    ws.psi[:, 1, idx0] = -1.0j * a

def apply_coin_all(ws: PDWorkspace, C: np.ndarray):
    a00, a01 = C[0, 0], C[0, 1]
    a10, a11 = C[1, 0], C[1, 1]
    psi0 = ws.psi[:, 0, :]
    psi1 = ws.psi[:, 1, :]
    out0 = ws.mid[:, 0, :]
    out1 = ws.mid[:, 1, :]
    out0[:] = a00 * psi0 + a01 * psi1
    out1[:] = a10 * psi0 + a11 * psi1

def apply_coin_freqmix(ws: PDWorkspace, C_A: np.ndarray, C_B: np.ndarray, chooseA: np.ndarray):
    # Compute B for all, overwrite A where chosen.
    b00, b01 = C_B[0, 0], C_B[0, 1]
    b10, b11 = C_B[1, 0], C_B[1, 1]
    a00, a01 = C_A[0, 0], C_A[0, 1]
    a10, a11 = C_A[1, 0], C_A[1, 1]
    psi0 = ws.psi[:, 0, :]
    psi1 = ws.psi[:, 1, :]
    out0 = ws.mid[:, 0, :]
    out1 = ws.mid[:, 1, :]

    out0[:] = b00 * psi0 + b01 * psi1
    out1[:] = b10 * psi0 + b11 * psi1
    if np.any(chooseA):
        out0[chooseA] = a00 * psi0[chooseA] + a01 * psi1[chooseA]
        out1[chooseA] = a10 * psi0[chooseA] + a11 * psi1[chooseA]

def apply_defect(ws: PDWorkspace, idx0: int, eiphi: complex):
    ws.mid[:, 0, idx0] *= eiphi
    ws.mid[:, 1, idx0] *= eiphi

def apply_pd_unravel(ws: PDWorkspace, p: float, rng: SplitMixRNG):
    if p <= 0.0:
        return
    if p >= 1.0:
        p = 1.0

    # Coin-0 population
    tmp = ws.tmp
    P0 = ws.P0
    np.square(ws.mid[:, 0, :].real, out=tmp)
    P0[:] = tmp.sum(axis=1)
    np.square(ws.mid[:, 0, :].imag, out=tmp)
    P0[:] += tmp.sum(axis=1)
    P0[:] = np.clip(P0, 0.0, 1.0)
    P1 = 1.0 - P0

    u = rng.random()
    q0 = 1.0 - p
    q1 = p * P0
    t1 = q0 + q1

    m1 = (u >= q0) & (u < t1)
    m2 = (u >= t1)

    eps = 1e-12
    if np.any(m1):
        denom = np.sqrt(np.maximum(P0, eps))
        ws.mid[m1, 0, :] /= denom[m1, None]
        ws.mid[m1, 1, :] = 0.0
    if np.any(m2):
        denom = np.sqrt(np.maximum(P1, eps))
        ws.mid[m2, 1, :] /= denom[m2, None]
        ws.mid[m2, 0, :] = 0.0

def apply_shift(ws: PDWorkspace):
    ws.psi.fill(0.0)
    ws.psi[:, 0, 1:] = ws.mid[:, 0, :-1]
    ws.psi[:, 1, :-1] = ws.mid[:, 1, 1:]

def measure_all(ws: PDWorkspace, idx0: int, x_float: np.ndarray, lo3: int, hi3: int, lo5: int, hi5: int):
    # prob = |psi0|^2 + |psi1|^2 using one temp buffer
    prob = ws.prob
    tmp = ws.tmp
    psi0 = ws.psi[:, 0, :]
    psi1 = ws.psi[:, 1, :]
    np.square(psi0.real, out=prob)
    np.square(psi0.imag, out=tmp); prob += tmp
    np.square(psi1.real, out=tmp); prob += tmp
    np.square(psi1.imag, out=tmp); prob += tmp

    x_mean = prob.dot(x_float)
    PR = prob[:, idx0 + 1 :].sum(axis=1)
    PL = prob[:, :idx0].sum(axis=1)
    deltaP = PR - PL
    w3 = prob[:, lo3:hi3].sum(axis=1)
    w5 = prob[:, lo5:hi5].sum(axis=1)
    P0 = prob[:, idx0].copy()
    return x_mean, deltaP, w3, w5, P0

def ols_slope(sum_x: np.ndarray, sum_tx: np.ndarray, n: int, sum_t: float, sum_tt: float) -> np.ndarray:
    denom = (n * sum_tt - sum_t * sum_t)
    return (n * sum_tx - sum_t * sum_x) / denom

def simulate_pd(ws: PDWorkspace, seq: str, C_A: np.ndarray, C_B: np.ndarray, phi: float, p_noise: float,
                T: int, T0: int, T1: int, idx0: int, x_float: np.ndarray,
                x0_primary: int, x0_sens: int,
                base_seed: int, channel_id: int, seq_id: int, i: int, j: int) -> dict:
    n_traj = ws.n_traj
    init_state(ws, idx0)
    eiphi = complex(math.cos(phi), math.sin(phi))

    lo3 = idx0 - x0_primary
    hi3 = idx0 + x0_primary + 1
    lo5 = idx0 - x0_sens
    hi5 = idx0 + x0_sens + 1

    traj_ids = np.arange(n_traj, dtype=np.uint64)
    rng = SplitMixRNG(seed_u64(base_seed, channel_id, seq_id, i, j, traj_ids))

    sum_x0 = np.zeros(n_traj, dtype=np.float64)
    sum_tx0 = np.zeros(n_traj, dtype=np.float64)
    sum_x1 = np.zeros(n_traj, dtype=np.float64)
    sum_tx1 = np.zeros(n_traj, dtype=np.float64)

    sum_w3 = np.zeros(n_traj, dtype=np.float64)
    sum_w5 = np.zeros(n_traj, dtype=np.float64)
    sum_P0 = np.zeros(n_traj, dtype=np.float64)
    sum_dP = np.zeros(n_traj, dtype=np.float64)
    late_count = 0

    times0 = np.arange(T0, T + 1, dtype=np.float64)
    n0 = int(times0.size)
    sum_t0 = float(times0.sum())
    sum_tt0 = float((times0 * times0).sum())

    times1 = np.arange(T1, T + 1, dtype=np.float64)
    n1 = int(times1.size)
    sum_t1 = float(times1.sum())
    sum_tt1 = float((times1 * times1).sum())

    x_late2 = np.zeros((n_traj, n1), dtype=np.float64)

    # t = 0 measurement
    x_mean, dP, w3, w5, P0 = measure_all(ws, idx0, x_float, lo3, hi3, lo5, hi5)
    if 0 >= T0:
        late_count += 1
        sum_w3 += w3; sum_w5 += w5; sum_P0 += P0; sum_dP += dP
    if 0 >= T0:
        sum_x0 += x_mean; sum_tx0 += 0.0 * x_mean
    if 0 >= T1:
        sum_x1 += x_mean; sum_tx1 += 0.0 * x_mean
        x_late2[:, 0] = x_mean

    # evolve
    pattern = seq
    pat_len = len(pattern)

    for step in range(T):
        t = step  # current time before applying step
        symbol = pattern[t % pat_len]
        C = C_A if symbol == "A" else C_B

        apply_coin_all(ws, C)
        apply_defect(ws, idx0, eiphi)
        apply_pd_unravel(ws, p_noise, rng)
        apply_shift(ws)

        t1 = step + 1
        x_mean, dP, w3, w5, P0 = measure_all(ws, idx0, x_float, lo3, hi3, lo5, hi5)

        if t1 >= T0:
            late_count += 1
            sum_w3 += w3; sum_w5 += w5; sum_P0 += P0; sum_dP += dP
            sum_x0 += x_mean
            sum_tx0 += float(t1) * x_mean
        if t1 >= T1:
            sum_x1 += x_mean
            sum_tx1 += float(t1) * x_mean
            x_late2[:, t1 - T1] = x_mean

    v_fit = ols_slope(sum_x0, sum_tx0, n0, sum_t0, sum_tt0)
    v_fit2 = ols_slope(sum_x1, sum_tx1, n1, sum_t1, sum_tt1)
    v_T = x_mean / float(T)

    # delta_v per plan: max_{t in [T1,T]} | vbar(t) - vbar(T) |
    tgrid = times1.copy()
    tgrid[tgrid < 1.0] = 1.0
    vbar = x_late2 / tgrid[None, :]
    dv = np.max(np.abs(vbar - v_T[:, None]), axis=1)

    w3_mean = sum_w3 / max(1, late_count)
    w5_mean = sum_w5 / max(1, late_count)
    P0bar = sum_P0 / max(1, late_count)
    dP_mean = sum_dP / max(1, late_count)

    return {
        "v_fit": v_fit,
        "v_fit2": v_fit2,
        "v_T": v_T,
        "delta_v": dv,
        "deltaP_late_mean": dP_mean,
        "w_loc_primary": w3_mean,
        "w_loc_sensitivity": w5_mean,
        "P0bar": P0bar,
    }

def simulate_freqmix(ws: PDWorkspace, C_A: np.ndarray, C_B: np.ndarray, phi: float, p_noise: float,
                     T: int, T0: int, T1: int, idx0: int, x_float: np.ndarray,
                     x0_primary: int, x0_sens: int, p_choose_A: float,
                     base_seed: int, channel_id: int, seq_id: int, i: int, j: int) -> dict:
    n_traj = ws.n_traj
    init_state(ws, idx0)
    eiphi = complex(math.cos(phi), math.sin(phi))

    lo3 = idx0 - x0_primary
    hi3 = idx0 + x0_primary + 1
    lo5 = idx0 - x0_sens
    hi5 = idx0 + x0_sens + 1

    traj_ids = np.arange(n_traj, dtype=np.uint64)
    rng = SplitMixRNG(seed_u64(base_seed, channel_id, seq_id, i, j, traj_ids))

    sum_x0 = np.zeros(n_traj, dtype=np.float64)
    sum_tx0 = np.zeros(n_traj, dtype=np.float64)
    sum_x1 = np.zeros(n_traj, dtype=np.float64)
    sum_tx1 = np.zeros(n_traj, dtype=np.float64)

    sum_w3 = np.zeros(n_traj, dtype=np.float64)
    sum_w5 = np.zeros(n_traj, dtype=np.float64)
    sum_P0 = np.zeros(n_traj, dtype=np.float64)
    sum_dP = np.zeros(n_traj, dtype=np.float64)
    late_count = 0

    times0 = np.arange(T0, T + 1, dtype=np.float64)
    n0 = int(times0.size)
    sum_t0 = float(times0.sum())
    sum_tt0 = float((times0 * times0).sum())

    times1 = np.arange(T1, T + 1, dtype=np.float64)
    n1 = int(times1.size)
    sum_t1 = float(times1.sum())
    sum_tt1 = float((times1 * times1).sum())

    x_late2 = np.zeros((n_traj, n1), dtype=np.float64)

    x_mean, dP, w3, w5, P0 = measure_all(ws, idx0, x_float, lo3, hi3, lo5, hi5)
    if 0 >= T0:
        late_count += 1
        sum_w3 += w3; sum_w5 += w5; sum_P0 += P0; sum_dP += dP
        sum_x0 += x_mean; sum_tx0 += 0.0 * x_mean
    if 0 >= T1:
        sum_x1 += x_mean; sum_tx1 += 0.0 * x_mean
        x_late2[:, 0] = x_mean

    for step in range(T):
        t1 = step + 1

        chooseA = (rng.random() < p_choose_A)
        apply_coin_freqmix(ws, C_A, C_B, chooseA)
        apply_defect(ws, idx0, eiphi)
        apply_pd_unravel(ws, p_noise, rng)
        apply_shift(ws)

        x_mean, dP, w3, w5, P0 = measure_all(ws, idx0, x_float, lo3, hi3, lo5, hi5)

        if t1 >= T0:
            late_count += 1
            sum_w3 += w3; sum_w5 += w5; sum_P0 += P0; sum_dP += dP
            sum_x0 += x_mean
            sum_tx0 += float(t1) * x_mean
        if t1 >= T1:
            sum_x1 += x_mean
            sum_tx1 += float(t1) * x_mean
            x_late2[:, t1 - T1] = x_mean

    v_fit = ols_slope(sum_x0, sum_tx0, n0, sum_t0, sum_tt0)
    v_fit2 = ols_slope(sum_x1, sum_tx1, n1, sum_t1, sum_tt1)
    v_T = x_mean / float(T)

    tgrid = times1.copy()
    tgrid[tgrid < 1.0] = 1.0
    vbar = x_late2 / tgrid[None, :]
    dv = np.max(np.abs(vbar - v_T[:, None]), axis=1)

    w3_mean = sum_w3 / max(1, late_count)
    w5_mean = sum_w5 / max(1, late_count)
    P0bar = sum_P0 / max(1, late_count)
    dP_mean = sum_dP / max(1, late_count)

    return {
        "v_fit": v_fit,
        "v_fit2": v_fit2,
        "v_T": v_T,
        "delta_v": dv,
        "deltaP_late_mean": dP_mean,
        "w_loc_primary": w3_mean,
        "w_loc_sensitivity": w5_mean,
        "P0bar": P0bar,
    }

def simulate_zcp(C_A: np.ndarray, C_B: np.ndarray, T: int, T0: int, T1: int, x0_primary: int, x0_sens: int,
                 base_seed: int, channel_id: int, seq_id: int, i: int, j: int, n_traj: int) -> dict:
    traj_ids = np.arange(n_traj, dtype=np.uint64)
    rng = SplitMixRNG(seed_u64(base_seed, channel_id, seq_id, i, j, traj_ids))

    pos = np.zeros(n_traj, dtype=np.int32)
    basis = np.full(n_traj, -1, dtype=np.int8)  # -1 means "init coin vector"

    init_coin = np.array([1.0 / math.sqrt(2.0), -1.0j / math.sqrt(2.0)], dtype=np.complex128)

    sum_x0 = np.zeros(n_traj, dtype=np.float64)
    sum_tx0 = np.zeros(n_traj, dtype=np.float64)
    sum_x1 = np.zeros(n_traj, dtype=np.float64)
    sum_tx1 = np.zeros(n_traj, dtype=np.float64)

    sum_w3 = np.zeros(n_traj, dtype=np.float64)
    sum_w5 = np.zeros(n_traj, dtype=np.float64)
    sum_P0 = np.zeros(n_traj, dtype=np.float64)
    sum_dP = np.zeros(n_traj, dtype=np.float64)
    late_count = 0

    times0 = np.arange(T0, T + 1, dtype=np.float64)
    n0 = int(times0.size)
    sum_t0 = float(times0.sum())
    sum_tt0 = float((times0 * times0).sum())

    times1 = np.arange(T1, T + 1, dtype=np.float64)
    n1 = int(times1.size)
    sum_t1 = float(times1.sum())
    sum_tt1 = float((times1 * times1).sum())

    x_late2 = np.zeros((n_traj, n1), dtype=np.float64)

    def deltaP_from_pos(p: np.ndarray) -> np.ndarray:
        out = np.zeros_like(p, dtype=np.float64)
        out[p > 0] = 1.0
        out[p < 0] = -1.0
        return out

    # t=0
    x_mean = pos.astype(np.float64)
    if 0 >= T0:
        late_count += 1
        sum_w3 += (np.abs(pos) <= x0_primary).astype(np.float64)
        sum_w5 += (np.abs(pos) <= x0_sens).astype(np.float64)
        sum_P0 += (pos == 0).astype(np.float64)
        sum_dP += deltaP_from_pos(pos)
        sum_x0 += x_mean
    if 0 >= T1:
        sum_x1 += x_mean
        x_late2[:, 0] = x_mean

    pattern = "ABB"
    pat_len = len(pattern)

    # precompute coin column probabilities
    def col_probs(C: np.ndarray):
        # Prob(coin=0 | basis=0) = |C00|^2; Prob(coin=0 | basis=1) = |C01|^2
        p00 = float((C[0, 0].conjugate() * C[0, 0]).real)
        p01 = float((C[0, 1].conjugate() * C[0, 1]).real)
        return p00, p01

    pA00, pA01 = col_probs(C_A)
    pB00, pB01 = col_probs(C_B)

    for step in range(T):
        t = step
        sym = pattern[t % pat_len]
        if sym == "A":
            p00, p01 = pA00, pA01
        else:
            p00, p01 = pB00, pB01

        u = rng.random()
        if step == 0:
            # first step uses init coin vector
            amp = (C_A if sym == "A" else C_B) @ init_coin
            p0 = float((amp[0].conjugate() * amp[0]).real)
            p0 = min(max(p0, 0.0), 1.0)
            basis[:] = (u >= p0).astype(np.int8)  # 0 if u<p0 else 1
        else:
            p0 = np.where(basis == 0, p00, p01)
            basis[:] = (u >= p0).astype(np.int8)

        # shift
        pos += (1 - 2 * basis.astype(np.int32))

        t1 = step + 1
        x_mean = pos.astype(np.float64)

        if t1 >= T0:
            late_count += 1
            sum_w3 += (np.abs(pos) <= x0_primary).astype(np.float64)
            sum_w5 += (np.abs(pos) <= x0_sens).astype(np.float64)
            sum_P0 += (pos == 0).astype(np.float64)
            sum_dP += deltaP_from_pos(pos)
            sum_x0 += x_mean
            sum_tx0 += float(t1) * x_mean
        if t1 >= T1:
            sum_x1 += x_mean
            sum_tx1 += float(t1) * x_mean
            x_late2[:, t1 - T1] = x_mean

    v_fit = ols_slope(sum_x0, sum_tx0, n0, sum_t0, sum_tt0)
    v_fit2 = ols_slope(sum_x1, sum_tx1, n1, sum_t1, sum_tt1)
    v_T = x_mean / float(T)

    tgrid = times1.copy()
    tgrid[tgrid < 1.0] = 1.0
    vbar = x_late2 / tgrid[None, :]
    dv = np.max(np.abs(vbar - v_T[:, None]), axis=1)

    w3_mean = sum_w3 / max(1, late_count)
    w5_mean = sum_w5 / max(1, late_count)
    P0bar = sum_P0 / max(1, late_count)
    dP_mean = sum_dP / max(1, late_count)

    return {
        "v_fit": v_fit,
        "v_fit2": v_fit2,
        "v_T": v_T,
        "delta_v": dv,
        "deltaP_late_mean": dP_mean,
        "w_loc_primary": w3_mean,
        "w_loc_sensitivity": w5_mean,
        "P0bar": P0bar,
    }

def bootstrap_ci_mean(values: np.ndarray, boot_idx: np.ndarray, alpha: float) -> tuple[float, float]:
    means = values[boot_idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi

def connected_cluster_sizes(mask: np.ndarray) -> list[int]:
    n_phi, n_p = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []

    def neigh(ii: int, jj: int):
        yield ((ii - 1) % n_phi, jj)
        yield ((ii + 1) % n_phi, jj)
        if jj - 1 >= 0:
            yield (ii, jj - 1)
        if jj + 1 < n_p:
            yield (ii, jj + 1)

    for i in range(n_phi):
        for j in range(n_p):
            if (not mask[i, j]) or visited[i, j]:
                continue
            q = [(i, j)]
            visited[i, j] = True
            count = 0
            while q:
                ci, cj = q.pop()
                count += 1
                for ni, nj in neigh(ci, cj):
                    if mask[ni, nj] and (not visited[ni, nj]):
                        visited[ni, nj] = True
                        q.append((ni, nj))
            sizes.append(count)

    sizes.sort(reverse=True)
    return sizes

# Globals set in worker initializer
G = {}

def _init_worker(cfg: dict):
    global G
    G = cfg

def _phi_partial_path(run_dir: Path, i: int) -> Path:
    return run_dir / "partials" / f"phi_{i:03d}.npz"

def _write_partial_atomic(path: Path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)

def compute_phi_slice(i: int) -> dict:
    run_dir = Path(G["run_dir"])
    resume = bool(G["resume"])
    partial_path = _phi_partial_path(run_dir, i)

    if resume and partial_path.exists():
        d = np.load(partial_path, allow_pickle=False)
        out = {k: d[k] for k in d.files}
        out["i"] = i
        return out

    phi = G["phi"]
    pgrid = G["p"]
    C_A = G["C_A"]
    C_B = G["C_B"]
    base_seed = int(G["base_seed"])
    T = int(G["T"])
    T0 = int(G["T0"])
    T1 = int(G["T1"])
    pad_m = int(G["pad_m"])
    x0_primary = int(G["x0_primary"])
    x0_sens = int(G["x0_sens"])
    n_traj = int(G["N_traj"])
    boot_idx = G["boot_idx"]
    alpha = float(G["alpha"])

    L = T + pad_m
    idx0 = L
    x_float = np.arange(-L, L + 1, dtype=np.float64)

    ws = PDWorkspace(n_traj, x_float.size)

    Np = pgrid.size

    def zf():
        return np.zeros((Np,), dtype=np.float64)

    # A
    vA = zf(); vA_lo = zf(); vA_hi = zf()
    vA2 = zf(); vAT = zf(); dVA = zf()
    # B
    vB = zf(); vB_lo = zf(); vB_hi = zf()
    vB2 = zf(); vBT = zf(); dVB = zf()
    # ABB
    vABB = zf(); vABB_lo = zf(); vABB_hi = zf()
    vABB2 = zf(); vABBT = zf(); dVABB = zf()
    dPABB = zf(); w3ABB = zf(); w5ABB = zf(); P0ABB = zf()

    # Nulls (only combined)
    n1_v = zf(); n1_v_std = zf(); n1_dp = zf(); n1_dp_std = zf()
    n2_v = zf(); n2_v_std = zf(); n2_dp = zf(); n2_dp_std = zf()

    phi_val = float(phi[i])

    for j in range(Np):
        p_val = float(pgrid[j])

        # PD main sequences
        A = simulate_pd(ws, "A", C_A, C_B, phi_val, p_val, T, T0, T1, idx0, x_float, x0_primary, x0_sens,
                        base_seed, 1, 1, i, j)
        vA[j] = float(A["v_fit"].mean())
        vA2[j] = float(A["v_fit2"].mean())
        vAT[j] = float(A["v_T"].mean())
        dVA[j] = float(A["delta_v"].mean())
        lo, hi = bootstrap_ci_mean(A["v_fit"], boot_idx, alpha)
        vA_lo[j] = lo; vA_hi[j] = hi

        B = simulate_pd(ws, "B", C_A, C_B, phi_val, p_val, T, T0, T1, idx0, x_float, x0_primary, x0_sens,
                        base_seed, 1, 2, i, j)
        vB[j] = float(B["v_fit"].mean())
        vB2[j] = float(B["v_fit2"].mean())
        vBT[j] = float(B["v_T"].mean())
        dVB[j] = float(B["delta_v"].mean())
        lo, hi = bootstrap_ci_mean(B["v_fit"], boot_idx, alpha)
        vB_lo[j] = lo; vB_hi[j] = hi

        ABB = simulate_pd(ws, "ABB", C_A, C_B, phi_val, p_val, T, T0, T1, idx0, x_float, x0_primary, x0_sens,
                          base_seed, 1, 3, i, j)
        vABB[j] = float(ABB["v_fit"].mean())
        vABB2[j] = float(ABB["v_fit2"].mean())
        vABBT[j] = float(ABB["v_T"].mean())
        dVABB[j] = float(ABB["delta_v"].mean())
        dPABB[j] = float(ABB["deltaP_late_mean"].mean())
        w3ABB[j] = float(ABB["w_loc_primary"].mean())
        w5ABB[j] = float(ABB["w_loc_sensitivity"].mean())
        P0ABB[j] = float(ABB["P0bar"].mean())
        lo, hi = bootstrap_ci_mean(ABB["v_fit"], boot_idx, alpha)
        vABB_lo[j] = lo; vABB_hi[j] = hi

        # Null N1: frequency-matched random mixture (A freq=1/3, B freq=2/3)
        N1 = simulate_freqmix(ws, C_A, C_B, phi_val, p_val, T, T0, T1, idx0, x_float, x0_primary, x0_sens,
                              1.0 / 3.0, base_seed, 2, 4, i, j)
        n1_v[j] = float(N1["v_fit"].mean())
        n1_v_std[j] = float(N1["v_fit"].std(ddof=1))
        n1_dp[j] = float(N1["deltaP_late_mean"].mean())
        n1_dp_std[j] = float(N1["deltaP_late_mean"].std(ddof=1))

        # Null N2: Z_cp (complete classicalization), computed per gridpoint for deterministic seeding
        N2 = simulate_zcp(C_A, C_B, T, T0, T1, x0_primary, x0_sens, base_seed, 3, 5, i, j, n_traj)
        n2_v[j] = float(N2["v_fit"].mean())
        n2_v_std[j] = float(N2["v_fit"].std(ddof=1))
        n2_dp[j] = float(N2["deltaP_late_mean"].mean())
        n2_dp_std[j] = float(N2["deltaP_late_mean"].std(ddof=1))

    run_dir.joinpath("partials").mkdir(parents=True, exist_ok=True)
    _write_partial_atomic(
        partial_path,
        vA=vA, vA_lo=vA_lo, vA_hi=vA_hi, vA2=vA2, vAT=vAT, dVA=dVA,
        vB=vB, vB_lo=vB_lo, vB_hi=vB_hi, vB2=vB2, vBT=vBT, dVB=dVB,
        vABB=vABB, vABB_lo=vABB_lo, vABB_hi=vABB_hi, vABB2=vABB2, vABBT=vABBT, dVABB=dVABB,
        dPABB=dPABB, w3ABB=w3ABB, w5ABB=w5ABB, P0ABB=P0ABB,
        n1_v=n1_v, n1_v_std=n1_v_std, n1_dp=n1_dp, n1_dp_std=n1_dp_std,
        n2_v=n2_v, n2_v_std=n2_v_std, n2_dp=n2_dp, n2_dp_std=n2_dp_std,
    )

    return {
        "i": i,
        "vA": vA, "vA_lo": vA_lo, "vA_hi": vA_hi, "vA2": vA2, "vAT": vAT, "dVA": dVA,
        "vB": vB, "vB_lo": vB_lo, "vB_hi": vB_hi, "vB2": vB2, "vBT": vBT, "dVB": dVB,
        "vABB": vABB, "vABB_lo": vABB_lo, "vABB_hi": vABB_hi, "vABB2": vABB2, "vABBT": vABBT, "dVABB": dVABB,
        "dPABB": dPABB, "w3ABB": w3ABB, "w5ABB": w5ABB, "P0ABB": P0ABB,
        "n1_v": n1_v, "n1_v_std": n1_v_std, "n1_dp": n1_dp, "n1_dp_std": n1_dp_std,
        "n2_v": n2_v, "n2_v_std": n2_v_std, "n2_dp": n2_dp, "n2_dp_std": n2_dp_std,
    }

def stability_mask(v: np.ndarray, v2: np.ndarray, vT: np.ndarray, dv: np.ndarray, rel_tol: float = 0.5) -> np.ndarray:
    eps = 1e-12
    denom = np.maximum(np.abs(v), eps)
    sign_ok = (np.sign(v) == np.sign(v2)) & (np.sign(v) == np.sign(vT)) & (np.sign(v) != 0)
    rel_ok = (np.abs(v2 - v) / denom <= rel_tol) & (np.abs(vT - v) / denom <= rel_tol) & (dv / denom <= rel_tol)
    return sign_ok & rel_ok

def sign_eps(x: np.ndarray, eps: float) -> np.ndarray:
    s = np.zeros_like(x, dtype=np.int8)
    s[x > eps] = 1
    s[x < -eps] = -1
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mrc.yaml")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_yaml(cfg_path)

    # Validate step_order
    expected_order = "coin -> defect -> coin-noise -> shift"
    step_order = str(cfg.get("step_order", "")).strip()
    if step_order != expected_order:
        raise ValueError(f"Unexpected step_order: {step_order!r} (expected {expected_order!r})")

    coins_cfg = cfg["coins"]
    A_angles = (float(coins_cfg["A"]["alpha_deg"]), float(coins_cfg["A"]["beta_deg"]), float(coins_cfg["A"]["gamma_deg"]))
    B_angles = (float(coins_cfg["B"]["alpha_deg"]), float(coins_cfg["B"]["beta_deg"]), float(coins_cfg["B"]["gamma_deg"]))

    stage3 = cfg["stage3_grid"]
    N_phi = int(stage3["N_phi"])
    N_p = int(stage3["N_p"])

    sim = cfg["simulation"]
    T = int(sim["horizons"]["T_long"])
    T0 = int(sim["late_windows"]["T0"])
    T1 = int(sim["late_windows"]["T1"])
    pad_m = int(sim["lattice"].get("padded_margin_m", 0))

    loc = sim["localization_overlay"]
    x0_primary = int(loc["x0_primary"])
    x0_sens = int(loc["x0_sensitivity"])
    w_thr_primary = float(loc["w_thr_primary"])
    w_thr_sens = float(loc["w_thr_sensitivity"])

    se = cfg["stochastic_estimation"]
    N_traj = int(se["N_traj"])
    B = int(se["bootstrap"]["B"])
    alpha = float(se["bootstrap"]["alpha"])
    base_seed = int(se["rng_policy"]["base_seed"])

    if args.smoke:
        N_phi = 6
        N_p = 7
        T = 40
        T0 = 20
        T1 = 30
        N_traj = 60
        B = 200
        pad_m = 1

    # Grids
    phi = np.linspace(0.0, 2.0 * math.pi, N_phi, endpoint=False, dtype=np.float64)
    pgrid = np.linspace(0.0, 1.0, N_p, endpoint=True, dtype=np.float64)

    # Coins
    C_A = su2_coin(*A_angles, degrees=True)
    C_B = su2_coin(*B_angles, degrees=True)

    # Run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
    else:
        paths = make_run_dir(tag="stage3_pd_confirmatory")
        run_dir = paths["run_dir"]
        fig_dir = paths["fig_dir"]

    # Save config + manifest
    write_text(run_dir / "config_used.yaml", cfg_path.read_text(encoding="utf-8", errors="ignore"))
    write_manifest(run_dir, tag="stage3_pd_confirmatory")

    logs = []
    def log(msg: str):
        print(msg, flush=True)
        logs.append(msg)

    jobs = args.jobs if args.jobs and args.jobs > 0 else max(1, (os.cpu_count() or 1) - 2)

    log("Stage3 PD confirmatory (CPU-only, parallel)")
    log(f"run_dir={run_dir}")
    log(f"config={cfg_path}")
    log(f"N_phi={N_phi}, N_p={N_p}, total={N_phi*N_p}")
    log(f"T={T}, T0={T0}, T1={T1}, N_traj={N_traj}, B={B}, alpha={alpha}, pad_m={pad_m}")
    log(f"jobs={jobs}")
    log(f"w_thr_primary={w_thr_primary} (x0_primary={x0_primary}), w_thr_sensitivity={w_thr_sens} (x0_sens={x0_sens})")
    log(f"resume={bool(args.resume)}")

    # Global bootstrap indices (deterministic)
    boot_seed = int(mix64(np.array([np.uint64(base_seed ^ 0xABCDEF01)], dtype=np.uint64))[0] & np.uint64(0xFFFFFFFF))
    boot_rng = np.random.default_rng(boot_seed)
    boot_idx = boot_rng.integers(0, N_traj, size=(B, N_traj), dtype=np.int32)

    # Allocate arrays
    shape = (N_phi, N_p)
    zf = lambda: np.zeros(shape, dtype=np.float64)

    vA = zf(); vA_lo = zf(); vA_hi = zf(); vA2 = zf(); vAT = zf(); dVA = zf()
    vB = zf(); vB_lo = zf(); vB_hi = zf(); vB2 = zf(); vBT = zf(); dVB = zf()
    vABB = zf(); vABB_lo = zf(); vABB_hi = zf(); vABB2 = zf(); vABBT = zf(); dVABB = zf()
    dPABB = zf(); w3ABB = zf(); w5ABB = zf(); P0ABB = zf()

    n1_v = zf(); n1_v_std = zf(); n1_dp = zf(); n1_dp_std = zf()
    n2_v = zf(); n2_v_std = zf(); n2_dp = zf(); n2_dp_std = zf()

    # Dispatch work
    cfg_pack = {
        "run_dir": str(run_dir),
        "resume": bool(args.resume),
        "phi": phi,
        "p": pgrid,
        "C_A": C_A,
        "C_B": C_B,
        "base_seed": base_seed,
        "T": T,
        "T0": T0,
        "T1": T1,
        "pad_m": pad_m,
        "x0_primary": x0_primary,
        "x0_sens": x0_sens,
        "N_traj": N_traj,
        "boot_idx": boot_idx,
        "alpha": alpha,
    }

    t_start = time.time()
    futures = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(cfg_pack,)) as ex:
        for i in range(N_phi):
            futures.append(ex.submit(compute_phi_slice, i))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="phi slices"):
            out = fut.result()
            i = int(out["i"])
            vA[i, :] = out["vA"]; vA_lo[i, :] = out["vA_lo"]; vA_hi[i, :] = out["vA_hi"]; vA2[i, :] = out["vA2"]; vAT[i, :] = out["vAT"]; dVA[i, :] = out["dVA"]
            vB[i, :] = out["vB"]; vB_lo[i, :] = out["vB_lo"]; vB_hi[i, :] = out["vB_hi"]; vB2[i, :] = out["vB2"]; vBT[i, :] = out["vBT"]; dVB[i, :] = out["dVB"]
            vABB[i, :] = out["vABB"]; vABB_lo[i, :] = out["vABB_lo"]; vABB_hi[i, :] = out["vABB_hi"]; vABB2[i, :] = out["vABB2"]; vABBT[i, :] = out["vABBT"]; dVABB[i, :] = out["dVABB"]
            dPABB[i, :] = out["dPABB"]; w3ABB[i, :] = out["w3ABB"]; w5ABB[i, :] = out["w5ABB"]; P0ABB[i, :] = out["P0ABB"]
            n1_v[i, :] = out["n1_v"]; n1_v_std[i, :] = out["n1_v_std"]; n1_dp[i, :] = out["n1_dp"]; n1_dp_std[i, :] = out["n1_dp_std"]
            n2_v[i, :] = out["n2_v"]; n2_v_std[i, :] = out["n2_v_std"]; n2_dp[i, :] = out["n2_dp"]; n2_dp_std[i, :] = out["n2_dp_std"]

    elapsed = time.time() - t_start
    log(f"Simulation finished in {elapsed:.2f} s")

    # Floors / dead-zones from null variability
    v_min_n1 = float(np.quantile(np.abs(n1_v), 0.95))
    v_min_n2 = float(np.quantile(np.abs(n2_v), 0.95))
    v_min = float(max(v_min_n1, v_min_n2))

    z = 1.96
    eps_v_n1 = float(np.quantile(z * (n1_v_std / math.sqrt(N_traj)), 0.95))
    eps_v_n2 = float(np.quantile(z * (n2_v_std / math.sqrt(N_traj)), 0.95))
    eps_P_n1 = float(np.quantile(z * (n1_dp_std / math.sqrt(N_traj)), 0.95))
    eps_P_n2 = float(np.quantile(z * (n2_dp_std / math.sqrt(N_traj)), 0.95))
    eps_v = float(max(eps_v_n1, eps_v_n2))
    eps_P = float(max(eps_P_n1, eps_P_n2))

    log(f"Derived v_min: n1={v_min_n1:.6g}, n2={v_min_n2:.6g}, v_min={v_min:.6g}")
    log(f"Derived eps_v: n1={eps_v_n1:.6g}, n2={eps_v_n2:.6g}, eps_v={eps_v:.6g}")
    log(f"Derived eps_P: n1={eps_P_n1:.6g}, n2={eps_P_n2:.6g}, eps_P={eps_P:.6g}")

    # Stability guardrails
    stA = stability_mask(vA, vA2, vAT, dVA)
    stB = stability_mask(vB, vB2, vBT, dVB)
    stABB = stability_mask(vABB, vABB2, vABBT, dVABB)

    # Decisive win/lose rules (CI excludes zero + magnitude floor + stability)
    loseA = (vA_hi < 0.0) & (np.abs(vA) > v_min) & stA
    loseB = (vB_hi < 0.0) & (np.abs(vB) > v_min) & stB
    winABB = (vABB_lo > 0.0) & (vABB > v_min) & stABB

    strict = loseA & loseB
    dt_primary = (w3ABB < w_thr_primary)
    dt_sens = (w5ABB < w_thr_sens)

    parr_primary = strict & winABB & dt_primary
    parr_sens = strict & winABB & dt_sens

    adv = vABB - np.maximum(vA, vB)

    # Sign mismatch with dead-zones
    sv = sign_eps(vABB, eps_v)
    sp = sign_eps(dPABB, eps_P)
    mismatch = (sv * sp == -1)
    mismatch_denom = int(((sv != 0) & (sp != 0)).sum())

    cl_primary = connected_cluster_sizes(parr_primary)
    cl_sens = connected_cluster_sizes(parr_sens)

    Smax_primary = int(cl_primary[0]) if cl_primary else 0
    Smax_sens = int(cl_sens[0]) if cl_sens else 0

    idx = np.unravel_index(int(np.argmax(adv)), adv.shape)
    i_max = int(idx[0]); j_max = int(idx[1])

    summary = {
        "stage": "stage3_pd_confirmatory",
        "run_dir": str(run_dir),
        "config_path": str(cfg_path),
        "grid": {"N_phi": int(N_phi), "N_p": int(N_p)},
        "sim": {"T": int(T), "T0": int(T0), "T1": int(T1), "N_traj": int(N_traj), "pad_m": int(pad_m), "step_order": step_order, "base_seed": int(base_seed)},
        "bootstrap": {"B": int(B), "alpha": float(alpha), "ci": "percentile CI on mean(v_fit)"},
        "thresholds": {
            "w_thr_primary": float(w_thr_primary),
            "w_thr_sensitivity": float(w_thr_sens),
            "x0_primary": int(x0_primary),
            "x0_sensitivity": int(x0_sens),
            "v_min": float(v_min),
            "eps_v": float(eps_v),
            "eps_P": float(eps_P),
        },
        "counts": {
            "strict_mask_points": int(strict.sum()),
            "directed_transport_points_primary": int(dt_primary.sum()),
            "directed_transport_points_sensitivity": int(dt_sens.sum()),
            "parrondo_positive_points_primary": int(parr_primary.sum()),
            "parrondo_positive_points_sensitivity": int(parr_sens.sum()),
            "mismatch_sign_points": int(mismatch.sum()),
            "mismatch_sign_denominator": int(mismatch_denom),
        },
        "clusters": {
            "primary_cluster_sizes": cl_primary[:20],
            "primary_S_max": Smax_primary,
            "sensitivity_cluster_sizes": cl_sens[:20],
            "sensitivity_S_max": Smax_sens,
        },
        "maxima": {
            "adv_v_max_ABB": {
                "value": float(adv[i_max, j_max]),
                "phi_index": i_max,
                "p_index": j_max,
                "phi": float(phi[i_max]),
                "phi_over_pi": float(phi[i_max] / math.pi),
                "p": float(pgrid[j_max]),
                "v_fit_A": float(vA[i_max, j_max]),
                "v_fit_B": float(vB[i_max, j_max]),
                "v_fit_ABB": float(vABB[i_max, j_max]),
            }
        },
        "timing_seconds": float(elapsed),
    }

    atlas_path = run_dir / "atlas.npz"
    np.savez_compressed(
        atlas_path,
        phi=phi,
        p=pgrid,
        v_fit_A=vA, v_fit_A_ci_lo=vA_lo, v_fit_A_ci_hi=vA_hi, v_fit2_A=vA2, v_T_A=vAT, delta_v_A=dVA,
        v_fit_B=vB, v_fit_B_ci_lo=vB_lo, v_fit_B_ci_hi=vB_hi, v_fit2_B=vB2, v_T_B=vBT, delta_v_B=dVB,
        v_fit_ABB=vABB, v_fit_ABB_ci_lo=vABB_lo, v_fit_ABB_ci_hi=vABB_hi, v_fit2_ABB=vABB2, v_T_ABB=vABBT, delta_v_ABB=dVABB,
        deltaP_late_mean_ABB=dPABB,
        w_loc_primary_ABB=w3ABB,
        w_loc_sensitivity_ABB=w5ABB,
        P0bar_ABB=P0ABB,
        adv_v_max_ABB=adv,
        strict_mask=strict.astype(np.uint8),
        directed_transport_primary=dt_primary.astype(np.uint8),
        directed_transport_sensitivity=dt_sens.astype(np.uint8),
        parrondo_positive_primary=parr_primary.astype(np.uint8),
        parrondo_positive_sensitivity=parr_sens.astype(np.uint8),
        mismatch_sign=mismatch.astype(np.uint8),
        null_n1_v_fit=n1_v, null_n1_v_fit_std=n1_v_std, null_n1_deltaP=n1_dp, null_n1_deltaP_std=n1_dp_std,
        null_n2_v_fit=n2_v, null_n2_v_fit_std=n2_v_std, null_n2_deltaP=n2_dp, null_n2_deltaP_std=n2_dp_std,
        thresholds=np.array([v_min, eps_v, eps_P], dtype=np.float64),
    )

    write_json(run_dir / "metrics_summary.json", summary)
    write_text(run_dir / "logs.txt", "\n".join(logs) + "\n")

    log(f"Wrote atlas: {atlas_path}")
    log(f"Wrote summary: {run_dir / 'metrics_summary.json'}")
    log(f"Wrote logs: {run_dir / 'logs.txt'}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
