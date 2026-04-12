#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from dtqw.io import load_yaml, write_json, write_manifest, write_text, write_yaml
from dtqw.metrics import fit_v_fit


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def make_run_dir(tag: str, base: str = "outputs") -> Dict[str, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"run_{stamp}_{tag}"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=False)
    return {"run_dir": run_dir, "fig_dir": fig_dir}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Config / reference loading
# -----------------------------------------------------------------------------


def parse_t_list(text: str) -> List[int]:
    out: List[int] = []
    for chunk in text.split(","):
        s = chunk.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        raise ValueError("t_list must contain at least one horizon")
    out = sorted(set(out))
    return out


def resolve_from_mrc(cfg: Dict[str, Any]) -> Dict[str, Any]:
    coins_cfg = cfg["coins"]
    coin_params = {
        "A": (
            float(coins_cfg["A"]["alpha_deg"]),
            float(coins_cfg["A"]["beta_deg"]),
            float(coins_cfg["A"]["gamma_deg"]),
        ),
        "B": (
            float(coins_cfg["B"]["alpha_deg"]),
            float(coins_cfg["B"]["beta_deg"]),
            float(coins_cfg["B"]["gamma_deg"]),
        ),
    }
    sim = cfg["simulation"]
    loc = sim["localization_overlay"]
    se = cfg["stochastic_estimation"]
    stage3 = cfg["stage3_grid"]
    return {
        "coin_params": coin_params,
        "T_stage3": int(sim["horizons"]["T_long"]),
        "T0_stage3": int(sim["late_windows"]["T0"]),
        "T1_stage3": int(sim["late_windows"]["T1"]),
        "x0_primary": int(loc["x0_primary"]),
        "x0_sensitivity": int(loc["x0_sensitivity"]),
        "w_thr_primary": float(loc["w_thr_primary"]),
        "w_thr_sensitivity": float(loc["w_thr_sensitivity"]),
        "base_seed": int(se["rng_policy"]["base_seed"]),
        "N_phi": int(stage3["N_phi"]),
        "N_p": int(stage3["N_p"]),
        "rel_tol": 0.5,
    }


def load_stage4_reference(stage4_run_dir: Path) -> Dict[str, Any]:
    summary_path = stage4_run_dir / "exact_p0_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Stage-4 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = summary["counts_exact_p0"]
    head = summary["headline_points"]
    max_adv_phi = float(head["max_advantage"]["phi"])
    pp_sens_phi = None
    if head.get("first_pp_sensitivity") is not None:
        pp_sens_phi = float(head["first_pp_sensitivity"]["phi"])
    thresholds = summary["thresholds_from_stage3"]
    return {
        "summary": summary,
        "counts": counts,
        "thresholds": thresholds,
        "max_adv_phi": max_adv_phi,
        "pp_sens_phi": pp_sens_phi,
        "reference_phis": [0.0, max_adv_phi] + ([pp_sens_phi] if pp_sens_phi is not None else []),
    }


# -----------------------------------------------------------------------------
# Coin / exact unitary helpers
# -----------------------------------------------------------------------------


def su2_coin(alpha: float, beta: float, gamma: float, degrees: bool = True) -> np.ndarray:
    if degrees:
        a = np.deg2rad(alpha)
        b = np.deg2rad(beta)
        g = np.deg2rad(gamma)
    else:
        a = float(alpha)
        b = float(beta)
        g = float(gamma)
    c = np.cos(b)
    s = np.sin(b)
    return np.array(
        [
            [np.exp(1j * a) * c, -np.exp(-1j * g) * s],
            [np.exp(1j * g) * s, np.exp(-1j * a) * c],
        ],
        dtype=np.complex128,
    )


def default_init_state_on_lattice(Tmax: int) -> np.ndarray:
    npos = 2 * Tmax + 1
    psi = np.zeros((2, npos), dtype=np.complex128)
    idx0 = Tmax
    psi[:, idx0] = np.array([1.0, -1.0j], dtype=np.complex128) / np.sqrt(2.0)
    return psi


def apply_coin_state(psi: np.ndarray, C: np.ndarray, out: np.ndarray) -> np.ndarray:
    # out = C @ psi on the coin axis
    out[0, :] = C[0, 0] * psi[0, :] + C[0, 1] * psi[1, :]
    out[1, :] = C[1, 0] * psi[0, :] + C[1, 1] * psi[1, :]
    return out


def apply_origin_defect_state(psi: np.ndarray, idx0: int, eiphi: complex) -> None:
    psi[:, idx0] *= eiphi


def apply_shift_no_wrap_state(psi: np.ndarray, out: np.ndarray) -> np.ndarray:
    out.fill(0.0)
    out[0, 1:] = psi[0, :-1]
    out[1, :-1] = psi[1, 1:]
    return out


def measure_state_scalar_series(
    psi: np.ndarray,
    x: np.ndarray,
    idx0: int,
    x0_primary: int,
    x0_sens: int,
) -> Tuple[float, float, float, float, float]:
    prob = (np.abs(psi[0]) ** 2 + np.abs(psi[1]) ** 2).real.astype(np.float64)
    x_mean = float(prob.dot(x))
    dP = float(prob[idx0 + 1 :].sum() - prob[:idx0].sum())
    w3 = float(prob[idx0 - x0_primary : idx0 + x0_primary + 1].sum())
    w5 = float(prob[idx0 - x0_sens : idx0 + x0_sens + 1].sum())
    p0 = float(prob[idx0])
    return x_mean, dP, w3, w5, p0


@dataclass
class ScalarSeries:
    x_mean_t: np.ndarray
    deltaP_t: np.ndarray
    w3_t: np.ndarray
    w5_t: np.ndarray
    P0_t: np.ndarray


@dataclass
class PointMetrics:
    v_fit: float
    v_fit2: float
    v_T: float
    delta_v: float
    deltaP_late_mean: float
    deltaP_final: float
    w_loc3: float
    w_loc5: float
    P0bar: float
    x_mean_final: float
    stable: bool



def scaled_windows(T: int) -> Tuple[int, int]:
    return int(math.floor((2.0 / 3.0) * T)), int(math.floor((5.0 / 6.0) * T))



def fit_tail_and_metrics(series: ScalarSeries, T: int, T0: int, T1: int, rel_tol: float) -> PointMetrics:
    x_mean_t = np.asarray(series.x_mean_t[: T + 1], dtype=np.float64)
    dP_t = np.asarray(series.deltaP_t[: T + 1], dtype=np.float64)
    w3_t = np.asarray(series.w3_t[: T + 1], dtype=np.float64)
    w5_t = np.asarray(series.w5_t[: T + 1], dtype=np.float64)
    P0_t = np.asarray(series.P0_t[: T + 1], dtype=np.float64)

    v_fit, _ = fit_v_fit(x_mean_t, T0=T0)
    v_fit2, _ = fit_v_fit(x_mean_t, T0=T1)
    v_T = float(x_mean_t[T] / float(T))

    tt = np.arange(T1, T + 1, dtype=np.float64)
    tt_safe = tt.copy()
    tt_safe[tt_safe < 1.0] = 1.0
    vbar = x_mean_t[T1:] / tt_safe
    delta_v = float(np.max(np.abs(vbar - v_T)))

    eps = 1e-12
    denom = max(abs(v_fit), eps)
    stable = (
        np.sign(v_fit) == np.sign(v_fit2)
        and np.sign(v_fit) == np.sign(v_T)
        and np.sign(v_fit) != 0
        and abs(v_fit2 - v_fit) / denom <= rel_tol
        and abs(v_T - v_fit) / denom <= rel_tol
        and delta_v / denom <= rel_tol
    )

    return PointMetrics(
        v_fit=float(v_fit),
        v_fit2=float(v_fit2),
        v_T=float(v_T),
        delta_v=float(delta_v),
        deltaP_late_mean=float(np.mean(dP_t[T0:])),
        deltaP_final=float(dP_t[-1]),
        w_loc3=float(np.mean(w3_t[T0:])),
        w_loc5=float(np.mean(w5_t[T0:])),
        P0bar=float(np.mean(P0_t[T0:])),
        x_mean_final=float(x_mean_t[-1]),
        stable=bool(stable),
    )



def sign_eps_scalar(x: float, eps: float) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0



def simulate_unitary_scalar_timeseries(
    seq: str,
    Tmax: int,
    phi: float,
    C_A: np.ndarray,
    C_B: np.ndarray,
    x0_primary: int,
    x0_sens: int,
) -> ScalarSeries:
    npos = 2 * Tmax + 1
    idx0 = Tmax
    x = np.arange(-Tmax, Tmax + 1, dtype=np.float64)

    psi = default_init_state_on_lattice(Tmax)
    mid = np.zeros_like(psi)
    nxt = np.zeros_like(psi)
    eiphi = complex(math.cos(phi), math.sin(phi))

    x_mean_t = np.zeros(Tmax + 1, dtype=np.float64)
    dP_t = np.zeros(Tmax + 1, dtype=np.float64)
    w3_t = np.zeros(Tmax + 1, dtype=np.float64)
    w5_t = np.zeros(Tmax + 1, dtype=np.float64)
    P0_t = np.zeros(Tmax + 1, dtype=np.float64)

    x_mean_t[0], dP_t[0], w3_t[0], w5_t[0], P0_t[0] = measure_state_scalar_series(psi, x, idx0, x0_primary, x0_sens)

    pat = list(seq)
    pat_len = len(pat)
    for step in range(Tmax):
        sym = pat[step % pat_len]
        C = C_A if sym == "A" else C_B
        apply_coin_state(psi, C, mid)
        apply_origin_defect_state(mid, idx0, eiphi)
        apply_shift_no_wrap_state(mid, nxt)
        psi, nxt = nxt, psi
        x_mean_t[step + 1], dP_t[step + 1], w3_t[step + 1], w5_t[step + 1], P0_t[step + 1] = measure_state_scalar_series(psi, x, idx0, x0_primary, x0_sens)

    return ScalarSeries(x_mean_t=x_mean_t, deltaP_t=dP_t, w3_t=w3_t, w5_t=w5_t, P0_t=P0_t)


# -----------------------------------------------------------------------------
# Exact density-matrix evolution for validation
# -----------------------------------------------------------------------------


def measure_density_scalar_series(
    rho: np.ndarray,
    x: np.ndarray,
    idx0: int,
    x0_primary: int,
    x0_sens: int,
) -> Tuple[float, float, float, float, float]:
    pdiag = np.real(np.diagonal(rho[0, :, 0, :]) + np.diagonal(rho[1, :, 1, :])).astype(np.float64)
    # tiny numerical negatives are harmless; keep diagnostics numerically stable
    pdiag = np.where(pdiag < 0.0, np.maximum(pdiag, -1e-14), pdiag)
    pdiag = np.maximum(pdiag, 0.0)
    s = float(pdiag.sum())
    if s > 0.0:
        pdiag = pdiag / s
    x_mean = float(pdiag.dot(x))
    dP = float(pdiag[idx0 + 1 :].sum() - pdiag[:idx0].sum())
    w3 = float(pdiag[idx0 - x0_primary : idx0 + x0_primary + 1].sum())
    w5 = float(pdiag[idx0 - x0_sens : idx0 + x0_sens + 1].sum())
    p0 = float(pdiag[idx0])
    return x_mean, dP, w3, w5, p0



def apply_coin_density(rho: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.einsum("ac,cxdy,bd->axby", C, rho, C.conj(), optimize=True)



def apply_origin_defect_density(rho: np.ndarray, idx0: int, eiphi: complex) -> None:
    rho[:, idx0, :, :] *= eiphi
    rho[:, :, :, idx0] *= np.conjugate(eiphi)



def apply_phase_damping_density(rho: np.ndarray, p: float) -> None:
    lam = 1.0 - float(p)
    if lam == 1.0:
        return
    rho[0, :, 1, :] *= lam
    rho[1, :, 0, :] *= lam



def apply_shift_density(rho: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rho)
    out[0, 1:, 0, 1:] = rho[0, :-1, 0, :-1]
    out[0, 1:, 1, :-1] = rho[0, :-1, 1, 1:]
    out[1, :-1, 0, 1:] = rho[1, 1:, 0, :-1]
    out[1, :-1, 1, :-1] = rho[1, 1:, 1, 1:]
    return out



def simulate_density_scalar_timeseries(
    seq: str,
    T: int,
    phi: float,
    p: float,
    C_A: np.ndarray,
    C_B: np.ndarray,
    x0_primary: int,
    x0_sens: int,
) -> ScalarSeries:
    npos = 2 * T + 1
    idx0 = T
    x = np.arange(-T, T + 1, dtype=np.float64)
    init_coin = np.array([1.0, -1.0j], dtype=np.complex128) / np.sqrt(2.0)

    rho = np.zeros((2, npos, 2, npos), dtype=np.complex128)
    rho[:, idx0, :, idx0] = np.outer(init_coin, init_coin.conj())
    eiphi = complex(math.cos(phi), math.sin(phi))

    x_mean_t = np.zeros(T + 1, dtype=np.float64)
    dP_t = np.zeros(T + 1, dtype=np.float64)
    w3_t = np.zeros(T + 1, dtype=np.float64)
    w5_t = np.zeros(T + 1, dtype=np.float64)
    P0_t = np.zeros(T + 1, dtype=np.float64)

    x_mean_t[0], dP_t[0], w3_t[0], w5_t[0], P0_t[0] = measure_density_scalar_series(rho, x, idx0, x0_primary, x0_sens)

    pat = list(seq)
    pat_len = len(pat)
    for step in range(T):
        sym = pat[step % pat_len]
        C = C_A if sym == "A" else C_B
        rho = apply_coin_density(rho, C)
        apply_origin_defect_density(rho, idx0, eiphi)
        apply_phase_damping_density(rho, p)
        rho = apply_shift_density(rho)
        x_mean_t[step + 1], dP_t[step + 1], w3_t[step + 1], w5_t[step + 1], P0_t[step + 1] = measure_density_scalar_series(rho, x, idx0, x0_primary, x0_sens)

    return ScalarSeries(x_mean_t=x_mean_t, deltaP_t=dP_t, w3_t=w3_t, w5_t=w5_t, P0_t=P0_t)


# -----------------------------------------------------------------------------
# Stage-3 simulator import for padding / MC validation
# -----------------------------------------------------------------------------


def import_stage3_module(repo_root: Path):
    stage3_path = repo_root / "scripts" / "stage3_confirmatory_pd_atlas_cpu.py"
    if not stage3_path.exists():
        raise FileNotFoundError(f"Could not find Stage-3 script for import: {stage3_path}")
    spec = importlib.util.spec_from_file_location("stage3_confirmatory_pd_atlas_cpu", stage3_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {stage3_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod



def stage3_point_metrics(
    stage3_mod,
    seq: str,
    phi: float,
    p: float,
    T: int,
    T0: int,
    T1: int,
    pad_m: int,
    x0_primary: int,
    x0_sens: int,
    N_traj: int,
    base_seed: int,
    point_seed_i: int,
    point_seed_j: int,
    C_A: np.ndarray,
    C_B: np.ndarray,
) -> Dict[str, float]:
    L = T + int(pad_m)
    idx0 = L
    x_float = np.arange(-L, L + 1, dtype=np.float64)
    ws = stage3_mod.PDWorkspace(int(N_traj), x_float.size)
    seq_id = {"A": 1, "B": 2, "ABB": 3}[seq]
    out = stage3_mod.simulate_pd(
        ws,
        seq,
        C_A,
        C_B,
        float(phi),
        float(p),
        int(T),
        int(T0),
        int(T1),
        int(idx0),
        x_float,
        int(x0_primary),
        int(x0_sens),
        int(base_seed),
        1,
        int(seq_id),
        int(point_seed_i),
        int(point_seed_j),
    )
    return {
        "v_fit": float(np.mean(out["v_fit"])),
        "v_fit2": float(np.mean(out["v_fit2"])),
        "v_T": float(np.mean(out["v_T"])),
        "delta_v": float(np.mean(out["delta_v"])),
        "deltaP_late_mean": float(np.mean(out["deltaP_late_mean"])),
        "w_loc3": float(np.mean(out["w_loc_primary"])),
        "w_loc5": float(np.mean(out["w_loc_sensitivity"])),
        "P0bar": float(np.mean(out["P0bar"])),
        "v_fit_std": float(np.std(out["v_fit"], ddof=1)),
        "w_loc3_std": float(np.std(out["w_loc_primary"], ddof=1)),
        "P0bar_std": float(np.std(out["P0bar"], ddof=1)),
    }


# -----------------------------------------------------------------------------
# Exact horizon sweep
# -----------------------------------------------------------------------------


@dataclass
class HorizonRow:
    T: int
    policy: str
    phi_index: int
    phi: float
    phi_over_pi: float
    T0: int
    T1: int
    v_fit_A: float
    v_fit_B: float
    v_fit_ABB: float
    v_fit2_A: float
    v_fit2_B: float
    v_fit2_ABB: float
    v_T_A: float
    v_T_B: float
    v_T_ABB: float
    delta_v_A: float
    delta_v_B: float
    delta_v_ABB: float
    deltaP_late_mean_ABB: float
    deltaP_final_ABB: float
    w_loc3_ABB: float
    w_loc5_ABB: float
    P0bar_ABB: float
    stable_A: int
    stable_B: int
    stable_ABB: int
    strict_raw: int
    drift_raw: int
    strict_stage3: int
    drift_stage3: int
    pp_primary: int
    pp_sensitivity: int
    mismatch_sign: int
    adv_v: float

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()



def run_exact_horizon_sweep(
    run_dir: Path,
    t_list: Sequence[int],
    ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    resume: bool,
) -> Dict[str, Any]:
    Tmax = int(max(t_list))
    Nphi = int(cfg_r["N_phi"])
    x0_primary = int(cfg_r["x0_primary"])
    x0_sens = int(cfg_r["x0_sensitivity"])
    rel_tol = float(cfg_r["rel_tol"])

    phi_vals = np.linspace(0.0, 2.0 * math.pi, Nphi, endpoint=False, dtype=np.float64)
    coins = cfg_r["coin_params"]
    C_A = su2_coin(*coins["A"], degrees=True)
    C_B = su2_coin(*coins["B"], degrees=True)

    thresholds = ref["thresholds"]
    v_min = float(thresholds["v_min"])
    eps_v = float(thresholds["eps_v"])
    eps_P = float(thresholds["eps_P"])
    w_thr_primary = float(thresholds["w_thr_primary"])
    w_thr_sensitivity = float(thresholds["w_thr_sensitivity"])

    partial_dir = ensure_dir(run_dir / "partials" / "exact_unitary")
    seqs = ["A", "B", "ABB"]
    series_cache: Dict[str, List[ScalarSeries]] = {seq: [] for seq in seqs}

    for seq in seqs:
        for i, phi in enumerate(phi_vals):
            partial_path = partial_dir / f"series_{seq}_phi_{i:03d}_Tmax_{Tmax}.npz"
            if resume and partial_path.exists():
                d = np.load(partial_path, allow_pickle=False)
                series = ScalarSeries(
                    x_mean_t=d["x_mean_t"].astype(np.float64),
                    deltaP_t=d["deltaP_t"].astype(np.float64),
                    w3_t=d["w3_t"].astype(np.float64),
                    w5_t=d["w5_t"].astype(np.float64),
                    P0_t=d["P0_t"].astype(np.float64),
                )
            else:
                print(f"[exact] seq={seq} phi_index={i:03d}/{Nphi-1:03d} phi/pi={phi/math.pi:.6f} Tmax={Tmax}", flush=True)
                series = simulate_unitary_scalar_timeseries(seq, Tmax, float(phi), C_A, C_B, x0_primary, x0_sens)
                np.savez_compressed(
                    partial_path,
                    x_mean_t=series.x_mean_t,
                    deltaP_t=series.deltaP_t,
                    w3_t=series.w3_t,
                    w5_t=series.w5_t,
                    P0_t=series.P0_t,
                )
            series_cache[seq].append(series)

    rows: List[Dict[str, Any]] = []
    counts_rows: List[Dict[str, Any]] = []

    # For persistence tracking.
    masks_stage3: Dict[str, Dict[str, np.ndarray]] = {
        "fixed": {
            "strict": np.ones((Nphi,), dtype=bool),
            "drift": np.ones((Nphi,), dtype=bool),
            "pp_primary": np.ones((Nphi,), dtype=bool),
            "pp_sens": np.ones((Nphi,), dtype=bool),
        },
        "scaled": {
            "strict": np.ones((Nphi,), dtype=bool),
            "drift": np.ones((Nphi,), dtype=bool),
            "pp_primary": np.ones((Nphi,), dtype=bool),
            "pp_sens": np.ones((Nphi,), dtype=bool),
        },
    }
    masks_raw: Dict[str, Dict[str, np.ndarray]] = {
        "fixed": {
            "strict": np.ones((Nphi,), dtype=bool),
            "drift": np.ones((Nphi,), dtype=bool),
        },
        "scaled": {
            "strict": np.ones((Nphi,), dtype=bool),
            "drift": np.ones((Nphi,), dtype=bool),
        },
    }

    per_policy_T_rows: Dict[Tuple[str, int], List[HorizonRow]] = {}

    for policy in ["fixed", "scaled"]:
        for T in t_list:
            if policy == "fixed":
                T0 = int(cfg_r["T0_stage3"])
                T1 = int(cfg_r["T1_stage3"])
                if T < T1:
                    continue
            else:
                T0, T1 = scaled_windows(int(T))

            T_rows: List[HorizonRow] = []
            for i, phi in enumerate(phi_vals):
                mA = fit_tail_and_metrics(series_cache["A"][i], T, T0, T1, rel_tol)
                mB = fit_tail_and_metrics(series_cache["B"][i], T, T0, T1, rel_tol)
                mABB = fit_tail_and_metrics(series_cache["ABB"][i], T, T0, T1, rel_tol)

                strict_raw = bool(mA.v_fit < 0.0 and mB.v_fit < 0.0)
                drift_raw = bool(strict_raw and mABB.v_fit > 0.0)
                strict_stage3 = bool(mA.v_fit < 0.0 and abs(mA.v_fit) > v_min and mA.stable and mB.v_fit < 0.0 and abs(mB.v_fit) > v_min and mB.stable)
                drift_stage3 = bool(strict_stage3 and mABB.v_fit > 0.0 and mABB.v_fit > v_min and mABB.stable)
                pp_primary = bool(drift_stage3 and mABB.w_loc3 < w_thr_primary)
                pp_sens = bool(drift_stage3 and mABB.w_loc5 < w_thr_sensitivity)

                sv = sign_eps_scalar(mABB.v_fit, eps_v)
                sp = sign_eps_scalar(mABB.deltaP_late_mean, eps_P)
                mismatch = int(sv * sp == -1)
                adv = float(mABB.v_fit - max(mA.v_fit, mB.v_fit))

                hrow = HorizonRow(
                    T=int(T),
                    policy=str(policy),
                    phi_index=int(i),
                    phi=float(phi),
                    phi_over_pi=float(phi / math.pi),
                    T0=int(T0),
                    T1=int(T1),
                    v_fit_A=float(mA.v_fit),
                    v_fit_B=float(mB.v_fit),
                    v_fit_ABB=float(mABB.v_fit),
                    v_fit2_A=float(mA.v_fit2),
                    v_fit2_B=float(mB.v_fit2),
                    v_fit2_ABB=float(mABB.v_fit2),
                    v_T_A=float(mA.v_T),
                    v_T_B=float(mB.v_T),
                    v_T_ABB=float(mABB.v_T),
                    delta_v_A=float(mA.delta_v),
                    delta_v_B=float(mB.delta_v),
                    delta_v_ABB=float(mABB.delta_v),
                    deltaP_late_mean_ABB=float(mABB.deltaP_late_mean),
                    deltaP_final_ABB=float(mABB.deltaP_final),
                    w_loc3_ABB=float(mABB.w_loc3),
                    w_loc5_ABB=float(mABB.w_loc5),
                    P0bar_ABB=float(mABB.P0bar),
                    stable_A=int(mA.stable),
                    stable_B=int(mB.stable),
                    stable_ABB=int(mABB.stable),
                    strict_raw=int(strict_raw),
                    drift_raw=int(drift_raw),
                    strict_stage3=int(strict_stage3),
                    drift_stage3=int(drift_stage3),
                    pp_primary=int(pp_primary),
                    pp_sensitivity=int(pp_sens),
                    mismatch_sign=int(mismatch),
                    adv_v=float(adv),
                )
                T_rows.append(hrow)

            per_policy_T_rows[(policy, int(T))] = T_rows
            rows.extend([r.as_dict() for r in T_rows])

            strict_raw_mask = np.array([bool(r.strict_raw) for r in T_rows], dtype=bool)
            drift_raw_mask = np.array([bool(r.drift_raw) for r in T_rows], dtype=bool)
            strict_stage3_mask = np.array([bool(r.strict_stage3) for r in T_rows], dtype=bool)
            drift_stage3_mask = np.array([bool(r.drift_stage3) for r in T_rows], dtype=bool)
            pp_primary_mask = np.array([bool(r.pp_primary) for r in T_rows], dtype=bool)
            pp_sens_mask = np.array([bool(r.pp_sensitivity) for r in T_rows], dtype=bool)
            mismatch_mask = np.array([bool(r.mismatch_sign) for r in T_rows], dtype=bool)
            mismatch_denom = int(sum((sign_eps_scalar(r.v_fit_ABB, eps_v) != 0) and (sign_eps_scalar(r.deltaP_late_mean_ABB, eps_P) != 0) for r in T_rows))
            i_max = int(np.argmax(np.array([r.adv_v for r in T_rows], dtype=np.float64)))
            max_row = T_rows[i_max]

            counts_rows.append(
                {
                    "policy": policy,
                    "T": int(T),
                    "T0": int(T0),
                    "T1": int(T1),
                    "strict_raw": int(strict_raw_mask.sum()),
                    "drift_raw": int(drift_raw_mask.sum()),
                    "strict_stage3": int(strict_stage3_mask.sum()),
                    "drift_stage3": int(drift_stage3_mask.sum()),
                    "pp_primary": int(pp_primary_mask.sum()),
                    "pp_sensitivity": int(pp_sens_mask.sum()),
                    "mismatch_sign": int(mismatch_mask.sum()),
                    "mismatch_sign_denominator": int(mismatch_denom),
                    "max_adv_phi": float(max_row.phi),
                    "max_adv_phi_over_pi": float(max_row.phi_over_pi),
                    "max_adv_v": float(max_row.adv_v),
                    "max_adv_w_loc3": float(max_row.w_loc3_ABB),
                    "max_adv_w_loc5": float(max_row.w_loc5_ABB),
                    "max_adv_v_fit_ABB": float(max_row.v_fit_ABB),
                }
            )

            masks_stage3[policy]["strict"] &= strict_stage3_mask
            masks_stage3[policy]["drift"] &= drift_stage3_mask
            masks_stage3[policy]["pp_primary"] &= pp_primary_mask
            masks_stage3[policy]["pp_sens"] &= pp_sens_mask
            masks_raw[policy]["strict"] &= strict_raw_mask
            masks_raw[policy]["drift"] &= drift_raw_mask

    grid_csv = run_dir / "exact_horizon_grid.csv"
    counts_csv = run_dir / "exact_horizon_counts.csv"
    write_csv(grid_csv, rows)
    write_csv(counts_csv, counts_rows)

    # Persistent points across all requested T.
    persistent_stage3_rows: List[Dict[str, Any]] = []
    persistent_raw_rows: List[Dict[str, Any]] = []
    T_last = int(max(t_list))
    for policy in ["fixed", "scaled"]:
        T0_last, T1_last = (int(cfg_r["T0_stage3"]), int(cfg_r["T1_stage3"])) if policy == "fixed" else scaled_windows(T_last)
        for i, phi in enumerate(phi_vals):
            mA = fit_tail_and_metrics(series_cache["A"][i], T_last, T0_last, T1_last, rel_tol)
            mB = fit_tail_and_metrics(series_cache["B"][i], T_last, T0_last, T1_last, rel_tol)
            mABB = fit_tail_and_metrics(series_cache["ABB"][i], T_last, T0_last, T1_last, rel_tol)
            adv = float(mABB.v_fit - max(mA.v_fit, mB.v_fit))
            if masks_stage3[policy]["strict"][i] or masks_stage3[policy]["drift"][i] or masks_stage3[policy]["pp_primary"][i] or masks_stage3[policy]["pp_sens"][i]:
                persistent_stage3_rows.append(
                    {
                        "policy": policy,
                        "phi_index": int(i),
                        "phi": float(phi),
                        "phi_over_pi": float(phi / math.pi),
                        "strict_all_T": int(masks_stage3[policy]["strict"][i]),
                        "drift_all_T": int(masks_stage3[policy]["drift"][i]),
                        "pp_primary_all_T": int(masks_stage3[policy]["pp_primary"][i]),
                        "pp_sensitivity_all_T": int(masks_stage3[policy]["pp_sens"][i]),
                        "T_last": int(T_last),
                        "v_fit_A_T_last": float(mA.v_fit),
                        "v_fit_B_T_last": float(mB.v_fit),
                        "v_fit_ABB_T_last": float(mABB.v_fit),
                        "w_loc3_ABB_T_last": float(mABB.w_loc3),
                        "w_loc5_ABB_T_last": float(mABB.w_loc5),
                        "adv_v_T_last": float(adv),
                    }
                )
            if masks_raw[policy]["strict"][i] or masks_raw[policy]["drift"][i]:
                persistent_raw_rows.append(
                    {
                        "policy": policy,
                        "phi_index": int(i),
                        "phi": float(phi),
                        "phi_over_pi": float(phi / math.pi),
                        "strict_raw_all_T": int(masks_raw[policy]["strict"][i]),
                        "drift_raw_all_T": int(masks_raw[policy]["drift"][i]),
                        "T_last": int(T_last),
                        "v_fit_A_T_last": float(mA.v_fit),
                        "v_fit_B_T_last": float(mB.v_fit),
                        "v_fit_ABB_T_last": float(mABB.v_fit),
                        "adv_v_T_last": float(adv),
                    }
                )

    write_csv(run_dir / "persistent_points_stage3.csv", persistent_stage3_rows)
    write_csv(run_dir / "persistent_points_raw.csv", persistent_raw_rows)

    # Selected exact time-series for representative phis.
    ts_dir = ensure_dir(run_dir / "selected_timeseries")
    selected_phis = [0.0, ref["max_adv_phi"]]
    if ref.get("pp_sens_phi") is not None:
        selected_phis.append(float(ref["pp_sens_phi"]))
    selected_indices = sorted(set(int(np.argmin(np.abs(phi_vals - phi))) for phi in selected_phis))
    selected_rows: List[Dict[str, Any]] = []
    for seq in seqs:
        for i in selected_indices:
            phi = float(phi_vals[i])
            ser = series_cache[seq][i]
            path = ts_dir / f"timeseries_seq_{seq}_phi_{i:03d}.csv"
            ts_rows = []
            for t in range(Tmax + 1):
                ts_rows.append(
                    {
                        "t": int(t),
                        "phi_index": int(i),
                        "phi": float(phi),
                        "phi_over_pi": float(phi / math.pi),
                        "sequence": seq,
                        "x_mean": float(ser.x_mean_t[t]),
                        "deltaP": float(ser.deltaP_t[t]),
                        "w_loc3_inst": float(ser.w3_t[t]),
                        "w_loc5_inst": float(ser.w5_t[t]),
                        "P0": float(ser.P0_t[t]),
                    }
                )
            write_csv(path, ts_rows)
            selected_rows.append(
                {
                    "sequence": seq,
                    "phi_index": int(i),
                    "phi": float(phi),
                    "phi_over_pi": float(phi / math.pi),
                    "csv": str(path.relative_to(run_dir)),
                }
            )
    write_csv(run_dir / "selected_timeseries_index.csv", selected_rows)

    # Summary
    counts_lookup = {(row["policy"], int(row["T"])): row for row in counts_rows}
    summary = {
        "T_list": [int(t) for t in t_list],
        "grid_csv": str(grid_csv.relative_to(run_dir)),
        "counts_csv": str(counts_csv.relative_to(run_dir)),
        "persistent_stage3_csv": "persistent_points_stage3.csv",
        "persistent_raw_csv": "persistent_points_raw.csv",
        "selected_timeseries_index_csv": "selected_timeseries_index.csv",
        "counts": counts_rows,
        "stage3_thresholded_persistent_counts": {
            policy: {
                key: int(mask.sum())
                for key, mask in masks_stage3[policy].items()
            }
            for policy in ["fixed", "scaled"]
        },
        "raw_persistent_counts": {
            policy: {
                key: int(mask.sum())
                for key, mask in masks_raw[policy].items()
            }
            for policy in ["fixed", "scaled"]
        },
        "T300_fixed_counts": counts_lookup.get(("fixed", 300)),
        "T300_scaled_counts": counts_lookup.get(("scaled", 300)),
        "Tlast_fixed_counts": counts_lookup.get(("fixed", T_last)),
        "Tlast_scaled_counts": counts_lookup.get(("scaled", T_last)),
    }
    write_json(run_dir / "exact_horizon_summary.json", summary)
    return summary


# -----------------------------------------------------------------------------
# Density-matrix validation against trajectory unraveling
# -----------------------------------------------------------------------------


@dataclass
class ValidationPoint:
    seq: str
    phi: float
    p: float
    point_id: int



def default_validation_points(ref: Dict[str, Any]) -> List[ValidationPoint]:
    phi_vals = [0.0, float(ref["max_adv_phi"])]
    if ref.get("pp_sens_phi") is not None:
        phi_vals.append(float(ref["pp_sens_phi"]))
    # Deduplicate to avoid repeated work if max_adv==pp_sens in another project state.
    phi_vals = list(dict.fromkeys(phi_vals))
    p_vals = [0.0, 0.5, 1.0]
    seqs = ["A", "B", "ABB"]
    out: List[ValidationPoint] = []
    pid = 0
    for seq in seqs:
        for phi in phi_vals:
            for p in p_vals:
                out.append(ValidationPoint(seq=seq, phi=float(phi), p=float(p), point_id=pid))
                pid += 1
    return out



def run_density_validation(
    repo_root: Path,
    run_dir: Path,
    ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    resume: bool,
    N_traj_validation: int,
    validation_T: int,
) -> Dict[str, Any]:
    stage3_mod = import_stage3_module(repo_root)
    T = int(validation_T)
    T0, T1 = scaled_windows(T)
    x0_primary = int(cfg_r["x0_primary"])
    x0_sens = int(cfg_r["x0_sensitivity"])
    base_seed = int(cfg_r["base_seed"])
    rel_tol = float(cfg_r["rel_tol"])

    coins = cfg_r["coin_params"]
    C_A = su2_coin(*coins["A"], degrees=True)
    C_B = su2_coin(*coins["B"], degrees=True)

    v_pass = 0.01
    w_pass = 0.02
    p0_pass = 0.02

    val_dir = ensure_dir(run_dir / "partials" / "density_validation")
    points = default_validation_points(ref)
    rows: List[Dict[str, Any]] = []

    for vp in points:
        partial_path = val_dir / f"val_{vp.point_id:03d}_{vp.seq}_phi_{vp.phi/math.pi:.6f}_p_{vp.p:.3f}.json"
        if resume and partial_path.exists():
            row = json.loads(partial_path.read_text(encoding="utf-8"))
            rows.append(row)
            continue

        print(
            f"[density] point={vp.point_id:03d}/{len(points)-1:03d} seq={vp.seq} phi/pi={vp.phi/math.pi:.6f} p={vp.p:.3f} N_traj={N_traj_validation}",
            flush=True,
        )

        exact_series = simulate_density_scalar_timeseries(vp.seq, T, vp.phi, vp.p, C_A, C_B, x0_primary, x0_sens)
        exact_metrics = fit_tail_and_metrics(exact_series, T, T0, T1, rel_tol)

        mc = stage3_point_metrics(
            stage3_mod,
            vp.seq,
            vp.phi,
            vp.p,
            T,
            T0,
            T1,
            0,
            x0_primary,
            x0_sens,
            N_traj_validation,
            base_seed,
            1000 + vp.point_id,
            2000 + vp.point_id,
            C_A,
            C_B,
        )

        row = {
            "point_id": int(vp.point_id),
            "sequence": vp.seq,
            "phi": float(vp.phi),
            "phi_over_pi": float(vp.phi / math.pi),
            "p": float(vp.p),
            "T": int(T),
            "T0": int(T0),
            "T1": int(T1),
            "T": int(T),
        "T0": int(T0),
        "T1": int(T1),
        "N_traj": int(N_traj_validation),
            "exact_v_fit": float(exact_metrics.v_fit),
            "mc_v_fit": float(mc["v_fit"]),
            "abs_diff_v_fit": float(abs(mc["v_fit"] - exact_metrics.v_fit)),
            "exact_w_loc3": float(exact_metrics.w_loc3),
            "mc_w_loc3": float(mc["w_loc3"]),
            "abs_diff_w_loc3": float(abs(mc["w_loc3"] - exact_metrics.w_loc3)),
            "exact_P0bar": float(exact_metrics.P0bar),
            "mc_P0bar": float(mc["P0bar"]),
            "abs_diff_P0bar": float(abs(mc["P0bar"] - exact_metrics.P0bar)),
            "exact_deltaP_late_mean": float(exact_metrics.deltaP_late_mean),
            "mc_deltaP_late_mean": float(mc["deltaP_late_mean"]),
            "abs_diff_deltaP_late_mean": float(abs(mc["deltaP_late_mean"] - exact_metrics.deltaP_late_mean)),
            "exact_w_loc5": float(exact_metrics.w_loc5),
            "mc_w_loc5": float(mc["w_loc5"]),
            "abs_diff_w_loc5": float(abs(mc["w_loc5"] - exact_metrics.w_loc5)),
            "mc_v_fit_std": float(mc["v_fit_std"]),
            "mc_w_loc3_std": float(mc["w_loc3_std"]),
            "mc_P0bar_std": float(mc["P0bar_std"]),
            "pass_v_fit": int(abs(mc["v_fit"] - exact_metrics.v_fit) <= v_pass),
            "pass_w_loc3": int(abs(mc["w_loc3"] - exact_metrics.w_loc3) <= w_pass),
            "pass_P0bar": int(abs(mc["P0bar"] - exact_metrics.P0bar) <= p0_pass),
            "pass_all": int(
                abs(mc["v_fit"] - exact_metrics.v_fit) <= v_pass
                and abs(mc["w_loc3"] - exact_metrics.w_loc3) <= w_pass
                and abs(mc["P0bar"] - exact_metrics.P0bar) <= p0_pass
            ),
        }
        partial_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)

    rows = sorted(rows, key=lambda r: int(r["point_id"]))
    write_csv(run_dir / "density_validation.csv", rows)

    p1_rows = [r for r in rows if abs(float(r["p"]) - 1.0) < 1e-12]
    phi_invariance_rows: List[Dict[str, Any]] = []
    for seq in ["A", "B", "ABB"]:
        seq_rows = [r for r in p1_rows if r["sequence"] == seq]
        if not seq_rows:
            continue
        ev = np.array([float(r["exact_v_fit"]) for r in seq_rows], dtype=float)
        ew3 = np.array([float(r["exact_w_loc3"]) for r in seq_rows], dtype=float)
        ep0 = np.array([float(r["exact_P0bar"]) for r in seq_rows], dtype=float)
        edp = np.array([float(r["exact_deltaP_late_mean"]) for r in seq_rows], dtype=float)
        phi_invariance_rows.append(
            {
                "sequence": seq,
                "p": 1.0,
                "n_phi_tested": int(len(seq_rows)),
                "max_abs_diff_exact_v_fit_across_phi": float(np.max(ev) - np.min(ev)),
                "max_abs_diff_exact_w_loc3_across_phi": float(np.max(ew3) - np.min(ew3)),
                "max_abs_diff_exact_P0bar_across_phi": float(np.max(ep0) - np.min(ep0)),
                "max_abs_diff_exact_deltaP_late_mean_across_phi": float(np.max(edp) - np.min(edp)),
            }
        )
    write_csv(run_dir / "p1_phi_invariance.csv", phi_invariance_rows)

    summary = {
        "n_points": int(len(rows)),
        "T": int(T),
        "T0": int(T0),
        "T1": int(T1),
        "validation_T": int(T),
        "validation_T0": int(T0),
        "validation_T1": int(T1),
        "N_traj": int(N_traj_validation),
        "thresholds": {
            "abs_diff_v_fit": v_pass,
            "abs_diff_w_loc3": w_pass,
            "abs_diff_P0bar": p0_pass,
        },
        "n_pass_all": int(sum(int(r["pass_all"]) for r in rows)),
        "max_abs_diff_v_fit": float(max(float(r["abs_diff_v_fit"]) for r in rows)) if rows else None,
        "max_abs_diff_w_loc3": float(max(float(r["abs_diff_w_loc3"]) for r in rows)) if rows else None,
        "max_abs_diff_P0bar": float(max(float(r["abs_diff_P0bar"]) for r in rows)) if rows else None,
        "max_abs_diff_w_loc5": float(max(float(r["abs_diff_w_loc5"]) for r in rows)) if rows else None,
        "max_abs_diff_deltaP_late_mean": float(max(float(r["abs_diff_deltaP_late_mean"]) for r in rows)) if rows else None,
        "csv": "density_validation.csv",
        "p1_phi_invariance_csv": "p1_phi_invariance.csv",
    }
    write_json(run_dir / "density_validation_summary.json", summary)
    return summary


# -----------------------------------------------------------------------------
# Padding invariance on the Stage-3 trajectory engine
# -----------------------------------------------------------------------------


@dataclass
class PaddingPoint:
    seq: str
    phi: float
    p: float
    point_id: int



def default_padding_points(ref: Dict[str, Any]) -> List[PaddingPoint]:
    phi0 = 0.0
    phi1 = float(ref["max_adv_phi"])
    phi2 = float(ref["pp_sens_phi"]) if ref.get("pp_sens_phi") is not None else float(ref["max_adv_phi"])
    specs = [
        ("A", phi0, 0.0),
        ("B", phi0, 0.0),
        ("ABB", phi0, 0.0),
        ("ABB", phi1, 0.0),
        ("ABB", phi2, 0.0),
        ("ABB", phi1, 0.5),
        ("ABB", phi2, 0.5),
        ("ABB", phi0, 1.0),
        ("A", phi1, 0.5),
        ("B", phi2, 0.5),
    ]
    return [PaddingPoint(seq=s, phi=float(phi), p=float(p), point_id=i) for i, (s, phi, p) in enumerate(specs)]



def run_padding_validation(
    repo_root: Path,
    run_dir: Path,
    ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    resume: bool,
    N_traj_padding: int,
) -> Dict[str, Any]:
    stage3_mod = import_stage3_module(repo_root)
    T = int(cfg_r["T_stage3"])
    T0 = int(cfg_r["T0_stage3"])
    T1 = int(cfg_r["T1_stage3"])
    x0_primary = int(cfg_r["x0_primary"])
    x0_sens = int(cfg_r["x0_sensitivity"])
    base_seed = int(cfg_r["base_seed"])
    coins = cfg_r["coin_params"]
    C_A = su2_coin(*coins["A"], degrees=True)
    C_B = su2_coin(*coins["B"], degrees=True)

    dv_pass = 0.005
    w_pass = 0.01
    p0_pass = 0.01

    pad_dir = ensure_dir(run_dir / "partials" / "padding_validation")
    points = default_padding_points(ref)
    rows: List[Dict[str, Any]] = []
    for vp in points:
        partial_path = pad_dir / f"pad_{vp.point_id:03d}_{vp.seq}_phi_{vp.phi/math.pi:.6f}_p_{vp.p:.3f}.json"
        if resume and partial_path.exists():
            row = json.loads(partial_path.read_text(encoding="utf-8"))
            rows.append(row)
            continue

        print(
            f"[padding] point={vp.point_id:03d}/{len(points)-1:03d} seq={vp.seq} phi/pi={vp.phi/math.pi:.6f} p={vp.p:.3f} N_traj={N_traj_padding}",
            flush=True,
        )
        m0 = stage3_point_metrics(
            stage3_mod,
            vp.seq,
            vp.phi,
            vp.p,
            T,
            T0,
            T1,
            0,
            x0_primary,
            x0_sens,
            N_traj_padding,
            base_seed,
            3000 + vp.point_id,
            4000 + vp.point_id,
            C_A,
            C_B,
        )
        m2 = stage3_point_metrics(
            stage3_mod,
            vp.seq,
            vp.phi,
            vp.p,
            T,
            T0,
            T1,
            2,
            x0_primary,
            x0_sens,
            N_traj_padding,
            base_seed,
            3000 + vp.point_id,
            4000 + vp.point_id,
            C_A,
            C_B,
        )
        m4 = stage3_point_metrics(
            stage3_mod,
            vp.seq,
            vp.phi,
            vp.p,
            T,
            T0,
            T1,
            4,
            x0_primary,
            x0_sens,
            N_traj_padding,
            base_seed,
            3000 + vp.point_id,
            4000 + vp.point_id,
            C_A,
            C_B,
        )

        diffs_02 = {
            "abs_diff_v_fit_0_vs_2": abs(m0["v_fit"] - m2["v_fit"]),
            "abs_diff_w_loc3_0_vs_2": abs(m0["w_loc3"] - m2["w_loc3"]),
            "abs_diff_P0bar_0_vs_2": abs(m0["P0bar"] - m2["P0bar"]),
            "abs_diff_deltaP_0_vs_2": abs(m0["deltaP_late_mean"] - m2["deltaP_late_mean"]),
        }
        diffs_24 = {
            "abs_diff_v_fit_2_vs_4": abs(m2["v_fit"] - m4["v_fit"]),
            "abs_diff_w_loc3_2_vs_4": abs(m2["w_loc3"] - m4["w_loc3"]),
            "abs_diff_P0bar_2_vs_4": abs(m2["P0bar"] - m4["P0bar"]),
            "abs_diff_deltaP_2_vs_4": abs(m2["deltaP_late_mean"] - m4["deltaP_late_mean"]),
        }
        pass_all = (
            diffs_02["abs_diff_v_fit_0_vs_2"] <= dv_pass
            and diffs_02["abs_diff_w_loc3_0_vs_2"] <= w_pass
            and diffs_02["abs_diff_P0bar_0_vs_2"] <= p0_pass
            and diffs_24["abs_diff_v_fit_2_vs_4"] <= dv_pass
            and diffs_24["abs_diff_w_loc3_2_vs_4"] <= w_pass
            and diffs_24["abs_diff_P0bar_2_vs_4"] <= p0_pass
        )
        row = {
            "point_id": int(vp.point_id),
            "sequence": vp.seq,
            "phi": float(vp.phi),
            "phi_over_pi": float(vp.phi / math.pi),
            "p": float(vp.p),
            "T": int(T),
            "T0": int(T0),
            "T1": int(T1),
            "N_traj": int(N_traj_padding),
            **{f"pad0_{k}": float(v) for k, v in m0.items() if not k.endswith("_std")},
            **{f"pad2_{k}": float(v) for k, v in m2.items() if not k.endswith("_std")},
            **{f"pad4_{k}": float(v) for k, v in m4.items() if not k.endswith("_std")},
            **{k: float(v) for k, v in diffs_02.items()},
            **{k: float(v) for k, v in diffs_24.items()},
            "pass_all": int(pass_all),
        }
        partial_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)

    rows = sorted(rows, key=lambda r: int(r["point_id"]))
    write_csv(run_dir / "padding_validation.csv", rows)

    summary = {
        "n_points": int(len(rows)),
        "N_traj": int(N_traj_padding),
        "thresholds": {
            "abs_diff_v_fit": dv_pass,
            "abs_diff_w_loc3": w_pass,
            "abs_diff_P0bar": p0_pass,
        },
        "n_pass_all": int(sum(int(r["pass_all"]) for r in rows)),
        "max_abs_diff_v_fit": float(max(max(float(r["abs_diff_v_fit_0_vs_2"]), float(r["abs_diff_v_fit_2_vs_4"])) for r in rows)) if rows else None,
        "max_abs_diff_w_loc3": float(max(max(float(r["abs_diff_w_loc3_0_vs_2"]), float(r["abs_diff_w_loc3_2_vs_4"])) for r in rows)) if rows else None,
        "max_abs_diff_P0bar": float(max(max(float(r["abs_diff_P0bar_0_vs_2"]), float(r["abs_diff_P0bar_2_vs_4"])) for r in rows)) if rows else None,
        "csv": "padding_validation.csv",
    }
    write_json(run_dir / "padding_validation_summary.json", summary)
    return summary


# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------



def write_rigour_report(
    run_dir: Path,
    stage4_ref: Dict[str, Any],
    horizon_summary: Dict[str, Any],
    density_summary: Dict[str, Any],
    padding_summary: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Stage-5 Rigour Suite Report")
    lines.append("")
    lines.append("## Stage-4 exact reference")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stage4_ref["counts"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("Reference phis used in this suite:")
    lines.append("```json")
    lines.append(json.dumps({"phi0": 0.0, "max_adv_phi": stage4_ref["max_adv_phi"], "pp_sens_phi": stage4_ref["pp_sens_phi"]}, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Exact p=0 horizon sweep")
    lines.append("")
    lines.append(f"- horizons tested: {horizon_summary['T_list']}")
    lines.append(f"- counts csv: `{horizon_summary['counts_csv']}`")
    lines.append(f"- persistent stage3 csv: `{horizon_summary['persistent_stage3_csv']}`")
    lines.append(f"- persistent raw csv: `{horizon_summary['persistent_raw_csv']}`")
    lines.append("")
    lines.append("T=300 fixed-window count row:")
    lines.append("```json")
    lines.append(json.dumps(horizon_summary.get("T300_fixed_counts"), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("Largest requested horizon (fixed-window) count row:")
    lines.append("```json")
    lines.append(json.dumps(horizon_summary.get("Tlast_fixed_counts"), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Density-matrix validation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(density_summary, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Padding invariance validation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(padding_summary, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Recommended writing use")
    lines.append("")
    lines.append("1. Use the exact horizon sweep to decide whether your p=0 drift-Parrondo points persist or shrink with T.")
    lines.append("2. Use the density-matrix validation table to justify the noisy trajectory engine as an implementation of the Kraus channel.")
    lines.append("3. Use the padding validation table to close the protocol gap between mrc.yaml and the code actually executed.")
    lines.append("4. Use p=1 phi-invariance from the density summary to support the claim that the defect phase is irrelevant after complete dephasing/classicalization.")
    write_text(run_dir / "rigour_report.md", "\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------



def main() -> None:
    ap = argparse.ArgumentParser(description="Long-run rigour suite for the PH4511 DTQW project")
    ap.add_argument("--config", default="mrc.yaml")
    ap.add_argument("--stage4_run_dir", required=True, help="Path to outputs/run_*_stage4_exact_p0_audit")
    ap.add_argument("--repo_root", default=".", help="Repository root (default: current directory)")
    ap.add_argument("--tag", default="stage5_rigour_suite_longrun")
    ap.add_argument("--run_dir", default="", help="Existing run dir to resume into")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--t_list", default="300,600,900,1200,1500", help="Comma-separated exact p=0 horizons")
    ap.add_argument("--validation_n_traj", type=int, default=4000, help="Trajectories per density-validation point")
    ap.add_argument("--validation_T", type=int, default=120, help="Horizon for exact density validation (uses scaled windows)")
    ap.add_argument("--padding_n_traj", type=int, default=400, help="Trajectories per padding-validation point")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    stage4_run_dir = Path(args.stage4_run_dir)
    if not stage4_run_dir.is_absolute():
        stage4_run_dir = (repo_root / stage4_run_dir).resolve()
    if not stage4_run_dir.exists():
        raise FileNotFoundError(f"Stage-4 run dir not found: {stage4_run_dir}")

    cfg = load_yaml(cfg_path)
    cfg_r = resolve_from_mrc(cfg)
    stage4_ref = load_stage4_reference(stage4_run_dir)
    t_list = parse_t_list(args.t_list)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = (repo_root / run_dir).resolve()
        ensure_dir(run_dir)
        ensure_dir(run_dir / "figures")
    else:
        paths = make_run_dir(args.tag, base=str(repo_root / "outputs"))
        run_dir = paths["run_dir"].resolve()

    write_yaml(run_dir / "config_used.yaml", cfg)
    write_manifest(run_dir, tag=args.tag, extra={"repo_root": str(repo_root), "stage4_reference": str(stage4_run_dir)})

    print("[stage5] run_dir:", run_dir, flush=True)
    print("[stage5] exact horizons:", t_list, flush=True)
    print("[stage5] validation_n_traj:", args.validation_n_traj, flush=True)
    print("[stage5] validation_T:", args.validation_T, flush=True)
    print("[stage5] padding_n_traj:", args.padding_n_traj, flush=True)

    horizon_summary = run_exact_horizon_sweep(run_dir, t_list, stage4_ref, cfg_r, resume=bool(args.resume))
    density_summary = run_density_validation(repo_root, run_dir, stage4_ref, cfg_r, resume=bool(args.resume), N_traj_validation=int(args.validation_n_traj), validation_T=int(args.validation_T))
    padding_summary = run_padding_validation(repo_root, run_dir, stage4_ref, cfg_r, resume=bool(args.resume), N_traj_padding=int(args.padding_n_traj))

    overall = {
        "stage4_reference_run": str(stage4_run_dir),
        "exact_horizon_summary": horizon_summary,
        "density_validation_summary": density_summary,
        "padding_validation_summary": padding_summary,
        "key_output_files": [
            "exact_horizon_summary.json",
            "exact_horizon_counts.csv",
            "persistent_points_stage3.csv",
            "persistent_points_raw.csv",
            "density_validation_summary.json",
            "density_validation.csv",
            "p1_phi_invariance.csv",
            "padding_validation_summary.json",
            "padding_validation.csv",
            "rigour_report.md",
        ],
    }
    write_json(run_dir / "rigour_summary.json", overall)
    write_rigour_report(run_dir, stage4_ref, horizon_summary, density_summary, padding_summary)

    print("\n[done] stage5 rigour suite complete")
    print("[done] summary:", run_dir / "rigour_summary.json")
    print("[done] exact horizon counts:", run_dir / "exact_horizon_counts.csv")
    print("[done] density validation:", run_dir / "density_validation.csv")
    print("[done] padding validation:", run_dir / "padding_validation.csv")
    print("[done] report:", run_dir / "rigour_report.md")


if __name__ == "__main__":
    main()
