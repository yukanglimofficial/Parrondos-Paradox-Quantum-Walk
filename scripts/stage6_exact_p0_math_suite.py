#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dtqw.io import load_yaml, write_json, write_manifest, write_text, write_yaml
from dtqw.metrics import fit_v_fit


# -----------------------------------------------------------------------------
# Basic IO helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def make_run_dir(tag: str, base: str = "outputs") -> Dict[str, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"run_{stamp}_{tag}"
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figures")
    return {"run_dir": run_dir, "fig_dir": run_dir / "figures"}



def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def append_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = list(rows[0].keys())
        mode = "a"
        write_header = False
    else:
        header = list(rows[0].keys())
        mode = "w"
        write_header = True
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)



def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# -----------------------------------------------------------------------------
# Config / reference loading
# -----------------------------------------------------------------------------


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for chunk in text.split(","):
        s = chunk.strip()
        if s:
            out.append(int(s))
    if not out:
        raise ValueError("expected a non-empty integer list")
    return sorted(set(out))



def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for chunk in text.split(","):
        s = chunk.strip()
        if s:
            out.append(float(s))
    if not out:
        raise ValueError("expected a non-empty float list")
    return sorted(set(out))



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
        "N_phi_stage3": int(stage3["N_phi"]),
        "rel_tol": 0.5,
    }



def load_stage4_reference(stage4_run_dir: Path) -> Dict[str, Any]:
    summary_path = stage4_run_dir / "exact_p0_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Stage-4 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw = summary["thresholds_from_stage3"]

    def pick(*names: str) -> float:
        for name in names:
            if name in raw:
                return float(raw[name])
        raise KeyError(f"Missing threshold. Tried keys={names}; available={sorted(raw.keys())}")

    thresholds = {
        "vmin": pick("vmin", "v_min"),
        "eps_v": pick("eps_v", "epsilon_v"),
        "eps_P": pick("eps_P", "eps_p", "epsilon_P"),
        "w_thr_primary": pick("w_thr_primary"),
        "w_thr_sensitivity": pick("w_thr_sensitivity"),
    }

    head = summary["headline_points"]
    counts = summary["counts_exact_p0"]
    return {
        "summary": summary,
        "thresholds": thresholds,
        "thresholds_raw": raw,
        "counts": counts,
        "max_adv_phi": float(head["max_advantage"]["phi"]),
        "pp_sens_phi": None if head.get("first_pp_sensitivity") is None else float(head["first_pp_sensitivity"]["phi"]),
    }



def load_stage5_reference(stage5_run_dir: Path) -> Dict[str, Any]:
    summary_path = stage5_run_dir / "rigour_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Stage-5 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    exact = summary["exact_horizon_summary"]
    grid_path = stage5_run_dir / exact["grid_csv"]
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing Stage-5 exact horizon grid: {grid_path}")
    grid_rows = read_csv_rows(grid_path)

    # Identify key phis from the exact long-horizon sweep.
    T300 = exact.get("T300_fixed_counts")
    Tlast = exact.get("Tlast_fixed_counts")
    T_last = int(Tlast["T"])

    pp_primary_phi_last = None
    pp_primary_rows = [
        r for r in grid_rows
        if r["policy"] == "fixed" and int(r["T"]) == T_last and int(r["pp_primary"]) == 1
    ]
    if pp_primary_rows:
        pp_primary_phi_last = float(pp_primary_rows[0]["phi"])

    drift_T300 = {
        int(r["phi_index"])
        for r in grid_rows
        if r["policy"] == "fixed" and int(r["T"]) == int(T300["T"]) and int(r["drift_stage3"]) == 1
    }
    drift_Tlast = {
        int(r["phi_index"])
        for r in grid_rows
        if r["policy"] == "fixed" and int(r["T"]) == T_last and int(r["drift_stage3"]) == 1
    }
    newly_added_drift = sorted(drift_Tlast - drift_T300)
    newly_added_drift_phi = None
    if newly_added_drift:
        for r in grid_rows:
            if r["policy"] == "fixed" and int(r["T"]) == T_last and int(r["phi_index"]) == newly_added_drift[0]:
                newly_added_drift_phi = float(r["phi"])
                break

    return {
        "summary": summary,
        "exact": exact,
        "T300_fixed": T300,
        "Tlast_fixed": Tlast,
        "T_last": T_last,
        "grid_rows": grid_rows,
        "pp_primary_phi_last": pp_primary_phi_last,
        "newly_added_drift_phi": newly_added_drift_phi,
    }


# -----------------------------------------------------------------------------
# Exact p=0 simulator and metric extraction on the infinite line
# -----------------------------------------------------------------------------


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
# Region refinement on p=0
# -----------------------------------------------------------------------------


@dataclass
class RegionSpec:
    name: str
    center_phi: float
    halfwidth: float
    n_points: int



def wrap_phi(phi: float) -> float:
    twopi = 2.0 * math.pi
    out = phi % twopi
    if out < 0.0:
        out += twopi
    return out



def make_region_phi_grid(center_phi: float, halfwidth: float, n_points: int) -> np.ndarray:
    grid = np.linspace(center_phi - halfwidth, center_phi + halfwidth, int(n_points), endpoint=True, dtype=np.float64)
    return np.mod(grid, 2.0 * math.pi)



def derive_region_specs(stage4_ref: Dict[str, Any], stage5_ref: Dict[str, Any], coarse_step: float, n_points: int) -> List[RegionSpec]:
    T300_phi = float(stage5_ref["T300_fixed"]["max_adv_phi"])
    Tlast_phi = float(stage5_ref["Tlast_fixed"]["max_adv_phi"])
    pp_phi = stage5_ref["pp_primary_phi_last"]
    if pp_phi is None:
        pp_phi = stage4_ref["pp_sens_phi"]
    if pp_phi is None:
        raise RuntimeError("Could not determine PP-centered phi from Stage-4/5 outputs")

    regions = [
        RegionSpec(
            name="max_region",
            center_phi=float(0.5 * (T300_phi + Tlast_phi)),
            halfwidth=float(2.0 * coarse_step),
            n_points=int(n_points),
        ),
        RegionSpec(
            name="pp_region",
            center_phi=float(pp_phi),
            halfwidth=float(1.5 * coarse_step),
            n_points=int(n_points),
        ),
    ]
    border_phi = stage5_ref.get("newly_added_drift_phi")
    if border_phi is not None:
        regions.append(
            RegionSpec(
                name="borderline_drift_region",
                center_phi=float(border_phi),
                halfwidth=float(1.0 * coarse_step),
                n_points=int(max(161, n_points // 2)),
            )
        )
    return regions



def compute_decision_flags(
    mA: PointMetrics,
    mB: PointMetrics,
    mABB: PointMetrics,
    thresholds: Dict[str, float],
    cfg_r: Dict[str, Any],
) -> Dict[str, int]:
    vmin = float(thresholds.get("vmin", thresholds.get("v_min")))
    eps_v = float(thresholds.get("eps_v", thresholds.get("epsilon_v")))
    eps_P = float(thresholds.get("eps_P", thresholds.get("eps_p", thresholds.get("epsilon_P"))))

    A_losing = bool(mA.stable and (mA.v_fit < -vmin))
    B_losing = bool(mB.stable and (mB.v_fit < -vmin))
    ABB_winning = bool(mABB.stable and (mABB.v_fit > vmin))

    strict_raw = int((mA.v_fit < 0.0) and (mB.v_fit < 0.0))
    drift_raw = int(strict_raw and (mABB.v_fit > 0.0))
    strict_stage3 = int(A_losing and B_losing)
    drift_stage3 = int(strict_stage3 and ABB_winning)
    pp_primary = int(drift_stage3 and (mABB.w_loc3 < cfg_r["w_thr_primary"]))
    pp_sensitivity = int(drift_stage3 and (mABB.w_loc5 < cfg_r["w_thr_sensitivity"]))
    mismatch = int(
        (sign_eps_scalar(mABB.v_fit, eps_v) != 0)
        and (sign_eps_scalar(mABB.deltaP_late_mean, eps_P) != 0)
        and (sign_eps_scalar(mABB.v_fit, eps_v) != sign_eps_scalar(mABB.deltaP_late_mean, eps_P))
    )

    return {
        "strict_raw": int(strict_raw),
        "drift_raw": int(drift_raw),
        "strict_stage3": int(strict_stage3),
        "drift_stage3": int(drift_stage3),
        "pp_primary": int(pp_primary),
        "pp_sensitivity": int(pp_sensitivity),
        "mismatch_sign": int(mismatch),
    }



def run_region_refinement(
    run_dir: Path,
    stage4_ref: Dict[str, Any],
    stage5_ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    T_refine_list: List[int],
    refine_points: int,
    resume: bool,
) -> Dict[str, Any]:
    region_csv = run_dir / "region_refine_grid.csv"
    summary_json = run_dir / "region_refine_summary.json"
    intervals_csv = run_dir / "region_refine_intervals.csv"

    coarse_step = 2.0 * math.pi / float(cfg_r["N_phi_stage3"])
    regions = derive_region_specs(stage4_ref, stage5_ref, coarse_step=coarse_step, n_points=refine_points)

    coin_params = cfg_r["coin_params"]
    C_A = su2_coin(*coin_params["A"], degrees=True)
    C_B = su2_coin(*coin_params["B"], degrees=True)
    Tmax = int(max(T_refine_list))
    thresholds = stage4_ref["thresholds"]
    rel_tol = float(cfg_r["rel_tol"])

    existing_rows = read_csv_rows(region_csv) if resume else []
    done_keys = {
        (r["region"], int(r["phi_grid_index"]))
        for r in existing_rows
    }

    if not resume and region_csv.exists():
        region_csv.unlink()

    for region in regions:
        phi_grid = make_region_phi_grid(region.center_phi, region.halfwidth, region.n_points)
        for i, phi in enumerate(phi_grid):
            key = (region.name, int(i))
            if key in done_keys:
                continue
            series_A = simulate_unitary_scalar_timeseries(
                seq="A",
                Tmax=Tmax,
                phi=float(phi),
                C_A=C_A,
                C_B=C_B,
                x0_primary=cfg_r["x0_primary"],
                x0_sens=cfg_r["x0_sensitivity"],
            )
            series_B = simulate_unitary_scalar_timeseries(
                seq="B",
                Tmax=Tmax,
                phi=float(phi),
                C_A=C_A,
                C_B=C_B,
                x0_primary=cfg_r["x0_primary"],
                x0_sens=cfg_r["x0_sensitivity"],
            )
            series_ABB = simulate_unitary_scalar_timeseries(
                seq="ABB",
                Tmax=Tmax,
                phi=float(phi),
                C_A=C_A,
                C_B=C_B,
                x0_primary=cfg_r["x0_primary"],
                x0_sens=cfg_r["x0_sensitivity"],
            )

            rows_to_append: List[Dict[str, Any]] = []
            for policy in ["fixed", "scaled"]:
                for T in T_refine_list:
                    if policy == "fixed":
                        T0 = int(cfg_r["T0_stage3"])
                        T1 = int(cfg_r["T1_stage3"])
                    else:
                        T0, T1 = scaled_windows(int(T))
                    if not (0 <= T0 < T1 <= T):
                        continue

                    mA = fit_tail_and_metrics(series_A, int(T), int(T0), int(T1), rel_tol)
                    mB = fit_tail_and_metrics(series_B, int(T), int(T0), int(T1), rel_tol)
                    mABB = fit_tail_and_metrics(series_ABB, int(T), int(T0), int(T1), rel_tol)
                    flags = compute_decision_flags(mA, mB, mABB, thresholds, cfg_r)
                    adv = float(mABB.v_fit - max(mA.v_fit, mB.v_fit))

                    rows_to_append.append(
                        {
                            "region": region.name,
                            "phi_grid_index": int(i),
                            "phi": float(phi),
                            "phi_over_pi": float(phi / math.pi),
                            "region_center_phi": float(region.center_phi),
                            "region_center_phi_over_pi": float(region.center_phi / math.pi),
                            "region_halfwidth": float(region.halfwidth),
                            "region_halfwidth_over_pi": float(region.halfwidth / math.pi),
                            "policy": policy,
                            "T": int(T),
                            "T0": int(T0),
                            "T1": int(T1),
                            "v_fit_A": float(mA.v_fit),
                            "v_fit_B": float(mB.v_fit),
                            "v_fit_ABB": float(mABB.v_fit),
                            "v_fit2_A": float(mA.v_fit2),
                            "v_fit2_B": float(mB.v_fit2),
                            "v_fit2_ABB": float(mABB.v_fit2),
                            "v_T_A": float(mA.v_T),
                            "v_T_B": float(mB.v_T),
                            "v_T_ABB": float(mABB.v_T),
                            "delta_v_A": float(mA.delta_v),
                            "delta_v_B": float(mB.delta_v),
                            "delta_v_ABB": float(mABB.delta_v),
                            "deltaP_late_mean_ABB": float(mABB.deltaP_late_mean),
                            "deltaP_final_ABB": float(mABB.deltaP_final),
                            "w_loc3_ABB": float(mABB.w_loc3),
                            "w_loc5_ABB": float(mABB.w_loc5),
                            "P0bar_ABB": float(mABB.P0bar),
                            "stable_A": int(mA.stable),
                            "stable_B": int(mB.stable),
                            "stable_ABB": int(mABB.stable),
                            **flags,
                            "adv_v": float(adv),
                        }
                    )
            append_csv_rows(region_csv, rows_to_append)
            done_keys.add(key)

    all_rows = read_csv_rows(region_csv)

    # Summaries: local maxima and sampled connected intervals for selected masks.
    summary: Dict[str, Any] = {
        "T_refine_list": [int(t) for t in T_refine_list],
        "coarse_step": float(coarse_step),
        "coarse_step_over_pi": float(coarse_step / math.pi),
        "region_csv": str(region_csv.relative_to(run_dir)),
        "intervals_csv": str(intervals_csv.relative_to(run_dir)),
        "regions": [],
    }
    interval_rows: List[Dict[str, Any]] = []
    mask_names = ["strict_stage3", "drift_stage3", "pp_primary", "pp_sensitivity"]

    for region in regions:
        region_payload: Dict[str, Any] = {
            "region": region.name,
            "center_phi": float(region.center_phi),
            "center_phi_over_pi": float(region.center_phi / math.pi),
            "halfwidth": float(region.halfwidth),
            "halfwidth_over_pi": float(region.halfwidth / math.pi),
            "policies": [],
        }
        for policy in ["fixed", "scaled"]:
            policy_payload: Dict[str, Any] = {"policy": policy, "T": []}
            for T in T_refine_list:
                sub = [
                    r for r in all_rows
                    if r["region"] == region.name and r["policy"] == policy and int(r["T"]) == int(T)
                ]
                sub_sorted = sorted(sub, key=lambda r: int(r["phi_grid_index"]))
                if not sub_sorted:
                    continue
                adv_vals = np.array([float(r["adv_v"]) for r in sub_sorted], dtype=np.float64)
                i_max = int(np.argmax(adv_vals))
                row_max = sub_sorted[i_max]
                T_payload: Dict[str, Any] = {
                    "T": int(T),
                    "T0": int(sub_sorted[0]["T0"]),
                    "T1": int(sub_sorted[0]["T1"]),
                    "max_adv_phi": float(row_max["phi"]),
                    "max_adv_phi_over_pi": float(row_max["phi_over_pi"]),
                    "max_adv_v": float(row_max["adv_v"]),
                    "max_adv_w_loc3": float(row_max["w_loc3_ABB"]),
                    "max_adv_w_loc5": float(row_max["w_loc5_ABB"]),
                    "max_adv_v_fit_ABB": float(row_max["v_fit_ABB"]),
                    "intervals": {},
                }
                for mask_name in mask_names:
                    mask = np.array([int(r[mask_name]) == 1 for r in sub_sorted], dtype=bool)
                    intervals: List[Tuple[int, int]] = []
                    start = None
                    for idx, flag in enumerate(mask):
                        if flag and start is None:
                            start = idx
                        elif (not flag) and start is not None:
                            intervals.append((start, idx - 1))
                            start = None
                    if start is not None:
                        intervals.append((start, len(mask) - 1))
                    T_payload["intervals"][mask_name] = []
                    for k, (a, b) in enumerate(intervals):
                        ra = sub_sorted[a]
                        rb = sub_sorted[b]
                        width = float((float(rb["phi"]) - float(ra["phi"])) if b >= a else 0.0)
                        payload = {
                            "index": int(k),
                            "phi_left": float(ra["phi"]),
                            "phi_right": float(rb["phi"]),
                            "phi_left_over_pi": float(ra["phi_over_pi"]),
                            "phi_right_over_pi": float(rb["phi_over_pi"]),
                            "n_points": int(b - a + 1),
                            "sampled_width": float(width),
                            "sampled_width_over_pi": float(width / math.pi),
                        }
                        T_payload["intervals"][mask_name].append(payload)
                        interval_rows.append(
                            {
                                "region": region.name,
                                "policy": policy,
                                "T": int(T),
                                "mask_name": mask_name,
                                "interval_index": int(k),
                                "phi_left": float(ra["phi"]),
                                "phi_right": float(rb["phi"]),
                                "phi_left_over_pi": float(ra["phi_over_pi"]),
                                "phi_right_over_pi": float(rb["phi_over_pi"]),
                                "n_points": int(b - a + 1),
                                "sampled_width": float(width),
                                "sampled_width_over_pi": float(width / math.pi),
                            }
                        )
                policy_payload["T"].append(T_payload)
            region_payload["policies"].append(policy_payload)
        summary["regions"].append(region_payload)

    write_csv(intervals_csv, interval_rows)
    write_json(summary_json, summary)
    return summary


# -----------------------------------------------------------------------------
# Exact long timeseries on the infinite line for representative phis
# -----------------------------------------------------------------------------


@dataclass
class TimeSeriesPoint:
    label: str
    phi: float



def derive_timeseries_points(stage4_ref: Dict[str, Any], stage5_ref: Dict[str, Any]) -> List[TimeSeriesPoint]:
    pts = [
        TimeSeriesPoint(label="paper_max_T300", phi=float(stage5_ref["T300_fixed"]["max_adv_phi"])),
        TimeSeriesPoint(label="long_max_Tlast", phi=float(stage5_ref["Tlast_fixed"]["max_adv_phi"])),
    ]
    pp_phi = stage5_ref.get("pp_primary_phi_last")
    if pp_phi is None:
        pp_phi = stage4_ref.get("pp_sens_phi")
    if pp_phi is not None:
        pts.append(TimeSeriesPoint(label="pp_primary_long" if stage5_ref.get("pp_primary_phi_last") is not None else "pp_sensitivity_stage4", phi=float(pp_phi)))
    # Deduplicate by rounded phi while keeping first label.
    out: List[TimeSeriesPoint] = []
    seen = set()
    for pt in pts:
        key = round(float(pt.phi), 14)
        if key in seen:
            continue
        seen.add(key)
        out.append(pt)
    return out



def default_checkpoint_list(Tmax: int) -> List[int]:
    base = [300, 600, 900, 1200, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
    out = [t for t in base if t <= Tmax]
    if Tmax not in out:
        out.append(Tmax)
    return sorted(set(out))



def run_exact_long_timeseries(
    run_dir: Path,
    stage4_ref: Dict[str, Any],
    stage5_ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    T_timeseries: int,
    resume: bool,
) -> Dict[str, Any]:
    ts_dir = ensure_dir(run_dir / "exact_long_timeseries")
    summary_csv = run_dir / "exact_long_windowed_summary.csv"
    summary_json = run_dir / "exact_long_timeseries_summary.json"

    if not resume and summary_csv.exists():
        summary_csv.unlink()

    existing_rows = read_csv_rows(summary_csv) if resume else []
    done_keys = {
        (r["label"], r["sequence"]) for r in existing_rows
    }

    coin_params = cfg_r["coin_params"]
    C_A = su2_coin(*coin_params["A"], degrees=True)
    C_B = su2_coin(*coin_params["B"], degrees=True)
    rel_tol = float(cfg_r["rel_tol"])
    thresholds = stage4_ref["thresholds"]
    points = derive_timeseries_points(stage4_ref, stage5_ref)
    checkpoints = default_checkpoint_list(int(T_timeseries))

    for pt in points:
        for seq in ["A", "B", "ABB"]:
            key = (pt.label, seq)
            if key in done_keys:
                continue
            ser = simulate_unitary_scalar_timeseries(
                seq=seq,
                Tmax=int(T_timeseries),
                phi=float(pt.phi),
                C_A=C_A,
                C_B=C_B,
                x0_primary=cfg_r["x0_primary"],
                x0_sens=cfg_r["x0_sensitivity"],
            )
            # Save raw timeseries csv for future plotting and writing.
            ts_rows: List[Dict[str, Any]] = []
            for t in range(int(T_timeseries) + 1):
                ts_rows.append(
                    {
                        "label": pt.label,
                        "sequence": seq,
                        "phi": float(pt.phi),
                        "phi_over_pi": float(pt.phi / math.pi),
                        "t": int(t),
                        "x_mean": float(ser.x_mean_t[t]),
                        "deltaP": float(ser.deltaP_t[t]),
                        "w_loc3_inst": float(ser.w3_t[t]),
                        "w_loc5_inst": float(ser.w5_t[t]),
                        "P0": float(ser.P0_t[t]),
                    }
                )
            write_csv(ts_dir / f"timeseries_{pt.label}_{seq}.csv", ts_rows)

            out_rows: List[Dict[str, Any]] = []
            for policy in ["fixed", "scaled"]:
                for T in checkpoints:
                    if policy == "fixed":
                        T0 = int(cfg_r["T0_stage3"])
                        T1 = int(cfg_r["T1_stage3"])
                    else:
                        T0, T1 = scaled_windows(int(T))
                    if not (0 <= T0 < T1 <= T):
                        continue
                    m = fit_tail_and_metrics(ser, int(T), int(T0), int(T1), rel_tol)
                    out_rows.append(
                        {
                            "label": pt.label,
                            "sequence": seq,
                            "phi": float(pt.phi),
                            "phi_over_pi": float(pt.phi / math.pi),
                            "policy": policy,
                            "T": int(T),
                            "T0": int(T0),
                            "T1": int(T1),
                            "v_fit": float(m.v_fit),
                            "v_fit2": float(m.v_fit2),
                            "v_T": float(m.v_T),
                            "delta_v": float(m.delta_v),
                            "deltaP_late_mean": float(m.deltaP_late_mean),
                            "deltaP_final": float(m.deltaP_final),
                            "w_loc3": float(m.w_loc3),
                            "w_loc5": float(m.w_loc5),
                            "P0bar": float(m.P0bar),
                            "x_mean_final": float(m.x_mean_final),
                            "stable": int(m.stable),
                        }
                    )
            append_csv_rows(summary_csv, out_rows)
            done_keys.add(key)

    all_rows = read_csv_rows(summary_csv)

    # Build high-level summaries for the paper-critical phis.
    points_payload: List[Dict[str, Any]] = []
    for pt in points:
        payload = {
            "label": pt.label,
            "phi": float(pt.phi),
            "phi_over_pi": float(pt.phi / math.pi),
            "sequence_summaries": {},
        }
        for seq in ["A", "B", "ABB"]:
            sub = [
                r for r in all_rows
                if r["label"] == pt.label and r["sequence"] == seq and r["policy"] == "fixed"
            ]
            sub_sorted = sorted(sub, key=lambda r: int(r["T"]))
            if sub_sorted:
                payload["sequence_summaries"][seq] = {
                    "T_values": [int(r["T"]) for r in sub_sorted],
                    "v_fit_values": [float(r["v_fit"]) for r in sub_sorted],
                    "w_loc3_values": [float(r["w_loc3"]) for r in sub_sorted],
                    "w_loc5_values": [float(r["w_loc5"]) for r in sub_sorted],
                    "P0bar_values": [float(r["P0bar"]) for r in sub_sorted],
                }
        points_payload.append(payload)

    summary = {
        "T_timeseries": int(T_timeseries),
        "summary_csv": str(summary_csv.relative_to(run_dir)),
        "timeseries_dir": str(ts_dir.relative_to(run_dir)),
        "points": points_payload,
    }
    write_json(summary_json, summary)
    return summary


# -----------------------------------------------------------------------------
# Spectral / Floquet proxy on a periodic ring
# -----------------------------------------------------------------------------



def apply_shift_periodic_state(psi: np.ndarray, out: np.ndarray) -> np.ndarray:
    out[0, :] = np.roll(psi[0, :], 1)
    out[1, :] = np.roll(psi[1, :], -1)
    return out



def build_step_matrix_on_ring(C: np.ndarray, phi: float, L: int) -> np.ndarray:
    N = 2 * int(L) + 1
    dim = 2 * N
    idx0 = int(L)
    eiphi = complex(math.cos(phi), math.sin(phi))
    step = np.zeros((dim, dim), dtype=np.complex128)
    basis = np.zeros((2, N), dtype=np.complex128)
    mid = np.zeros_like(basis)
    nxt = np.zeros_like(basis)
    for col in range(dim):
        basis.fill(0.0)
        coin_idx = 0 if col < N else 1
        pos_idx = col if col < N else (col - N)
        basis[coin_idx, pos_idx] = 1.0 + 0.0j
        apply_coin_state(basis, C, mid)
        apply_origin_defect_state(mid, idx0, eiphi)
        apply_shift_periodic_state(mid, nxt)
        step[:, col] = nxt.reshape(-1)
    # Unitarity check with mild tolerance.
    eye = np.eye(dim, dtype=np.complex128)
    err = np.linalg.norm(step.conj().T @ step - eye)
    if err > 1e-9:
        raise RuntimeError(f"Constructed ring step is not unitary enough; L={L}, err={err}")
    return step



def build_period_operator_and_prefixes(seq: str, C_A: np.ndarray, C_B: np.ndarray, phi: float, L: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    pat = list(seq)
    if not pat:
        raise ValueError("sequence must be non-empty")
    step_mats: List[np.ndarray] = []
    for sym in pat:
        C = C_A if sym == "A" else C_B
        step_mats.append(build_step_matrix_on_ring(C=C, phi=phi, L=L))
    dim = step_mats[0].shape[0]
    F = np.eye(dim, dtype=np.complex128)
    for U in step_mats:
        F = U @ F
    prefixes: List[np.ndarray] = [np.eye(dim, dtype=np.complex128)]
    W = np.eye(dim, dtype=np.complex128)
    for U in step_mats[:-1]:
        W = U @ W
        prefixes.append(W.copy())
    return F, prefixes



def initial_state_on_ring(L: int) -> np.ndarray:
    N = 2 * int(L) + 1
    psi = np.zeros((2, N), dtype=np.complex128)
    psi[:, int(L)] = np.array([1.0, -1.0j], dtype=np.complex128) / np.sqrt(2.0)
    return psi.reshape(-1)



def diag_prob_and_observables(vec: np.ndarray, L: int, x0_primary: int, x0_sens: int) -> Dict[str, float]:
    N = 2 * int(L) + 1
    amp = vec.reshape(2, N)
    prob = (np.abs(amp[0]) ** 2 + np.abs(amp[1]) ** 2).real.astype(np.float64)
    # Normalize defensively.
    s = float(prob.sum())
    if s > 0.0:
        prob = prob / s
    idx0 = int(L)
    w3 = float(prob[idx0 - x0_primary : idx0 + x0_primary + 1].sum())
    w5 = float(prob[idx0 - x0_sens : idx0 + x0_sens + 1].sum())
    p0 = float(prob[idx0])
    ipr = float(np.sum(prob**2))
    return {"w_loc3": w3, "w_loc5": w5, "P0": p0, "ipr": ipr}



def diagonal_ensemble_period_average(
    eigvecs: np.ndarray,
    overlaps: np.ndarray,
    prefixes: List[np.ndarray],
    L: int,
    x0_primary: int,
    x0_sens: int,
) -> Dict[str, float]:
    dim = eigvecs.shape[0]
    n = eigvecs.shape[1]
    # Each column of eigvecs is an eigenvector.
    if overlaps.shape != (n,):
        raise ValueError("overlaps has wrong shape")
    obs_m = []
    for W in prefixes:
        w3 = 0.0
        w5 = 0.0
        p0 = 0.0
        for j in range(n):
            vec_m = W @ eigvecs[:, j]
            obs = diag_prob_and_observables(vec_m, L=L, x0_primary=x0_primary, x0_sens=x0_sens)
            weight = float(overlaps[j])
            w3 += weight * obs["w_loc3"]
            w5 += weight * obs["w_loc5"]
            p0 += weight * obs["P0"]
        obs_m.append((w3, w5, p0))
    arr = np.asarray(obs_m, dtype=np.float64)
    return {
        "diag_period_avg_w_loc3": float(np.mean(arr[:, 0])),
        "diag_period_avg_w_loc5": float(np.mean(arr[:, 1])),
        "diag_period_avg_P0": float(np.mean(arr[:, 2])),
    }



def simulate_on_ring_period_average(
    seq: str,
    C_A: np.ndarray,
    C_B: np.ndarray,
    phi: float,
    L: int,
    x0_primary: int,
    x0_sens: int,
    n_cycles: int,
    burn_cycles: int,
) -> Dict[str, float]:
    if burn_cycles >= n_cycles:
        raise ValueError("burn_cycles must be smaller than n_cycles")
    N = 2 * int(L) + 1
    idx0 = int(L)
    psi = initial_state_on_ring(L).reshape(2, N)
    pat = list(seq)
    eiphi = complex(math.cos(phi), math.sin(phi))
    mid = np.zeros_like(psi)
    nxt = np.zeros_like(psi)

    w3_vals: List[float] = []
    w5_vals: List[float] = []
    p0_vals: List[float] = []
    for cyc in range(int(n_cycles)):
        for sym in pat:
            C = C_A if sym == "A" else C_B
            apply_coin_state(psi, C, mid)
            apply_origin_defect_state(mid, idx0, eiphi)
            apply_shift_periodic_state(mid, nxt)
            psi, nxt = nxt, psi
            if cyc >= int(burn_cycles):
                prob = (np.abs(psi[0]) ** 2 + np.abs(psi[1]) ** 2).real.astype(np.float64)
                prob = prob / float(prob.sum())
                w3_vals.append(float(prob[idx0 - x0_primary : idx0 + x0_primary + 1].sum()))
                w5_vals.append(float(prob[idx0 - x0_sens : idx0 + x0_sens + 1].sum()))
                p0_vals.append(float(prob[idx0]))
    return {
        "ring_direct_period_avg_w_loc3": float(np.mean(np.asarray(w3_vals, dtype=np.float64))),
        "ring_direct_period_avg_w_loc5": float(np.mean(np.asarray(w5_vals, dtype=np.float64))),
        "ring_direct_period_avg_P0": float(np.mean(np.asarray(p0_vals, dtype=np.float64))),
    }



def run_spectral_proxy(
    run_dir: Path,
    stage4_ref: Dict[str, Any],
    stage5_ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    spectral_L_list: List[int],
    spectral_sequences: List[str],
    top_k_modes: int,
    n_cycles_ring: int,
    burn_cycles_ring: int,
) -> Dict[str, Any]:
    top_modes_csv = run_dir / "spectral_top_modes.csv"
    conv_csv = run_dir / "spectral_convergence.csv"
    summary_json = run_dir / "spectral_summary.json"

    coin_params = cfg_r["coin_params"]
    C_A = su2_coin(*coin_params["A"], degrees=True)
    C_B = su2_coin(*coin_params["B"], degrees=True)

    selected_points = derive_timeseries_points(stage4_ref, stage5_ref)
    top_rows: List[Dict[str, Any]] = []
    conv_rows: List[Dict[str, Any]] = []

    for seq in spectral_sequences:
        for pt in selected_points:
            prev_diag = None
            for L in spectral_L_list:
                F, prefixes = build_period_operator_and_prefixes(seq=seq, C_A=C_A, C_B=C_B, phi=pt.phi, L=int(L))
                eigvals, eigvecs = np.linalg.eig(F)

                # Normalize columns defensively.
                for j in range(eigvecs.shape[1]):
                    nrm = np.linalg.norm(eigvecs[:, j])
                    if nrm > 0.0:
                        eigvecs[:, j] /= nrm

                psi0 = initial_state_on_ring(int(L))
                coeffs = eigvecs.conj().T @ psi0
                overlaps = (np.abs(coeffs) ** 2).real.astype(np.float64)
                if overlaps.sum() > 0.0:
                    overlaps /= overlaps.sum()

                mode_obs: List[Dict[str, Any]] = []
                for j in range(eigvecs.shape[1]):
                    obs = diag_prob_and_observables(eigvecs[:, j], L=int(L), x0_primary=cfg_r["x0_primary"], x0_sens=cfg_r["x0_sensitivity"])
                    phase = float(np.angle(eigvals[j]))
                    mode_obs.append(
                        {
                            "mode_index": int(j),
                            "eigenphase": phase,
                            "abs_eigenvalue_minus_1": float(abs(abs(eigvals[j]) - 1.0)),
                            "w_loc3": float(obs["w_loc3"]),
                            "w_loc5": float(obs["w_loc5"]),
                            "P0": float(obs["P0"]),
                            "ipr": float(obs["ipr"]),
                            "overlap_weight": float(overlaps[j]),
                        }
                    )
                mode_obs_sorted = sorted(mode_obs, key=lambda r: (r["w_loc5"], r["ipr"], r["overlap_weight"]), reverse=True)
                for rank, rec in enumerate(mode_obs_sorted[: int(top_k_modes)]):
                    top_rows.append(
                        {
                            "sequence": seq,
                            "label": pt.label,
                            "phi": float(pt.phi),
                            "phi_over_pi": float(pt.phi / math.pi),
                            "L": int(L),
                            "rank_by_w_loc5": int(rank + 1),
                            **rec,
                        }
                    )

                diag_avg = diagonal_ensemble_period_average(
                    eigvecs=eigvecs,
                    overlaps=overlaps,
                    prefixes=prefixes,
                    L=int(L),
                    x0_primary=cfg_r["x0_primary"],
                    x0_sens=cfg_r["x0_sensitivity"],
                )
                ring_avg = simulate_on_ring_period_average(
                    seq=seq,
                    C_A=C_A,
                    C_B=C_B,
                    phi=pt.phi,
                    L=int(L),
                    x0_primary=cfg_r["x0_primary"],
                    x0_sens=cfg_r["x0_sensitivity"],
                    n_cycles=int(n_cycles_ring),
                    burn_cycles=int(burn_cycles_ring),
                )

                topk_overlap = float(sum(r["overlap_weight"] for r in mode_obs_sorted[: int(top_k_modes)]))
                row = {
                    "sequence": seq,
                    "label": pt.label,
                    "phi": float(pt.phi),
                    "phi_over_pi": float(pt.phi / math.pi),
                    "L": int(L),
                    "topk_overlap_weight": float(topk_overlap),
                    **diag_avg,
                    **ring_avg,
                    "abs_diff_diag_vs_ring_w_loc3": float(abs(diag_avg["diag_period_avg_w_loc3"] - ring_avg["ring_direct_period_avg_w_loc3"])),
                    "abs_diff_diag_vs_ring_w_loc5": float(abs(diag_avg["diag_period_avg_w_loc5"] - ring_avg["ring_direct_period_avg_w_loc5"])),
                    "abs_diff_diag_vs_ring_P0": float(abs(diag_avg["diag_period_avg_P0"] - ring_avg["ring_direct_period_avg_P0"])),
                }
                if prev_diag is None:
                    row.update(
                        {
                            "delta_from_prev_L_w_loc3": np.nan,
                            "delta_from_prev_L_w_loc5": np.nan,
                            "delta_from_prev_L_P0": np.nan,
                            "prev_L": np.nan,
                        }
                    )
                else:
                    row.update(
                        {
                            "delta_from_prev_L_w_loc3": float(abs(diag_avg["diag_period_avg_w_loc3"] - prev_diag["diag_period_avg_w_loc3"])),
                            "delta_from_prev_L_w_loc5": float(abs(diag_avg["diag_period_avg_w_loc5"] - prev_diag["diag_period_avg_w_loc5"])),
                            "delta_from_prev_L_P0": float(abs(diag_avg["diag_period_avg_P0"] - prev_diag["diag_period_avg_P0"])),
                            "prev_L": int(prev_diag["L"]),
                        }
                    )
                conv_rows.append(row)
                prev_diag = {**diag_avg, "L": int(L)}

    write_csv(top_modes_csv, top_rows)
    write_csv(conv_csv, conv_rows)

    summary = {
        "selected_points": [
            {"label": pt.label, "phi": float(pt.phi), "phi_over_pi": float(pt.phi / math.pi)}
            for pt in selected_points
        ],
        "spectral_sequences": spectral_sequences,
        "spectral_L_list": [int(L) for L in spectral_L_list],
        "top_k_modes": int(top_k_modes),
        "n_cycles_ring": int(n_cycles_ring),
        "burn_cycles_ring": int(burn_cycles_ring),
        "top_modes_csv": str(top_modes_csv.relative_to(run_dir)),
        "convergence_csv": str(conv_csv.relative_to(run_dir)),
    }
    write_json(summary_json, summary)
    return summary


# -----------------------------------------------------------------------------
# Write a compact report
# -----------------------------------------------------------------------------



def build_report(
    run_dir: Path,
    stage4_ref: Dict[str, Any],
    stage5_ref: Dict[str, Any],
    region_summary: Dict[str, Any],
    timeseries_summary: Dict[str, Any],
    spectral_summary: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Stage-6 Exact p=0 mathematics suite report")
    lines.append("")
    lines.append("## Starting point from Stage 4 and Stage 5")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({
        "stage4_counts_exact_p0": stage4_ref["counts"],
        "stage5_T300_fixed": stage5_ref["T300_fixed"],
        "stage5_Tlast_fixed": stage5_ref["Tlast_fixed"],
    }, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Region refinement outputs")
    lines.append("")
    lines.append(f"- summary json: `{(run_dir / 'region_refine_summary.json').name}`")
    lines.append(f"- full refined grid: `{(run_dir / 'region_refine_grid.csv').name}`")
    lines.append(f"- sampled mask intervals: `{(run_dir / 'region_refine_intervals.csv').name}`")
    lines.append("")
    lines.append("## Long exact timeseries outputs")
    lines.append("")
    lines.append(f"- summary json: `{(run_dir / 'exact_long_timeseries_summary.json').name}`")
    lines.append(f"- windowed summary csv: `{(run_dir / 'exact_long_windowed_summary.csv').name}`")
    lines.append(f"- raw timeseries dir: `exact_long_timeseries/`")
    lines.append("")
    lines.append("## Spectral / Floquet proxy outputs")
    lines.append("")
    lines.append(f"- summary json: `{(run_dir / 'spectral_summary.json').name}`")
    lines.append(f"- top localized modes: `{(run_dir / 'spectral_top_modes.csv').name}`")
    lines.append(f"- convergence and diagonal-ensemble comparison: `{(run_dir / 'spectral_convergence.csv').name}`")
    lines.append("")
    lines.append("## What to look for next in the outputs")
    lines.append("")
    lines.append("1. Does the PP-primary mask around the long-horizon point at phi/pi≈5/9 open into a nonzero interval under local refinement?")
    lines.append("2. Does the high-advantage peak stay near the T=1500 long-horizon location rather than the original T=300 location?")
    lines.append("3. In the spectral proxy, do the high-advantage points have larger overlap with top localized Floquet modes than the PP-primary point?")
    lines.append("4. Do the diagonal-ensemble localization predictions stabilize as the ring size L increases and agree with direct ring time averages?")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 6 exact p=0 mathematics suite: local refinement, long exact timeseries, and spectral/Floquet localization proxy.")
    p.add_argument("--config", required=True, help="Path to mrc.yaml")
    p.add_argument("--stage4_run_dir", required=True, help="Path to Stage-4 exact p=0 audit run directory")
    p.add_argument("--stage5_run_dir", required=True, help="Path to Stage-5 rigour-suite run directory")
    p.add_argument("--run_dir", default=None, help="Optional existing/new output directory")
    p.add_argument("--resume", action="store_true", help="Resume from partially written CSV outputs where possible")
    p.add_argument("--T_refine_list", default="1500,3000", help="Comma-separated exact horizons for local phi refinement")
    p.add_argument("--refine_points", type=int, default=241, help="Points per local phi refinement window")
    p.add_argument("--T_timeseries", type=int, default=6000, help="Maximum exact horizon for representative infinite-line timeseries")
    p.add_argument("--spectral_L_list", default="80,120,160", help="Comma-separated ring half-sizes L for Floquet spectral proxy")
    p.add_argument("--spectral_sequences", default="ABB", help="Comma-separated sequences for spectral proxy, e.g. ABB or A,B,ABB")
    p.add_argument("--top_k_modes", type=int, default=8, help="How many most localized Floquet modes to save per spectral job")
    p.add_argument("--n_cycles_ring", type=int, default=1500, help="Number of periods for direct ring averaging in the spectral proxy")
    p.add_argument("--burn_cycles_ring", type=int, default=300, help="Burn-in periods discarded in direct ring averaging")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg_r = resolve_from_mrc(cfg)
    stage4_ref = load_stage4_reference(Path(args.stage4_run_dir))
    stage5_ref = load_stage5_reference(Path(args.stage5_run_dir))

    if args.run_dir is None:
        rd = make_run_dir("stage6_exact_p0_math_suite")
        run_dir = rd["run_dir"]
    else:
        run_dir = Path(args.run_dir)
        ensure_dir(run_dir)
        ensure_dir(run_dir / "figures")

    write_yaml(run_dir / "config_used.yaml", cfg)

    T_refine_list = parse_int_list(args.T_refine_list)
    spectral_L_list = parse_int_list(args.spectral_L_list)
    spectral_sequences = [s.strip() for s in args.spectral_sequences.split(",") if s.strip()]
    if not spectral_sequences:
        raise ValueError("spectral_sequences must contain at least one sequence")

    region_summary = run_region_refinement(
        run_dir=run_dir,
        stage4_ref=stage4_ref,
        stage5_ref=stage5_ref,
        cfg_r=cfg_r,
        T_refine_list=T_refine_list,
        refine_points=int(args.refine_points),
        resume=bool(args.resume),
    )

    timeseries_summary = run_exact_long_timeseries(
        run_dir=run_dir,
        stage4_ref=stage4_ref,
        stage5_ref=stage5_ref,
        cfg_r=cfg_r,
        T_timeseries=int(args.T_timeseries),
        resume=bool(args.resume),
    )

    spectral_summary = run_spectral_proxy(
        run_dir=run_dir,
        stage4_ref=stage4_ref,
        stage5_ref=stage5_ref,
        cfg_r=cfg_r,
        spectral_L_list=spectral_L_list,
        spectral_sequences=spectral_sequences,
        top_k_modes=int(args.top_k_modes),
        n_cycles_ring=int(args.n_cycles_ring),
        burn_cycles_ring=int(args.burn_cycles_ring),
    )

    report = build_report(
        run_dir=run_dir,
        stage4_ref=stage4_ref,
        stage5_ref=stage5_ref,
        region_summary=region_summary,
        timeseries_summary=timeseries_summary,
        spectral_summary=spectral_summary,
    )
    write_text(run_dir / "stage6_report.md", report)

    summary = {
        "stage4_run_dir": str(Path(args.stage4_run_dir)),
        "stage5_run_dir": str(Path(args.stage5_run_dir)),
        "region_refine_summary": region_summary,
        "exact_long_timeseries_summary": timeseries_summary,
        "spectral_summary": spectral_summary,
        "report_md": "stage6_report.md",
    }
    write_json(run_dir / "stage6_summary.json", summary)
    write_manifest(
        run_dir,
        "stage6_exact_p0_math_suite",
        {
            "config": str(Path(args.config)),
            "stage4_run_dir": str(Path(args.stage4_run_dir)),
            "stage5_run_dir": str(Path(args.stage5_run_dir)),
            "T_refine_list": T_refine_list,
            "refine_points": int(args.refine_points),
            "T_timeseries": int(args.T_timeseries),
            "spectral_L_list": spectral_L_list,
            "spectral_sequences": spectral_sequences,
            "top_k_modes": int(args.top_k_modes),
            "n_cycles_ring": int(args.n_cycles_ring),
            "burn_cycles_ring": int(args.burn_cycles_ring),
        },
    )

    print(f"[done] wrote stage-6 suite to: {run_dir}")
    print(f"[done] summary: {run_dir / 'stage6_summary.json'}")
    print(f"[done] report: {run_dir / 'stage6_report.md'}")
    print(f"[done] refined phi grid: {run_dir / 'region_refine_grid.csv'}")
    print(f"[done] long timeseries summary: {run_dir / 'exact_long_windowed_summary.csv'}")
    print(f"[done] spectral convergence: {run_dir / 'spectral_convergence.csv'}")


if __name__ == "__main__":
    main()
