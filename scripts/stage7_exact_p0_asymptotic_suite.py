
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dtqw.io import load_yaml, write_json, write_manifest, write_text, write_yaml

from stage6_exact_p0_math_suite import (
    ensure_dir,
    make_run_dir,
    write_csv,
    append_csv_rows,
    read_csv_rows,
    parse_int_list,
    resolve_from_mrc,
    load_stage4_reference,
    load_stage5_reference,
    su2_coin,
    scaled_windows,
    simulate_unitary_scalar_timeseries,
    fit_tail_and_metrics,
    build_period_operator_and_prefixes,
    initial_state_on_ring,
    diag_prob_and_observables,
    diagonal_ensemble_period_average,
    simulate_on_ring_period_average,
)




# ---------------------------------------------------------------------
# Robust Stage-4 / Stage-5 loaders (more tolerant than the Stage-6 helper)
# ---------------------------------------------------------------------


def load_stage4_reference(stage4_run_dir: Path) -> Dict[str, Any]:
    summary_path = stage4_run_dir / "exact_p0_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Stage-4 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw = dict(summary.get("thresholds_from_stage3", {}))

    def pick(*names: str, default: Optional[float] = None) -> float:
        for name in names:
            if name in raw:
                return float(raw[name])
        if default is not None:
            return float(default)
        raise KeyError(f"Missing threshold. Tried keys={names}; available={sorted(raw.keys())}")

    thresholds = {
        "vmin": pick("vmin", "v_min"),
        "eps_v": pick("eps_v", "epsilon_v"),
        "eps_P": pick("eps_P", "eps_p", "epsilon_P"),
        "w_thr_primary": pick("w_thr_primary", default=0.10),
        "w_thr_sensitivity": pick("w_thr_sensitivity", default=0.15),
    }
    head = summary.get("headline_points", {})
    counts = summary.get("counts_exact_p0", {})
    return {
        "summary": summary,
        "thresholds": thresholds,
        "counts": counts,
        "max_adv_phi": None if head.get("max_advantage") is None else float(head["max_advantage"]["phi"]),
        "pp_sens_phi": None if head.get("first_pp_sensitivity") is None else float(head["first_pp_sensitivity"]["phi"]),
    }


def load_stage5_reference(stage5_run_dir: Path) -> Dict[str, Any]:
    summary_path = stage5_run_dir / "rigour_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Stage-5 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    exact = summary["exact_horizon_summary"]
    counts = exact["counts"]
    # Pull out the T=300 fixed row and the largest-T fixed row directly from the summary.
    fixed_rows = [row for row in counts if row["policy"] == "fixed"]
    if not fixed_rows:
        raise RuntimeError("Stage-5 summary contains no fixed-policy rows")
    fixed_rows_sorted = sorted(fixed_rows, key=lambda r: int(r["T"]))
    T300 = None
    for row in fixed_rows_sorted:
        if int(row["T"]) == 300:
            T300 = row
            break
    if T300 is None:
        T300 = fixed_rows_sorted[0]
    Tlast = fixed_rows_sorted[-1]
    return {
        "summary": summary,
        "exact": exact,
        "T300_fixed": T300,
        "Tlast_fixed": Tlast,
        "T_last": int(Tlast["T"]),
        "grid_rows": [],
        "pp_primary_phi_last": None,
        "newly_added_drift_phi": None,
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _f(x: Any) -> float:
    return float(x)


def _i(x: Any) -> int:
    return int(float(x))


def threshold_value(thresholds: Dict[str, Any], *names: str) -> float:
    for name in names:
        if name in thresholds:
            return float(thresholds[name])
    raise KeyError(f"Missing threshold among candidates: {names}")


def sign_eps_scalar(x: float, eps: float) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def compute_decision_flags_stage7(
    mA: Any,
    mB: Any,
    mABB: Any,
    thresholds: Dict[str, Any],
    cfg_r: Dict[str, Any],
) -> Dict[str, int]:
    vmin = threshold_value(thresholds, "vmin", "v_min")
    eps_v = threshold_value(thresholds, "eps_v", "epsilon_v")
    eps_P = threshold_value(thresholds, "eps_P", "epsilon_P")

    A_losing = bool(mA.stable and (mA.v_fit < -vmin))
    B_losing = bool(mB.stable and (mB.v_fit < -vmin))
    ABB_winning = bool(mABB.stable and (mABB.v_fit > vmin))

    strict_raw = int((mA.v_fit < 0.0) and (mB.v_fit < 0.0))
    drift_raw = int(strict_raw and (mABB.v_fit > 0.0))
    strict_stage3 = int(A_losing and B_losing)
    drift_stage3 = int(strict_stage3 and ABB_winning)
    pp_primary = int(drift_stage3 and (mABB.w_loc3 < cfg_r["w_thr_primary"]))
    pp_sensitivity = int(drift_stage3 and (mABB.w_loc5 < cfg_r["w_thr_sensitivity"]))

    mismatch_eligible = int(
        (sign_eps_scalar(mABB.v_fit, eps_v) != 0)
        and (sign_eps_scalar(mABB.deltaP_late_mean, eps_P) != 0)
    )
    mismatch = int(
        mismatch_eligible
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
        "mismatch_eligible": int(mismatch_eligible),
    }


@dataclass
class Bundle:
    mA: Any
    mB: Any
    mABB: Any
    flags: Dict[str, int]
    adv_v: float


class ExactSeriesCache:
    """
    Lightweight exact-slice cache.
    Keys are (sequence, rounded_phi, Tmax). Values are ScalarSeries objects.
    """

    def __init__(self, C_A: np.ndarray, C_B: np.ndarray, cfg_r: Dict[str, Any]) -> None:
        self.C_A = C_A
        self.C_B = C_B
        self.cfg_r = cfg_r
        self._cache: Dict[Tuple[str, float, int], Any] = {}

    def get(self, seq: str, phi: float, Tmax: int) -> Any:
        key = (str(seq), round(float(phi), 15), int(Tmax))
        if key not in self._cache:
            self._cache[key] = simulate_unitary_scalar_timeseries(
                seq=str(seq),
                Tmax=int(Tmax),
                phi=float(phi),
                C_A=self.C_A,
                C_B=self.C_B,
                x0_primary=int(self.cfg_r["x0_primary"]),
                x0_sens=int(self.cfg_r["x0_sensitivity"]),
            )
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()


def evaluate_bundle(
    phi: float,
    T: int,
    policy: str,
    cache: ExactSeriesCache,
    cfg_r: Dict[str, Any],
    thresholds: Dict[str, Any],
    Tmax_cache: Optional[int] = None,
) -> Bundle:
    if Tmax_cache is None:
        Tmax_cache = int(T)
    Tmax_cache = max(int(Tmax_cache), int(T))
    if policy == "fixed":
        T0 = int(cfg_r["T0_stage3"])
        T1 = int(cfg_r["T1_stage3"])
    elif policy == "scaled":
        T0, T1 = scaled_windows(int(T))
    else:
        raise ValueError(f"Unknown policy: {policy}")

    mA = fit_tail_and_metrics(cache.get("A", phi, Tmax_cache), int(T), int(T0), int(T1), float(cfg_r["rel_tol"]))
    mB = fit_tail_and_metrics(cache.get("B", phi, Tmax_cache), int(T), int(T0), int(T1), float(cfg_r["rel_tol"]))
    mABB = fit_tail_and_metrics(cache.get("ABB", phi, Tmax_cache), int(T), int(T0), int(T1), float(cfg_r["rel_tol"]))
    flags = compute_decision_flags_stage7(mA, mB, mABB, thresholds, cfg_r)
    adv_v = float(mABB.v_fit - max(mA.v_fit, mB.v_fit))
    return Bundle(mA=mA, mB=mB, mABB=mABB, flags=flags, adv_v=adv_v)


def evaluate_abb_metrics(
    phi: float,
    T: int,
    policy: str,
    cache: ExactSeriesCache,
    cfg_r: Dict[str, Any],
    Tmax_cache: Optional[int] = None,
) -> Any:
    if Tmax_cache is None:
        Tmax_cache = int(T)
    Tmax_cache = max(int(Tmax_cache), int(T))
    if policy == "fixed":
        T0 = int(cfg_r["T0_stage3"])
        T1 = int(cfg_r["T1_stage3"])
    elif policy == "scaled":
        T0, T1 = scaled_windows(int(T))
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return fit_tail_and_metrics(cache.get("ABB", phi, Tmax_cache), int(T), int(T0), int(T1), float(cfg_r["rel_tol"]))


def default_checkpoint_list(Tmax: int) -> List[int]:
    base = [300, 600, 900, 1200, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 15000, 20000]
    out = [t for t in base if t <= int(Tmax)]
    if int(Tmax) not in out:
        out.append(int(Tmax))
    return sorted(set(out))


def fit_inverse_series(T_values: Sequence[int], y_values: Sequence[float], degree: int) -> Dict[str, Any]:
    T = np.asarray(T_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if T.ndim != 1 or y.ndim != 1 or T.size != y.size:
        raise ValueError("T_values and y_values must be same-length 1D sequences")
    if T.size < degree + 2:
        return {"degree": int(degree), "ok": False}
    X_cols = [np.ones_like(T)]
    for k in range(1, degree + 1):
        X_cols.append(T ** (-k))
    X = np.vstack(X_cols).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    payload = {
        "degree": int(degree),
        "ok": True,
        "coefficients": [float(b) for b in beta],
        "limit_estimate": float(beta[0]),
        "rmse": float(rmse),
    }
    return payload


# ---------------------------------------------------------------------
# Load Stage-6 reference
# ---------------------------------------------------------------------


def load_stage6_reference(stage6_run_dir: Path) -> Dict[str, Any]:
    summary_path = stage6_run_dir / "stage6_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Stage-6 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    intervals_path = stage6_run_dir / "region_refine_intervals.csv"
    if not intervals_path.exists():
        raise FileNotFoundError(f"Missing Stage-6 intervals csv: {intervals_path}")
    grid_path = stage6_run_dir / "region_refine_grid.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing Stage-6 grid csv: {grid_path}")
    intervals_rows = read_csv_rows(intervals_path)
    grid_rows = read_csv_rows(grid_path)

    pp_region_meta = None
    max_region_meta = None
    for region in summary["region_refine_summary"]["regions"]:
        if region["region"] == "pp_region":
            pp_region_meta = region
        if region["region"] == "max_region":
            max_region_meta = region
    if pp_region_meta is None or max_region_meta is None:
        raise RuntimeError("Stage-6 summary missing pp_region or max_region")

    return {
        "summary": summary,
        "intervals_rows": intervals_rows,
        "grid_rows": grid_rows,
        "pp_region_meta": pp_region_meta,
        "max_region_meta": max_region_meta,
    }


# ---------------------------------------------------------------------
# Module 1: extend exact coarse counts on the stage-3 phi grid
# ---------------------------------------------------------------------


def run_coarse_exact_counts(
    run_dir: Path,
    stage4_ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    T_coarse_list: List[int],
    resume: bool,
) -> Dict[str, Any]:
    out_csv = run_dir / "coarse_exact_grid.csv"
    counts_csv = run_dir / "coarse_exact_counts.csv"
    summary_json = run_dir / "coarse_exact_summary.json"

    if (not resume) and out_csv.exists():
        out_csv.unlink()

    existing_rows = read_csv_rows(out_csv) if resume else []
    done_phi = {_i(r["phi_index"]) for r in existing_rows}

    coin_params = cfg_r["coin_params"]
    C_A = su2_coin(*coin_params["A"], degrees=True)
    C_B = su2_coin(*coin_params["B"], degrees=True)
    cache = ExactSeriesCache(C_A=C_A, C_B=C_B, cfg_r=cfg_r)

    thresholds = stage4_ref["thresholds"]
    N_phi = int(cfg_r["N_phi_stage3"])
    phi_grid = np.linspace(0.0, 2.0 * math.pi, N_phi, endpoint=False, dtype=np.float64)
    Tmax = int(max(T_coarse_list))

    for j, phi in enumerate(phi_grid):
        if int(j) in done_phi:
            continue
        rows_to_append: List[Dict[str, Any]] = []
        # Cache all three sequences once for this phi at Tmax.
        _ = cache.get("A", float(phi), Tmax)
        _ = cache.get("B", float(phi), Tmax)
        _ = cache.get("ABB", float(phi), Tmax)
        for policy in ["fixed", "scaled"]:
            for T in T_coarse_list:
                bundle = evaluate_bundle(
                    phi=float(phi),
                    T=int(T),
                    policy=policy,
                    cache=cache,
                    cfg_r=cfg_r,
                    thresholds=thresholds,
                    Tmax_cache=Tmax,
                )
                if policy == "fixed":
                    T0 = int(cfg_r["T0_stage3"])
                    T1 = int(cfg_r["T1_stage3"])
                else:
                    T0, T1 = scaled_windows(int(T))
                rows_to_append.append(
                    {
                        "phi_index": int(j),
                        "phi": float(phi),
                        "phi_over_pi": float(phi / math.pi),
                        "policy": policy,
                        "T": int(T),
                        "T0": int(T0),
                        "T1": int(T1),
                        "v_fit_A": float(bundle.mA.v_fit),
                        "v_fit_B": float(bundle.mB.v_fit),
                        "v_fit_ABB": float(bundle.mABB.v_fit),
                        "deltaP_late_mean_ABB": float(bundle.mABB.deltaP_late_mean),
                        "w_loc3_ABB": float(bundle.mABB.w_loc3),
                        "w_loc5_ABB": float(bundle.mABB.w_loc5),
                        "P0bar_ABB": float(bundle.mABB.P0bar),
                        "stable_A": int(bundle.mA.stable),
                        "stable_B": int(bundle.mB.stable),
                        "stable_ABB": int(bundle.mABB.stable),
                        **bundle.flags,
                        "adv_v": float(bundle.adv_v),
                    }
                )
        append_csv_rows(out_csv, rows_to_append)
        done_phi.add(int(j))

    rows = read_csv_rows(out_csv)
    count_rows: List[Dict[str, Any]] = []
    for policy in ["fixed", "scaled"]:
        for T in T_coarse_list:
            sub = [r for r in rows if r["policy"] == policy and _i(r["T"]) == int(T)]
            if not sub:
                continue
            adv_vals = np.asarray([_f(r["adv_v"]) for r in sub], dtype=np.float64)
            i_max = int(np.argmax(adv_vals))
            rmax = sub[i_max]
            count_rows.append(
                {
                    "policy": policy,
                    "T": int(T),
                    "T0": _i(sub[0]["T0"]),
                    "T1": _i(sub[0]["T1"]),
                    "strict_raw": int(sum(_i(r["strict_raw"]) for r in sub)),
                    "drift_raw": int(sum(_i(r["drift_raw"]) for r in sub)),
                    "strict_stage3": int(sum(_i(r["strict_stage3"]) for r in sub)),
                    "drift_stage3": int(sum(_i(r["drift_stage3"]) for r in sub)),
                    "pp_primary": int(sum(_i(r["pp_primary"]) for r in sub)),
                    "pp_sensitivity": int(sum(_i(r["pp_sensitivity"]) for r in sub)),
                    "mismatch_sign": int(sum(_i(r["mismatch_sign"]) for r in sub)),
                    "mismatch_sign_denominator": int(sum(_i(r["mismatch_eligible"]) for r in sub)),
                    "max_adv_phi": float(rmax["phi"]),
                    "max_adv_phi_over_pi": float(rmax["phi_over_pi"]),
                    "max_adv_v": float(rmax["adv_v"]),
                    "max_adv_w_loc3": float(rmax["w_loc3_ABB"]),
                    "max_adv_w_loc5": float(rmax["w_loc5_ABB"]),
                    "max_adv_v_fit_ABB": float(rmax["v_fit_ABB"]),
                }
            )
    write_csv(counts_csv, count_rows)
    summary = {
        "T_coarse_list": [int(T) for T in T_coarse_list],
        "grid_csv": str(out_csv.relative_to(run_dir)),
        "counts_csv": str(counts_csv.relative_to(run_dir)),
        "counts": count_rows,
    }
    write_json(summary_json, summary)
    return summary


# ---------------------------------------------------------------------
# Module 2: exact localization-threshold boundary solver near pp region
# ---------------------------------------------------------------------


def pp_region_bounds(stage6_ref: Dict[str, Any]) -> Tuple[float, float]:
    meta = stage6_ref["pp_region_meta"]
    center = float(meta["center_phi"])
    half = float(meta["halfwidth"])
    return center - half, center + half


def max_region_bounds(stage6_ref: Dict[str, Any]) -> Tuple[float, float]:
    meta = stage6_ref["max_region_meta"]
    center = float(meta["center_phi"])
    half = float(meta["halfwidth"])
    return center - half, center + half


def find_mask_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    start = None
    for k, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = k
        elif (not flag) and start is not None:
            out.append((start, k - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def bisection_root(
    func,
    left: float,
    right: float,
    tol_abs: float,
    max_iter: int = 60,
) -> Tuple[float, float, float]:
    fL = float(func(left))
    fR = float(func(right))
    if fL == 0.0:
        return float(left), float(fL), float(fR)
    if fR == 0.0:
        return float(right), float(fL), float(fR)
    if fL * fR > 0.0:
        raise ValueError(f"Bisection bracket does not change sign: f(left)={fL}, f(right)={fR}")
    a = float(left)
    b = float(right)
    fa = float(fL)
    fb = float(fR)
    for _ in range(int(max_iter)):
        m = 0.5 * (a + b)
        fm = float(func(m))
        if abs(b - a) <= float(tol_abs):
            return float(m), float(fa), float(fb)
        if fm == 0.0:
            return float(m), float(fa), float(fb)
        if fa * fm <= 0.0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    return float(0.5 * (a + b)), float(fa), float(fb)


def run_pp_boundary_suite(
    run_dir: Path,
    stage4_ref: Dict[str, Any],
    stage6_ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    T_boundary_list: List[int],
    scan_points: int,
    tol_pi: float,
) -> Dict[str, Any]:
    root_csv = run_dir / "pp_boundary_estimates.csv"
    scan_csv = run_dir / "pp_boundary_scan.csv"
    profile_csv = run_dir / "pp_boundary_profile_samples.csv"
    summary_json = run_dir / "pp_boundary_summary.json"

    coin_params = cfg_r["coin_params"]
    C_A = su2_coin(*coin_params["A"], degrees=True)
    C_B = su2_coin(*coin_params["B"], degrees=True)
    cache = ExactSeriesCache(C_A=C_A, C_B=C_B, cfg_r=cfg_r)

    pp_left, pp_right = pp_region_bounds(stage6_ref)
    phi_scan = np.linspace(pp_left, pp_right, int(scan_points), endpoint=True, dtype=np.float64)
    Tmax = int(max(T_boundary_list))
    thresholds = stage4_ref["thresholds"]
    vmin = threshold_value(thresholds, "vmin", "v_min")

    scan_rows: List[Dict[str, Any]] = []
    root_rows: List[Dict[str, Any]] = []
    profile_rows: List[Dict[str, Any]] = []

    for policy in ["fixed", "scaled"]:
        for T in T_boundary_list:
            w3_vals: List[float] = []
            w5_vals: List[float] = []
            for k, phi in enumerate(phi_scan):
                mABB = evaluate_abb_metrics(
                    phi=float(phi),
                    T=int(T),
                    policy=policy,
                    cache=cache,
                    cfg_r=cfg_r,
                    Tmax_cache=Tmax,
                )
                w3 = float(mABB.w_loc3)
                w5 = float(mABB.w_loc5)
                w3_vals.append(w3)
                w5_vals.append(w5)
                scan_rows.append(
                    {
                        "policy": policy,
                        "T": int(T),
                        "phi_scan_index": int(k),
                        "phi": float(phi),
                        "phi_over_pi": float(phi / math.pi),
                        "w_loc3_ABB": float(w3),
                        "w_loc5_ABB": float(w5),
                        "primary_mask_from_scan": int(w3 < cfg_r["w_thr_primary"]),
                        "sensitivity_mask_from_scan": int(w5 < cfg_r["w_thr_sensitivity"]),
                    }
                )

            for mask_name, thr, values in [
                ("pp_primary", float(cfg_r["w_thr_primary"]), np.asarray(w3_vals, dtype=np.float64)),
                ("pp_sensitivity", float(cfg_r["w_thr_sensitivity"]), np.asarray(w5_vals, dtype=np.float64)),
            ]:
                mask = values < float(thr)
                intervals = find_mask_intervals(mask)
                if not intervals:
                    continue
                # We only expect one main interval in this region; keep the widest one.
                intervals = sorted(intervals, key=lambda ab: (ab[1] - ab[0] + 1), reverse=True)
                ia, ib = intervals[0]
                if ia <= 0 or ib >= len(phi_scan) - 1:
                    raise RuntimeError(
                        f"Scan interval for {mask_name} at policy={policy}, T={T} touches scan boundary; "
                        "increase scan range or scan_points."
                    )

                left_out = float(phi_scan[ia - 1])
                left_in = float(phi_scan[ia])
                right_in = float(phi_scan[ib])
                right_out = float(phi_scan[ib + 1])

                def g(phi: float) -> float:
                    mABB_local = evaluate_abb_metrics(
                        phi=float(phi),
                        T=int(T),
                        policy=policy,
                        cache=cache,
                        cfg_r=cfg_r,
                        Tmax_cache=Tmax,
                    )
                    if mask_name == "pp_primary":
                        return float(mABB_local.w_loc3 - cfg_r["w_thr_primary"])
                    return float(mABB_local.w_loc5 - cfg_r["w_thr_sensitivity"])

                left_root, _, _ = bisection_root(g, left_out, left_in, tol_abs=float(tol_pi) * math.pi)
                right_root, _, _ = bisection_root(g, right_in, right_out, tol_abs=float(tol_pi) * math.pi)

                mid = 0.5 * (left_root + right_root)
                # Sample strict/win margins on the interval to confirm transport boundary is
                # created by localization thresholds rather than by loss of drift predicates.
                tol_abs = float(tol_pi) * math.pi
                prof_left = float(left_root + tol_abs)
                prof_right = float(right_root - tol_abs)
                if prof_right <= prof_left:
                    prof_left = float(left_root)
                    prof_right = float(right_root)
                phi_profile = np.linspace(prof_left, prof_right, 17, endpoint=True, dtype=np.float64)
                all_primary_ok = True
                all_sens_ok = True
                all_strict_ok = True
                all_drift_ok = True
                for j, phi in enumerate(phi_profile):
                    bundle = evaluate_bundle(
                        phi=float(phi),
                        T=int(T),
                        policy=policy,
                        cache=cache,
                        cfg_r=cfg_r,
                        thresholds=thresholds,
                        Tmax_cache=Tmax,
                    )
                    row = {
                        "mask_name": mask_name,
                        "policy": policy,
                        "T": int(T),
                        "profile_index": int(j),
                        "phi": float(phi),
                        "phi_over_pi": float(phi / math.pi),
                        "v_fit_A": float(bundle.mA.v_fit),
                        "v_fit_B": float(bundle.mB.v_fit),
                        "v_fit_ABB": float(bundle.mABB.v_fit),
                        "margin_A_losing": float((-vmin) - bundle.mA.v_fit),
                        "margin_B_losing": float((-vmin) - bundle.mB.v_fit),
                        "margin_ABB_winning": float(bundle.mABB.v_fit - vmin),
                        "margin_primary_transport": float(cfg_r["w_thr_primary"] - bundle.mABB.w_loc3),
                        "margin_sensitivity_transport": float(cfg_r["w_thr_sensitivity"] - bundle.mABB.w_loc5),
                        "strict_stage3": int(bundle.flags["strict_stage3"]),
                        "drift_stage3": int(bundle.flags["drift_stage3"]),
                        "pp_primary": int(bundle.flags["pp_primary"]),
                        "pp_sensitivity": int(bundle.flags["pp_sensitivity"]),
                    }
                    profile_rows.append(row)
                    all_primary_ok = all_primary_ok and bool(bundle.flags["pp_primary"] == 1)
                    all_sens_ok = all_sens_ok and bool(bundle.flags["pp_sensitivity"] == 1)
                    all_strict_ok = all_strict_ok and bool(bundle.flags["strict_stage3"] == 1)
                    all_drift_ok = all_drift_ok and bool(bundle.flags["drift_stage3"] == 1)

                root_rows.extend(
                    [
                        {
                            "mask_name": mask_name,
                            "policy": policy,
                            "T": int(T),
                            "side": "left",
                            "phi_root": float(left_root),
                            "phi_root_over_pi": float(left_root / math.pi),
                            "scan_bracket_left": float(left_out),
                            "scan_bracket_right": float(left_in),
                            "sample_interval_left": float(phi_scan[ia]),
                            "sample_interval_right": float(phi_scan[ib]),
                            "interval_width": float(right_root - left_root),
                            "interval_width_over_pi": float((right_root - left_root) / math.pi),
                            "midpoint_phi": float(mid),
                            "midpoint_phi_over_pi": float(mid / math.pi),
                            "strict_stage3_all_profile_points": int(all_strict_ok),
                            "drift_stage3_all_profile_points": int(all_drift_ok),
                            "pp_primary_all_profile_points": int(all_primary_ok),
                            "pp_sensitivity_all_profile_points": int(all_sens_ok),
                        },
                        {
                            "mask_name": mask_name,
                            "policy": policy,
                            "T": int(T),
                            "side": "right",
                            "phi_root": float(right_root),
                            "phi_root_over_pi": float(right_root / math.pi),
                            "scan_bracket_left": float(right_in),
                            "scan_bracket_right": float(right_out),
                            "sample_interval_left": float(phi_scan[ia]),
                            "sample_interval_right": float(phi_scan[ib]),
                            "interval_width": float(right_root - left_root),
                            "interval_width_over_pi": float((right_root - left_root) / math.pi),
                            "midpoint_phi": float(mid),
                            "midpoint_phi_over_pi": float(mid / math.pi),
                            "strict_stage3_all_profile_points": int(all_strict_ok),
                            "drift_stage3_all_profile_points": int(all_drift_ok),
                            "pp_primary_all_profile_points": int(all_primary_ok),
                            "pp_sensitivity_all_profile_points": int(all_sens_ok),
                        },
                    ]
                )

    write_csv(scan_csv, scan_rows)
    write_csv(root_csv, root_rows)
    write_csv(profile_csv, profile_rows)

    summary = {
        "T_boundary_list": [int(T) for T in T_boundary_list],
        "scan_points": int(scan_points),
        "tol_pi": float(tol_pi),
        "root_csv": str(root_csv.relative_to(run_dir)),
        "scan_csv": str(scan_csv.relative_to(run_dir)),
        "profile_csv": str(profile_csv.relative_to(run_dir)),
        "intervals": {},
    }
    for mask_name in ["pp_primary", "pp_sensitivity"]:
        summary["intervals"][mask_name] = {}
        for policy in ["fixed", "scaled"]:
            summary["intervals"][mask_name][policy] = {}
            for T in T_boundary_list:
                sub = [
                    r for r in root_rows
                    if r["mask_name"] == mask_name and r["policy"] == policy and _i(r["T"]) == int(T)
                ]
                if len(sub) != 2:
                    continue
                left_row = [r for r in sub if r["side"] == "left"][0]
                right_row = [r for r in sub if r["side"] == "right"][0]
                summary["intervals"][mask_name][policy][str(int(T))] = {
                    "left_phi": float(left_row["phi_root"]),
                    "right_phi": float(right_row["phi_root"]),
                    "left_phi_over_pi": float(left_row["phi_root_over_pi"]),
                    "right_phi_over_pi": float(right_row["phi_root_over_pi"]),
                    "width": float(left_row["interval_width"]),
                    "width_over_pi": float(left_row["interval_width_over_pi"]),
                    "midpoint_phi": float(left_row["midpoint_phi"]),
                    "midpoint_phi_over_pi": float(left_row["midpoint_phi_over_pi"]),
                    "strict_stage3_all_profile_points": int(left_row["strict_stage3_all_profile_points"]),
                    "drift_stage3_all_profile_points": int(left_row["drift_stage3_all_profile_points"]),
                    "pp_primary_all_profile_points": int(left_row["pp_primary_all_profile_points"]),
                    "pp_sensitivity_all_profile_points": int(left_row["pp_sensitivity_all_profile_points"]),
                }
    write_json(summary_json, summary)
    return summary


# ---------------------------------------------------------------------
# Module 3: refine the exact maximum-advantage location beyond Stage-6
# ---------------------------------------------------------------------


def run_maximum_refinement(
    run_dir: Path,
    stage4_ref: Dict[str, Any],
    stage6_ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    T_max_list: List[int],
) -> Dict[str, Any]:
    out_csv = run_dir / "max_refinement.csv"
    summary_json = run_dir / "max_refinement_summary.json"

    coin_params = cfg_r["coin_params"]
    C_A = su2_coin(*coin_params["A"], degrees=True)
    C_B = su2_coin(*coin_params["B"], degrees=True)
    cache = ExactSeriesCache(C_A=C_A, C_B=C_B, cfg_r=cfg_r)
    thresholds = stage4_ref["thresholds"]

    left0, right0 = max_region_bounds(stage6_ref)
    Tmax = int(max(T_max_list))
    rows: List[Dict[str, Any]] = []

    for policy in ["fixed", "scaled"]:
        for T in T_max_list:
            # Three-stage nested search by dense sampling.
            left = float(left0)
            right = float(right0)
            best_phi = None
            best_bundle = None
            for n_points in [41, 21, 21]:
                phi_grid = np.linspace(left, right, int(n_points), endpoint=True, dtype=np.float64)
                bundles: List[Tuple[float, Bundle]] = []
                for phi in phi_grid:
                    bundle = evaluate_bundle(
                        phi=float(phi),
                        T=int(T),
                        policy=policy,
                        cache=cache,
                        cfg_r=cfg_r,
                        thresholds=thresholds,
                        Tmax_cache=Tmax,
                    )
                    bundles.append((float(phi), bundle))
                idx = int(np.argmax(np.asarray([b.adv_v for _, b in bundles], dtype=np.float64)))
                best_phi, best_bundle = bundles[idx]
                step = float((right - left) / (int(n_points) - 1))
                left = max(float(left0), float(best_phi - 2.0 * step))
                right = min(float(right0), float(best_phi + 2.0 * step))
            if best_phi is None or best_bundle is None:
                continue
            rows.append(
                {
                    "policy": policy,
                    "T": int(T),
                    "phi": float(best_phi),
                    "phi_over_pi": float(best_phi / math.pi),
                    "adv_v": float(best_bundle.adv_v),
                    "v_fit_A": float(best_bundle.mA.v_fit),
                    "v_fit_B": float(best_bundle.mB.v_fit),
                    "v_fit_ABB": float(best_bundle.mABB.v_fit),
                    "w_loc3_ABB": float(best_bundle.mABB.w_loc3),
                    "w_loc5_ABB": float(best_bundle.mABB.w_loc5),
                    "P0bar_ABB": float(best_bundle.mABB.P0bar),
                    "strict_stage3": int(best_bundle.flags["strict_stage3"]),
                    "drift_stage3": int(best_bundle.flags["drift_stage3"]),
                    "pp_primary": int(best_bundle.flags["pp_primary"]),
                    "pp_sensitivity": int(best_bundle.flags["pp_sensitivity"]),
                }
            )

    write_csv(out_csv, rows)
    summary = {
        "T_max_list": [int(T) for T in T_max_list],
        "max_region_bounds": {
            "phi_left": float(left0),
            "phi_right": float(right0),
            "phi_left_over_pi": float(left0 / math.pi),
            "phi_right_over_pi": float(right0 / math.pi),
        },
        "csv": str(out_csv.relative_to(run_dir)),
        "rows": rows,
    }
    write_json(summary_json, summary)
    return summary


# ---------------------------------------------------------------------
# Module 4: longer exact timeseries + asymptotic extrapolations
# ---------------------------------------------------------------------


def pick_timeseries_points(
    stage5_ref: Dict[str, Any],
    boundary_summary: Dict[str, Any],
    max_summary: Dict[str, Any],
) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    out.append(("paper_max_T300", float(stage5_ref["T300_fixed"]["max_adv_phi"])))
    # Prefer the T=6000 fixed refined max if present; else fall back to first row.
    max_rows = max_summary.get("rows", [])
    fixed_rows = [r for r in max_rows if r["policy"] == "fixed"]
    if fixed_rows:
        fixed_rows = sorted(fixed_rows, key=lambda r: int(r["T"]))
        out.append((f"refined_max_T{int(fixed_rows[-1]['T'])}", float(fixed_rows[-1]["phi"])))
    elif max_rows:
        out.append((f"refined_max_T{int(max_rows[-1]['T'])}", float(max_rows[-1]["phi"])))

    # Use primary interval midpoint and boundaries at the largest fixed T.
    primary_fixed = boundary_summary["intervals"].get("pp_primary", {}).get("fixed", {})
    if primary_fixed:
        T_key = sorted(primary_fixed.keys(), key=lambda s: int(s))[-1]
        rec = primary_fixed[T_key]
        out.append((f"pp_primary_mid_T{T_key}", float(rec["midpoint_phi"])))
        out.append((f"pp_primary_left_T{T_key}", float(rec["left_phi"])))
        out.append((f"pp_primary_right_T{T_key}", float(rec["right_phi"])))

    # Deduplicate by rounded phi.
    dedup: List[Tuple[str, float]] = []
    seen = set()
    for label, phi in out:
        key = round(float(phi), 14)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((label, float(phi)))
    return dedup


def run_asymptotic_timeseries(
    run_dir: Path,
    stage5_ref: Dict[str, Any],
    boundary_summary: Dict[str, Any],
    max_summary: Dict[str, Any],
    cfg_r: Dict[str, Any],
    T_timeseries: int,
) -> Dict[str, Any]:
    ts_dir = ensure_dir(run_dir / "asymptotic_timeseries")
    summary_csv = run_dir / "asymptotic_windowed_summary.csv"
    extrap_csv = run_dir / "asymptotic_extrapolation.csv"
    summary_json = run_dir / "asymptotic_timeseries_summary.json"

    coin_params = cfg_r["coin_params"]
    C_A = su2_coin(*coin_params["A"], degrees=True)
    C_B = su2_coin(*coin_params["B"], degrees=True)
    cache = ExactSeriesCache(C_A=C_A, C_B=C_B, cfg_r=cfg_r)

    points = pick_timeseries_points(stage5_ref, boundary_summary, max_summary)
    checkpoints = default_checkpoint_list(int(T_timeseries))
    summary_rows: List[Dict[str, Any]] = []
    extrap_rows: List[Dict[str, Any]] = []

    for label, phi in points:
        for seq in ["A", "B", "ABB"]:
            ser = cache.get(seq, float(phi), int(T_timeseries))
            ts_rows: List[Dict[str, Any]] = []
            for t in range(int(T_timeseries) + 1):
                ts_rows.append(
                    {
                        "label": label,
                        "sequence": seq,
                        "phi": float(phi),
                        "phi_over_pi": float(phi / math.pi),
                        "t": int(t),
                        "x_mean": float(ser.x_mean_t[t]),
                        "deltaP": float(ser.deltaP_t[t]),
                        "w_loc3_inst": float(ser.w3_t[t]),
                        "w_loc5_inst": float(ser.w5_t[t]),
                        "P0": float(ser.P0_t[t]),
                    }
                )
            write_csv(ts_dir / f"timeseries_{label}_{seq}.csv", ts_rows)

            for policy in ["fixed", "scaled"]:
                vals_for_extrap: Dict[str, List[Tuple[int, float]]] = {
                    "v_fit": [],
                    "deltaP_late_mean": [],
                    "w_loc3": [],
                    "w_loc5": [],
                    "P0bar": [],
                }
                for T in checkpoints:
                    if policy == "fixed":
                        T0 = int(cfg_r["T0_stage3"])
                        T1 = int(cfg_r["T1_stage3"])
                    else:
                        T0, T1 = scaled_windows(int(T))
                    m = fit_tail_and_metrics(ser, int(T), int(T0), int(T1), float(cfg_r["rel_tol"]))
                    row = {
                        "label": label,
                        "sequence": seq,
                        "phi": float(phi),
                        "phi_over_pi": float(phi / math.pi),
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
                    summary_rows.append(row)
                    if int(T) >= max(3000, checkpoints[min(len(checkpoints) - 1, 3)]):
                        vals_for_extrap["v_fit"].append((int(T), float(m.v_fit)))
                        vals_for_extrap["deltaP_late_mean"].append((int(T), float(m.deltaP_late_mean)))
                        vals_for_extrap["w_loc3"].append((int(T), float(m.w_loc3)))
                        vals_for_extrap["w_loc5"].append((int(T), float(m.w_loc5)))
                        vals_for_extrap["P0bar"].append((int(T), float(m.P0bar)))

                for metric_name, pairs in vals_for_extrap.items():
                    if len(pairs) < 3:
                        continue
                    Tvals = [p[0] for p in pairs]
                    yvals = [p[1] for p in pairs]
                    fit1 = fit_inverse_series(Tvals, yvals, degree=1)
                    fit2 = fit_inverse_series(Tvals, yvals, degree=2)
                    extrap_rows.append(
                        {
                            "label": label,
                            "sequence": seq,
                            "phi": float(phi),
                            "phi_over_pi": float(phi / math.pi),
                            "policy": policy,
                            "metric_name": metric_name,
                            "n_points": int(len(Tvals)),
                            "T_min": int(min(Tvals)),
                            "T_max": int(max(Tvals)),
                            "limit_estimate_deg1": None if not fit1["ok"] else float(fit1["limit_estimate"]),
                            "rmse_deg1": None if not fit1["ok"] else float(fit1["rmse"]),
                            "limit_estimate_deg2": None if not fit2["ok"] else float(fit2["limit_estimate"]),
                            "rmse_deg2": None if not fit2["ok"] else float(fit2["rmse"]),
                        }
                    )

    write_csv(summary_csv, summary_rows)
    write_csv(extrap_csv, extrap_rows)
    summary = {
        "T_timeseries": int(T_timeseries),
        "points": [{"label": label, "phi": float(phi), "phi_over_pi": float(phi / math.pi)} for label, phi in points],
        "summary_csv": str(summary_csv.relative_to(run_dir)),
        "extrapolation_csv": str(extrap_csv.relative_to(run_dir)),
        "timeseries_dir": str(ts_dir.relative_to(run_dir)),
    }
    write_json(summary_json, summary)
    return summary


# ---------------------------------------------------------------------
# Module 5: extended spectral proxy at paper-critical phis
# ---------------------------------------------------------------------


def pick_spectral_points(
    asymptotic_summary: Dict[str, Any],
) -> List[Tuple[str, float]]:
    pts: List[Tuple[str, float]] = []
    for rec in asymptotic_summary["points"]:
        pts.append((str(rec["label"]), float(rec["phi"])))
    # Deduplicate by rounded phi.
    out: List[Tuple[str, float]] = []
    seen = set()
    for label, phi in pts:
        key = round(float(phi), 14)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, float(phi)))
    return out


def run_extended_spectral_proxy(
    run_dir: Path,
    asymptotic_summary: Dict[str, Any],
    cfg_r: Dict[str, Any],
    spectral_L_list: List[int],
    top_k_modes: int,
    n_cycles_ring: int,
    burn_cycles_ring: int,
) -> Dict[str, Any]:
    top_modes_csv = run_dir / "spectral_extended_top_modes.csv"
    conv_csv = run_dir / "spectral_extended_convergence.csv"
    summary_json = run_dir / "spectral_extended_summary.json"

    coin_params = cfg_r["coin_params"]
    C_A = su2_coin(*coin_params["A"], degrees=True)
    C_B = su2_coin(*coin_params["B"], degrees=True)

    selected_points = pick_spectral_points(asymptotic_summary)
    top_rows: List[Dict[str, Any]] = []
    conv_rows: List[Dict[str, Any]] = []

    for label, phi in selected_points:
        prev_diag = None
        for L in spectral_L_list:
            F, prefixes = build_period_operator_and_prefixes(seq="ABB", C_A=C_A, C_B=C_B, phi=float(phi), L=int(L))
            eigvals, eigvecs = np.linalg.eig(F)
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
            overlap_w5_gt_05 = 0.0
            overlap_w5_gt_09 = 0.0
            overlap_w3_gt_05 = 0.0
            for j in range(eigvecs.shape[1]):
                obs = diag_prob_and_observables(
                    eigvecs[:, j],
                    L=int(L),
                    x0_primary=int(cfg_r["x0_primary"]),
                    x0_sens=int(cfg_r["x0_sensitivity"]),
                )
                weight = float(overlaps[j])
                if obs["w_loc5"] > 0.5:
                    overlap_w5_gt_05 += weight
                if obs["w_loc5"] > 0.9:
                    overlap_w5_gt_09 += weight
                if obs["w_loc3"] > 0.5:
                    overlap_w3_gt_05 += weight
                mode_obs.append(
                    {
                        "mode_index": int(j),
                        "eigenphase": float(np.angle(eigvals[j])),
                        "abs_eigenvalue_minus_1": float(abs(abs(eigvals[j]) - 1.0)),
                        "w_loc3": float(obs["w_loc3"]),
                        "w_loc5": float(obs["w_loc5"]),
                        "P0": float(obs["P0"]),
                        "ipr": float(obs["ipr"]),
                        "overlap_weight": float(weight),
                    }
                )
            mode_obs_sorted = sorted(mode_obs, key=lambda r: (r["w_loc5"], r["ipr"], r["overlap_weight"]), reverse=True)
            for rank, rec in enumerate(mode_obs_sorted[: int(top_k_modes)]):
                top_rows.append(
                    {
                        "label": label,
                        "phi": float(phi),
                        "phi_over_pi": float(phi / math.pi),
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
                x0_primary=int(cfg_r["x0_primary"]),
                x0_sens=int(cfg_r["x0_sensitivity"]),
            )
            ring_avg = simulate_on_ring_period_average(
                seq="ABB",
                C_A=C_A,
                C_B=C_B,
                phi=float(phi),
                L=int(L),
                x0_primary=int(cfg_r["x0_primary"]),
                x0_sens=int(cfg_r["x0_sensitivity"]),
                n_cycles=int(n_cycles_ring),
                burn_cycles=int(burn_cycles_ring),
            )
            row = {
                "label": label,
                "phi": float(phi),
                "phi_over_pi": float(phi / math.pi),
                "L": int(L),
                "topk_overlap_weight": float(sum(r["overlap_weight"] for r in mode_obs_sorted[: int(top_k_modes)])),
                "overlap_w5_gt_05": float(overlap_w5_gt_05),
                "overlap_w5_gt_09": float(overlap_w5_gt_09),
                "overlap_w3_gt_05": float(overlap_w3_gt_05),
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
        "selected_points": [{"label": label, "phi": float(phi), "phi_over_pi": float(phi / math.pi)} for label, phi in selected_points],
        "spectral_L_list": [int(L) for L in spectral_L_list],
        "top_k_modes": int(top_k_modes),
        "n_cycles_ring": int(n_cycles_ring),
        "burn_cycles_ring": int(burn_cycles_ring),
        "top_modes_csv": str(top_modes_csv.relative_to(run_dir)),
        "convergence_csv": str(conv_csv.relative_to(run_dir)),
    }
    write_json(summary_json, summary)
    return summary


# ---------------------------------------------------------------------
# Module 6: write patch-ready notes for the paper
# ---------------------------------------------------------------------


def build_paper_patch_notes(
    coarse_summary: Dict[str, Any],
    boundary_summary: Dict[str, Any],
    max_summary: Dict[str, Any],
    asymptotic_summary: Dict[str, Any],
    spectral_summary: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Stage-7 paper patch notes")
    lines.append("")
    lines.append("## What changed relative to the current PDF")
    lines.append("")
    lines.append("1. The locked Stage-3 atlas statement remains true for T=300, but the exact long-horizon noiseless follow-up no longer supports the blanket phrase 'no primary PP point'.")
    lines.append("2. Stage 6 already opened a nonzero primary PP interval near phi/pi≈5/9; Stage 7 should be used to quote interval endpoints rather than a single coarse-grid point.")
    lines.append("3. The strongest drift-advantage location is not exactly phi/pi=1/3 in the exact long-horizon/refined analysis; it sits in a nearby localized band around phi/pi≈0.296.")
    lines.append("")
    lines.append("## Suggested revised thesis sentence")
    lines.append("")
    lines.append("> For the Jan coins, an origin phase defect produces an exact drift-based Parrondo reversal in the noiseless DTQW. At long exact horizons, a narrow primary directed-transport interval emerges near phi/pi≈5/9, but the largest drift-advantage region remains strongly localized and is therefore distinct from robust transport.")
    lines.append("")
    lines.append("## Numbers to update in the Results section")
    lines.append("")
    if coarse_summary.get("counts"):
        lines.append("### Exact coarse-grid counts")
        lines.append("```json")
        lines.append(json.dumps(coarse_summary["counts"], indent=2))
        lines.append("```")
        lines.append("")
    lines.append("### Primary and sensitivity interval estimates near phi/pi≈5/9")
    lines.append("```json")
    lines.append(json.dumps(boundary_summary.get("intervals", {}), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Refined maximum-advantage estimates")
    lines.append("```json")
    lines.append(json.dumps(max_summary.get("rows", []), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Suggested new subsection headings")
    lines.append("")
    lines.append("1. Exact noiseless p=0 follow-up: horizon stability and interval structure")
    lines.append("2. Transport interval near phi/pi≈5/9: threshold-defined boundaries from w_loc")
    lines.append("3. Localization mechanism: finite-volume Floquet proxy and localized-mode overlap")
    lines.append("")
    lines.append("## What to say in Discussion")
    lines.append("")
    lines.append("- The pre-specified Stage-3 atlas is a finite-horizon statement.")
    lines.append("- The long exact p=0 follow-up separates two phenomena: a broad, highly localized high-advantage band and a much narrower transport-permitting interval near 5/9 pi.")
    lines.append("- The spectral proxy should be described as supporting, not proving, a localized-mode explanation.")
    lines.append("")
    return "\n".join(lines)


def build_math_appendix_notes(
    boundary_summary: Dict[str, Any],
) -> str:
    primary_fixed = boundary_summary.get("intervals", {}).get("pp_primary", {}).get("fixed", {})
    T_key = sorted(primary_fixed.keys(), key=lambda s: int(s))[-1] if primary_fixed else None
    primary_rec = primary_fixed[T_key] if T_key is not None else None

    lines: List[str] = []
    lines.append("# Math appendix notes")
    lines.append("")
    lines.append("## Proposition 1 (phase damping CPTP map)")
    lines.append("")
    lines.append("Let")
    lines.append(r"$$K_0=\sqrt{1-p}\,I,\qquad K_1=\sqrt{p}\,|0\rangle\langle 0|,\qquad K_2=\sqrt{p}\,|1\rangle\langle 1|.$$")
    lines.append(r"Then $\sum_k K_k^\dagger K_k = I$, so the map $\mathcal E_p(\rho)=\sum_k K_k\rho K_k^\dagger$ is CPTP. For")
    lines.append(r"$$\rho=\begin{pmatrix}a&c\\ c^\ast & b\end{pmatrix},$$")
    lines.append(r"direct multiplication gives")
    lines.append(r"$$\mathcal E_p(\rho)=\begin{pmatrix}a&(1-p)c\\ (1-p)c^\ast & b\end{pmatrix}.$$")
    lines.append("")
    lines.append("## Proposition 2 (phi-irrelevance after complete dephasing)")
    lines.append("")
    lines.append(r"In the fully classicalized $Z_{cp}$ null, all off-diagonal phases are removed each step, so the origin defect contributes only a phase factor with no effect on classical position probabilities. Therefore the $p=1$ classicalized null is $\phi$-invariant.")
    lines.append("")
    lines.append("## Lemma 3 (bias and drift can disagree)")
    lines.append("")
    lines.append(r"A simple counterexample: place probability $0.51$ at $x=+1$ and probability $0.49$ at $x=-100$. Then $\Delta P=0.51-0.49>0$, but $\langle x\rangle=0.51-49<0$. Hence right-heavy bias does not imply positive mean-position drift.")
    lines.append("")
    lines.append("## Proposition 4 (finite-horizon primary transport interval at p=0)")
    lines.append("")
    if primary_rec is not None:
        lines.append(
            f"At fixed-window horizon T={T_key}, the exact follow-up estimates a nonzero primary PP interval "
            f"approximately [{primary_rec['left_phi_over_pi']:.6f}, {primary_rec['right_phi_over_pi']:.6f}]·pi "
            f"with width {primary_rec['width_over_pi']:.6f}·pi."
        )
    else:
        lines.append("Insert the Stage-7 primary interval estimate here once available.")
    lines.append("")
    lines.append("State this explicitly as an exact finite-horizon proposition, not as an infinite-time theorem.")
    lines.append("")
    lines.append("## Wording caution")
    lines.append("")
    lines.append("Use phrases such as 'exact finite-horizon', 'long-horizon follow-up', and 'finite-volume Floquet proxy'. Do not overstate these computations as closed-form infinite-time proofs.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 7 exact p=0 asymptotic rescue suite: coarse exact counts, transport-interval boundary estimates, longer exact timeseries, extended spectral proxy, and patch-ready math notes."
    )
    p.add_argument("--config", required=True, help="Path to mrc.yaml")
    p.add_argument("--stage4_run_dir", required=True, help="Path to Stage-4 exact p=0 audit directory")
    p.add_argument("--stage5_run_dir", required=True, help="Path to Stage-5 rigour-suite directory")
    p.add_argument("--stage6_run_dir", required=True, help="Path to Stage-6 exact p=0 mathematics-suite directory")
    p.add_argument("--run_dir", default=None, help="Optional output directory")
    p.add_argument("--resume", action="store_true", help="Resume coarse-grid CSV if partially written")
    p.add_argument("--T_coarse_list", default="3000,6000", help="Comma-separated exact horizons for the full 72-point p=0 coarse grid")
    p.add_argument("--T_boundary_list", default="3000,6000", help="Comma-separated exact horizons for localization-threshold boundary solving")
    p.add_argument("--pp_scan_points", type=int, default=81, help="Scan points across the pp window before bisection")
    p.add_argument("--pp_tol_pi", type=float, default=5e-6, help="Absolute bisection tolerance expressed in units of pi")
    p.add_argument("--T_max_list", default="6000", help="Comma-separated exact horizons for refined maximum-advantage search")
    p.add_argument("--T_timeseries", type=int, default=12000, help="Maximum horizon for representative exact timeseries")
    p.add_argument("--spectral_L_list", default="80,120,160,220", help="Comma-separated ring half-sizes L for the extended spectral proxy")
    p.add_argument("--top_k_modes", type=int, default=12, help="How many top localized modes to save per point/L")
    p.add_argument("--n_cycles_ring", type=int, default=3000, help="Direct ring-averaging periods for the spectral proxy")
    p.add_argument("--burn_cycles_ring", type=int, default=600, help="Burn-in periods discarded in direct ring averaging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg_r = resolve_from_mrc(cfg)
    stage4_ref = load_stage4_reference(Path(args.stage4_run_dir))
    stage5_ref = load_stage5_reference(Path(args.stage5_run_dir))
    stage6_ref = load_stage6_reference(Path(args.stage6_run_dir))

    if args.run_dir is None:
        rd = make_run_dir("stage7_exact_p0_asymptotic_suite")
        run_dir = rd["run_dir"]
    else:
        run_dir = Path(args.run_dir)
        ensure_dir(run_dir)
        ensure_dir(run_dir / "figures")

    write_yaml(run_dir / "config_used.yaml", cfg)

    T_coarse_list = parse_int_list(args.T_coarse_list)
    T_boundary_list = parse_int_list(args.T_boundary_list)
    T_max_list = parse_int_list(args.T_max_list)
    spectral_L_list = parse_int_list(args.spectral_L_list)

    coarse_summary = run_coarse_exact_counts(
        run_dir=run_dir,
        stage4_ref=stage4_ref,
        cfg_r=cfg_r,
        T_coarse_list=T_coarse_list,
        resume=bool(args.resume),
    )

    boundary_summary = run_pp_boundary_suite(
        run_dir=run_dir,
        stage4_ref=stage4_ref,
        stage6_ref=stage6_ref,
        cfg_r=cfg_r,
        T_boundary_list=T_boundary_list,
        scan_points=int(args.pp_scan_points),
        tol_pi=float(args.pp_tol_pi),
    )

    max_summary = run_maximum_refinement(
        run_dir=run_dir,
        stage4_ref=stage4_ref,
        stage6_ref=stage6_ref,
        cfg_r=cfg_r,
        T_max_list=T_max_list,
    )

    asymptotic_summary = run_asymptotic_timeseries(
        run_dir=run_dir,
        stage5_ref=stage5_ref,
        boundary_summary=boundary_summary,
        max_summary=max_summary,
        cfg_r=cfg_r,
        T_timeseries=int(args.T_timeseries),
    )

    spectral_summary = run_extended_spectral_proxy(
        run_dir=run_dir,
        asymptotic_summary=asymptotic_summary,
        cfg_r=cfg_r,
        spectral_L_list=spectral_L_list,
        top_k_modes=int(args.top_k_modes),
        n_cycles_ring=int(args.n_cycles_ring),
        burn_cycles_ring=int(args.burn_cycles_ring),
    )

    patch_notes = build_paper_patch_notes(
        coarse_summary=coarse_summary,
        boundary_summary=boundary_summary,
        max_summary=max_summary,
        asymptotic_summary=asymptotic_summary,
        spectral_summary=spectral_summary,
    )
    write_text(run_dir / "paper_patch_notes.md", patch_notes)

    math_notes = build_math_appendix_notes(boundary_summary=boundary_summary)
    write_text(run_dir / "math_appendix_notes.md", math_notes)

    summary = {
        "stage4_run_dir": str(Path(args.stage4_run_dir)),
        "stage5_run_dir": str(Path(args.stage5_run_dir)),
        "stage6_run_dir": str(Path(args.stage6_run_dir)),
        "coarse_exact_summary": coarse_summary,
        "pp_boundary_summary": boundary_summary,
        "max_refinement_summary": max_summary,
        "asymptotic_timeseries_summary": asymptotic_summary,
        "spectral_extended_summary": spectral_summary,
        "paper_patch_notes_md": "paper_patch_notes.md",
        "math_appendix_notes_md": "math_appendix_notes.md",
    }
    write_json(run_dir / "stage7_summary.json", summary)
    write_manifest(
        run_dir,
        "stage7_exact_p0_asymptotic_suite",
        {
            "config": str(Path(args.config)),
            "stage4_run_dir": str(Path(args.stage4_run_dir)),
            "stage5_run_dir": str(Path(args.stage5_run_dir)),
            "stage6_run_dir": str(Path(args.stage6_run_dir)),
            "T_coarse_list": T_coarse_list,
            "T_boundary_list": T_boundary_list,
            "pp_scan_points": int(args.pp_scan_points),
            "pp_tol_pi": float(args.pp_tol_pi),
            "T_max_list": T_max_list,
            "T_timeseries": int(args.T_timeseries),
            "spectral_L_list": spectral_L_list,
            "top_k_modes": int(args.top_k_modes),
            "n_cycles_ring": int(args.n_cycles_ring),
            "burn_cycles_ring": int(args.burn_cycles_ring),
        },
    )

    print(f"[done] wrote stage-7 suite to: {run_dir}")
    print(f"[done] summary: {run_dir / 'stage7_summary.json'}")
    print(f"[done] coarse exact counts: {run_dir / 'coarse_exact_counts.csv'}")
    print(f"[done] pp boundary estimates: {run_dir / 'pp_boundary_estimates.csv'}")
    print(f"[done] max refinement: {run_dir / 'max_refinement.csv'}")
    print(f"[done] asymptotic summary: {run_dir / 'asymptotic_windowed_summary.csv'}")
    print(f"[done] spectral convergence: {run_dir / 'spectral_extended_convergence.csv'}")
    print(f"[done] patch notes: {run_dir / 'paper_patch_notes.md'}")


if __name__ == "__main__":
    main()
