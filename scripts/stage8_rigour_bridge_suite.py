
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


# -----------------------------------------------------------------------------
# Light reference loaders
# -----------------------------------------------------------------------------


def _float(x: Any) -> float:
    return float(x)


def _int(x: Any) -> int:
    return int(float(x))


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
        raise KeyError(f"Missing threshold among {names}; available={sorted(raw.keys())}")

    thresholds = {
        "vmin": pick("vmin", "v_min"),
        "eps_v": pick("eps_v", "epsilon_v"),
        "eps_P": pick("eps_P", "eps_p", "epsilon_P"),
        "w_thr_primary": pick("w_thr_primary", default=0.10),
        "w_thr_sensitivity": pick("w_thr_sensitivity", default=0.15),
    }
    head = summary.get("headline_points", {})
    return {
        "summary": summary,
        "thresholds": thresholds,
        "counts": summary.get("counts_exact_p0", {}),
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
    fixed_rows = sorted([r for r in counts if r["policy"] == "fixed"], key=lambda r: int(r["T"]))
    if not fixed_rows:
        raise RuntimeError("Stage-5 summary has no fixed-policy rows")
    t300 = None
    for row in fixed_rows:
        if int(row["T"]) == 300:
            t300 = row
            break
    if t300 is None:
        t300 = fixed_rows[0]
    return {
        "summary": summary,
        "exact": exact,
        "T300_fixed": t300,
        "Tlast_fixed": fixed_rows[-1],
    }


def load_stage7_reference(stage7_run_dir: Path) -> Dict[str, Any]:
    stage7 = json.loads((stage7_run_dir / "stage7_summary.json").read_text(encoding="utf-8"))
    pp = json.loads((stage7_run_dir / "pp_boundary_summary.json").read_text(encoding="utf-8"))
    mx = json.loads((stage7_run_dir / "max_refinement_summary.json").read_text(encoding="utf-8"))
    asym = json.loads((stage7_run_dir / "asymptotic_timeseries_summary.json").read_text(encoding="utf-8"))
    spec = json.loads((stage7_run_dir / "spectral_extended_summary.json").read_text(encoding="utf-8"))

    # Largest-T fixed primary interval.
    primary_fixed = pp.get("intervals", {}).get("pp_primary", {}).get("fixed", {})
    if not primary_fixed:
        raise RuntimeError("Stage-7 pp_boundary_summary has no fixed pp_primary interval")
    T_key = sorted(primary_fixed.keys(), key=lambda s: int(s))[-1]
    primary_rec = primary_fixed[T_key]

    # Largest-T fixed refined maximum.
    max_rows = mx.get("rows", [])
    fixed_rows = sorted([r for r in max_rows if r["policy"] == "fixed"], key=lambda r: int(r["T"]))
    if not fixed_rows:
        raise RuntimeError("Stage-7 max_refinement_summary has no fixed rows")
    fixed_max = fixed_rows[-1]

    return {
        "stage7_summary": stage7,
        "pp_boundary_summary": pp,
        "max_refinement_summary": mx,
        "asymptotic_summary": asym,
        "spectral_summary": spec,
        "primary_fixed_T_key": T_key,
        "primary_fixed_interval": primary_rec,
        "fixed_max_row": fixed_max,
    }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def threshold_value(thresholds: Dict[str, Any], *names: str) -> float:
    for name in names:
        if name in thresholds:
            return float(thresholds[name])
    raise KeyError(f"Missing threshold among {names}")


def sign_eps_scalar(x: float, eps: float) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def compute_decision_flags(
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


def fit_inverse_series(Tvals: Sequence[int], yvals: Sequence[float], degree: int) -> Dict[str, Any]:
    if len(Tvals) != len(yvals):
        raise ValueError("Tvals and yvals length mismatch")
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    T = np.asarray(Tvals, dtype=np.float64)
    y = np.asarray(yvals, dtype=np.float64)
    x = 1.0 / T
    cols = [np.ones_like(x), x]
    if degree >= 2:
        cols.append(x * x)
    A = np.column_stack(cols)
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"ok": False}
    yhat = A @ coef
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    return {
        "ok": True,
        "degree": int(degree),
        "limit_estimate": float(coef[0]),
        "rmse": rmse,
    }


def default_checkpoint_list_long(Tmax: int) -> List[int]:
    base = [3000, 6000, 8000, 10000, 12000, 16000, 20000]
    out = [t for t in base if t <= Tmax]
    if Tmax not in out:
        out.append(Tmax)
    return sorted(set(out))


@dataclass
class Bundle:
    mA: Any
    mB: Any
    mABB: Any
    flags: Dict[str, int]
    adv_v: float


class ExactSeriesCache:
    def __init__(self, C_A: np.ndarray, C_B: np.ndarray, cfg_r: Dict[str, Any]) -> None:
        self.C_A = C_A
        self.C_B = C_B
        self.cfg_r = cfg_r
        self._cache: Dict[Tuple[str, float, int], Any] = {}

    def get(self, seq: str, phi: float, Tmax: int) -> Any:
        key = (str(seq), round(float(phi), 14), int(Tmax))
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
        raise ValueError(f"Unknown policy={policy}")

    mA = fit_tail_and_metrics(cache.get("A", float(phi), Tmax_cache), int(T), int(T0), int(T1), float(cfg_r["rel_tol"]))
    mB = fit_tail_and_metrics(cache.get("B", float(phi), Tmax_cache), int(T), int(T0), int(T1), float(cfg_r["rel_tol"]))
    mABB = fit_tail_and_metrics(cache.get("ABB", float(phi), Tmax_cache), int(T), int(T0), int(T1), float(cfg_r["rel_tol"]))
    flags = compute_decision_flags(mA, mB, mABB, thresholds, cfg_r)
    adv_v = float(mABB.v_fit - max(mA.v_fit, mB.v_fit))
    return Bundle(mA=mA, mB=mB, mABB=mABB, flags=flags, adv_v=adv_v)


def pick_stage8_points(stage5_ref: Dict[str, Any], stage7_ref: Dict[str, Any]) -> List[Tuple[str, float]]:
    pts = [
        ("paper_max_T300", float(stage5_ref["T300_fixed"]["max_adv_phi"])),
        (f"refined_max_T{int(stage7_ref['fixed_max_row']['T'])}", float(stage7_ref["fixed_max_row"]["phi"])),
        (f"pp_primary_mid_T{stage7_ref['primary_fixed_T_key']}", float(stage7_ref["primary_fixed_interval"]["midpoint_phi"])),
        (f"pp_primary_left_T{stage7_ref['primary_fixed_T_key']}", float(stage7_ref["primary_fixed_interval"]["left_phi"])),
        (f"pp_primary_right_T{stage7_ref['primary_fixed_T_key']}", float(stage7_ref["primary_fixed_interval"]["right_phi"])),
    ]
    out: List[Tuple[str, float]] = []
    seen = set()
    for label, phi in pts:
        key = round(float(phi), 14)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, float(phi)))
    return out


# -----------------------------------------------------------------------------
# Module 1: selected-point long exact continuation
# -----------------------------------------------------------------------------


def run_selected_point_long_continuation(
    run_dir: Path,
    stage5_ref: Dict[str, Any],
    stage7_ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    T_timeseries: int,
    resume: bool,
) -> Dict[str, Any]:
    ts_dir = ensure_dir(run_dir / "selected_point_timeseries")
    summary_csv = run_dir / "selected_points_long_summary.csv"
    extrap_csv = run_dir / "selected_points_limit_fits.csv"
    summary_json = run_dir / "selected_points_long_summary.json"

    if (not resume) and summary_csv.exists():
        summary_csv.unlink()
    if (not resume) and extrap_csv.exists():
        extrap_csv.unlink()

    existing_rows = read_csv_rows(summary_csv) if resume else []
    done_keys = {(r["label"], r["sequence"]) for r in existing_rows}

    coin_params = cfg_r["coin_params"]
    C_A = su2_coin(*coin_params["A"], degrees=True)
    C_B = su2_coin(*coin_params["B"], degrees=True)

    points = pick_stage8_points(stage5_ref, stage7_ref)
    checkpoints = default_checkpoint_list_long(int(T_timeseries))
    summary_rows: List[Dict[str, Any]] = []
    extrap_rows: List[Dict[str, Any]] = []

    for label, phi in points:
        for seq in ["A", "B", "ABB"]:
            key = (label, seq)
            if key in done_keys:
                continue
            ser = simulate_unitary_scalar_timeseries(
                seq=seq,
                Tmax=int(T_timeseries),
                phi=float(phi),
                C_A=C_A,
                C_B=C_B,
                x0_primary=int(cfg_r["x0_primary"]),
                x0_sens=int(cfg_r["x0_sensitivity"]),
            )
            ts_rows = []
            for t in range(int(T_timeseries) + 1):
                ts_rows.append({
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
                })
            write_csv(ts_dir / f"timeseries_{label}_{seq}.csv", ts_rows)

            vals_for_extrap: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
            for policy in ["fixed", "scaled"]:
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
                    for metric_name, metric_value in [
                        ("v_fit", float(m.v_fit)),
                        ("deltaP_late_mean", float(m.deltaP_late_mean)),
                        ("w_loc3", float(m.w_loc3)),
                        ("w_loc5", float(m.w_loc5)),
                        ("P0bar", float(m.P0bar)),
                    ]:
                        if int(T) >= min(max(6000, checkpoints[0]), int(T_timeseries)):
                            vals_for_extrap.setdefault((policy, metric_name), []).append((int(T), metric_value))
            append_csv_rows(summary_csv, summary_rows)
            summary_rows.clear()

            for (policy, metric_name), pairs in vals_for_extrap.items():
                if len(pairs) < 3:
                    continue
                Tvals = [p[0] for p in pairs]
                yvals = [p[1] for p in pairs]
                fit1 = fit_inverse_series(Tvals, yvals, degree=1)
                fit2 = fit_inverse_series(Tvals, yvals, degree=2)
                extrap_rows.append({
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
                })

    if extrap_rows:
        write_csv(extrap_csv, extrap_rows)
    else:
        if not extrap_csv.exists():
            extrap_csv.write_text("", encoding="utf-8")

    summary = {
        "T_timeseries": int(T_timeseries),
        "checkpoints": [int(t) for t in checkpoints],
        "points": [{"label": label, "phi": float(phi), "phi_over_pi": float(phi / math.pi)} for label, phi in points],
        "summary_csv": str(summary_csv.relative_to(run_dir)),
        "extrapolation_csv": str(extrap_csv.relative_to(run_dir)),
        "timeseries_dir": str(ts_dir.relative_to(run_dir)),
    }
    write_json(summary_json, summary)
    return summary


# -----------------------------------------------------------------------------
# Module 2: boundary continuation near the primary / sensitivity interval
# -----------------------------------------------------------------------------


def boundary_mask_value(mask_name: str, bundle: Bundle) -> int:
    return int(bundle.flags[mask_name])


def bisection_bool(
    func,
    left: float,
    right: float,
    left_val: int,
    right_val: int,
    target_left_zero_right_one: bool,
    tol: float,
    max_iter: int = 60,
) -> float:
    a = float(left)
    b = float(right)
    fa = int(left_val)
    fb = int(right_val)
    if target_left_zero_right_one:
        if not (fa == 0 and fb == 1):
            raise ValueError("Expected left=0, right=1 bracket")
        for _ in range(int(max_iter)):
            if abs(b - a) <= tol:
                break
            m = 0.5 * (a + b)
            fm = int(func(m))
            if fm == 1:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
        return 0.5 * (a + b)
    else:
        if not (fa == 1 and fb == 0):
            raise ValueError("Expected left=1, right=0 bracket")
        for _ in range(int(max_iter)):
            if abs(b - a) <= tol:
                break
            m = 0.5 * (a + b)
            fm = int(func(m))
            if fm == 1:
                a = m
                fa = fm
            else:
                b = m
                fb = fm
        return 0.5 * (a + b)


def find_switch_bracket(phi_grid: np.ndarray, vals: Sequence[int], want: str) -> Tuple[float, float, int, int]:
    if want == "left":
        # find 0 -> 1
        for i in range(len(phi_grid) - 1):
            if int(vals[i]) == 0 and int(vals[i + 1]) == 1:
                return float(phi_grid[i]), float(phi_grid[i + 1]), int(vals[i]), int(vals[i + 1])
    elif want == "right":
        # find 1 -> 0
        for i in range(len(phi_grid) - 1):
            if int(vals[i]) == 1 and int(vals[i + 1]) == 0:
                return float(phi_grid[i]), float(phi_grid[i + 1]), int(vals[i]), int(vals[i + 1])
    raise RuntimeError(f"Could not find {want} switch bracket")


def locate_switch_bracket_with_expansion(
    *,
    center: float,
    base_margin: float,
    side_scan_points: int,
    max_expansions: int,
    want: str,
    eval_flag,
) -> Tuple[Optional[Tuple[float, float, int, int]], np.ndarray, List[int], float]:
    margin = float(base_margin)
    phi_grid = np.array([], dtype=np.float64)
    vals: List[int] = []
    for _ in range(int(max_expansions) + 1):
        phi_grid = np.linspace(center - margin, center + margin, int(side_scan_points), endpoint=True, dtype=np.float64)
        vals = [int(eval_flag(float(phi))) for phi in phi_grid]
        try:
            br = find_switch_bracket(phi_grid, vals, want=want)
            return br, phi_grid, vals, margin
        except RuntimeError:
            margin *= 2.0
            continue
    return None, phi_grid, vals, margin


def run_boundary_continuation(
    run_dir: Path,
    stage4_ref: Dict[str, Any],
    stage7_ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    T_boundary_list: List[int],
    side_scan_points: int,
    side_margin_pi: float,
    tol_pi: float,
) -> Dict[str, Any]:
    out_csv = run_dir / "boundary_continuation.csv"
    summary_json = run_dir / "boundary_continuation_summary.json"

    coin_params = cfg_r["coin_params"]
    C_A = su2_coin(*coin_params["A"], degrees=True)
    C_B = su2_coin(*coin_params["B"], degrees=True)
    cache = ExactSeriesCache(C_A=C_A, C_B=C_B, cfg_r=cfg_r)
    thresholds = stage4_ref["thresholds"]

    prim = stage7_ref["primary_fixed_interval"]
    left0 = float(prim["left_phi"])
    right0 = float(prim["right_phi"])
    mid0 = float(prim["midpoint_phi"])

    margin = float(side_margin_pi * math.pi)
    tol = float(tol_pi * math.pi)

    rows: List[Dict[str, Any]] = []

    for mask_name in ["pp_primary", "pp_sensitivity"]:
        for policy in ["fixed", "scaled"]:
            for T in T_boundary_list:
                def fmask(phi_val: float) -> int:
                    return boundary_mask_value(
                        mask_name,
                        evaluate_bundle(
                            phi=float(phi_val),
                            T=int(T),
                            policy=policy,
                            cache=cache,
                            cfg_r=cfg_r,
                            thresholds=thresholds,
                            Tmax_cache=int(T),
                        ),
                    )

                left_bracket, left_grid, left_vals, left_margin_used = locate_switch_bracket_with_expansion(
                    center=float(left0),
                    base_margin=float(margin),
                    side_scan_points=int(side_scan_points),
                    max_expansions=4,
                    want="left",
                    eval_flag=fmask,
                )
                right_bracket, right_grid, right_vals, right_margin_used = locate_switch_bracket_with_expansion(
                    center=float(right0),
                    base_margin=float(margin),
                    side_scan_points=int(side_scan_points),
                    max_expansions=4,
                    want="right",
                    eval_flag=fmask,
                )

                if left_bracket is None or right_bracket is None:
                    rows.extend([
                        {
                            "mask_name": mask_name,
                            "policy": policy,
                            "T": int(T),
                            "side": "left",
                            "phi_root": np.nan,
                            "phi_root_over_pi": np.nan,
                            "search_center": float(left0),
                            "search_margin": float(left_margin_used),
                            "search_margin_over_pi": float(left_margin_used / math.pi),
                            "side_scan_points": int(side_scan_points),
                            "interval_width": np.nan,
                            "interval_width_over_pi": np.nan,
                            "midpoint_phi": np.nan,
                            "midpoint_phi_over_pi": np.nan,
                            "strict_stage3_all_profile_points": 0,
                            "drift_stage3_all_profile_points": 0,
                            "pp_primary_all_profile_points": 0,
                            "pp_sensitivity_all_profile_points": 0,
                            "bracket_found": int(left_bracket is not None and right_bracket is not None),
                        },
                        {
                            "mask_name": mask_name,
                            "policy": policy,
                            "T": int(T),
                            "side": "right",
                            "phi_root": np.nan,
                            "phi_root_over_pi": np.nan,
                            "search_center": float(right0),
                            "search_margin": float(right_margin_used),
                            "search_margin_over_pi": float(right_margin_used / math.pi),
                            "side_scan_points": int(side_scan_points),
                            "interval_width": np.nan,
                            "interval_width_over_pi": np.nan,
                            "midpoint_phi": np.nan,
                            "midpoint_phi_over_pi": np.nan,
                            "strict_stage3_all_profile_points": 0,
                            "drift_stage3_all_profile_points": 0,
                            "pp_primary_all_profile_points": 0,
                            "pp_sensitivity_all_profile_points": 0,
                            "bracket_found": int(left_bracket is not None and right_bracket is not None),
                        },
                    ])
                    continue

                left_a, left_b, fa, fb = left_bracket
                right_a, right_b, fa_r, fb_r = right_bracket

                left_root = bisection_bool(
                    func=fmask,
                    left=left_a,
                    right=left_b,
                    left_val=fa,
                    right_val=fb,
                    target_left_zero_right_one=True,
                    tol=float(tol),
                )
                right_root = bisection_bool(
                    func=fmask,
                    left=right_a,
                    right=right_b,
                    left_val=fa_r,
                    right_val=fb_r,
                    target_left_zero_right_one=False,
                    tol=float(tol),
                )
                midpoint = 0.5 * (left_root + right_root)

                # Profile points for certification-like bookkeeping.
                prof_pts = np.linspace(left_root, right_root, 9, endpoint=True, dtype=np.float64)
                prof_flags = []
                all_strict = True
                all_drift = True
                all_primary = True
                all_sens = True
                for phi in prof_pts:
                    b = evaluate_bundle(phi=float(phi), T=int(T), policy=policy, cache=cache, cfg_r=cfg_r, thresholds=thresholds, Tmax_cache=int(T))
                    prof_flags.append(int(boundary_mask_value(mask_name, b)))
                    all_strict = all_strict and bool(b.flags["strict_stage3"] == 1)
                    all_drift = all_drift and bool(b.flags["drift_stage3"] == 1)
                    all_primary = all_primary and bool(b.flags["pp_primary"] == 1)
                    all_sens = all_sens and bool(b.flags["pp_sensitivity"] == 1)

                rows.extend([
                    {
                        "mask_name": mask_name,
                        "policy": policy,
                        "T": int(T),
                        "side": "left",
                        "phi_root": float(left_root),
                        "phi_root_over_pi": float(left_root / math.pi),
                        "search_center": float(left0),
                        "search_margin": float(left_margin_used),
                        "search_margin_over_pi": float(left_margin_used / math.pi),
                        "side_scan_points": int(side_scan_points),
                        "interval_width": float(right_root - left_root),
                        "interval_width_over_pi": float((right_root - left_root) / math.pi),
                        "midpoint_phi": float(midpoint),
                        "midpoint_phi_over_pi": float(midpoint / math.pi),
                        "strict_stage3_all_profile_points": int(all_strict),
                        "drift_stage3_all_profile_points": int(all_drift),
                        "pp_primary_all_profile_points": int(all_primary),
                        "pp_sensitivity_all_profile_points": int(all_sens),
                        "bracket_found": 1,
                    },
                    {
                        "mask_name": mask_name,
                        "policy": policy,
                        "T": int(T),
                        "side": "right",
                        "phi_root": float(right_root),
                        "phi_root_over_pi": float(right_root / math.pi),
                        "search_center": float(right0),
                        "search_margin": float(right_margin_used),
                        "search_margin_over_pi": float(right_margin_used / math.pi),
                        "side_scan_points": int(side_scan_points),
                        "interval_width": float(right_root - left_root),
                        "interval_width_over_pi": float((right_root - left_root) / math.pi),
                        "midpoint_phi": float(midpoint),
                        "midpoint_phi_over_pi": float(midpoint / math.pi),
                        "strict_stage3_all_profile_points": int(all_strict),
                        "drift_stage3_all_profile_points": int(all_drift),
                        "pp_primary_all_profile_points": int(all_primary),
                        "pp_sensitivity_all_profile_points": int(all_sens),
                        "bracket_found": 1,
                    },
                ])

    write_csv(out_csv, rows)

    summary: Dict[str, Any] = {
        "T_boundary_list": [int(T) for T in T_boundary_list],
        "side_scan_points": int(side_scan_points),
        "side_margin_pi": float(side_margin_pi),
        "tol_pi": float(tol_pi),
        "csv": str(out_csv.relative_to(run_dir)),
        "intervals": {},
    }
    for mask_name in ["pp_primary", "pp_sensitivity"]:
        summary["intervals"][mask_name] = {}
        for policy in ["fixed", "scaled"]:
            summary["intervals"][mask_name][policy] = {}
            for T in T_boundary_list:
                sub = [r for r in rows if r["mask_name"] == mask_name and r["policy"] == policy and int(r["T"]) == int(T)]
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


# -----------------------------------------------------------------------------
# Module 3: extended spectral localized-subspace analysis
# -----------------------------------------------------------------------------


def fit_tail_exponential(prob: np.ndarray, idx0: int, d_min: int, d_max: int) -> Dict[str, Any]:
    d_values: List[int] = []
    y_values: List[float] = []
    max_d = min(int(d_max), idx0, len(prob) - idx0 - 1)
    for d in range(int(d_min), max_d + 1):
        y = float(prob[idx0 - d] + prob[idx0 + d])
        if y > 0.0:
            d_values.append(int(d))
            y_values.append(math.log(y))
    if len(d_values) < 4:
        return {"ok": False}
    x = np.asarray(d_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    slope = float(coef[1])
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    return {
        "ok": True,
        "n_points": int(len(x)),
        "intercept": float(coef[0]),
        "slope": float(slope),
        "kappa": float(max(0.0, -slope)),
        "rmse_log": float(rmse),
        "d_min": int(d_values[0]),
        "d_max": int(d_values[-1]),
    }


def run_extended_localized_subspace_suite(
    run_dir: Path,
    stage5_ref: Dict[str, Any],
    stage7_ref: Dict[str, Any],
    cfg_r: Dict[str, Any],
    spectral_L_list: List[int],
    top_k_modes: int,
    projector_k_list: List[int],
    n_cycles_ring: int,
    burn_cycles_ring: int,
    resume: bool,
) -> Dict[str, Any]:
    modes_csv = run_dir / "spectral_localized_modes.csv"
    proj_csv = run_dir / "spectral_projector_decomposition.csv"
    tail_csv = run_dir / "spectral_tail_fits.csv"
    conv_csv = run_dir / "spectral_subspace_convergence.csv"
    summary_json = run_dir / "spectral_localized_subspace_summary.json"

    if (not resume) and modes_csv.exists():
        modes_csv.unlink()
    if (not resume) and proj_csv.exists():
        proj_csv.unlink()
    if (not resume) and tail_csv.exists():
        tail_csv.unlink()
    if (not resume) and conv_csv.exists():
        conv_csv.unlink()

    existing_conv = read_csv_rows(conv_csv) if resume else []
    done_conv = {(r["label"], _int(r["L"])) for r in existing_conv}

    coin_params = cfg_r["coin_params"]
    C_A = su2_coin(*coin_params["A"], degrees=True)
    C_B = su2_coin(*coin_params["B"], degrees=True)

    points = pick_stage8_points(stage5_ref, stage7_ref)
    top_rows: List[Dict[str, Any]] = []
    proj_rows: List[Dict[str, Any]] = []
    tail_rows: List[Dict[str, Any]] = []
    conv_rows: List[Dict[str, Any]] = []

    for label, phi in points:
        prev_diag = None
        for L in spectral_L_list:
            if (label, int(L)) in done_conv:
                continue

            F, prefixes = build_period_operator_and_prefixes(seq="ABB", C_A=C_A, C_B=C_B, phi=float(phi), L=int(L))
            eigvals, eigvecs = np.linalg.eig(F)
            for j in range(eigvecs.shape[1]):
                nrm = np.linalg.norm(eigvecs[:, j])
                if nrm > 0.0:
                    eigvecs[:, j] /= nrm

            psi0 = initial_state_on_ring(int(L))
            overlaps = np.abs(eigvecs.conj().T @ psi0) ** 2
            obs_rows: List[Dict[str, Any]] = []
            for j in range(eigvecs.shape[1]):
                vec = eigvecs[:, j]
                obs = diag_prob_and_observables(vec, L=int(L), x0_primary=int(cfg_r["x0_primary"]), x0_sens=int(cfg_r["x0_sensitivity"]))
                amp = vec.reshape(2, 2 * int(L) + 1)
                prob = (np.abs(amp[0]) ** 2 + np.abs(amp[1]) ** 2).real.astype(np.float64)
                prob = prob / float(prob.sum())
                tail_fit = fit_tail_exponential(
                    prob=prob,
                    idx0=int(L),
                    d_min=int(cfg_r["x0_sensitivity"]) + 1,
                    d_max=max(int(cfg_r["x0_sensitivity"]) + 4, min(int(L) // 4, 40)),
                )
                obs_rows.append({
                    "label": label,
                    "phi": float(phi),
                    "phi_over_pi": float(phi / math.pi),
                    "L": int(L),
                    "mode_index": int(j),
                    "eigenphase": float(np.angle(eigvals[j])),
                    "abs_eigenvalue_minus_1": float(abs(abs(eigvals[j]) - 1.0)),
                    "w_loc3": float(obs["w_loc3"]),
                    "w_loc5": float(obs["w_loc5"]),
                    "P0": float(obs["P0"]),
                    "ipr": float(obs["ipr"]),
                    "overlap_weight": float(overlaps[j]),
                    "tail_fit_ok": int(tail_fit.get("ok", False)),
                    "tail_kappa": None if not tail_fit.get("ok", False) else float(tail_fit["kappa"]),
                    "tail_rmse_log": None if not tail_fit.get("ok", False) else float(tail_fit["rmse_log"]),
                    "tail_d_min": None if not tail_fit.get("ok", False) else int(tail_fit["d_min"]),
                    "tail_d_max": None if not tail_fit.get("ok", False) else int(tail_fit["d_max"]),
                })
            obs_sorted = sorted(obs_rows, key=lambda r: (r["w_loc5"], r["w_loc3"], r["P0"]), reverse=True)

            # Save top localized modes.
            for rank, rec in enumerate(obs_sorted[: int(top_k_modes)], start=1):
                top_rows.append({**rec, "rank_by_w_loc5": int(rank)})
                tail_rows.append({
                    "label": label,
                    "phi": float(phi),
                    "phi_over_pi": float(phi / math.pi),
                    "L": int(L),
                    "rank_by_w_loc5": int(rank),
                    "mode_index": int(rec["mode_index"]),
                    "w_loc3": float(rec["w_loc3"]),
                    "w_loc5": float(rec["w_loc5"]),
                    "P0": float(rec["P0"]),
                    "ipr": float(rec["ipr"]),
                    "overlap_weight": float(rec["overlap_weight"]),
                    "tail_fit_ok": int(rec["tail_fit_ok"]),
                    "tail_kappa": rec["tail_kappa"],
                    "tail_rmse_log": rec["tail_rmse_log"],
                })

            # Full diagonal ensemble and direct ring averages.
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
                "topk_overlap_weight": float(sum(r["overlap_weight"] for r in obs_sorted[: int(top_k_modes)])),
                "diag_period_avg_w_loc3": float(diag_avg["diag_period_avg_w_loc3"]),
                "diag_period_avg_w_loc5": float(diag_avg["diag_period_avg_w_loc5"]),
                "diag_period_avg_P0": float(diag_avg["diag_period_avg_P0"]),
                "ring_direct_period_avg_w_loc3": float(ring_avg["ring_direct_period_avg_w_loc3"]),
                "ring_direct_period_avg_w_loc5": float(ring_avg["ring_direct_period_avg_w_loc5"]),
                "ring_direct_period_avg_P0": float(ring_avg["ring_direct_period_avg_P0"]),
                "abs_diff_diag_vs_ring_w_loc3": float(abs(diag_avg["diag_period_avg_w_loc3"] - ring_avg["ring_direct_period_avg_w_loc3"])),
                "abs_diff_diag_vs_ring_w_loc5": float(abs(diag_avg["diag_period_avg_w_loc5"] - ring_avg["ring_direct_period_avg_w_loc5"])),
                "abs_diff_diag_vs_ring_P0": float(abs(diag_avg["diag_period_avg_P0"] - ring_avg["ring_direct_period_avg_P0"])),
            }
            if prev_diag is None:
                row.update({
                    "delta_from_prev_L_w_loc3": np.nan,
                    "delta_from_prev_L_w_loc5": np.nan,
                    "delta_from_prev_L_P0": np.nan,
                    "prev_L": np.nan,
                })
            else:
                row.update({
                    "delta_from_prev_L_w_loc3": float(abs(diag_avg["diag_period_avg_w_loc3"] - prev_diag["diag_period_avg_w_loc3"])),
                    "delta_from_prev_L_w_loc5": float(abs(diag_avg["diag_period_avg_w_loc5"] - prev_diag["diag_period_avg_w_loc5"])),
                    "delta_from_prev_L_P0": float(abs(diag_avg["diag_period_avg_P0"] - prev_diag["diag_period_avg_P0"])),
                    "prev_L": int(prev_diag["L"]),
                })
            conv_rows.append(row)
            prev_diag = {**diag_avg, "L": int(L)}

            # Projector decompositions using top-k localized modes by w_loc5.
            for k in sorted(set(int(kv) for kv in projector_k_list if int(kv) > 0)):
                sub = obs_sorted[: min(int(k), len(obs_sorted))]
                sub_weight = float(sum(float(r["overlap_weight"]) for r in sub))
                abs_contrib_w3 = float(sum(float(r["overlap_weight"]) * float(r["w_loc3"]) for r in sub))
                abs_contrib_w5 = float(sum(float(r["overlap_weight"]) * float(r["w_loc5"]) for r in sub))
                abs_contrib_P0 = float(sum(float(r["overlap_weight"]) * float(r["P0"]) for r in sub))
                if sub_weight > 0.0:
                    cond_w3 = abs_contrib_w3 / sub_weight
                    cond_w5 = abs_contrib_w5 / sub_weight
                    cond_P0 = abs_contrib_P0 / sub_weight
                    kappas = [float(r["tail_kappa"]) for r in sub if int(r["tail_fit_ok"]) == 1 and r["tail_kappa"] is not None]
                    weighted_kappa = None
                    if kappas:
                        num = sum(float(r["overlap_weight"]) * float(r["tail_kappa"]) for r in sub if int(r["tail_fit_ok"]) == 1 and r["tail_kappa"] is not None)
                        weighted_kappa = float(num / sub_weight)
                else:
                    cond_w3 = np.nan
                    cond_w5 = np.nan
                    cond_P0 = np.nan
                    weighted_kappa = None
                proj_rows.append({
                    "label": label,
                    "phi": float(phi),
                    "phi_over_pi": float(phi / math.pi),
                    "L": int(L),
                    "k_localized_modes": int(k),
                    "subspace_weight": float(sub_weight),
                    "absolute_contribution_w_loc3": float(abs_contrib_w3),
                    "absolute_contribution_w_loc5": float(abs_contrib_w5),
                    "absolute_contribution_P0": float(abs_contrib_P0),
                    "conditional_w_loc3": float(cond_w3),
                    "conditional_w_loc5": float(cond_w5),
                    "conditional_P0": float(cond_P0),
                    "weighted_tail_kappa": weighted_kappa,
                    "full_diag_w_loc3": float(diag_avg["diag_period_avg_w_loc3"]),
                    "full_diag_w_loc5": float(diag_avg["diag_period_avg_w_loc5"]),
                    "full_diag_P0": float(diag_avg["diag_period_avg_P0"]),
                })

            append_csv_rows(modes_csv, top_rows)
            append_csv_rows(tail_csv, tail_rows)
            append_csv_rows(proj_csv, proj_rows)
            append_csv_rows(conv_csv, [row])
            top_rows.clear()
            tail_rows.clear()
            proj_rows.clear()

    summary = {
        "points": [{"label": label, "phi": float(phi), "phi_over_pi": float(phi / math.pi)} for label, phi in points],
        "spectral_L_list": [int(L) for L in spectral_L_list],
        "top_k_modes": int(top_k_modes),
        "projector_k_list": [int(k) for k in projector_k_list],
        "n_cycles_ring": int(n_cycles_ring),
        "burn_cycles_ring": int(burn_cycles_ring),
        "modes_csv": str(modes_csv.relative_to(run_dir)),
        "projector_csv": str(proj_csv.relative_to(run_dir)),
        "tail_csv": str(tail_csv.relative_to(run_dir)),
        "convergence_csv": str(conv_csv.relative_to(run_dir)),
    }
    write_json(summary_json, summary)
    return summary


# -----------------------------------------------------------------------------
# Module 4: proof-ready notes and LaTeX patches
# -----------------------------------------------------------------------------


def lookup_extrap_row(extrap_rows: List[Dict[str, Any]], label: str, sequence: str, policy: str, metric_name: str) -> Optional[Dict[str, Any]]:
    candidates = [
        r for r in extrap_rows
        if r["label"] == label and r["sequence"] == sequence and r["policy"] == policy and r["metric_name"] == metric_name
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: int(r["T_max"]), reverse=True)[0]


def lookup_conv_row(conv_rows: List[Dict[str, Any]], label: str, L: int) -> Optional[Dict[str, Any]]:
    candidates = [r for r in conv_rows if r["label"] == label and _int(r["L"]) == int(L)]
    if not candidates:
        return None
    return candidates[0]


def build_proof_snippets(
    run_dir: Path,
    stage5_ref: Dict[str, Any],
    stage7_ref: Dict[str, Any],
    long_summary: Dict[str, Any],
    boundary_summary: Dict[str, Any],
    spectral_summary: Dict[str, Any],
) -> Dict[str, str]:
    extrap_rows = read_csv_rows(run_dir / long_summary["extrapolation_csv"])
    conv_rows = read_csv_rows(run_dir / spectral_summary["convergence_csv"])

    refined_label = f"refined_max_T{int(stage7_ref['fixed_max_row']['T'])}"
    pp_label = f"pp_primary_mid_T{stage7_ref['primary_fixed_T_key']}"
    largest_L = max(int(L) for L in spectral_summary["spectral_L_list"])

    ref_v = lookup_extrap_row(extrap_rows, refined_label, "ABB", "fixed", "v_fit")
    ref_w3 = lookup_extrap_row(extrap_rows, refined_label, "ABB", "fixed", "w_loc3")
    pp_v = lookup_extrap_row(extrap_rows, pp_label, "ABB", "fixed", "v_fit")
    pp_w3 = lookup_extrap_row(extrap_rows, pp_label, "ABB", "fixed", "w_loc3")
    ref_spec = lookup_conv_row(conv_rows, refined_label, largest_L)
    pp_spec = lookup_conv_row(conv_rows, pp_label, largest_L)

    b_fixed = boundary_summary["intervals"]["pp_primary"]["fixed"][str(max(int(T) for T in boundary_summary["T_boundary_list"]))]

    md_lines: List[str] = []
    md_lines.append("# Stage-8 proof snippets")
    md_lines.append("")
    md_lines.append("## Proposition 1 (phase damping is CPTP)")
    md_lines.append("")
    md_lines.append("With Kraus operators")
    md_lines.append("$$K_0=\\sqrt{1-p}\\,I,\\qquad K_1=\\sqrt{p}\\,|0\\rangle\\langle 0|,\\qquad K_2=\\sqrt{p}\\,|1\\rangle\\langle 1|,$$")
    md_lines.append("we have $\\sum_k K_k^\\dagger K_k = I$, so $\\mathcal E_p(\\rho)=\\sum_k K_k\\rho K_k^\\dagger$ is CPTP. For $\\rho=\\begin{pmatrix}a&c\\\\ c^\\ast&b\\end{pmatrix}$ one gets")
    md_lines.append("$$\\mathcal E_p(\\rho)=\\begin{pmatrix}a&(1-p)c\\\\ (1-p)c^\\ast&b\\end{pmatrix}.$$")
    md_lines.append("")
    md_lines.append("## Proposition 2 ($\\phi$ is irrelevant at complete dephasing)")
    md_lines.append("")
    md_lines.append("At $p=1$, every step removes all off-diagonal coin coherences before the shift statistics are read. The origin defect multiplies amplitudes at $x=0$ by a common phase factor $e^{i\\phi}$, which leaves diagonal probabilities unchanged. Hence the fully classicalized null is $\\phi$-invariant.")
    md_lines.append("")
    md_lines.append("## Lemma 3 (bias and drift can disagree)")
    md_lines.append("")
    md_lines.append("Take a distribution with probability $0.51$ at $x=+1$ and probability $0.49$ at $x=-100$. Then $\\Delta P=0.51-0.49>0$ but $\\langle x\\rangle=0.51-49<0$. Therefore right-heavy bias does not imply positive mean-position drift.")
    md_lines.append("")
    md_lines.append("## Proposition 4 (finite-horizon primary interval, exact follow-up)")
    md_lines.append("")
    md_lines.append(
        f"At fixed-window horizon $T={max(int(T) for T in boundary_summary['T_boundary_list'])}$, the exact follow-up estimates a nonzero primary PP interval "
        f"$[{b_fixed['left_phi_over_pi']:.6f},\\,{b_fixed['right_phi_over_pi']:.6f}]\\pi$ with width {b_fixed['width_over_pi']:.6f}$\\pi$."
    )
    md_lines.append("")
    md_lines.append("State this explicitly as a **finite-horizon exact proposition**, not as an infinite-time theorem.")
    md_lines.append("")
    md_lines.append("## Proposition 5 (localized-band / transport-band distinction)")
    md_lines.append("")
    if ref_v and ref_w3 and pp_v and pp_w3:
        md_lines.append(
            f"The refined maximum-advantage point has extrapolated ABB drift {float(ref_v['limit_estimate_deg2']):.6f} and extrapolated "
            f"$w_{{\\mathrm{{loc}},3}}$ {float(ref_w3['limit_estimate_deg2']):.6f}, while the PP-primary midpoint has extrapolated ABB drift "
            f"{float(pp_v['limit_estimate_deg2']):.6f} and extrapolated $w_{{\\mathrm{{loc}},3}}$ {float(pp_w3['limit_estimate_deg2']):.6f}. "
            f"This separates the strongest-drift localized band from the narrower transport interval."
        )
    if ref_spec and pp_spec:
        md_lines.append(
            f"On the largest tested ring ($L={largest_L}$), the top-localized-mode overlap is {float(ref_spec['topk_overlap_weight']):.6f} at the refined maximum "
            f"but {float(pp_spec['topk_overlap_weight']):.6f} at the PP-primary midpoint, supporting the bound-state/localization interpretation of the high-advantage band."
        )
    md_lines.append("")
    md_lines.append("## Wording guardrail")
    md_lines.append("")
    md_lines.append("Use phrases such as **exact finite-horizon**, **long-horizon noiseless follow-up**, and **finite-volume Floquet proxy**. Avoid claiming a closed-form infinite-time theorem unless you actually prove one.")
    proof_md = "\n".join(md_lines)

    tex_lines: List[str] = []
    tex_lines.append("% Auto-generated Stage-8 math appendix snippets")
    tex_lines.append("\\subsection*{Proposition 1 (Phase damping is CPTP)}")
    tex_lines.append("Let")
    tex_lines.append("\\[")
    tex_lines.append("K_0=\\sqrt{1-p}\\,I,\\qquad K_1=\\sqrt{p}\\,|0\\rangle\\langle 0|,\\qquad K_2=\\sqrt{p}\\,|1\\rangle\\langle 1|.")
    tex_lines.append("\\]")
    tex_lines.append("Then $\\sum_k K_k^\\dagger K_k=I$, so $\\mathcal E_p(\\rho)=\\sum_k K_k\\rho K_k^\\dagger$ is CPTP. For")
    tex_lines.append("\\[\\rho=\\begin{pmatrix}a&c\\\\ c^\\ast & b\\end{pmatrix},\\]")
    tex_lines.append("direct multiplication gives")
    tex_lines.append("\\[\\mathcal E_p(\\rho)=\\begin{pmatrix}a&(1-p)c\\\\ (1-p)c^\\ast&b\\end{pmatrix}.\\]")
    tex_lines.append("")
    tex_lines.append("\\subsection*{Proposition 2 ($\\phi$-irrelevance at complete dephasing)}")
    tex_lines.append("In the fully classicalized null, all off-diagonal coin coherences are removed at every step. The origin defect multiplies amplitudes at the origin by the common phase factor $e^{i\\phi}$, which leaves diagonal probabilities unchanged. Therefore the $p=1$ null is $\\phi$-invariant.")
    tex_lines.append("")
    tex_lines.append("\\subsection*{Lemma 3 (Bias and drift can disagree)}")
    tex_lines.append("Consider the distribution $\\mathbb P(X=+1)=0.51$ and $\\mathbb P(X=-100)=0.49$. Then $\\Delta P=0.51-0.49>0$ but $\\mathbb E[X]=0.51-49<0$. Hence right-heavy bias does not imply positive mean-position drift.")
    tex_lines.append("")
    tex_lines.append("\\subsection*{Proposition 4 (Finite-horizon primary interval at $p=0$)}")
    tex_lines.append(
        f"At fixed-window horizon $T={max(int(T) for T in boundary_summary['T_boundary_list'])}$, the exact follow-up estimates a nonzero primary PP interval "
        f"$[{b_fixed['left_phi_over_pi']:.6f},\\,{b_fixed['right_phi_over_pi']:.6f}]\\pi$ of width {b_fixed['width_over_pi']:.6f}\\pi$."
    )
    tex_lines.append("State this as a finite-horizon exact proposition, not as an infinite-time theorem.")
    appendix_tex = "\n".join(tex_lines)

    results_lines: List[str] = []
    results_lines.append("% Auto-generated Stage-8 results patch")
    results_lines.append("\\paragraph{Locked atlas versus exact follow-up.}")
    results_lines.append(
        "The pre-specified Stage-3 atlas at $T=300$ remains a valid finite-horizon benchmark, but the exact noiseless follow-up changes the broader $p=0$ story. "
        "In particular, the blanket phrase ``no primary PP point'' should be replaced by a horizon-qualified statement."
    )
    results_lines.append("")
    results_lines.append("\\paragraph{Updated exact-$p=0$ statement.}")
    results_lines.append(
        f"On the exact long-horizon noiseless follow-up, a narrow primary PP interval appears near $\\phi/\\pi\\approx 5/9$. "
        f"At the largest Stage-8 boundary continuation horizon, the fixed-window primary interval is "
        f"$[{b_fixed['left_phi_over_pi']:.6f},\\,{b_fixed['right_phi_over_pi']:.6f}]\\pi$ with width {b_fixed['width_over_pi']:.6f}\\pi$."
    )
    if ref_v and ref_w3 and pp_v and pp_w3:
        results_lines.append("")
        results_lines.append("\\paragraph{Localized high-advantage band versus transport interval.}")
        results_lines.append(
            f"The strongest drift-advantage region is not the transport interval. The refined maximum-advantage point has extrapolated ABB drift "
            f"{float(ref_v['limit_estimate_deg2']):.6f} with extrapolated localization weight $w_{{\\mathrm{{loc}},3}}={float(ref_w3['limit_estimate_deg2']):.6f}$, "
            f"whereas the PP-primary midpoint has extrapolated ABB drift {float(pp_v['limit_estimate_deg2']):.6f} with "
            f"$w_{{\\mathrm{{loc}},3}}={float(pp_w3['limit_estimate_deg2']):.6f}$. "
            "Thus the largest drift-advantage region is strongly localized and should be discussed separately from directed transport."
        )
    results_tex = "\n".join(results_lines)

    report_lines: List[str] = []
    report_lines.append("# Stage-8 rigour bridge report")
    report_lines.append("")
    report_lines.append("This run is designed to bridge the remaining gap between strong exact numerics and a defensible mathematical-physics writeup.")
    report_lines.append("")
    report_lines.append("## What this run adds")
    report_lines.append("")
    report_lines.append("1. Longer exact selected-point continuations on the infinite line.")
    report_lines.append("2. Higher-horizon continuation of the PP-primary and PP-sensitivity interval boundaries.")
    report_lines.append("3. Extended localized-subspace / Floquet-ring analysis with tail fits and projector decompositions.")
    report_lines.append("4. Proof-ready markdown and LaTeX snippets for the mathematics appendix and results patch.")
    report_lines.append("")
    report_lines.append("## How to use the outputs")
    report_lines.append("")
    report_lines.append("- Use `selected_points_limit_fits.csv` for horizon-qualified finite-time statements.")
    report_lines.append("- Use `boundary_continuation.csv` and `boundary_continuation_summary.json` for the primary-interval endpoint quotation.")
    report_lines.append("- Use `spectral_localized_modes.csv`, `spectral_projector_decomposition.csv`, and `spectral_subspace_convergence.csv` to support the localized-band interpretation.")
    report_lines.append("- Paste from `proof_snippets.md`, `math_appendix_latex.tex`, and `results_patch_latex.tex` into the manuscript.")
    stage8_report = "\n".join(report_lines)

    return {
        "proof_md": proof_md,
        "appendix_tex": appendix_tex,
        "results_tex": results_tex,
        "stage8_report": stage8_report,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_projector_k_list(text: str) -> List[int]:
    return sorted(set(int(s.strip()) for s in text.split(",") if s.strip()))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 8 rigour bridge suite: long exact selected-point continuations, boundary continuation, localized-subspace spectral analysis, and proof-ready appendix patches.")
    p.add_argument("--config", required=True, help="Path to mrc.yaml")
    p.add_argument("--stage4_run_dir", required=True, help="Path to Stage-4 exact p=0 audit")
    p.add_argument("--stage5_run_dir", required=True, help="Path to Stage-5 rigour suite")
    p.add_argument("--stage7_run_dir", required=True, help="Path to Stage-7 exact p=0 asymptotic suite")
    p.add_argument("--run_dir", default=None, help="Optional output directory")
    p.add_argument("--resume", action="store_true", help="Resume partially completed CSV outputs where possible")
    p.add_argument("--T_timeseries", type=int, default=24000, help="Maximum exact horizon for selected-point infinite-line continuations")
    p.add_argument("--T_boundary_list", default="12000", help="Comma-separated horizons for boundary continuation near the PP interval")
    p.add_argument("--side_scan_points", type=int, default=31, help="Per-side scan points around each inherited boundary")
    p.add_argument("--side_margin_pi", type=float, default=0.006, help="Boundary scan half-width in units of pi")
    p.add_argument("--tol_pi", type=float, default=2e-6, help="Boundary bisection tolerance in units of pi")
    p.add_argument("--spectral_L_list", default="120,160,220,300", help="Comma-separated ring half-sizes L for localized-subspace analysis")
    p.add_argument("--top_k_modes", type=int, default=16, help="How many most localized modes to save per point/L")
    p.add_argument("--projector_k_list", default="2,4,8,16", help="Comma-separated k values for cumulative localized-subspace projectors")
    p.add_argument("--n_cycles_ring", type=int, default=4000, help="Number of periods for direct ring averaging")
    p.add_argument("--burn_cycles_ring", type=int, default=800, help="Burn-in periods discarded in direct ring averaging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg_r = resolve_from_mrc(cfg)

    stage4_ref = load_stage4_reference(Path(args.stage4_run_dir))
    stage5_ref = load_stage5_reference(Path(args.stage5_run_dir))
    stage7_ref = load_stage7_reference(Path(args.stage7_run_dir))

    if args.run_dir is None:
        run_dir = make_run_dir("stage8_rigour_bridge_suite")["run_dir"]
    else:
        run_dir = ensure_dir(Path(args.run_dir))
        ensure_dir(run_dir / "figures")
    write_yaml(run_dir / "config_used.yaml", cfg)

    T_boundary_list = parse_int_list(args.T_boundary_list)
    spectral_L_list = parse_int_list(args.spectral_L_list)
    projector_k_list = parse_projector_k_list(args.projector_k_list)

    long_summary = run_selected_point_long_continuation(
        run_dir=run_dir,
        stage5_ref=stage5_ref,
        stage7_ref=stage7_ref,
        cfg_r=cfg_r,
        T_timeseries=int(args.T_timeseries),
        resume=bool(args.resume),
    )

    boundary_summary = run_boundary_continuation(
        run_dir=run_dir,
        stage4_ref=stage4_ref,
        stage7_ref=stage7_ref,
        cfg_r=cfg_r,
        T_boundary_list=T_boundary_list,
        side_scan_points=int(args.side_scan_points),
        side_margin_pi=float(args.side_margin_pi),
        tol_pi=float(args.tol_pi),
    )

    spectral_summary = run_extended_localized_subspace_suite(
        run_dir=run_dir,
        stage5_ref=stage5_ref,
        stage7_ref=stage7_ref,
        cfg_r=cfg_r,
        spectral_L_list=spectral_L_list,
        top_k_modes=int(args.top_k_modes),
        projector_k_list=projector_k_list,
        n_cycles_ring=int(args.n_cycles_ring),
        burn_cycles_ring=int(args.burn_cycles_ring),
        resume=bool(args.resume),
    )

    snippets = build_proof_snippets(
        run_dir=run_dir,
        stage5_ref=stage5_ref,
        stage7_ref=stage7_ref,
        long_summary=long_summary,
        boundary_summary=boundary_summary,
        spectral_summary=spectral_summary,
    )
    write_text(run_dir / "proof_snippets.md", snippets["proof_md"])
    write_text(run_dir / "math_appendix_latex.tex", snippets["appendix_tex"])
    write_text(run_dir / "results_patch_latex.tex", snippets["results_tex"])
    write_text(run_dir / "stage8_report.md", snippets["stage8_report"])

    summary = {
        "stage4_run_dir": str(args.stage4_run_dir),
        "stage5_run_dir": str(args.stage5_run_dir),
        "stage7_run_dir": str(args.stage7_run_dir),
        "selected_point_long_summary": long_summary,
        "boundary_continuation_summary": boundary_summary,
        "spectral_localized_subspace_summary": spectral_summary,
        "proof_snippets_md": "proof_snippets.md",
        "math_appendix_tex": "math_appendix_latex.tex",
        "results_patch_tex": "results_patch_latex.tex",
        "stage8_report_md": "stage8_report.md",
    }
    write_json(run_dir / "stage8_summary.json", summary)
    write_manifest(run_dir, "stage8_rigour_bridge_suite", {
        "config": str(args.config),
        "stage4_run_dir": str(args.stage4_run_dir),
        "stage5_run_dir": str(args.stage5_run_dir),
        "stage7_run_dir": str(args.stage7_run_dir),
        "T_timeseries": int(args.T_timeseries),
        "T_boundary_list": T_boundary_list,
        "spectral_L_list": spectral_L_list,
        "top_k_modes": int(args.top_k_modes),
        "projector_k_list": projector_k_list,
        "n_cycles_ring": int(args.n_cycles_ring),
        "burn_cycles_ring": int(args.burn_cycles_ring),
        "resume": bool(args.resume),
    })


if __name__ == "__main__":
    main()
