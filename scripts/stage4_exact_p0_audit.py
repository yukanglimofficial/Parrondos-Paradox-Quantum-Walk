#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from dtqw.io import load_yaml, write_json, write_manifest, write_text, write_yaml
from dtqw.metrics import compute_metrics_from_P_t, prob_from_psi_t, fit_v_fit, w_loc, P0bar, deltaP
from dtqw.simulate import run_sequence_unitary


def make_run_dir(tag: str) -> Dict[str, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / f"run_{stamp}_{tag}"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=False)
    return {"run_dir": run_dir, "fig_dir": fig_dir}


def _fit_tail_and_stability(x_mean_t: np.ndarray, T0: int, T1: int) -> Dict[str, float]:
    v_fit, _ = fit_v_fit(x_mean_t, T0=T0)
    v_fit2, _ = fit_v_fit(x_mean_t, T0=T1)
    T = int(len(x_mean_t) - 1)
    v_T = float(x_mean_t[T] / T)
    tt = np.arange(T1, T + 1, dtype=float)
    ratio = x_mean_t[T1:] / tt
    delta_v = float(np.max(np.abs(ratio - v_T)))
    return {
        "v_fit": float(v_fit),
        "v_fit2": float(v_fit2),
        "v_T": float(v_T),
        "delta_v": float(delta_v),
    }


def _stability_mask(v: np.ndarray, v2: np.ndarray, vT: np.ndarray, dv: np.ndarray, rel_tol: float = 0.5) -> np.ndarray:
    eps = 1e-12
    denom = np.maximum(np.abs(v), eps)
    sign_ok = (np.sign(v) == np.sign(v2)) & (np.sign(v) == np.sign(vT)) & (np.sign(v) != 0)
    rel_ok = (np.abs(v2 - v) / denom <= rel_tol) & (np.abs(vT - v) / denom <= rel_tol) & (dv / denom <= rel_tol)
    return sign_ok & rel_ok


def _sign_eps(x: np.ndarray, eps: float) -> np.ndarray:
    s = np.zeros_like(x, dtype=np.int8)
    s[x > eps] = 1
    s[x < -eps] = -1
    return s


def _seq_metrics_for_phi_grid(
    seq: str,
    phi_vals: np.ndarray,
    T: int,
    T0: int,
    T1: int,
    coin_params: Dict[str, tuple[float, float, float]],
) -> Dict[str, np.ndarray]:
    Nphi = int(phi_vals.size)
    out = {
        "v_fit": np.zeros(Nphi, dtype=float),
        "v_fit2": np.zeros(Nphi, dtype=float),
        "v_T": np.zeros(Nphi, dtype=float),
        "delta_v": np.zeros(Nphi, dtype=float),
        "deltaP_late_mean": np.zeros(Nphi, dtype=float),
        "deltaP_final": np.zeros(Nphi, dtype=float),
        "w_loc3": np.zeros(Nphi, dtype=float),
        "w_loc5": np.zeros(Nphi, dtype=float),
        "P0bar": np.zeros(Nphi, dtype=float),
        "x_mean_final": np.zeros(Nphi, dtype=float),
        "x_mean_late_mean": np.zeros(Nphi, dtype=float),
    }

    for i, phi in enumerate(phi_vals):
        sim = run_sequence_unitary(sequence=seq, T=T, phi=float(phi), coin_params=coin_params)
        x = sim["x"]
        P_t = prob_from_psi_t(sim["psi_t"])
        met = compute_metrics_from_P_t(P_t, x, T0=T0, x0_loc=3)
        tail = _fit_tail_and_stability(met["x_mean_t"], T0=T0, T1=T1)
        dP_t = np.asarray(met["deltaP_t"], dtype=float)

        out["v_fit"][i] = tail["v_fit"]
        out["v_fit2"][i] = tail["v_fit2"]
        out["v_T"][i] = tail["v_T"]
        out["delta_v"][i] = tail["delta_v"]
        out["deltaP_late_mean"][i] = float(np.mean(dP_t[T0:]))
        out["deltaP_final"][i] = float(dP_t[-1])
        out["w_loc3"][i] = float(w_loc(P_t, x, x0=3, T0=T0))
        out["w_loc5"][i] = float(w_loc(P_t, x, x0=5, T0=T0))
        out["P0bar"][i] = float(P0bar(P_t, x, T0=T0))
        out["x_mean_final"][i] = float(met["x_mean_t"][-1])
        out["x_mean_late_mean"][i] = float(np.mean(np.asarray(met["x_mean_t"], dtype=float)[T0:]))

    return out


def _resolve_from_mrc(cfg: Dict[str, Any]) -> Dict[str, Any]:
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
    T = int(sim["horizons"]["T_long"])
    T0 = int(sim["late_windows"]["T0"])
    T1 = int(sim["late_windows"]["T1"])
    N_phi = int(cfg["stage3_grid"]["N_phi"])
    phi_vals = np.linspace(0.0, 2.0 * math.pi, N_phi, endpoint=False, dtype=float)
    rel_tol = 0.5
    return {
        "coin_params": coin_params,
        "T": T,
        "T0": T0,
        "T1": T1,
        "N_phi": N_phi,
        "phi_vals": phi_vals,
        "rel_tol": rel_tol,
    }


def _load_stage3_thresholds_and_p0(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "metrics_summary.json"
    atlas_path = run_dir / "atlas.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing metrics_summary.json: {summary_path}")
    if not atlas_path.exists():
        raise FileNotFoundError(f"Missing atlas.npz: {atlas_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    atlas = np.load(atlas_path, allow_pickle=False)
    p = atlas["p"].astype(float)
    if not np.isclose(p[0], 0.0):
        raise ValueError("Expected first p grid point to be 0.")
    return {
        "summary": summary,
        "atlas": atlas,
        "thresholds": summary["thresholds"],
        "p0_index": 0,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact noiseless p=0 audit for the PH4511 DTQW project.")
    ap.add_argument("--config", default="mrc.yaml", help="Path to mrc.yaml")
    ap.add_argument(
        "--stage3_run_dir",
        default="outputs/run_20260212_090815_stage3_pd_confirmatory",
        help="Existing Stage-3 run dir containing atlas.npz and metrics_summary.json",
    )
    ap.add_argument(
        "--tag",
        default="stage4_exact_p0_audit",
        help="Suffix for outputs/run_*_<tag>",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = load_yaml(cfg_path)
    r = _resolve_from_mrc(cfg)

    stage3_run_dir = Path(args.stage3_run_dir)
    if not stage3_run_dir.exists():
        raise FileNotFoundError(f"Stage-3 run dir not found: {stage3_run_dir}")
    stage3 = _load_stage3_thresholds_and_p0(stage3_run_dir)
    thresholds = stage3["thresholds"]
    atlas = stage3["atlas"]

    paths = make_run_dir(tag=args.tag)
    run_dir = paths["run_dir"]

    write_yaml(run_dir / "config_used.yaml", cfg)
    write_manifest(run_dir, tag=args.tag)

    T = int(r["T"])
    T0 = int(r["T0"])
    T1 = int(r["T1"])
    phi_vals = np.asarray(r["phi_vals"], dtype=float)
    rel_tol = float(r["rel_tol"])
    coin_params = r["coin_params"]

    seqs = ["A", "B", "ABB"]
    seq_metrics: Dict[str, Dict[str, np.ndarray]] = {}
    for seq in seqs:
        print(f"[compute] exact p=0 phi-grid for sequence {seq} ...", flush=True)
        seq_metrics[seq] = _seq_metrics_for_phi_grid(seq, phi_vals, T, T0, T1, coin_params)

    vA = seq_metrics["A"]["v_fit"]
    vB = seq_metrics["B"]["v_fit"]
    vABB = seq_metrics["ABB"]["v_fit"]
    vA2 = seq_metrics["A"]["v_fit2"]
    vB2 = seq_metrics["B"]["v_fit2"]
    vABB2 = seq_metrics["ABB"]["v_fit2"]
    vAT = seq_metrics["A"]["v_T"]
    vBT = seq_metrics["B"]["v_T"]
    vABBT = seq_metrics["ABB"]["v_T"]
    dVA = seq_metrics["A"]["delta_v"]
    dVB = seq_metrics["B"]["delta_v"]
    dVABB = seq_metrics["ABB"]["delta_v"]

    stA = _stability_mask(vA, vA2, vAT, dVA, rel_tol=rel_tol)
    stB = _stability_mask(vB, vB2, vBT, dVB, rel_tol=rel_tol)
    stABB = _stability_mask(vABB, vABB2, vABBT, dVABB, rel_tol=rel_tol)

    v_min = float(thresholds["v_min"])
    eps_v = float(thresholds["eps_v"])
    eps_P = float(thresholds["eps_P"])
    w_thr_primary = float(thresholds["w_thr_primary"])
    w_thr_sensitivity = float(thresholds["w_thr_sensitivity"])

    loseA = (vA < 0.0) & (np.abs(vA) > v_min) & stA
    loseB = (vB < 0.0) & (np.abs(vB) > v_min) & stB
    winABB = (vABB > 0.0) & (vABB > v_min) & stABB
    strict = loseA & loseB
    dt_primary = seq_metrics["ABB"]["w_loc3"] < w_thr_primary
    dt_sens = seq_metrics["ABB"]["w_loc5"] < w_thr_sensitivity
    pp_primary = strict & winABB & dt_primary
    pp_sens = strict & winABB & dt_sens
    adv = vABB - np.maximum(vA, vB)

    sv = _sign_eps(vABB, eps_v)
    sp = _sign_eps(seq_metrics["ABB"]["deltaP_late_mean"], eps_P)
    mismatch = (sv * sp == -1)
    mismatch_denom = int(((sv != 0) & (sp != 0)).sum())

    rows = []
    for i, phi in enumerate(phi_vals):
        rows.append(
            {
                "phi_index": int(i),
                "phi": float(phi),
                "phi_over_pi": float(phi / math.pi),
                "v_fit_A": float(vA[i]),
                "v_fit2_A": float(vA2[i]),
                "v_T_A": float(vAT[i]),
                "delta_v_A": float(dVA[i]),
                "stable_A": int(stA[i]),
                "v_fit_B": float(vB[i]),
                "v_fit2_B": float(vB2[i]),
                "v_T_B": float(vBT[i]),
                "delta_v_B": float(dVB[i]),
                "stable_B": int(stB[i]),
                "v_fit_ABB": float(vABB[i]),
                "v_fit2_ABB": float(vABB2[i]),
                "v_T_ABB": float(vABBT[i]),
                "delta_v_ABB": float(dVABB[i]),
                "stable_ABB": int(stABB[i]),
                "adv_v": float(adv[i]),
                "deltaP_late_mean_ABB": float(seq_metrics["ABB"]["deltaP_late_mean"][i]),
                "deltaP_final_ABB": float(seq_metrics["ABB"]["deltaP_final"][i]),
                "w_loc3_ABB": float(seq_metrics["ABB"]["w_loc3"][i]),
                "w_loc5_ABB": float(seq_metrics["ABB"]["w_loc5"][i]),
                "P0bar_ABB": float(seq_metrics["ABB"]["P0bar"][i]),
                "strict": int(strict[i]),
                "win_ABB": int(winABB[i]),
                "pp_primary": int(pp_primary[i]),
                "pp_sensitivity": int(pp_sens[i]),
                "mismatch_sign": int(mismatch[i]),
            }
        )
    _write_csv(run_dir / "exact_p0_grid.csv", rows)

    def _filter_rows(mask: np.ndarray) -> List[Dict[str, Any]]:
        return [rows[i] for i in range(len(rows)) if bool(mask[i])]

    _write_csv(run_dir / "exact_p0_strict_points.csv", _filter_rows(strict))
    _write_csv(run_dir / "exact_p0_drift_parrondo_points.csv", _filter_rows(strict & winABB))
    _write_csv(run_dir / "exact_p0_pp_primary_points.csv", _filter_rows(pp_primary))
    _write_csv(run_dir / "exact_p0_pp_sensitivity_points.csv", _filter_rows(pp_sens))
    _write_csv(run_dir / "exact_p0_mismatch_points.csv", _filter_rows(mismatch))

    # Top-20 advantage points.
    top_idx = np.argsort(adv)[::-1][:20]
    _write_csv(run_dir / "exact_p0_top20_advantage_points.csv", [rows[int(i)] for i in top_idx])

    # Stage-0 exact benchmark at phi=0, T=50 for A/B/ABB to repair Table 1.
    stage0_rows: List[Dict[str, Any]] = []
    T_stage0 = 50
    T0_stage0 = 25
    for seq in seqs:
        sim = run_sequence_unitary(sequence=seq, T=T_stage0, phi=0.0, coin_params=coin_params)
        x = sim["x"]
        P_t = prob_from_psi_t(sim["psi_t"])
        met = compute_metrics_from_P_t(P_t, x, T0=T0_stage0, x0_loc=2)
        dP_t = np.asarray(met["deltaP_t"], dtype=float)
        stage0_rows.append(
            {
                "sequence": seq,
                "T": T_stage0,
                "phi": 0.0,
                "T0": T0_stage0,
                "v_fit": float(met["v_fit"]),
                "deltaP_final": float(dP_t[-1]),
                "deltaP_late_mean": float(np.mean(dP_t[T0_stage0:])),
                "w_loc_x0eq2": float(met["w_loc"]),
                "P0bar": float(met["P0bar"]),
            }
        )
    _write_csv(run_dir / "stage0_exact_phi0_T50.csv", stage0_rows)

    # Comparison against included Stage-3 atlas p=0 slice.
    p0 = int(stage3["p0_index"])
    diffs = {
        "max_abs_diff_v_fit_A": float(np.max(np.abs(vA - atlas["v_fit_A"][:, p0]))),
        "max_abs_diff_v_fit_B": float(np.max(np.abs(vB - atlas["v_fit_B"][:, p0]))),
        "max_abs_diff_v_fit_ABB": float(np.max(np.abs(vABB - atlas["v_fit_ABB"][:, p0]))),
        "max_abs_diff_v_fit2_A": float(np.max(np.abs(vA2 - atlas["v_fit2_A"][:, p0]))),
        "max_abs_diff_v_fit2_B": float(np.max(np.abs(vB2 - atlas["v_fit2_B"][:, p0]))),
        "max_abs_diff_v_fit2_ABB": float(np.max(np.abs(vABB2 - atlas["v_fit2_ABB"][:, p0]))),
        "max_abs_diff_v_T_A": float(np.max(np.abs(vAT - atlas["v_T_A"][:, p0]))),
        "max_abs_diff_v_T_B": float(np.max(np.abs(vBT - atlas["v_T_B"][:, p0]))),
        "max_abs_diff_v_T_ABB": float(np.max(np.abs(vABBT - atlas["v_T_ABB"][:, p0]))),
        "max_abs_diff_delta_v_A": float(np.max(np.abs(dVA - atlas["delta_v_A"][:, p0]))),
        "max_abs_diff_delta_v_B": float(np.max(np.abs(dVB - atlas["delta_v_B"][:, p0]))),
        "max_abs_diff_delta_v_ABB": float(np.max(np.abs(dVABB - atlas["delta_v_ABB"][:, p0]))),
        "max_abs_diff_deltaP_late_mean_ABB": float(np.max(np.abs(seq_metrics["ABB"]["deltaP_late_mean"] - atlas["deltaP_late_mean_ABB"][:, p0]))),
        "max_abs_diff_w_loc3_ABB": float(np.max(np.abs(seq_metrics["ABB"]["w_loc3"] - atlas["w_loc_primary_ABB"][:, p0]))),
        "max_abs_diff_w_loc5_ABB": float(np.max(np.abs(seq_metrics["ABB"]["w_loc5"] - atlas["w_loc_sensitivity_ABB"][:, p0]))),
        "max_abs_diff_P0bar_ABB": float(np.max(np.abs(seq_metrics["ABB"]["P0bar"] - atlas["P0bar_ABB"][:, p0]))),
    }

    i_max = int(np.argmax(adv))
    i_pp_sens = np.flatnonzero(pp_sens)
    i_pp_primary = np.flatnonzero(pp_primary)
    first_pp_sens = int(i_pp_sens[0]) if i_pp_sens.size else None
    first_pp_primary = int(i_pp_primary[0]) if i_pp_primary.size else None

    summary = {
        "purpose": "Exact deterministic audit of the noiseless p=0 slice plus Stage-0 benchmark repair",
        "stage3_reference_run": str(stage3_run_dir),
        "config": {
            "config_path": str(cfg_path),
            "T": T,
            "T0": T0,
            "T1": T1,
            "N_phi": int(phi_vals.size),
            "rel_tol": rel_tol,
        },
        "thresholds_from_stage3": {
            "v_min": v_min,
            "eps_v": eps_v,
            "eps_P": eps_P,
            "w_thr_primary": w_thr_primary,
            "w_thr_sensitivity": w_thr_sensitivity,
        },
        "counts_exact_p0": {
            "strict": int(strict.sum()),
            "drift_parrondo": int((strict & winABB).sum()),
            "pp_primary": int(pp_primary.sum()),
            "pp_sensitivity": int(pp_sens.sum()),
            "mismatch_sign": int(mismatch.sum()),
            "mismatch_sign_denominator": mismatch_denom,
        },
        "headline_points": {
            "max_advantage": rows[i_max],
            "first_pp_primary": rows[first_pp_primary] if first_pp_primary is not None else None,
            "first_pp_sensitivity": rows[first_pp_sens] if first_pp_sens is not None else None,
            "phi0_point": rows[0],
        },
        "comparison_to_stage3_p0_slice": diffs,
        "stage0_exact_benchmark": stage0_rows,
        "output_files": [
            "exact_p0_grid.csv",
            "exact_p0_strict_points.csv",
            "exact_p0_drift_parrondo_points.csv",
            "exact_p0_pp_primary_points.csv",
            "exact_p0_pp_sensitivity_points.csv",
            "exact_p0_mismatch_points.csv",
            "exact_p0_top20_advantage_points.csv",
            "stage0_exact_phi0_T50.csv",
        ],
    }
    write_json(run_dir / "exact_p0_summary.json", summary)

    log_lines = [
        f"run_dir={run_dir}",
        f"stage3_reference_run={stage3_run_dir}",
        f"counts_exact_p0.strict={int(strict.sum())}",
        f"counts_exact_p0.drift_parrondo={int((strict & winABB).sum())}",
        f"counts_exact_p0.pp_primary={int(pp_primary.sum())}",
        f"counts_exact_p0.pp_sensitivity={int(pp_sens.sum())}",
        f"counts_exact_p0.mismatch_sign={int(mismatch.sum())}/{mismatch_denom}",
        f"max_advantage.phi_over_pi={rows[i_max]['phi_over_pi']}",
        f"max_advantage.adv_v={rows[i_max]['adv_v']}",
        f"max_abs_diff_v_fit_ABB={diffs['max_abs_diff_v_fit_ABB']}",
    ]
    write_text(run_dir / "logs.txt", "\n".join(log_lines) + "\n")

    print("\n[done] wrote exact audit to:", run_dir)
    print("[done] summary:", run_dir / "exact_p0_summary.json")
    print("[done] exact p=0 grid:", run_dir / "exact_p0_grid.csv")
    print("[done] stage-0 benchmark:", run_dir / "stage0_exact_phi0_T50.csv")
    print("[done] stage-3 p=0 max abs diff v_fit_ABB:", diffs["max_abs_diff_v_fit_ABB"])
    print("[done] exact counts: strict=%d, drift_parrondo=%d, pp_primary=%d, pp_sensitivity=%d, mismatch=%d/%d" % (
        int(strict.sum()), int((strict & winABB).sum()), int(pp_primary.sum()), int(pp_sens.sum()), int(mismatch.sum()), mismatch_denom
    ))


if __name__ == "__main__":
    main()
