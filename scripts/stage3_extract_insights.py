#!/usr/bin/env python3
"""
Extract paper-ready tables + a short Stage3 insights report from an existing Stage3 run_dir.

Writes:
  reports/stage3_insights_report.md
  reports/stage3_counts_by_p.csv
  reports/stage3_top_adv_points.csv
  reports/stage3_parrondo_points.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _get(atlas, *names):
    for n in names:
        if n in atlas.files:
            return atlas[n]
    raise KeyError(f"Missing expected array; tried: {names}")


def _phi_over_pi(phi_val):
    return float(phi_val / math.pi)


def _row_for_point(i, j, phi, p, vA, vB, vABB, adv, w3, w5, P0, dP, masks, ci):
    out = {
        "phi_index": int(i),
        "p_index": int(j),
        "phi": float(phi[i]),
        "phi_over_pi": _phi_over_pi(phi[i]),
        "p": float(p[j]),
        "v_fit_A": float(vA[i, j]),
        "v_fit_B": float(vB[i, j]),
        "v_fit_ABB": float(vABB[i, j]),
        "adv_v": float(adv[i, j]),
        "w_loc_primary": float(w3[i, j]),
        "w_loc_sensitivity": float(w5[i, j]),
        "P0bar": float(P0[i, j]),
        "deltaP_late_mean": float(dP[i, j]),
        "strict": int(masks["strict"][i, j]),
        "dt_primary": int(masks["dt_primary"][i, j]),
        "dt_sens": int(masks["dt_sens"][i, j]),
        "pp_primary": int(masks["pp_primary"][i, j]),
        "pp_sens": int(masks["pp_sens"][i, j]),
        "mismatch": int(masks["mismatch"][i, j]),
    }
    if ci is not None:
        out.update({
            "vA_ci_lo": float(ci["vA_lo"][i, j]),
            "vA_ci_hi": float(ci["vA_hi"][i, j]),
            "vB_ci_lo": float(ci["vB_lo"][i, j]),
            "vB_ci_hi": float(ci["vB_hi"][i, j]),
            "vABB_ci_lo": float(ci["vABB_lo"][i, j]),
            "vABB_ci_hi": float(ci["vABB_hi"][i, j]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    atlas_path = run_dir / "atlas.npz"
    summary_path = run_dir / "metrics_summary.json"
    if not atlas_path.exists():
        raise FileNotFoundError(f"Missing atlas.npz: {atlas_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing metrics_summary.json: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    atlas = np.load(atlas_path, allow_pickle=False)

    phi = _get(atlas, "phi").astype(float)
    p = _get(atlas, "p", "p_grid").astype(float)

    vA = _get(atlas, "v_fit_A").astype(float)
    vB = _get(atlas, "v_fit_B").astype(float)
    vABB = _get(atlas, "v_fit_ABB").astype(float)
    adv = _get(atlas, "adv_v_max_ABB").astype(float)

    w3 = _get(atlas, "w_loc_primary_ABB").astype(float)
    w5 = _get(atlas, "w_loc_sensitivity_ABB").astype(float)
    P0 = _get(atlas, "P0bar_ABB").astype(float)
    dP = _get(atlas, "deltaP_late_mean_ABB").astype(float)

    masks = {
        "strict": _get(atlas, "strict_mask").astype(bool),
        "dt_primary": _get(atlas, "directed_transport_primary").astype(bool),
        "dt_sens": _get(atlas, "directed_transport_sensitivity").astype(bool),
        "pp_primary": _get(atlas, "parrondo_positive_primary").astype(bool),
        "pp_sens": _get(atlas, "parrondo_positive_sensitivity").astype(bool),
        "mismatch": _get(atlas, "mismatch_sign").astype(bool),
    }

    ci = None
    try:
        ci = {
            "vA_lo": _get(atlas, "v_fit_A_ci_lo").astype(float),
            "vA_hi": _get(atlas, "v_fit_A_ci_hi").astype(float),
            "vB_lo": _get(atlas, "v_fit_B_ci_lo").astype(float),
            "vB_hi": _get(atlas, "v_fit_B_ci_hi").astype(float),
            "vABB_lo": _get(atlas, "v_fit_ABB_ci_lo").astype(float),
            "vABB_hi": _get(atlas, "v_fit_ABB_ci_hi").astype(float),
        }
    except KeyError:
        ci = None

    # Recompute counts to sanity-check summary
    recomputed = {
        "strict_mask_points": int(masks["strict"].sum()),
        "directed_transport_points_primary": int(masks["dt_primary"].sum()),
        "directed_transport_points_sensitivity": int(masks["dt_sens"].sum()),
        "parrondo_positive_points_primary": int(masks["pp_primary"].sum()),
        "parrondo_positive_points_sensitivity": int(masks["pp_sens"].sum()),
        "mismatch_sign_points": int(masks["mismatch"].sum()),
    }

    # Max advantage point
    i_max, j_max = np.unravel_index(int(np.argmax(adv)), adv.shape)

    # Parrondo points
    pp1_idx = np.argwhere(masks["pp_primary"])
    pp2_idx = np.argwhere(masks["pp_sens"])

    # Top-K adv points
    K = 20
    flat = adv.ravel()
    top_flat_idx = np.argsort(flat)[::-1][:K]
    top_points = [np.unravel_index(int(fi), adv.shape) for fi in top_flat_idx]

    # Counts by p
    counts_by_p = []
    for j in range(p.size):
        counts_by_p.append({
            "p_index": int(j),
            "p": float(p[j]),
            "strict": int(masks["strict"][:, j].sum()),
            "dt_primary": int(masks["dt_primary"][:, j].sum()),
            "dt_sens": int(masks["dt_sens"][:, j].sum()),
            "pp_primary": int(masks["pp_primary"][:, j].sum()),
            "pp_sens": int(masks["pp_sens"][:, j].sum()),
            "mismatch": int(masks["mismatch"][:, j].sum()),
        })

    # Output dirs
    repo_root = run_dir.resolve().parents[1]
    paper_data = repo_root / "reports"
    paper_data.mkdir(parents=True, exist_ok=True)

    # Write CSVs
    top_csv = paper_data / "stage3_top_adv_points.csv"
    parr_csv = paper_data / "stage3_parrondo_points.csv"
    p_csv = paper_data / "stage3_counts_by_p.csv"

    def write_csv(path, rows):
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    top_rows = [_row_for_point(i, j, phi, p, vA, vB, vABB, adv, w3, w5, P0, dP, masks, ci) for (i, j) in top_points]
    write_csv(top_csv, top_rows)

    parr_rows = []
    for (i, j) in pp1_idx:
        parr_rows.append(_row_for_point(int(i), int(j), phi, p, vA, vB, vABB, adv, w3, w5, P0, dP, masks, ci))
    for (i, j) in pp2_idx:
        parr_rows.append(_row_for_point(int(i), int(j), phi, p, vA, vB, vABB, adv, w3, w5, P0, dP, masks, ci))
    write_csv(parr_csv, parr_rows)

    write_csv(p_csv, counts_by_p)

    # Markdown report
    report_path = paper_data / "stage3_insights_report.md"
    lines = []
    lines.append("# Stage-3 Insights Report")
    lines.append("")
    lines.append(f"- run_dir: {summary.get('run_dir', str(run_dir))}")
    lines.append(f"- grid: N_phi={phi.size}, N_p={p.size}, total={phi.size * p.size}")
    lines.append(f"- sim: {json.dumps(summary.get('sim', {}), indent=2)}")
    lines.append(f"- thresholds: {json.dumps(summary.get('thresholds', {}), indent=2)}")
    lines.append("")
    lines.append("## Count sanity-check")
    lines.append("")
    lines.append("metrics_summary.json counts:")
    lines.append("```")
    lines.append(json.dumps(summary.get("counts", {}), indent=2))
    lines.append("```")
    lines.append("recomputed from atlas masks:")
    lines.append("```")
    lines.append(json.dumps(recomputed, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Headline maximum advantage point")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps(_row_for_point(int(i_max), int(j_max), phi, p, vA, vB, vABB, adv, w3, w5, P0, dP, masks, ci), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Parrondo-positive points")
    lines.append("")
    lines.append(f"- primary count: {int(pp1_idx.shape[0])}")
    lines.append(f"- sensitivity count: {int(pp2_idx.shape[0])}")
    lines.append(f"- CSV: {parr_csv.name}")
    lines.append("")
    lines.append("## Noise dependence quick check")
    lines.append("")
    any_pp_sens_p_gt0 = bool(masks["pp_sens"][:, 1:].any()) if p.size > 1 else False
    lines.append(f"- any PP_sens at p>0? {any_pp_sens_p_gt0}")
    lines.append(f"- counts-by-p CSV: {p_csv.name}")
    lines.append("")
    lines.append("## Top-20 advantage points (CSV)")
    lines.append("")
    lines.append(f"- {top_csv.name}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n[Stage3 Insights] Wrote:")
    print(" -", report_path)
    print(" -", top_csv)
    print(" -", parr_csv)
    print(" -", p_csv)

    print("\n[Stage3 Insights] Headline max-adv point:")
    print(json.dumps(_row_for_point(int(i_max), int(j_max), phi, p, vA, vB, vABB, adv, w3, w5, P0, dP, masks, ci), indent=2))

    if pp2_idx.shape[0] > 0:
        i0, j0 = pp2_idx[0]
        print("\n[Stage3 Insights] First PP_sens point:")
        print(json.dumps(_row_for_point(int(i0), int(j0), phi, p, vA, vB, vABB, adv, w3, w5, P0, dP, masks, ci), indent=2))


if __name__ == "__main__":
    main()
