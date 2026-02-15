from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def latest_run(pattern: str):
    runs = sorted(Path("outputs").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_mrc():
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    p = Path("mrc.yaml")
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text(encoding="utf-8-sig"))


def fmt(x) -> str:
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def main() -> None:
    Path("notes").mkdir(parents=True, exist_ok=True)

    stage0 = latest_run("run_*_stage0_*")
    stage1 = latest_run("run_*_stage1_*")
    stage2 = latest_run("run_*_stage2_pd_atlas")

    lines = []
    lines.append("# Paper readiness report")
    lines.append("")
    lines.append("Generated automatically by scripts/paper_readiness_report.py")
    lines.append("")

    lines.append("## Latest run folders detected")
    lines.append(f"- Stage0: {stage0}" if stage0 else "- Stage0: (not found)")
    lines.append(f"- Stage1: {stage1}" if stage1 else "- Stage1: (not found)")
    lines.append(f"- Stage2: {stage2}" if stage2 else "- Stage2: (not found)")
    lines.append("")

    def run_checks(run: Path, extra_required=None):
        req = ["metrics_summary.json", "config_used.yaml", "manifest.json", "logs.txt"]
        if extra_required:
            req += list(extra_required)
        ok = {}
        for r in req:
            ok[r] = (run / r).exists()
        return ok

    lines.append("## Reproducibility file check (bootstrap_plan expects these per run)")
    for label, run, extra in [
        ("Stage0", stage0, []),
        ("Stage1", stage1, ["phi_scan_ABB.npz"]),
        ("Stage2", stage2, ["atlas.npz"]),
    ]:
        if not run:
            lines.append(f"- {label}: (missing run folder)")
            continue
        ok = run_checks(run, extra_required=extra)
        missing = [k for k, v in ok.items() if not v]
        if missing:
            lines.append(f"- {label}: MISSING -> {', '.join(missing)}")
        else:
            lines.append(f"- {label}: OK (all expected files present)")
    lines.append("")

    if stage0 and (stage0 / "metrics_summary.json").exists():
        j0 = read_json(stage0 / "metrics_summary.json")
        lines.append("## Stage0 key results (Jan-style replication, dev run)")
        results = j0.get("results", {})
        for seq in ["A", "B", "ABB"]:
            if seq in results:
                r = results[seq]
                lines.append(
                    f"- {seq}: v_fit={fmt(r.get('v_fit'))}, "
                    f"deltaP_late_mean={fmt(r.get('deltaP_late_mean'))}, "
                    f"w_loc={fmt(r.get('w_loc'))}, P0bar={fmt(r.get('P0bar'))}"
                )
        lines.append("")

    if stage1 and (stage1 / "metrics_summary.json").exists():
        j1 = read_json(stage1 / "metrics_summary.json")
        lines.append("## Stage1 key results (defect-only phi scan at p=0, dev run)")
        r1 = j1.get("results", {}).get("ABB", {})
        if r1:
            lines.append(f"- ABB: phi_at_max_abs_v_fit={fmt(r1.get('phi_at_max_abs_v_fit'))}, max_abs_v_fit={fmt(r1.get('max_abs_v_fit'))}")
            lines.append(f"- phi_at_phi_start={fmt(r1.get('phi_at_phi_start'))}, v_fit_at_phi_start={fmt(r1.get('v_fit_at_phi_start'))}, deltaP_late_mean_at_phi_start={fmt(r1.get('deltaP_late_mean_at_phi_start'))}")
        lines.append("")

    if stage2 and (stage2 / "atlas.npz").exists():
        d = np.load(stage2 / "atlas.npz", allow_pickle=True)
        phi = d["phi"].astype(float)
        p = d["p"].astype(float)

        vA = d["v_fit_A"].astype(float) if "v_fit_A" in d else None
        vB = d["v_fit_B"].astype(float) if "v_fit_B" in d else None
        vABB = d["v_fit_ABB"].astype(float) if "v_fit_ABB" in d else None
        w = d["w_loc_ABB"].astype(float) if "w_loc_ABB" in d else None
        adv = d["adv_v_max_ABB"].astype(float) if "adv_v_max_ABB" in d else None
        dP = d["deltaP_late_mean_ABB"].astype(float) if "deltaP_late_mean_ABB" in d else None

        w_thr = float(d["w_thr"]) if "w_thr" in d else 0.10
        delta = float(d["delta_w_sensitivity"]) if "delta_w_sensitivity" in d else 0.05
        w_thr_sens = float(d["w_thr_sensitivity"]) if "w_thr_sensitivity" in d else (w_thr + delta)

        strict = None
        if "strict_mask" in d:
            strict = (d["strict_mask"].astype(np.uint8) != 0)
        elif (vA is not None) and (vB is not None):
            strict = (vA < 0.0) & (vB < 0.0)

        lines.append("## Stage2 key results (PD atlas, dev run)")
        lines.append(f"- Grid: Nphi={phi.size}, Np={p.size} (total={phi.size * p.size})")
        if w is not None:
            lines.append(f"- w_loc_ABB: min={fmt(np.min(w))}, max={fmt(np.max(w))}")
        else:
            lines.append("- w_loc_ABB: NA (missing in atlas)")
        lines.append(f"- Directed transport thresholds: w_thr_primary={fmt(w_thr)}, w_thr_sensitivity={fmt(w_thr_sens)}")

        if w is not None:
            lines.append(f"- directed_transport_points_primary = {int(np.sum(w < w_thr))}")
            lines.append(f"- directed_transport_points_sensitivity = {int(np.sum(w < w_thr_sens))}")

        if strict is not None:
            lines.append(f"- strict_mask_points = {int(np.sum(strict))}")

        if (strict is not None) and (vABB is not None) and (w is not None):
            parr_primary = strict & (vABB > 0.0) & (w < w_thr)
            parr_sens = strict & (vABB > 0.0) & (w < w_thr_sens)
            lines.append(f"- parrondo_positive_points_primary = {int(np.sum(parr_primary))}")
            lines.append(f"- parrondo_positive_points_sensitivity = {int(np.sum(parr_sens))}")

        if adv is not None:
            idx = np.unravel_index(np.argmax(adv), adv.shape)
            i, j = int(idx[0]), int(idx[1])
            lines.append(f"- max_adv_v_max_ABB = {fmt(adv[idx])} at phi={fmt(phi[i])} (phi_over_pi={fmt(phi[i] / np.pi)}), p={fmt(p[j])}")

        if vABB is not None:
            idx = np.unravel_index(np.argmax(vABB), vABB.shape)
            i, j = int(idx[0]), int(idx[1])
            lines.append(f"- max_v_fit_ABB = {fmt(vABB[idx])} at phi={fmt(phi[i])} (phi_over_pi={fmt(phi[i] / np.pi)}), p={fmt(p[j])}")

        if w is not None:
            idx = np.unravel_index(np.argmin(w), w.shape)
            i, j = int(idx[0]), int(idx[1])
            lines.append(f"- min_w_loc_ABB = {fmt(w[idx])} at phi={fmt(phi[i])} (phi_over_pi={fmt(phi[i] / np.pi)}), p={fmt(p[j])}")

        if (dP is not None) and (vABB is not None):
            mismatch = ((vABB > 0.0) & (dP < 0.0)) | ((vABB < 0.0) & (dP > 0.0))
            lines.append(f"- mismatch_sign(v_fit_ABB) vs sign(deltaP_late_mean_ABB): {int(np.sum(mismatch))} / {mismatch.size}")

        lines.append("")
    else:
        lines.append("## Stage2 key results (PD atlas, dev run)")
        lines.append("- Stage2 atlas.npz not found; cannot compute gridpoint counts.")
        lines.append("")

    mrc = load_mrc()
    lines.append("## Confirmatory settings (from mrc.yaml, should match plan.txt Box MR)")
    if not mrc:
        lines.append("- mrc.yaml not found or could not be parsed.")
    else:
        step_order = mrc.get("step_order", None)
        stage3_grid = mrc.get("stage3_grid", {})
        sim = mrc.get("simulation", {})
        horizons = sim.get("horizons", {})
        late = sim.get("late_windows", {})
        overlay = sim.get("localization_overlay", {})
        stoch = mrc.get("stochastic_estimation", {})
        boot = stoch.get("bootstrap", {}) if isinstance(stoch, dict) else {}
        rng = stoch.get("rng_policy", {}) if isinstance(stoch, dict) else {}

        lines.append(f"- step_order: {step_order}")
        lines.append(f"- T_long: {horizons.get('T_long')}, T0: {late.get('T0')}, T1: {late.get('T1')}")
        lines.append(f"- stage3_grid: N_phi={stage3_grid.get('N_phi')}, N_p={stage3_grid.get('N_p')}")
        lines.append(f"- N_traj: {stoch.get('N_traj')}, bootstrap_B: {boot.get('B')}, bootstrap_alpha: {boot.get('alpha')}, base_seed: {rng.get('base_seed')}")
        lines.append(f"- w_thr_primary: {overlay.get('w_thr_primary')}, w_thr_sensitivity: {overlay.get('w_thr_sensitivity')}, x0_primary: {overlay.get('x0_primary')}, x0_sensitivity: {overlay.get('x0_sensitivity')}")
    lines.append("")

    lines.append("## What is still missing to finish the paper (bootstrap_plan minimal done)")
    lines.append("- Fill in notes/paper_draft.md (methods, results for 3 panels, limitations, references).")
    lines.append("- Decide whether to run Milestone 8 robustness modules (bootstrap CIs, null baselines, region-level cluster overlay).")
    lines.append("- If making region claims, run confirmatory Stage3 with the locked settings above (and save config_used.yaml + manifest.json).")
    lines.append("")

    out_path = Path("notes") / "READINESS_REPORT.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("Wrote:", out_path)

    print("")
    print("=== QUICK SUMMARY ===")
    if stage2 and (stage2 / "metrics_summary.json").exists():
        j2 = read_json(stage2 / "metrics_summary.json")
        print("Stage2 counts from metrics_summary.json:", j2.get("counts", {}))
    if stage2 and (stage2 / "atlas.npz").exists():
        print("Stage2 atlas present:", stage2 / "atlas.npz")
    print("=====================")


if __name__ == "__main__":
    main()
