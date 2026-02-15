from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def latest_stage2_dir(base: str = "outputs") -> Path:
    runs = sorted(Path(base).glob("run_*_stage2_pd_atlas"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit("No outputs/run_*_stage2_pd_atlas folder found.")
    return runs[0]


def imshow_phi_p(phi_over_pi: np.ndarray, p: np.ndarray, z: np.ndarray, title: str, out_png: Path) -> None:
    extent = [float(phi_over_pi[0]), float(phi_over_pi[0] + 2.0), float(p[0]), float(p[-1])]

    plt.figure()
    plt.imshow(
        z.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )
    plt.xlabel("phi / pi")
    plt.ylabel("p")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def overlay_points(phi_over_pi: np.ndarray, p: np.ndarray, background: np.ndarray, mask: np.ndarray, title: str, out_png: Path) -> None:
    extent = [float(phi_over_pi[0]), float(phi_over_pi[0] + 2.0), float(p[0]), float(p[-1])]

    plt.figure()
    plt.imshow(
        background.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )
    ii, jj = np.where(mask)
    if ii.size > 0:
        plt.scatter(phi_over_pi[ii], p[jj], s=10, marker="o")
    plt.xlabel("phi / pi")
    plt.ylabel("p")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="", help="Path to outputs/run_*_stage2_pd_atlas (default: newest)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_stage2_dir()
    atlas_path = run_dir / "atlas.npz"
    if not atlas_path.exists():
        raise SystemExit(f"Missing: {atlas_path}")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(atlas_path, allow_pickle=True)

    phi = d["phi"].astype(float)
    p = d["p"].astype(float)
    phi_over_pi = phi / np.pi

    if "v_fit_ABB" in d:
        imshow_phi_p(phi_over_pi, p, d["v_fit_ABB"].astype(float), "raw v_fit(ABB) over (phi,p)", fig_dir / "L1_raw_v_fit_ABB.png")

    if "adv_v_max_ABB" in d:
        imshow_phi_p(phi_over_pi, p, d["adv_v_max_ABB"].astype(float), "Adv_v = v_fit(ABB) - max(v_fit(A), v_fit(B))", fig_dir / "L2_adv_v_max.png")

    vA = d["v_fit_A"].astype(float) if "v_fit_A" in d else None
    vB = d["v_fit_B"].astype(float) if "v_fit_B" in d else None

    if vA is not None:
        imshow_phi_p(phi_over_pi, p, np.sign(vA), "sign(v_fit(A)) (dev proxy for decisiveness)", fig_dir / "L3_sign_v_fit_A.png")
    if vB is not None:
        imshow_phi_p(phi_over_pi, p, np.sign(vB), "sign(v_fit(B)) (dev proxy for decisiveness)", fig_dir / "L4_sign_v_fit_B.png")

    if "strict_mask" in d:
        strict = (d["strict_mask"].astype(np.uint8) != 0)
    elif vA is not None and vB is not None:
        strict = (vA < 0.0) & (vB < 0.0)
    else:
        strict = None

    if "directed_transport_ABB" in d:
        directed = (d["directed_transport_ABB"].astype(np.uint8) != 0)
    else:
        w_thr = float(d["w_thr"]) if "w_thr" in d else 0.1
        directed = (d["w_loc_ABB"].astype(float) < w_thr) if "w_loc_ABB" in d else None

    # Preregistered (plan.txt) localization-threshold sensitivity:
    #   w_thr_primary = 0.10
    #   w_thr_sensitivity = 0.15 (i.e., w_thr_primary + delta_w_sensitivity, delta=0.05)
    w_thr_primary = float(d["w_thr"]) if "w_thr" in d else 0.10
    delta_w_sensitivity = float(d["delta_w_sensitivity"]) if "delta_w_sensitivity" in d else 0.05
    w_thr_sensitivity = float(d["w_thr_sensitivity"]) if "w_thr_sensitivity" in d else (w_thr_primary + delta_w_sensitivity)

    directed_sens = None
    if "w_loc_ABB" in d:
        w = d["w_loc_ABB"].astype(float)
        directed_sens = (w < w_thr_sensitivity)

    if strict is not None:
        imshow_phi_p(phi_over_pi, p, strict.astype(float), "strict eligibility mask (A losing & B losing)", fig_dir / "L5_mask_strict.png")
    if directed is not None:
        imshow_phi_p(phi_over_pi, p, directed.astype(float), "directed transport mask (w_loc_ABB < w_thr)", fig_dir / "L6_mask_directed_transport.png")

    if directed_sens is not None:
        imshow_phi_p(
            phi_over_pi,
            p,
            directed_sens.astype(float),
            "directed transport sensitivity mask (w_loc_ABB < w_thr_sensitivity)",
            fig_dir / "L11_mask_directed_transport_sensitivity.png",
        )

    if "parrondo_positive_ABB" in d:
        parr = (d["parrondo_positive_ABB"].astype(np.uint8) != 0)
    elif strict is not None and directed is not None and "v_fit_ABB" in d:
        parr = strict & (d["v_fit_ABB"].astype(float) > 0.0) & directed
    else:
        parr = None

    if parr is not None and "v_fit_ABB" in d:
        overlay_points(phi_over_pi, p, d["v_fit_ABB"].astype(float), parr, "Parrondo-positive overlay (dev predicate)", fig_dir / "L7_parrondo_positive_overlay_dev.png")

    parr_sens = None
    if strict is not None and directed_sens is not None and "v_fit_ABB" in d:
        parr_sens = strict & (d["v_fit_ABB"].astype(float) > 0.0) & directed_sens

        overlay_points(
            phi_over_pi,
            p,
            d["v_fit_ABB"].astype(float),
            parr_sens,
            "Parrondo-positive overlay (sensitivity w_thr_sensitivity)",
            fig_dir / "L12_parrondo_positive_overlay_sensitivity.png",
        )

    if "deltaP_late_mean_ABB" in d and "v_fit_ABB" in d:
        dP = d["deltaP_late_mean_ABB"].astype(float)
        v = d["v_fit_ABB"].astype(float)
        mismatch = ((v > 0.0) & (dP < 0.0)) | ((v < 0.0) & (dP > 0.0))
        imshow_phi_p(phi_over_pi, p, mismatch.astype(float), "mismatch: sign(v_fit_ABB) != sign(deltaP_late_mean_ABB)", fig_dir / "L8_mismatch_sign_v_vs_deltaP.png")

    if "w_loc_ABB" in d:
        imshow_phi_p(phi_over_pi, p, d["w_loc_ABB"].astype(float), "w_loc(ABB) over (phi,p)", fig_dir / "L9_w_loc_ABB.png")
    if "P0bar_ABB" in d:
        imshow_phi_p(phi_over_pi, p, d["P0bar_ABB"].astype(float), "P0bar(ABB) over (phi,p)", fig_dir / "L10_P0bar_ABB.png")

    if "w_loc_ABB" in d:
        w = d["w_loc_ABB"].astype(float)
        w_thr = float(d["w_thr"]) if "w_thr" in d else 0.1
        print("Run dir:", run_dir)
        print("w_loc_ABB min/max:", float(w.min()), float(w.max()), "w_thr_primary:", w_thr, "w_thr_sensitivity:", float(w_thr_sensitivity))
        print("directed_transport_points (primary):", int((w < w_thr).sum()))
        print("directed_transport_points (sensitivity):", int((w < float(w_thr_sensitivity)).sum()))
    if parr is not None:
        print("parrondo_positive_points:", int(np.sum(parr)))

    if parr_sens is not None:
        print("parrondo_positive_points (sensitivity):", int(np.sum(parr_sens)))


if __name__ == "__main__":
    main()
