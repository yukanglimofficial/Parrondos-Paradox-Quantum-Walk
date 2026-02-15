#!/usr/bin/env python3
"""
Stage 3 postprocess: generate figure layers from Stage3 atlas.npz.

Run:
  python scripts/stage3_postprocess_figures.py --run_dir outputs/run_*_stage3_pd_confirmatory
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def imshow_phi_p(fig_path, phi, p, Z, title, cbar_label=""):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    extent = [float(p.min()), float(p.max()), float(phi.min() / math.pi), float(phi.max() / math.pi)]
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent)
    ax.set_xlabel("p (noise strength)")
    ax.set_ylabel("phi / pi")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    if cbar_label:
        cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

def overlay_points(ax, phi, p, mask, marker="o", ms=24.0, alpha=0.9):
    ii, jj = np.where(mask)
    if len(ii) == 0:
        return
    ax.scatter(p[jj], (phi[ii] / math.pi), marker=marker, s=ms, alpha=alpha)

def connected_cluster_sizes(mask):
    n_phi, n_p = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    sizes = []

    def neigh(i, j):
        yield ((i - 1) % n_phi, j)
        yield ((i + 1) % n_phi, j)
        if j - 1 >= 0: yield (i, j - 1)
        if j + 1 < n_p: yield (i, j + 1)

    for i in range(n_phi):
        for j in range(n_p):
            if not mask[i, j] or visited[i, j]:
                continue
            q = [(i, j)]
            visited[i, j] = True
            count = 0
            while q:
                ci, cj = q.pop()
                count += 1
                for ni, nj in neigh(ci, cj):
                    if mask[ni, nj] and not visited[ni, nj]:
                        visited[ni, nj] = True
                        q.append((ni, nj))
            sizes.append(count)
    sizes.sort(reverse=True)
    return sizes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    atlas_path = run_dir / "atlas.npz"
    if not atlas_path.exists():
        raise FileNotFoundError(f"Missing atlas.npz: {atlas_path}")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    atlas = np.load(atlas_path, allow_pickle=False)
    phi = atlas["phi"].astype(float)
    p = atlas["p"].astype(float)

    vA = atlas["v_fit_A"].astype(float)
    vB = atlas["v_fit_B"].astype(float)
    vABB = atlas["v_fit_ABB"].astype(float)
    adv = atlas["adv_v_max_ABB"].astype(float)

    w3 = atlas["w_loc_primary_ABB"].astype(float)
    w5 = atlas["w_loc_sensitivity_ABB"].astype(float)
    P0 = atlas["P0bar_ABB"].astype(float)
    dP = atlas["deltaP_late_mean_ABB"].astype(float)

    strict = atlas["strict_mask"].astype(bool)
    dt1 = atlas["directed_transport_primary"].astype(bool)
    dt2 = atlas["directed_transport_sensitivity"].astype(bool)
    pp1 = atlas["parrondo_positive_primary"].astype(bool)
    pp2 = atlas["parrondo_positive_sensitivity"].astype(bool)
    mismatch = atlas["mismatch_sign"].astype(bool)

    imshow_phi_p(fig_dir / "S3_L1_v_fit_ABB.png", phi, p, vABB, "Stage3: v_fit(ABB)", "v_fit")
    imshow_phi_p(fig_dir / "S3_L2_adv_v_max_ABB.png", phi, p, adv, "Stage3: adv_v_max(ABB) = v_fit(ABB) - max(v_fit(A), v_fit(B))", "adv_v")
    imshow_phi_p(fig_dir / "S3_L3_sign_v_fit_A.png", phi, p, np.sign(vA), "Stage3: sign(v_fit(A))", "sign")
    imshow_phi_p(fig_dir / "S3_L4_sign_v_fit_B.png", phi, p, np.sign(vB), "Stage3: sign(v_fit(B))", "sign")
    imshow_phi_p(fig_dir / "S3_L5_mask_strict.png", phi, p, strict.astype(float), "Stage3: strict eligibility (A losing & B losing)", "mask")
    imshow_phi_p(fig_dir / "S3_L6_mask_directed_primary.png", phi, p, dt1.astype(float), "Stage3: directed transport (primary)", "mask")

    # Overlay primary
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    extent = [float(p.min()), float(p.max()), float(phi.min() / math.pi), float(phi.max() / math.pi)]
    im = ax.imshow(adv, origin="lower", aspect="auto", extent=extent)
    ax.set_xlabel("p (noise strength)")
    ax.set_ylabel("phi / pi")
    ax.set_title("Stage3: adv_v_max(ABB) + Parrondo-positive overlay (primary)")
    overlay_points(ax, phi, p, pp1, marker="o", ms=26.0, alpha=0.9)
    fig.colorbar(im, ax=ax).set_label("adv_v")
    fig.tight_layout()
    fig.savefig(fig_dir / "S3_L7_parrondo_overlay_primary.png", dpi=200)
    plt.close(fig)

    imshow_phi_p(fig_dir / "S3_L8_mismatch_sign.png", phi, p, mismatch.astype(float), "Stage3: sign mismatch v_fit(ABB) vs deltaP_late_mean(ABB)", "mismatch")
    imshow_phi_p(fig_dir / "S3_L9_w_loc_primary_ABB.png", phi, p, w3, "Stage3: w_loc (primary window)", "w_loc")
    imshow_phi_p(fig_dir / "S3_L10_P0bar_ABB.png", phi, p, P0, "Stage3: P0bar (late window origin probability)", "P0bar")
    imshow_phi_p(fig_dir / "S3_L11_mask_directed_sensitivity.png", phi, p, dt2.astype(float), "Stage3: directed transport (sensitivity)", "mask")
    imshow_phi_p(fig_dir / "S3_L12_deltaP_late_mean_ABB.png", phi, p, dP, "Stage3: deltaP_late_mean(ABB)", "deltaP")

    # Overlay sensitivity
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(adv, origin="lower", aspect="auto", extent=extent)
    ax.set_xlabel("p (noise strength)")
    ax.set_ylabel("phi / pi")
    ax.set_title("Stage3: adv_v_max(ABB) + Parrondo-positive overlay (sensitivity)")
    overlay_points(ax, phi, p, pp2, marker="o", ms=26.0, alpha=0.9)
    fig.colorbar(im, ax=ax).set_label("adv_v")
    fig.tight_layout()
    fig.savefig(fig_dir / "S3_L13_parrondo_overlay_sensitivity.png", dpi=200)
    plt.close(fig)

    sidecar = {
        "parrondo_positive_primary_points": int(pp1.sum()),
        "primary_cluster_sizes": connected_cluster_sizes(pp1)[:20],
        "parrondo_positive_sensitivity_points": int(pp2.sum()),
        "sensitivity_cluster_sizes": connected_cluster_sizes(pp2)[:20],
    }
    (run_dir / "figure_sidecar_summary.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print("Wrote figures to:", fig_dir)
    print("Wrote:", run_dir / "figure_sidecar_summary.json")

if __name__ == "__main__":
    main()
