from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dtqw.io import load_yaml, make_run_dir, write_json, write_manifest, write_text, write_yaml
from dtqw.metrics import compute_metrics_from_P_t
from dtqw.noise import run_sequence_pd_trajectories


def _resolve(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg2 = cfg.get("stage2_pd_atlas") or {}

    T = int(cfg2.get("T", cfg.get("T", 100)))
    N_traj = int(cfg2.get("N_traj", cfg.get("N_traj", 100)))
    seed = int(cfg2.get("seed", cfg.get("seed", 0)))

    sequences = cfg2.get("sequences", ["A", "B", "ABB"])
    if isinstance(sequences, str):
        sequences = [sequences]
    sequences = [str(s) for s in sequences]
    if len(sequences) == 0:
        raise ValueError("stage2_pd_atlas.sequences must be non-empty")

    Nphi = int(cfg2.get("Nphi", 24))
    Np = int(cfg2.get("Np", 9))
    if Nphi <= 0 or Np <= 1:
        raise ValueError("Nphi must be >0 and Np must be >1")

    phi_start = float(cfg2.get("phi_start", 0.0))
    phi_stop = float(cfg2.get("phi_stop", 2.0 * np.pi))
    p_start = float(cfg2.get("p_start", 0.0))
    p_stop = float(cfg2.get("p_stop", 1.0))

    coin_params_raw = cfg2.get("coin_params_degrees", cfg.get("coin_params_degrees"))
    if not isinstance(coin_params_raw, dict):
        raise ValueError("Missing coin_params_degrees mapping in dev.yaml")
    coin_params: Dict[str, Tuple[float, float, float]] = {
        str(k): (float(v[0]), float(v[1]), float(v[2])) for k, v in coin_params_raw.items()
    }

    m2 = cfg2.get("metrics") or {}
    m_top = cfg.get("metrics") or {}
    m = m2 if isinstance(m2, dict) else (m_top if isinstance(m_top, dict) else {})
    T0 = int(m.get("T0", max(1, T // 2)))
    x0_loc = int(m.get("x0_loc", 3))

    w_thr = float(cfg2.get("w_thr", 0.10))

    return {
        "T": T,
        "N_traj": N_traj,
        "seed": seed,
        "sequences": sequences,
        "Nphi": Nphi,
        "Np": Np,
        "phi_start": phi_start,
        "phi_stop": phi_stop,
        "p_start": p_start,
        "p_stop": p_stop,
        "coin_params_degrees": coin_params,
        "metrics": {"T0": T0, "x0_loc": x0_loc},
        "w_thr": w_thr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dev.yaml", help="Path to YAML config (default: dev.yaml)")
    ap.add_argument("--tag", default="stage2_pd_atlas", help="Run tag suffix used in outputs/run_*_<tag>/")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    r = _resolve(cfg)

    paths = make_run_dir(tag=args.tag)
    run_dir = paths["run_dir"]
    fig_dir = paths["fig_dir"]

    write_yaml(run_dir / "config_used.yaml", cfg)
    write_yaml(run_dir / "resolved_config.yaml", r)
    write_manifest(run_dir, tag=args.tag, extra={"resolved": {k: r[k] for k in ["T","N_traj","Nphi","Np","w_thr"]}})

    T = int(r["T"])
    N_traj = int(r["N_traj"])
    base_seed = int(r["seed"])
    sequences: List[str] = list(r["sequences"])
    coin_params = r["coin_params_degrees"]
    T0 = int(r["metrics"]["T0"])
    x0_loc = int(r["metrics"]["x0_loc"])
    w_thr = float(r["w_thr"])

    phi_vals = np.linspace(float(r["phi_start"]), float(r["phi_stop"]), int(r["Nphi"]), endpoint=False).astype(float)
    p_vals = np.linspace(float(r["p_start"]), float(r["p_stop"]), int(r["Np"]), endpoint=True).astype(float)

    Nphi = phi_vals.size
    Np = p_vals.size

    # Storage per sequence
    maps: Dict[str, Dict[str, np.ndarray]] = {}
    for seq in sequences:
        maps[seq] = {
            "v_fit": np.zeros((Nphi, Np), dtype=float),
            "w_loc": np.zeros((Nphi, Np), dtype=float),
            "P0bar": np.zeros((Nphi, Np), dtype=float),
            "deltaP_late_mean": np.zeros((Nphi, Np), dtype=float),
        }

    # Run grid
    total_cells = len(sequences) * Nphi * Np
    pbar = tqdm(total=total_cells, desc="Stage2 PD atlas", unit="cell")

    for s_idx, seq in enumerate(sequences):
        for i, phi in enumerate(phi_vals):
            for j, p in enumerate(p_vals):
                # Deterministic per-cell seed (avoid cross-cell correlated RNG)
                cell_seed = (base_seed + 1000000 * s_idx + 1000 * i + j) % (2**32)

                out = run_sequence_pd_trajectories(
                    sequence=seq,
                    T=T,
                    phi=float(phi),
                    p=float(p),
                    coin_params=coin_params,
                    N_traj=N_traj,
                    seed=int(cell_seed),
                )
                x = out["x"]
                P_t = out["P_t"]

                met = compute_metrics_from_P_t(P_t, x, T0=T0, x0_loc=x0_loc)

                maps[seq]["v_fit"][i, j] = float(met["v_fit"])
                maps[seq]["w_loc"][i, j] = float(met["w_loc"])
                maps[seq]["P0bar"][i, j] = float(met["P0bar"])

                dP_t = np.asarray(met["deltaP_t"], dtype=float)
                maps[seq]["deltaP_late_mean"][i, j] = float(np.mean(dP_t[T0:]))

                pbar.update(1)

    pbar.close()

    # Derived maps for the canonical Parrondo trio if present
    have_ABB = ("ABB" in maps)
    have_A = ("A" in maps)
    have_B = ("B" in maps)

    adv_v = None
    strict_mask = None
    directed_transport = None
    parrondo_positive = None

    if have_ABB and have_A and have_B:
        vA = maps["A"]["v_fit"]
        vB = maps["B"]["v_fit"]
        vABB = maps["ABB"]["v_fit"]

        # Conservative advantage definition: ABB beats the better of A,B
        adv_v = vABB - np.maximum(vA, vB)

        strict_mask = (vA < 0.0) & (vB < 0.0)
        directed_transport = maps["ABB"]["w_loc"] < float(w_thr)
        ABB_win = vABB > 0.0
        parrondo_positive = strict_mask & ABB_win & directed_transport

    # Save raw atlas
    save_dict = {
        "phi": phi_vals,
        "p": p_vals,
        "x": np.arange(-T, T + 1, dtype=int),
        "T": T,
        "N_traj": N_traj,
        "T0": T0,
        "x0_loc": x0_loc,
        "w_thr": float(w_thr),
        "sequences": np.array(sequences, dtype=object),
    }
    for seq in sequences:
        for k, arr in maps[seq].items():
            save_dict[f"{k}_{seq}"] = arr

    if adv_v is not None:
        save_dict["adv_v_max_ABB"] = adv_v
        save_dict["strict_mask"] = strict_mask.astype(np.uint8)
        save_dict["directed_transport_ABB"] = directed_transport.astype(np.uint8)
        save_dict["parrondo_positive_ABB"] = parrondo_positive.astype(np.uint8)

    np.savez(run_dir / "atlas.npz", **save_dict)

    # Summary JSON
    summary: Dict[str, Any] = {
        "params": {
            "T": T,
            "N_traj": N_traj,
            "phi_grid": {"Nphi": int(phi_vals.size), "endpoint": False, "phi_start": float(phi_vals[0]), "phi_stop": float(r["phi_stop"])},
            "p_grid": {"Np": int(p_vals.size), "endpoint": True, "p_start": float(p_vals[0]), "p_stop": float(p_vals[-1])},
            "T0": T0,
            "x0_loc": x0_loc,
            "w_thr": float(w_thr),
            "sequences": sequences,
        },
        "counts": {},
    }

    if parrondo_positive is not None:
        summary["counts"] = {
            "strict_mask_points": int(np.sum(strict_mask)),
            "directed_transport_points": int(np.sum(directed_transport)),
            "parrondo_positive_points": int(np.sum(parrondo_positive)),
        }
        imax = np.unravel_index(int(np.argmax(adv_v)), adv_v.shape)
        summary["max_adv_v"] = {
            "adv_v": float(adv_v[imax]),
            "phi": float(phi_vals[imax[0]]),
            "phi_over_pi": float(phi_vals[imax[0]] / np.pi),
            "p": float(p_vals[imax[1]]),
        }

    write_json(run_dir / "metrics_summary.json", summary)

    # Figures (simple heatmaps)
    phi_over_pi = phi_vals / np.pi

    def _imshow(z: np.ndarray, title: str, fname: str) -> None:
        plt.figure()
        plt.imshow(
            z.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[float(phi_over_pi[0]), float(phi_over_pi[0] + 2.0), float(p_vals[0]), float(p_vals[-1])],
        )
        plt.xlabel("phi / pi")
        plt.ylabel("p")
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=200)
        plt.close()

    if "ABB" in maps:
        _imshow(maps["ABB"]["v_fit"], "v_fit(ABB) over (phi,p)", "v_fit_ABB.png")
        _imshow(maps["ABB"]["w_loc"], "w_loc(ABB) over (phi,p)", "w_loc_ABB.png")
        _imshow(maps["ABB"]["P0bar"], "P0bar(ABB) over (phi,p)", "P0bar_ABB.png")

    if adv_v is not None:
        _imshow(adv_v, "Adv_v = v_fit(ABB) - max(v_fit(A), v_fit(B))", "adv_v_max.png")

        # Overlay: Parrondo-positive points on v_fit(ABB)
        z = maps["ABB"]["v_fit"]
        plt.figure()
        plt.imshow(
            z.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[float(phi_over_pi[0]), float(phi_over_pi[0] + 2.0), float(p_vals[0]), float(p_vals[-1])],
        )
        ii, jj = np.where(parrondo_positive)
        if ii.size > 0:
            plt.scatter(phi_over_pi[ii], p_vals[jj], s=10, marker="o")
        plt.xlabel("phi / pi")
        plt.ylabel("p")
        plt.title("Parrondo-positive overlay (strict & ABB win & directed transport)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(fig_dir / "parrondo_positive_overlay.png", dpi=200)
        plt.close()

    write_text(run_dir / "logs.txt", f"Run dir: {run_dir}\n")
    print(f"Wrote Stage 2 PD atlas to: {run_dir}")
    print(f"Summary: {run_dir / 'metrics_summary.json'}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
