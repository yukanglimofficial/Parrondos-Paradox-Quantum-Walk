from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from dtqw.io import load_yaml, make_run_dir, write_json, write_manifest, write_text, write_yaml
from dtqw.metrics import compute_metrics_from_P_t, prob_from_psi_t
from dtqw.simulate import run_sequence_unitary


def _resolve(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg1 = cfg.get("stage1_defect") or {}

    T = int(cfg1.get("T", cfg.get("T", 200)))

    sequences = cfg1.get("sequences", [cfg.get("sequence", "ABB")])
    if isinstance(sequences, str):
        sequences = [sequences]
    if not isinstance(sequences, list) or len(sequences) == 0:
        raise ValueError("stage1_defect.sequences must be a non-empty list (e.g. ['ABB']).")
    sequences = [str(s) for s in sequences]

    Nphi = int(cfg1.get("Nphi", cfg.get("Nphi", 72)))
    if Nphi <= 0:
        raise ValueError("Nphi must be positive.")

    phi_start = float(cfg1.get("phi_start", 0.0))
    phi_stop = float(cfg1.get("phi_stop", 2.0 * np.pi))

    coin_params_raw = cfg1.get("coin_params_degrees", cfg.get("coin_params_degrees"))
    if not isinstance(coin_params_raw, dict):
        raise ValueError("Missing coin_params_degrees mapping in dev.yaml.")

    coin_params: Dict[str, tuple[float, float, float]] = {}
    for k, v in coin_params_raw.items():
        if not (isinstance(v, (list, tuple)) and len(v) == 3):
            raise ValueError(f"coin_params_degrees[{k!r}] must be a 3-list [alpha,beta,gamma].")
        coin_params[str(k)] = (float(v[0]), float(v[1]), float(v[2]))

    # metrics: prefer stage1_defect.metrics, else top-level metrics, else defaults
    m1 = cfg1.get("metrics")
    m_top = cfg.get("metrics")
    m = (m1 if isinstance(m1, dict) else None) or (m_top if isinstance(m_top, dict) else {}) or {}

    T0 = int(m.get("T0", max(1, T // 2)))
    x0_loc = int(m.get("x0_loc", 2))

    return {
        "T": T,
        "sequences": sequences,
        "Nphi": Nphi,
        "phi_start": phi_start,
        "phi_stop": phi_stop,
        "coin_params_degrees": coin_params,
        "metrics": {"T0": T0, "x0_loc": x0_loc},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dev.yaml", help="Path to YAML config (default: dev.yaml)")
    ap.add_argument("--tag", default="stage1_defect_phi", help="Run tag suffix used in outputs/run_*_<tag>/")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    r = _resolve(cfg)

    paths = make_run_dir(tag=args.tag)
    run_dir = paths["run_dir"]
    fig_dir = paths["fig_dir"]

    # Required run artifacts
    write_yaml(run_dir / "config_used.yaml", cfg)
    write_yaml(run_dir / "resolved_config.yaml", r)
    write_manifest(run_dir, tag=args.tag, extra={"resolved": r})

    T = int(r["T"])
    Nphi = int(r["Nphi"])
    phi_start = float(r["phi_start"])
    phi_stop = float(r["phi_stop"])
    sequences: List[str] = list(r["sequences"])
    coin_params = r["coin_params_degrees"]
    T0 = int(r["metrics"]["T0"])
    x0_loc = int(r["metrics"]["x0_loc"])

    phi_vals = np.linspace(phi_start, phi_stop, Nphi, endpoint=False).astype(float)

    summary: Dict[str, Any] = {
        "params": {
            "T": T,
            "phi_grid": {"Nphi": Nphi, "phi_start": phi_start, "phi_stop": phi_stop, "endpoint": False},
            "sequences": sequences,
            "metrics": {"T0": T0, "x0_loc": x0_loc},
        },
        "results": {},
    }

    for seq in sequences:
        v_fit = np.zeros(Nphi, dtype=float)
        wloc = np.zeros(Nphi, dtype=float)
        p0bar = np.zeros(Nphi, dtype=float)
        dP_late = np.zeros(Nphi, dtype=float)
        dP_final = np.zeros(Nphi, dtype=float)

        # Store final distributions for exemplars and later plotting (small enough)
        x_any = None
        P_final = None

        for i, phi in enumerate(phi_vals):
            out = run_sequence_unitary(sequence=seq, T=T, phi=float(phi), coin_params=coin_params)
            x = out["x"]
            P_t = prob_from_psi_t(out["psi_t"])

            if x_any is None:
                x_any = x.copy()
                P_final = np.zeros((Nphi, x.size), dtype=float)

            met = compute_metrics_from_P_t(P_t, x, T0=T0, x0_loc=x0_loc)

            v_fit[i] = float(met["v_fit"])
            wloc[i] = float(met["w_loc"])
            p0bar[i] = float(met["P0bar"])

            dP_t = np.asarray(met["deltaP_t"], dtype=float)
            dP_final[i] = float(dP_t[-1])
            dP_late[i] = float(np.mean(dP_t[T0:]))

            P_final[i, :] = P_t[-1].astype(float)

        # Save numeric arrays
        np.savez(
            run_dir / f"phi_scan_{seq}.npz",
            phi=phi_vals,
            x=x_any,
            v_fit=v_fit,
            w_loc=wloc,
            P0bar=p0bar,
            deltaP_late_mean=dP_late,
            deltaP_final=dP_final,
            P_final=P_final,
            T=T,
            T0=T0,
            x0_loc=x0_loc,
            sequence=seq,
        )

        # Summaries for quick inspection
        imax_v = int(np.argmax(np.abs(v_fit)))
        imax_w = int(np.argmax(wloc))
        i0 = 0  # phi_start assumed to be 0 in our defaults, but still fine as a reference point

        summary["results"][seq] = {
            "phi_at_max_abs_v_fit": float(phi_vals[imax_v]),
            "max_abs_v_fit": float(v_fit[imax_v]),
            "phi_at_max_w_loc": float(phi_vals[imax_w]),
            "max_w_loc": float(wloc[imax_w]),
            "phi_at_phi_start": float(phi_vals[i0]),
            "v_fit_at_phi_start": float(v_fit[i0]),
            "deltaP_late_mean_at_phi_start": float(dP_late[i0]),
            "P0bar_at_phi_start": float(p0bar[i0]),
        }

        # Figures
        phi_over_pi = phi_vals / np.pi

        plt.figure()
        plt.plot(phi_over_pi, v_fit)
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("phi / pi")
        plt.ylabel("v_fit (OLS slope of <x>(t) over t in [T0, T])")
        plt.title(f"Stage 1 defect scan (seq={seq}) : v_fit vs phi")
        plt.tight_layout()
        plt.savefig(fig_dir / f"v_fit_vs_phi_{seq}.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(phi_over_pi, dP_late)
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("phi / pi")
        plt.ylabel("mean(deltaP(t)) over t in [T0, T]")
        plt.title(f"Stage 1 defect scan (seq={seq}) : deltaP late mean vs phi")
        plt.tight_layout()
        plt.savefig(fig_dir / f"deltaP_late_vs_phi_{seq}.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(phi_over_pi, wloc, label="w_loc (|x|<=x0_loc, late mean)")
        plt.plot(phi_over_pi, p0bar, label="P0bar (x=0, late mean)")
        plt.xlabel("phi / pi")
        plt.ylabel("mass")
        plt.title(f"Stage 1 defect scan (seq={seq}) : localization diagnostics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"localization_vs_phi_{seq}.png", dpi=200)
        plt.close()

        plt.figure()
        plt.scatter(v_fit, dP_late, s=12)
        plt.axhline(0.0, linewidth=1)
        plt.axvline(0.0, linewidth=1)
        plt.xlabel("v_fit")
        plt.ylabel("deltaP late mean")
        plt.title(f"Stage 1 defect scan (seq={seq}) : drift vs deltaP mismatch check")
        plt.tight_layout()
        plt.savefig(fig_dir / f"vfit_vs_deltaP_scatter_{seq}.png", dpi=200)
        plt.close()

        # Exemplar final distributions
        exemplar_idx = []
        for j in [i0, imax_w, imax_v]:
            if j not in exemplar_idx:
                exemplar_idx.append(j)

        plt.figure()
        for j in exemplar_idx:
            plt.plot(x_any, P_final[j, :], label=f"phi/pi={phi_over_pi[j]:.3f}")
        plt.xlabel("x")
        plt.ylabel("P(x, T)")
        plt.title(f"Stage 1 defect scan (seq={seq}) : exemplar final distributions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"P_final_exemplars_{seq}.png", dpi=200)
        plt.close()

    write_json(run_dir / "metrics_summary.json", summary)
    write_text(run_dir / "logs.txt", f"Run dir: {run_dir}\n")
    print(f"Wrote Stage 1 defect-phi scan to: {run_dir}")
    print(f"Summary: {run_dir / 'metrics_summary.json'}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
