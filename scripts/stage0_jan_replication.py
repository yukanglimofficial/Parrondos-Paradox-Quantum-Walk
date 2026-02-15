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
    cfg0 = cfg.get("stage0_jan") or {}

    T = int(cfg0.get("T", cfg.get("T", 50)))
    phi = float(cfg0.get("phi", cfg.get("phi", 0.0)))

    sequences = cfg0.get("sequences", ["A", "B", "ABB"])
    if not isinstance(sequences, list) or len(sequences) == 0:
        raise ValueError("stage0_jan.sequences must be a non-empty list (e.g., ['A','B','ABB']).")

    coin_params_raw = cfg0.get("coin_params_degrees", cfg.get("coin_params_degrees"))
    if not isinstance(coin_params_raw, dict):
        raise ValueError("Missing coin_params_degrees mapping in dev.yaml.")

    coin_params: Dict[str, tuple[float, float, float]] = {}
    for k, v in coin_params_raw.items():
        if not (isinstance(v, (list, tuple)) and len(v) == 3):
            raise ValueError(f"coin_params_degrees[{k!r}] must be a 3-list [alpha,beta,gamma].")
        coin_params[str(k)] = (float(v[0]), float(v[1]), float(v[2]))

    # metrics window resolution: prefer stage0_jan.metrics, else top-level metrics, else defaults
    m0 = cfg0.get("metrics")
    m_top = cfg.get("metrics")
    m = (m0 if isinstance(m0, dict) else None) or (m_top if isinstance(m_top, dict) else {}) or {}

    T0 = int(m.get("T0", max(1, T // 2)))
    x0_loc = int(m.get("x0_loc", 2))

    return {
        "T": T,
        "phi": phi,
        "sequences": sequences,
        "coin_params_degrees": coin_params,
        "T0": T0,
        "x0_loc": x0_loc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dev.yaml", help="Path to YAML config (default: dev.yaml)")
    ap.add_argument("--tag", default="stage0_jan", help="Run tag suffix used in outputs/run_*_<tag>/")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    r = _resolve(cfg)

    paths = make_run_dir(tag=args.tag)
    run_dir = paths["run_dir"]
    fig_dir = paths["fig_dir"]

    # Required run artifacts
    write_yaml(run_dir / "config_used.yaml", cfg)
    write_yaml(run_dir / "resolved_config.yaml", r)
    write_manifest(run_dir, tag=args.tag, extra={"resolved": {k: r[k] for k in ["T", "phi", "T0", "x0_loc"]}})

    T = int(r["T"])
    phi = float(r["phi"])
    T0 = int(r["T0"])
    x0_loc = int(r["x0_loc"])
    sequences: List[str] = [str(s) for s in r["sequences"]]
    coin_params = r["coin_params_degrees"]

    per_seq: Dict[str, Dict[str, Any]] = {}
    per_seq_Pt: Dict[str, np.ndarray] = {}

    for seq in sequences:
        out = run_sequence_unitary(sequence=seq, T=T, phi=phi, coin_params=coin_params)
        x = out["x"]
        psi_t = out["psi_t"]

        P_t = prob_from_psi_t(psi_t)
        met = compute_metrics_from_P_t(P_t, x, T0=T0, x0_loc=x0_loc)

        per_seq[seq] = met
        per_seq_Pt[seq] = P_t

        np.savez(
            run_dir / f"metrics_{seq}.npz",
            x=x,
            P_t=P_t,
            x_mean_t=met["x_mean_t"],
            deltaP_t=met["deltaP_t"],
            sigma_t=met["sigma_t"],
            ipr_t=met["ipr_t"],
            v_fit=met["v_fit"],
            v_intercept=met["v_intercept"],
            w_loc=met["w_loc"],
            P0bar=met["P0bar"],
            T0=met["T0"],
            x0_loc=met["x0_loc"],
        )

    # JSON summary for quick inspection
    def summarize(seq: str) -> Dict[str, Any]:
        met = per_seq[seq]
        dP = np.asarray(met["deltaP_t"], dtype=float)
        mx = np.asarray(met["x_mean_t"], dtype=float)
        return {
            "v_fit": float(met["v_fit"]),
            "deltaP_final": float(dP[-1]),
            "deltaP_late_mean": float(np.mean(dP[T0:])),
            "x_mean_final": float(mx[-1]),
            "x_mean_late_mean": float(np.mean(mx[T0:])),
            "w_loc": float(met["w_loc"]),
            "P0bar": float(met["P0bar"]),
        }

    summary = {
        "params": {
            "T": T,
            "phi": phi,
            "sequences": sequences,
            "coin_params_degrees": coin_params,
            "metrics": {"T0": T0, "x0_loc": x0_loc},
        },
        "results": {seq: summarize(seq) for seq in sequences},
    }
    write_json(run_dir / "metrics_summary.json", summary)

    # Figures
    t = np.arange(T + 1)

    plt.figure()
    for seq in sequences:
        plt.plot(t, per_seq[seq]["x_mean_t"], label=seq)
    plt.xlabel("t")
    plt.ylabel("<x>(t)")
    plt.title("Stage 0 (Jan replication): mean position")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "x_mean_t.png", dpi=200)
    plt.close()

    plt.figure()
    for seq in sequences:
        plt.plot(t, per_seq[seq]["deltaP_t"], label=seq)
    plt.xlabel("t")
    plt.ylabel("deltaP(t) = P_R - P_L")
    plt.title("Stage 0 (Jan replication): deltaP time series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "deltaP_t.png", dpi=200)
    plt.close()

    # Final distribution overlay
    plt.figure()
    for seq in sequences:
        plt.plot(x, per_seq_Pt[seq][-1], label=seq)
    plt.xlabel("x")
    plt.ylabel("P(x, T)")
    plt.title("Stage 0 (Jan replication): final distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "P_final_overlay.png", dpi=200)
    plt.close()

    # Log
    write_text(run_dir / "logs.txt", f"Run dir: {run_dir}\n")
    print(f"Wrote Stage 0 run to: {run_dir}")
    print(f"Summary: {run_dir / 'metrics_summary.json'}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
