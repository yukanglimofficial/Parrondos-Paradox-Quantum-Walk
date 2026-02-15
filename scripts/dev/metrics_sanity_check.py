from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dtqw.metrics import compute_metrics_from_P_t, prob_from_psi_t
from dtqw.noise import run_sequence_pd_trajectories
from dtqw.simulate import run_sequence_unitary


def main() -> None:
    # Use utf-8-sig to tolerate BOM if dev.yaml was created with Windows PowerShell UTF8
    cfg = yaml.safe_load(Path("dev.yaml").read_text(encoding="utf-8-sig"))

    T = int(cfg.get("T", 50))
    N_traj = int(cfg.get("N_traj", 100))
    phi = float(cfg.get("phi", 0.0))
    p = float(cfg.get("p", 0.0))
    seed = int(cfg.get("seed", 0))
    sequence = str(cfg.get("sequence", "A"))

    coin_params_raw = cfg.get("coin_params_degrees", {"A": [0.0, 45.0, 0.0]})
    coin_params = {k: tuple(float(x) for x in v) for k, v in coin_params_raw.items()}

    # Late-window start + localization window (configurable)
    metrics_cfg = cfg.get("metrics", {}) or {}
    T0 = int(metrics_cfg.get("T0", cfg.get("T0", max(1, T // 2))))
    x0_loc = int(metrics_cfg.get("x0_loc", cfg.get("x0_loc", 2)))

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = Path("outputs") / f"run_{stamp}_metrics_sanity"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Unitary run ----
    out_u = run_sequence_unitary(sequence=sequence, T=T, phi=phi, coin_params=coin_params)
    x = out_u["x"]
    psi_t = out_u["psi_t"]
    P_t_u = prob_from_psi_t(psi_t)
    met_u = compute_metrics_from_P_t(P_t_u, x, T0=T0, x0_loc=x0_loc)

    # ---- PD run (if p>0 or even if p==0, still useful) ----
    out_pd = run_sequence_pd_trajectories(
        sequence=sequence,
        T=T,
        phi=phi,
        p=p,
        coin_params=coin_params,
        N_traj=N_traj,
        seed=seed,
    )
    P_t_pd = out_pd["P_t"]
    met_pd = compute_metrics_from_P_t(P_t_pd, x, T0=T0, x0_loc=x0_loc)

    # Save arrays
    np.savez(run_dir / "metrics_unitary.npz", x=x, P_t=P_t_u, **met_u)
    np.savez(run_dir / "metrics_pd.npz", x=x, P_t=P_t_pd, **met_pd)

    # Small scalar summary
    def summarize(m: dict) -> dict:
        dP_late = float(np.mean(np.asarray(m["deltaP_t"])[T0:]))
        mx_late = float(np.mean(np.asarray(m["x_mean_t"])[T0:]))
        sig_final = float(np.asarray(m["sigma_t"])[-1])
        return {
            "v_fit": float(m["v_fit"]),
            "deltaP_late_mean": dP_late,
            "x_mean_late_mean": mx_late,
            "sigma_final": sig_final,
            "w_loc": float(m["w_loc"]),
            "P0bar": float(m["P0bar"]),
            "T0": int(m["T0"]),
            "x0_loc": int(m["x0_loc"]),
        }

    summary = {
        "unitary": summarize(met_u),
        "pd": summarize(met_pd),
        "params": {"T": T, "phi": phi, "p": p, "N_traj": N_traj, "seed": seed, "sequence": sequence},
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save config + manifest
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": "metrics_sanity",
        "python": sys.version.replace("\n", " "),
        "packages": {"numpy": np.__version__, "matplotlib": plt.matplotlib.__version__},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Figures
    t = np.arange(T + 1)

    plt.figure()
    plt.plot(t, met_u["x_mean_t"], label="unitary")
    plt.plot(t, met_pd["x_mean_t"], label="pd_mean")
    plt.xlabel("t")
    plt.ylabel("<x>(t)")
    plt.title("Mean position time series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "x_mean_t.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, met_u["deltaP_t"], label="unitary")
    plt.plot(t, met_pd["deltaP_t"], label="pd_mean")
    plt.xlabel("t")
    plt.ylabel("deltaP(t)")
    plt.title("DeltaP time series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "deltaP_t.png", dpi=200)
    plt.close()

    (run_dir / "logs.txt").write_text(f"Run dir: {run_dir}\n", encoding="utf-8")

    print("Wrote metrics sanity run to:", run_dir)
    print("Summary:", run_dir / "metrics_summary.json")
    print("Figures:", fig_dir)


if __name__ == "__main__":
    main()
