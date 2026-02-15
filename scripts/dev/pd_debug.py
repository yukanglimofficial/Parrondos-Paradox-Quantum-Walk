from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dtqw.noise import run_sequence_pd_trajectories


def main() -> None:
    cfg = yaml.safe_load(Path("dev.yaml").read_text(encoding="utf-8"))

    T = int(cfg["T"])
    N_traj = int(cfg["N_traj"])

    p = float(cfg["p"])
    phi = float(cfg["phi"])
    sequence = str(cfg["sequence"])
    seed = int(cfg.get("seed", 0))

    coin_params_raw = cfg["coin_params_degrees"]
    coin_params = {k: tuple(float(x) for x in v) for k, v in coin_params_raw.items()}

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = Path("outputs") / f"run_{stamp}_pd_debug"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    result = run_sequence_pd_trajectories(
        sequence=sequence,
        T=T,
        phi=phi,
        p=p,
        coin_params=coin_params,
        N_traj=N_traj,
        seed=seed,
    )

    x = result["x"]
    P_t = result["P_t"]

    np.savez(run_dir / "metrics.npz", x=x, P_t=P_t)

    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": "pd_debug",
        "python": sys.version.replace("\n", " "),
        "packages": {"numpy": np.__version__, "matplotlib": plt.matplotlib.__version__},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    plt.figure()
    plt.bar(x, P_t[-1])
    plt.xlabel("x")
    plt.ylabel("P_mean(x, T)")
    plt.title(f"PD DTQW mean distribution (seq={sequence}, T={T}, phi={phi}, p={p}, N_traj={N_traj})")
    plt.tight_layout()
    plt.savefig(fig_dir / "P_final_mean.png", dpi=200)
    plt.close()

    (run_dir / "logs.txt").write_text(f"Run dir: {run_dir}\n", encoding="utf-8")

    print(f"Wrote PD debug run to: {run_dir}")
    print(f"Figure: {fig_dir / 'P_final_mean.png'}")


if __name__ == "__main__":
    main()
