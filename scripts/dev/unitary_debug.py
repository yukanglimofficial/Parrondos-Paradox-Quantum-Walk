from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dtqw.simulate import run_sequence_unitary


def main() -> None:
    T = 20
    phi = 0.0
    sequence = "A"
    coin_params = {"A": (0.0, 45.0, 0.0)}

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = Path("outputs") / f"run_{stamp}_unitary_debug"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    result = run_sequence_unitary(sequence=sequence, T=T, phi=phi, coin_params=coin_params)
    x = result["x"]
    psi_t = result["psi_t"]
    P_t = np.sum(np.abs(psi_t) ** 2, axis=1)  # shape (T+1, Npos)

    np.savez(run_dir / "metrics.npz", x=x, psi_t=psi_t, P_t=P_t)

    config = {
        "T": T,
        "phi": float(phi),
        "sequence": sequence,
        "coin_params_degrees": coin_params,
    }
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": "unitary_debug",
        "python": sys.version.replace("\n", " "),
        "packages": {"numpy": np.__version__, "matplotlib": plt.matplotlib.__version__},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    plt.figure()
    plt.bar(x, P_t[-1])
    plt.xlabel("x")
    plt.ylabel("P(x, T)")
    plt.title(f"Unitary DTQW (sequence={sequence}, T={T}, phi={phi})")
    plt.tight_layout()
    plt.savefig(fig_dir / "P_final.png", dpi=200)
    plt.close()

    (run_dir / "logs.txt").write_text(f"Run dir: {run_dir}\n", encoding="utf-8")

    print(f"Wrote debug run to: {run_dir}")
    print(f"Figure: {fig_dir / 'P_final.png'}")


if __name__ == "__main__":
    main()
