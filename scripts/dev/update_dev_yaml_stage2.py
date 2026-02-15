from __future__ import annotations

from pathlib import Path
import yaml

p = Path("dev.yaml")
cfg = yaml.safe_load(p.read_text(encoding="utf-8-sig")) if p.exists() else {}
cfg = cfg or {}

stage2 = cfg.get("stage2_pd_atlas") or {}

defaults = {
    # Start small (dev) then scale later per bootstrap_plan appendix
    "T": 100,
    "N_traj": 100,
    "seed": 0,
    "sequences": ["A", "B", "ABB"],
    "Nphi": 24,
    "Np": 9,
    "phi_start": 0.0,
    "phi_stop": 6.283185307179586,  # 2*pi
    "p_start": 0.0,
    "p_stop": 1.0,
    "metrics": {"T0": 50, "x0_loc": 3},
    # Directed-transport threshold (plan uses w_thr_primary=0.10 in MRC; use same here)
    "w_thr": 0.10,
}

# Merge defaults without overwriting user-provided keys
for k, v in defaults.items():
    if k == "metrics":
        m = stage2.get("metrics") or {}
        for mk, mv in v.items():
            if mk not in m:
                m[mk] = mv
        stage2["metrics"] = m
    else:
        if k not in stage2:
            stage2[k] = v

cfg["stage2_pd_atlas"] = stage2

p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("Updated dev.yaml: ensured stage2_pd_atlas section exists (no duplicate keys).")
