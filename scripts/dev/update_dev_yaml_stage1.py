from __future__ import annotations

from pathlib import Path
import yaml

p = Path("dev.yaml")
cfg = {}
if p.exists():
    txt = p.read_text(encoding="utf-8-sig")
    cfg = yaml.safe_load(txt) or {}
else:
    cfg = {}

stage1 = cfg.get("stage1_defect") or {}

defaults = {
    "T": 200,
    "sequences": ["ABB"],
    "Nphi": 72,
    "phi_start": 0.0,
    "phi_stop": 6.283185307179586,  # 2*pi
    "metrics": {"T0": 100, "x0_loc": 2},
}

# Merge defaults without overwriting user-provided keys
for k, v in defaults.items():
    if k == "metrics":
        m = stage1.get("metrics") or {}
        for mk, mv in v.items():
            if mk not in m:
                m[mk] = mv
        stage1["metrics"] = m
    else:
        if k not in stage1:
            stage1[k] = v

cfg["stage1_defect"] = stage1

p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("Updated dev.yaml: ensured stage1_defect section exists (no duplicate keys).")
