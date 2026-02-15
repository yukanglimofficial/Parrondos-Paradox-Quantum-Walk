from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import importlib.metadata
import yaml


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stamp_local() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    # utf-8-sig tolerates BOM if a file was created with Windows PowerShell UTF8
    return yaml.safe_load(p.read_text(encoding="utf-8-sig"))


def write_yaml(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.write_text(text, encoding="utf-8")


def package_versions(dist_names: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in dist_names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def make_run_dir(tag: str, base: str = "outputs") -> Dict[str, Path]:
    run_dir = Path(base) / f"run_{_stamp_local()}_{tag}"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"run_dir": run_dir, "fig_dir": fig_dir}


def write_manifest(run_dir: Path, tag: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "created_utc": _utc_now_iso(),
        "tag": tag,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": package_versions(["numpy", "matplotlib", "PyYAML", "tqdm", "pytest"]),
    }
    if extra:
        manifest.update(extra)
    write_json(run_dir / "manifest.json", manifest)
    return manifest
