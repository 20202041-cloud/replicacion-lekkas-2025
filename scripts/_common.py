"""Argumentos comunes para los scripts del pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lekkas_rep.utils import get_logger, load_config, seed_everything  # noqa: E402


def parse(description: str, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="procesar solo N casos (prueba piloto)")
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["data"]["seed"])
    return args, cfg, get_logger()


def out(cfg, *parts) -> Path:
    p = Path(cfg["paths"]["outputs_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
