"""Pruebas: unidades clave y pipeline completo de punta a punta con datos sintéticos (CPU).

Ejecutar desde la raíz del repositorio:  python -m pytest -q
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lekkas_rep.data import assign_split  # noqa: E402
from lekkas_rep.evaluation import bootstrap_ci, delong_test, metrics_at  # noqa: E402
from lekkas_rep.models import CBAMResNet18_3D, DeepRadiomicFusion  # noqa: E402
from lekkas_rep.preprocess import crop_or_pad  # noqa: E402


def test_vendor_split():
    df = pd.DataFrame({"scanner": ["SIEMENS", "Philips", "TOSHIBA", "GE MEDICAL SYSTEMS"] * 5,
                       "y": [0, 1] * 10})
    out = assign_split(df, ["SIEMENS", "Philips"], ["TOSHIBA"], n_folds=2, seed=0)
    assert set(out.loc[out.scanner == "TOSHIBA", "split"]) == {"external"}
    assert set(out.loc[out.scanner.isin(["SIEMENS", "Philips"]), "split"]) == {"dev"}
    assert set(out.loc[out.scanner.str.startswith("GE"), "split"]) == {"excluded"}
    assert (out.loc[out.split == "dev", "fold"] >= 0).all()


def test_crop_or_pad_shapes():
    a = np.arange(5 * 6 * 7).reshape(5, 6, 7)
    assert crop_or_pad(a, (2, 3, 3), (8, 8, 8), -1).shape == (8, 8, 8)
    assert crop_or_pad(a, (0, 0, 0), (4, 4, 4), -1)[0, 0, 0] == -1


def test_models_forward():
    x = torch.randn(2, 1, 24, 32, 48)
    b = CBAMResNet18_3D(base=8)
    logits, e = b(x)
    assert logits.shape == (2, 2) and e.shape == (2, 64)
    f = DeepRadiomicFusion(b, n_radiomic=16, proj_dim=8, se_reduction=4, mode="frozen")
    lo, w = f(x, torch.randn(2, 16))
    assert lo.shape == (2, 2) and w.shape[1] == 64 + 8


def test_metrics_and_delong():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(60), np.zeros(140)].astype(int)
    p1 = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    p2 = np.clip(y * 0.2 + rng.normal(0.4, 0.2, 200), 0, 1)
    m = metrics_at(y, p1, 0.5)
    assert 0.5 < m["auc"] <= 1
    ci = bootstrap_ci(y, p1, 0.5, n_boot=50)
    assert ci["auc"][0] <= m["auc"] <= ci["auc"][1]
    t = delong_test(y, p1, p2)
    assert t["auc_1"] > t["auc_2"] and t["p_value"] < 0.05


@pytest.mark.slow
def test_end_to_end(tmp_path):
    syn = ROOT / "tests" / "_synthetic"
    shutil.rmtree(syn, ignore_errors=True)
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_synthetic.py"), str(syn)], check=True, cwd=ROOT)
    cfg = str(ROOT / "configs" / "smoke_test.yaml")
    steps = ["01_build_manifest.py", "02_preprocess.py", "03_extract_radiomics.py", "04_flow_a.py",
             "05_train_cnn.py", "06_train_fusion.py", "07_evaluate.py"]
    for s in steps:
        subprocess.run([sys.executable, str(ROOT / "scripts" / s), "--config", cfg], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "06_train_fusion.py"), "--config", cfg, "--mode", "end_to_end"],
                   check=True, cwd=ROOT)
    res = syn / "outputs" / "results"
    assert (res / "tabla_original_vs_replicacion.md").exists()
    assert (res / "metricas.json").exists()
    assert (res / "externa_roc.png").exists()
