"""Native-init R-parity test — regression guard for the 0.1.x bug where
py-mclustR's `hc()` used scipy Ward linkage instead of mclust's
model-based agglomerative VVV criterion (Fraley 1998 / Banfield-Raftery).

The bug was caught while porting TSCAN: on its `lpsdata` PCA-reduced fixture,
R Mclust produced clusters (44/57/30, loglik=-1727.97) while py-mclustR's
NATIVE init (no z_init supplied) produced (48/5/78, loglik=-1628.23) — a
degenerate local optimum.

The 251 existing R-parity tests under tests/_rparity/ all PASS because they
supply ``z_init`` from R hclust output. This test covers the case where the
user lets py-mclustR generate its own HC init — the path actually exercised
by callers like py-TSCAN, py-SC3, etc.

Reference dataset
-----------------
Saved as tests/_native_init/lpsdata_pca.csv along with the expected R Mclust
output (clusters, loglik). Regenerate with::

    python tests/_native_init/regen.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from mclust_py import Mclust, hc, hclass

DATA_DIR = Path(__file__).resolve().parent / "_native_init"


@pytest.fixture(scope="module")
def lpsdata_pca():
    csv = DATA_DIR / "lpsdata_pca.csv"
    if not csv.exists():
        pytest.skip(f"Fixture missing: {csv}. Regenerate with regen.py")
    return np.loadtxt(csv, delimiter=",", skiprows=1)


@pytest.fixture(scope="module")
def expected_clusterid():
    csv = DATA_DIR / "expected_clusterid.csv"
    if not csv.exists():
        pytest.skip(f"Fixture missing: {csv}")
    return np.loadtxt(csv, skiprows=1, dtype=int)


@pytest.fixture(scope="module")
def expected_hc_partition_g3():
    csv = DATA_DIR / "expected_hc_partition_g3.csv"
    if not csv.exists():
        pytest.skip(f"Fixture missing: {csv}")
    return np.loadtxt(csv, skiprows=1, dtype=int)


def test_native_hc_matches_R_VVV_SVD(lpsdata_pca, expected_hc_partition_g3):
    """py-mclustR's hc(use='SVD', model_name='VVV') at G=3 must match R's hc()."""
    res = hc(lpsdata_pca, use="SVD", model_name="VVV")
    py_part = hclass(res, G=3)
    ari = adjusted_rand_score(expected_hc_partition_g3, py_part)
    assert ari >= 0.99, f"HC partition ARI {ari:.4f} < 0.99 — regression in _hcvvv_linkage"


def test_native_Mclust_matches_R_lpsdata(lpsdata_pca, expected_clusterid):
    """Full Mclust pipeline with native HC init must match R Mclust on lpsdata."""
    res = Mclust(lpsdata_pca, G=[3], model_names=["VVV"])
    py_clu = res.classification.astype(int)
    if py_clu.min() == 0:
        py_clu += 1
    ari = adjusted_rand_score(expected_clusterid, py_clu)
    assert ari >= 0.99, (
        f"Mclust final ARI {ari:.4f} < 0.99 — regression in native init parity. "
        f"py sizes: {dict(zip(*np.unique(py_clu, return_counts=True)))}; "
        f"R sizes: {dict(zip(*np.unique(expected_clusterid, return_counts=True)))}"
    )


def test_native_Mclust_loglik_matches_R(lpsdata_pca):
    """Loglik must match R's -1727.969 to 0.1 absolute."""
    res = Mclust(lpsdata_pca, G=[3], model_names=["VVV"])
    expected_loglik = -1727.9685
    assert abs(res.loglik - expected_loglik) < 0.1, (
        f"loglik {res.loglik} drifted from R {expected_loglik}"
    )
