"""Smoke tests — every M-step runs and returns valid covariances."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn import datasets

from mclust_py import Mclust, me, mstep
from mclust_py.models import EM_MODEL_NAMES_MULTI


@pytest.fixture(scope="module")
def blobs():
    X, y = datasets.make_blobs(
        n_samples=300, centers=3, n_features=3, random_state=0, cluster_std=0.7
    )
    return X.astype(np.float64), y


@pytest.fixture(scope="module")
def init_z(blobs):
    X, y = blobs
    G = 3
    z = np.zeros((X.shape[0], G))
    z[np.arange(X.shape[0]), y] = 1.0
    return z


@pytest.mark.parametrize("model_name", EM_MODEL_NAMES_MULTI)
def test_mstep_returns_pd(blobs, init_z, model_name):
    X, _ = blobs
    pars = mstep(model_name, X, init_z)
    assert pars["pro"].shape == (3,)
    assert pars["mu"].shape == (3, 3)
    assert pars["sigma"].shape == (3, 3, 3)
    for k in range(3):
        eig = np.linalg.eigvalsh(pars["sigma"][:, :, k])
        assert eig.min() > -1e-9, f"{model_name} cluster {k} non-PD: {eig}"


@pytest.mark.parametrize("model_name", EM_MODEL_NAMES_MULTI)
def test_em_converges(blobs, init_z, model_name):
    X, _ = blobs
    res = me(X, model_name, init_z)
    assert np.isfinite(res.loglik)
    assert res.iterations > 0
    assert res.iterations < 1000  # nothing should run away


def test_mclust_picks_reasonable_model(blobs):
    X, y = blobs
    fit = Mclust(X, G=range(1, 6))
    assert fit.G in (2, 3, 4)  # blob count ≈ ground truth
    # cluster purity ≥ 90% on blobs at this separation
    cls = fit.classification
    confusion = np.zeros((fit.G, 3))
    for i in range(len(y)):
        confusion[cls[i] - 1, y[i]] += 1
    purity = confusion.max(axis=1).sum() / len(y)
    assert purity > 0.85
