"""Quickstart example for mclust-py.

Run on three classic datasets to show the BIC scan + best-model output.
"""
from __future__ import annotations

import numpy as np
from sklearn import datasets

from mclust_py import Mclust, predict_mclust


def run(name: str, X: np.ndarray, G_range=range(1, 7)) -> None:
    fit = Mclust(X, G=G_range)
    print(f"== {name}: {X.shape[0]} × {X.shape[1]} ==")
    print(f"  best model: {fit.model_name}, G = {fit.G}")
    print(f"  loglik     = {fit.loglik:.4f}")
    print(f"  BIC        = {fit.bic:.4f}")
    print(f"  cluster sizes: {dict(zip(*np.unique(fit.classification, return_counts=True)))}")
    print()


if __name__ == "__main__":
    iris = datasets.load_iris()
    wine = datasets.load_wine()
    run("iris", iris.data)
    run("wine", wine.data[:, :4])  # truncate for speed
    rng = np.random.default_rng(0)
    blobs, _ = datasets.make_blobs(n_samples=400, centers=4, n_features=3, random_state=0, cluster_std=0.7)
    run("blobs (n=400, K=4)", blobs)
