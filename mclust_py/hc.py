"""Initial-partition utilities for EM.

R's mclust uses model-based agglomerative hierarchical clustering (MBAHC)
to initialise EM. The full Fortran implementation (`hcvvv`, `hceee`, …)
optimises the classification log-likelihood under each model — porting
those routines bitwise is out of scope here. Instead this module provides:

1. :func:`hc` — Ward-linkage agglomeration on SVD-whitened data, which
   matches mclust's default ``use = "SVD"`` and reproduces the same
   *partitions* as ``hcVVV`` on well-separated test data. For exact
   parity with R, run R's ``hc()`` once and feed the resulting
   ``classification`` (or ``z`` matrix) directly to :func:`mclust_py.me`
   / :func:`mclust_py.Mclust` via the ``z_init`` / ``initialization``
   argument.
2. :func:`hclass` — extract a flat partition of size ``G`` from the tree.
3. :func:`partition_to_z` — convert integer labels to a hard-assignment
   responsibility matrix suitable as EM seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


@dataclass
class HCResult:
    linkage_matrix: np.ndarray  # scipy linkage Z (n-1, 4)
    initial_partition: np.ndarray
    n: int
    d: int
    model_name: str
    use: str


def _whiten(X: np.ndarray, use: str) -> np.ndarray:
    """Pre-process the data exactly as `mclust::hc(use=...)` does."""
    use = use.upper()
    if use == "VARS":
        return X.copy()
    Xc = X - X.mean(axis=0, keepdims=True)
    if use == "STD":
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        return (X - X.mean(axis=0)) / sd
    if use == "PCS":
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt.T
    if use == "PCR":
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Z = (X - X.mean(axis=0)) / sd
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        return Z @ Vt.T
    if use == "SPH":
        # Whitening by inverse SVD of centred Σ
        n, p = Xc.shape
        Sigma = (Xc.T @ Xc) / n
        U, S, Vt = np.linalg.svd(Sigma, full_matrices=False)
        # 1/sqrt(d) on the diagonal — guard against zero singular values
        s_inv = np.where(S > 0, 1.0 / np.sqrt(S), 0.0)
        return Xc @ Vt.T @ np.diag(s_inv)
    if use == "SVD":
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Z = (X - X.mean(axis=0)) / sd
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        s_inv = np.where(S > 0, 1.0 / np.sqrt(S), 0.0)
        return Z @ Vt.T @ np.diag(s_inv)
    raise ValueError(f"unknown 'use' option: {use!r}")


def hc(
    X: np.ndarray,
    *,
    model_name: str = "VVV",
    use: str = "SVD",
    method: str = "ward",
) -> HCResult:
    """Compute a hierarchical clustering tree for EM initialisation.

    Parameters
    ----------
    X : (n, d) array
    model_name : {'EII','VII','EEE','VVV','E','V'}
        Mostly cosmetic — we surface the same letter codes as mclust so
        downstream code can detect the init style. ``'VVV'`` (the mclust
        default) corresponds to a determinant criterion that Ward linkage
        on SVD-whitened data approximates well.
    use : str
        Pre-processing mode (``"VARS","STD","SPH","PCS","PCR","SVD"``).
    method : str
        Underlying scipy linkage method. Ward is the closest fit to the
        Banfield–Raftery model-based criterion in low dimensions.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    Z = _whiten(X, use)
    L = linkage(Z, method=method)
    return HCResult(
        linkage_matrix=L,
        initial_partition=np.arange(n, dtype=int),
        n=n,
        d=d,
        model_name=model_name,
        use=use,
    )


def hclass(hc_result: HCResult, G: int | list[int]) -> np.ndarray:
    """Cut the tree into ``G`` clusters (mirrors R's `hclass`).

    For a single ``G`` returns a 1-D array of length ``n``; for multiple
    Gs returns a 2-D matrix with one column per requested ``G``.
    """
    if isinstance(G, (int, np.integer)):
        return fcluster(hc_result.linkage_matrix, t=int(G), criterion="maxclust")
    cols = [fcluster(hc_result.linkage_matrix, t=int(g), criterion="maxclust") for g in G]
    out = np.column_stack(cols)
    return out


def partition_to_z(partition: np.ndarray, G: int | None = None) -> np.ndarray:
    """One-hot encoding of a hard label vector — EM warm-start matrix."""
    p = np.asarray(partition).ravel()
    uniq = np.unique(p)
    # mclust uses 1-based consecutive labels; replicate that convention.
    label_index = {v: i for i, v in enumerate(uniq)}
    n = len(p)
    G = len(uniq) if G is None else int(G)
    z = np.zeros((n, G), dtype=np.float64)
    for i, v in enumerate(p):
        idx = label_index[v]
        if idx < G:
            z[i, idx] = 1.0
    return z
