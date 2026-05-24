"""Initial-partition utilities for EM.

R's mclust uses model-based agglomerative hierarchical clustering (MBAHC)
to initialise EM. The mclust default is ``hcVVV`` — Fraley (1998)'s
agglomerative clustering with the **log-determinant** classification-
likelihood criterion for the VVV (full-covariance per cluster) Gaussian
model.

This module ports the algorithm:

1. :func:`hc` — dispatches by ``model_name``:
   - ``"VVV"`` (default): Fraley 1998 log-det agglomerative (matches R's
     ``hcVVV`` on partition; see :func:`_hcvvv_linkage`).
   - ``"EII"`` / ``"VII"`` / ``"EEE"`` / others: Ward linkage on
     pre-whitened data (closer to R's ``hcEII``).
   - Single-dimension data: scipy Ward linkage on whitened scalar.

2. :func:`hclass` — extract a flat partition of size ``G`` from the tree.

3. :func:`partition_to_z` — convert integer labels to a hard-assignment
   responsibility matrix suitable as EM seed.

History
-------
v0.1.x: Ward linkage in all cases. This approximates the VVV criterion but
diverges on real data (e.g., TSCAN's lpsdata fixture, where it produced a
degenerate 5-cell cluster vs R's 44/57/30 split). v0.2 ports Fraley 1998
directly.
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


def _hcvvv_alpha(X: np.ndarray) -> float:
    """Match R `hcVVV`'s default alpha exactly::

        alpha = max(traceW(X / sqrt(n*p)), eps)
              = sum((X/sqrt(n*p) - colMeans)^2)
              = (1/(n*p)) * sum((X - colMeans)^2)

    For SVD-whitened input, this scales the small-Bayes prior to the
    natural variance scale of the whitened data.
    """
    n, p = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    alpha = float(np.sum(Xc * Xc)) / (n * p)
    return max(alpha, np.finfo(np.float64).eps)


def _hcvvv_linkage(X: np.ndarray, alpha: float | None = None, beta: float = 1.0) -> np.ndarray:
    """Fraley 1998 / Banfield-Raftery agglomerative HC for the VVV criterion.

    Matches mclust's `hcVVV` Fortran implementation, including the dual-mode
    cost function from `vvvtij`:

        For n_k ≤ p (under-determined, trace-only):
            cost_k = n_k * log( beta * (tr(W_k) + alpha) / n_k )

        For n_k > p (full-rank, log-det blend):
            cost_k = n_k * log( exp(detlog) + beta*(tr(W_k)+alpha)/n_k )
            where detlog = slogdet(W_k / n_k).

    Greedy: at each step merge the pair (a, b) that minimises::

        Δ = cost_c − cost_a − cost_b

    with the Lance-Williams scatter recurrence::

        W_c = W_a + W_b + (n_a n_b / n_c) (μ_a − μ_b)(μ_a − μ_b)^T

    Singletons (n=1) have W=0 → cost_k = 1 * log(alpha) ⋅ (canceled by log-1 in
    Fortran), matching mclust's `ABLOG = log(alpha*beta)` initial term.

    Returns a scipy-compatible linkage matrix with monotone heights.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if p == 0:
        raise ValueError("X must have at least one column")
    if n < 2:
        return np.empty((0, 4), dtype=np.float64)

    if alpha is None:
        alpha = _hcvvv_alpha(X)
    log_ab = np.log(beta * alpha)        # = ABLOG in Fortran

    def _term(W: np.ndarray, n_k: int) -> float:
        """vvvtij at cluster size n_k. Returns n_k * vvvtij_inner."""
        trac = float(np.trace(W))
        if n_k <= p:
            # under-determined: trace criterion only
            return n_k * np.log(beta * (trac + alpha) / n_k)
        # full-rank: log-det blend, numerically-stable
        sign, detlog = np.linalg.slogdet(W / n_k)
        rhs = beta * (trac + alpha) / n_k
        if not np.isfinite(detlog) or detlog == -np.inf:
            return n_k * np.log(rhs)
        if detlog <= 0:
            return n_k * np.log(np.exp(detlog) + rhs)
        return n_k * (np.log(1.0 + np.exp(-detlog) * rhs) + detlog)

    # Active clusters
    mus = [X[i].copy() for i in range(n)]
    Ws = [np.zeros((p, p)) for _ in range(n)]
    ns = [1 for _ in range(n)]
    # Singleton init cost: n=1 ≤ p, W=0, tr(W)=0 → cost = 1 * log(beta*alpha/1) = log_ab.
    costs = [log_ab for _ in range(n)]
    cids = list(range(n))

    next_id = n
    Z_rows: list[list[float]] = []

    while len(mus) > 1:
        m = len(mus)
        best_delta = np.inf
        best = None
        for i in range(m):
            mu_a, W_a, n_a, cost_a = mus[i], Ws[i], ns[i], costs[i]
            for j in range(i + 1, m):
                mu_b, W_b, n_b, cost_b = mus[j], Ws[j], ns[j], costs[j]
                diff = mu_a - mu_b
                n_c = n_a + n_b
                W_c = W_a + W_b + (n_a * n_b / n_c) * np.outer(diff, diff)
                cost_c = _term(W_c, n_c)
                delta = cost_c - cost_a - cost_b
                if delta < best_delta:
                    best_delta = delta
                    mu_c = (n_a * mu_a + n_b * mu_b) / n_c
                    best = (i, j, mu_c, W_c, n_c, cost_c)

        i, j, mu_c, W_c, n_c, cost_c = best
        Z_rows.append([float(cids[i]), float(cids[j]), float(best_delta), float(n_c)])
        for idx in sorted([i, j], reverse=True):
            mus.pop(idx); Ws.pop(idx); ns.pop(idx); costs.pop(idx); cids.pop(idx)
        mus.append(mu_c); Ws.append(W_c); ns.append(n_c); costs.append(cost_c)
        cids.append(next_id)
        next_id += 1

    Z = np.array(Z_rows, dtype=np.float64)
    Z[:, 2] = np.maximum.accumulate(Z[:, 2])
    if Z[:, 2].min() < 0:
        Z[:, 2] = Z[:, 2] - Z[:, 2].min()
    return Z


def hc(
    X: np.ndarray,
    *,
    model_name: str = "VVV",
    use: str = "SVD",
    method: str = "ward",
    ridge: float | None = None,
) -> HCResult:
    """Compute a hierarchical clustering tree for EM initialisation.

    Parameters
    ----------
    X : (n, d) array
    model_name : {'EII','VII','EEE','VVV','E','V'}
        Selects the agglomeration criterion. ``'VVV'`` (mclust default)
        uses the Fraley 1998 log-determinant criterion via
        :func:`_hcvvv_linkage` — matches R's ``hcVVV`` partition on
        well-conditioned data. Other models fall back to Ward linkage
        on the pre-whitened data (closer to R's ``hcEII``).
    use : str
        Pre-processing mode (``"VARS","STD","SPH","PCS","PCR","SVD"``).
    method : str
        Scipy linkage method for non-VVV models (default ``"ward"``).
    ridge : float | None
        Deprecated alias for VVV singleton regulariser. The current VVV path
        uses mclust's data-adaptive ``alpha`` automatically; this param is
        kept only for API back-compat.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    Z_data = _whiten(X, use)

    if model_name.upper() == "VVV" and d > 1:
        # `ridge` from caller maps to `alpha`. None ⇒ R's hcVVV default alpha.
        L = _hcvvv_linkage(Z_data, alpha=ridge)
    else:
        L = linkage(Z_data, method=method)
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
