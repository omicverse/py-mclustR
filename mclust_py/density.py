"""Per-component log-density for the 14 Gaussian parameterizations.

The E-step needs ``log φ(x_i | μ_k, Σ_k)`` — a thin wrapper over the
Cholesky factor that's robust to (near-)singular covariances. The
parameter shapes mirror those returned by :mod:`mclust_py.models`.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

LOG2PI = float(np.log(2.0 * np.pi))


def _safe_chol(sigma: np.ndarray) -> np.ndarray | None:
    """Return upper-triangular Cholesky factor or ``None`` if not PD."""
    try:
        # numpy returns lower; we want a stable factorization for solves.
        L = np.linalg.cholesky(sigma)
        return L
    except np.linalg.LinAlgError:
        return None


def _logdet_from_chol(L: np.ndarray) -> float:
    return 2.0 * float(np.sum(np.log(np.diag(L))))


def log_dens_component(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Log multivariate normal density of each row of ``X`` under (μ, Σ)."""
    n, d = X.shape
    L = _safe_chol(sigma)
    if L is None:
        # Add a tiny ridge — matches Fortran's numerical safety net so the EM
        # doesn't crash on ill-conditioned covariances during early iterations.
        ridge = 1e-12 * float(np.trace(sigma)) / d
        L = np.linalg.cholesky(sigma + ridge * np.eye(d))
    diff = X - mu  # (n, d)
    z = np.linalg.solve(L, diff.T)  # (d, n)
    maha = np.sum(z * z, axis=0)
    logdet = _logdet_from_chol(L)
    return -0.5 * (d * LOG2PI + logdet + maha)


def log_dens_all(
    X: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Log densities for every component.

    Parameters
    ----------
    X : (n, d) array
    mean : (d, G) array  — column ``k`` is the mean of component ``k``
    sigma : (d, d, G) array  — slice ``[:,:,k]`` is Σ_k

    Returns
    -------
    (n, G) array of ``log φ(x_i | μ_k, Σ_k)``.
    """
    n, d = X.shape
    G = mean.shape[1]
    out = np.empty((n, G), dtype=np.float64)
    for k in range(G):
        out[:, k] = log_dens_component(X, mean[:, k], sigma[:, :, k])
    return out
