"""M-step for the 14 Gaussian mixture parameterizations of mclust.

The decomposition is  Σ_k = λ_k · D_k · A_k · D_k^T  (Banfield & Raftery 1993).
Each model fixes a subset of (λ, D, A) to be common across components. The
14 letter codes (E = equal, V = variable, I = identity) correspond to
constraints on (volume, shape, orientation):

==========  ========  ========  ============
modelName   λ         A         D
==========  ========  ========  ============
EII         equal     I         —
VII         var       I         —
EEI         equal     equal     diag (axes)
VEI         var       equal     diag
EVI         equal     var       diag
VVI         var       var       diag
EEE         equal     equal     equal (full)
VEE         var       equal     equal
EVE         equal     var       equal
VVE         var       var       equal
EEV         equal     equal     var
VEV         var       equal     var
EVV         equal     var       var
VVV         var       var       var
==========  ========  ========  ============

Closed-form models (EII, VII, EEI, EVI, VVI, EEE, VVV, EEV, EVV) follow
Celeux & Govaert (1995) directly. Iterative models (VEI, VEV, EVE, VEE,
VVE) use the Flury-style alternating updates that the CRAN Fortran code
implements.

Every M-step returns:
    pro    : (G,)       mixing proportions
    mu     : (d, G)     component means
    sigma  : (d, d, G)  full covariance matrices
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------- helpers


def _weights(z: np.ndarray) -> tuple[np.ndarray, float]:
    """Column sums of z (== n_k) and total (== n). Standard EM bookkeeping."""
    n_k = z.sum(axis=0)
    n = z.sum()
    return n_k, n


def _means(X: np.ndarray, z: np.ndarray, n_k: np.ndarray) -> np.ndarray:
    """μ_k = Σ_i z_ik x_i / Σ_i z_ik  →  shape (d, G)."""
    # X: (n, d), z: (n, G) → mu: (G, d)
    mu = (z.T @ X) / np.maximum(n_k[:, None], np.finfo(float).tiny)
    return mu.T  # (d, G)


def _scatter_per_component(
    X: np.ndarray, z: np.ndarray, mu: np.ndarray
) -> np.ndarray:
    """W_k = Σ_i z_ik (x_i - μ_k)(x_i - μ_k)^T  →  (d, d, G)."""
    n, d = X.shape
    G = mu.shape[1]
    W = np.empty((d, d, G), dtype=np.float64)
    for k in range(G):
        diff = X - mu[:, k]
        W[:, :, k] = (diff * z[:, k][:, None]).T @ diff
    return W


def _proportions(n_k: np.ndarray, n: float, equal_pro: bool) -> np.ndarray:
    if equal_pro:
        G = len(n_k)
        return np.full(G, 1.0 / G, dtype=np.float64)
    return n_k / n


def _broadcast_sigma(sigma_one: np.ndarray, G: int) -> np.ndarray:
    """Stack a single (d,d) matrix G times along axis 2."""
    out = np.repeat(sigma_one[:, :, None], G, axis=2)
    return out


def _stack_diag(diag_vec: np.ndarray) -> np.ndarray:
    """Build a (d, d) diagonal matrix from a 1-D array."""
    return np.diag(diag_vec)


# ---------------------------------------------------------------- diagonal & spherical


def mstep_EII(X, z, *, equal_pro: bool = False):
    """λI common volume, identity shape — single scalar variance."""
    n, d = X.shape
    n_k, n = _weights(z)
    mu = _means(X, z, n_k)
    W = _scatter_per_component(X, z, mu)
    trW = float(np.einsum("iik->", W))
    sigmasq = trW / (n * d)
    sigma = _broadcast_sigma(sigmasq * np.eye(d), z.shape[1])
    return {"pro": _proportions(n_k, n, equal_pro), "mu": mu, "sigma": sigma}


def mstep_VII(X, z, *, equal_pro: bool = False):
    """λ_k I — spherical, per-component volume."""
    n, d = X.shape
    n_k, n = _weights(z)
    mu = _means(X, z, n_k)
    W = _scatter_per_component(X, z, mu)
    G = mu.shape[1]
    sigma = np.empty((d, d, G))
    for k in range(G):
        s_k = np.trace(W[:, :, k]) / (n_k[k] * d)
        sigma[:, :, k] = s_k * np.eye(d)
    return {"pro": _proportions(n_k, n, equal_pro), "mu": mu, "sigma": sigma}


def mstep_EEI(X, z, *, equal_pro: bool = False):
    """λA, A diagonal — common diagonal covariance."""
    n, d = X.shape
    n_k, n = _weights(z)
    mu = _means(X, z, n_k)
    W = _scatter_per_component(X, z, mu)
    diag_pool = np.einsum("iik->i", W) / n  # pooled diagonal of W
    sigma_one = np.diag(diag_pool)
    sigma = _broadcast_sigma(sigma_one, z.shape[1])
    return {"pro": _proportions(n_k, n, equal_pro), "mu": mu, "sigma": sigma}


def mstep_VVI(X, z, *, equal_pro: bool = False):
    """λ_k A_k diag — diagonal per component."""
    n, d = X.shape
    n_k, n = _weights(z)
    mu = _means(X, z, n_k)
    G = mu.shape[1]
    sigma = np.zeros((d, d, G))
    for k in range(G):
        diff = X - mu[:, k]
        diag_k = (z[:, k][:, None] * diff * diff).sum(axis=0) / max(n_k[k], np.finfo(float).tiny)
        sigma[:, :, k] = np.diag(diag_k)
    return {"pro": _proportions(n_k, n, equal_pro), "mu": mu, "sigma": sigma}


def mstep_EVI(X, z, *, equal_pro: bool = False):
    """λ A_k diag, |A_k| = 1 — common volume, varying diagonal shape."""
    n, d = X.shape
    n_k, n = _weights(z)
    mu = _means(X, z, n_k)
    G = mu.shape[1]
    # diag of each W_k (un-normalised within-component diagonal scatter)
    diag_W = np.empty((d, G))
    for k in range(G):
        diff = X - mu[:, k]
        diag_W[:, k] = (z[:, k][:, None] * diff * diff).sum(axis=0)
    # A_k = diag(diag_W[:,k]) / |·|^(1/d)
    A = np.empty_like(diag_W)
    for k in range(G):
        det_k = np.prod(diag_W[:, k])
        scale = det_k ** (1.0 / d)
        A[:, k] = diag_W[:, k] / max(scale, np.finfo(float).tiny)
    # λ from pooling
    lam = np.sum(diag_W / np.maximum(A, np.finfo(float).tiny)) / (n * d)
    sigma = np.zeros((d, d, G))
    for k in range(G):
        sigma[:, :, k] = lam * np.diag(A[:, k])
    return {"pro": _proportions(n_k, n, equal_pro), "mu": mu, "sigma": sigma}


def mstep_VEI(
    X, z, *, equal_pro: bool = False, max_iter: int = 100, tol: float = 1e-8,
):
    """λ_k A diag, |A| = 1 — varying volume, common diagonal shape.

    Closed-form alternating updates (Celeux & Govaert 1995, eq. 22).
    """
    n, d = X.shape
    n_k, n = _weights(z)
    mu = _means(X, z, n_k)
    G = mu.shape[1]
    diag_W = np.empty((d, G))
    for k in range(G):
        diff = X - mu[:, k]
        diag_W[:, k] = (z[:, k][:, None] * diff * diff).sum(axis=0)
    # init A as the unit-determinant pooled diagonal
    pool = diag_W.sum(axis=1)
    A = pool / max(np.prod(pool) ** (1.0 / d), np.finfo(float).tiny)
    for _ in range(max_iter):
        # λ_k = (1/(n_k d)) Σ_j W_kjj / A_j
        lam = (diag_W / A[:, None]).sum(axis=0) / (n_k * d)
        # A_j ∝ Σ_k W_kjj / λ_k, normalised to det 1
        new_A = (diag_W / np.maximum(lam[None, :], np.finfo(float).tiny)).sum(axis=1)
        det = np.prod(new_A)
        new_A = new_A / max(det ** (1.0 / d), np.finfo(float).tiny)
        if float(np.max(np.abs(new_A - A))) < tol:
            A = new_A
            break
        A = new_A
    lam = (diag_W / A[:, None]).sum(axis=0) / (n_k * d)
    sigma = np.zeros((d, d, G))
    for k in range(G):
        sigma[:, :, k] = lam[k] * np.diag(A)
    return {"pro": _proportions(n_k, n, equal_pro), "mu": mu, "sigma": sigma}


# ---------------------------------------------------------------- full / shared


def mstep_EEE(X, z, *, equal_pro: bool = False):
    """Σ — common full covariance."""
    n, d = X.shape
    n_k, n = _weights(z)
    mu = _means(X, z, n_k)
    W = _scatter_per_component(X, z, mu)
    sigma_one = W.sum(axis=2) / n
    sigma = _broadcast_sigma(sigma_one, z.shape[1])
    return {"pro": _proportions(n_k, n, equal_pro), "mu": mu, "sigma": sigma}


def mstep_VVV(X, z, *, equal_pro: bool = False):
    """Σ_k — fully general per-component covariance."""
    n, d = X.shape
    n_k, n = _weights(z)
    mu = _means(X, z, n_k)
    W = _scatter_per_component(X, z, mu)
    G = mu.shape[1]
    sigma = np.empty_like(W)
    for k in range(G):
        sigma[:, :, k] = W[:, :, k] / max(n_k[k], np.finfo(float).tiny)
    return {"pro": _proportions(n_k, n, equal_pro), "mu": mu, "sigma": sigma}


# ---------------------------------------------------------------- eigen / orientation models


def _eig_descending(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomp of a symmetric matrix sorted by eigenvalue (descending)."""
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def mstep_EEV(X, z, *, equal_pro: bool = False):
    """Σ_k = λ A D_k^T … D_k — common λ and shape A, varying orientation."""
    n, d = X.shape
    n_k, n = _weights(z)
    mu = _means(X, z, n_k)
    W = _scatter_per_component(X, z, mu)
    G = mu.shape[1]
    D = np.empty((d, d, G))
    diag_eig = np.empty((d, G))  # eigenvalues of W_k (descending)
    for k in range(G):
        vals, vecs = _eig_descending(W[:, :, k])
        D[:, :, k] = vecs
        diag_eig[:, k] = vals
    # A: pooled normalised eigenvalues, det 1
    pooled = diag_eig.sum(axis=1)
    detA = np.prod(pooled) ** (1.0 / d)
    A = pooled / max(detA, np.finfo(float).tiny)
    lam = float(np.sum(diag_eig / A[:, None])) / (n * d)
    sigma = np.empty_like(W)
    for k in range(G):
        sigma[:, :, k] = lam * D[:, :, k] @ np.diag(A) @ D[:, :, k].T
    return {"pro": _proportions(n_k, n, equal_pro), "mu": mu, "sigma": sigma}


def mstep_VEV(
    X, z, *, equal_pro: bool = False, max_iter: int = 100, tol: float = 1e-8,
):
    """Σ_k = λ_k D_k A D_k^T — varying volume + orientation, common shape A."""
    n, d = X.shape
    n_k, n = _weights(z)
    mu = _means(X, z, n_k)
    W = _scatter_per_component(X, z, mu)
    G = mu.shape[1]
    D = np.empty((d, d, G))
    diag_eig = np.empty((d, G))
    for k in range(G):
        vals, vecs = _eig_descending(W[:, :, k])
        D[:, :, k] = vecs
        diag_eig[:, k] = vals
    # init A (det 1) from pooled eigenvalues
    pool = diag_eig.sum(axis=1)
    A = pool / max(np.prod(pool) ** (1.0 / d), np.finfo(float).tiny)
    for _ in range(max_iter):
        lam = (diag_eig / A[:, None]).sum(axis=0) / (n_k * d)
        new_A = (diag_eig / np.maximum(lam[None, :], np.finfo(float).tiny)).sum(axis=1)
        det = np.prod(new_A)
        new_A = new_A / max(det ** (1.0 / d), np.finfo(float).tiny)
        if float(np.max(np.abs(new_A - A))) < tol:
            A = new_A
            break
        A = new_A
    lam = (diag_eig / A[:, None]).sum(axis=0) / (n_k * d)
    sigma = np.empty_like(W)
    for k in range(G):
        sigma[:, :, k] = lam[k] * D[:, :, k] @ np.diag(A) @ D[:, :, k].T
    return {"pro": _proportions(n_k, n, equal_pro), "mu": mu, "sigma": sigma}


def mstep_EVV(X, z, *, equal_pro: bool = False):
    """Σ_k = λ D_k A_k D_k^T, |A_k| = 1 — common volume, free shape & orientation."""
    n, d = X.shape
    n_k, n = _weights(z)
    mu = _means(X, z, n_k)
    W = _scatter_per_component(X, z, mu)
    G = mu.shape[1]
    sigma = np.empty_like(W)
    detW = np.array([np.linalg.det(W[:, :, k]) for k in range(G)])
    # Celeux–Govaert (1995): Σ_k = λ · W_k / |W_k|^{1/d}, with λ = Σ_k |W_k|^{1/d} / n.
    lam = float(np.sum(detW ** (1.0 / d))) / n
    for k in range(G):
        sigma[:, :, k] = lam * W[:, :, k] / max(detW[k] ** (1.0 / d), np.finfo(float).tiny)
    return {"pro": _proportions(n_k, n, equal_pro), "mu": mu, "sigma": sigma}


# ---------------------------------------------------------------- common-orientation iterative models
#
# For VEE / EVE / VVE the orientation D is shared, while the shape and/or
# volume vary. Closed form does not exist; we use the Flury-style
# alternating maximisation that the Fortran backend uses.


def _flury_common_orientation(
    W: np.ndarray,
    n_k: np.ndarray,
    *,
    shape_per_component: bool,
    volume_per_component: bool,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | float]:
    """Solve for (D, A, λ) when D is common across components.

    Two phases:
      1. **Joint-eigen warm start.** Use the eigenvectors of the pooled
         within-cluster scatter as the initial D — this is exact when all
         W_k are simultaneously diagonalisable and a strong starting point
         otherwise.
      2. **Stiefel gradient descent.** Refine D on the orthogonal manifold
         by parameterising D ← D · expm(α · skew_step) where ``skew_step``
         is the skew-symmetric part of the gradient. This is provably
         monotone (objective decreases each step) — no oscillation.

    Closed-form A_k and λ_k updates follow Celeux–Govaert (1995) given a
    fixed D.
    """
    d, _, G = W.shape
    eps = np.finfo(float).eps

    # ---- 1. Warm start from the pooled-scatter eigendecomposition ----
    pooled = W.sum(axis=2)
    _, D = _eig_descending(pooled)  # (d, d), columns are eigenvectors

    if shape_per_component:
        A = np.ones((d, G), dtype=np.float64)
    else:
        A = np.ones(d, dtype=np.float64)
    if volume_per_component:
        lam = np.ones(G, dtype=np.float64)
    else:
        lam = 1.0

    def _refresh_AL():
        nonlocal A, lam
        diag_W = np.empty((d, G))
        for k in range(G):
            diag_W[:, k] = np.diag(D.T @ W[:, :, k] @ D)
        diag_W = np.maximum(diag_W, eps)  # safety net for collapsed clusters
        if shape_per_component and volume_per_component:  # VVE
            for k in range(G):
                shape_k = diag_W[:, k] / max(n_k[k], eps)
                scale_k = float(np.prod(shape_k) ** (1.0 / d))
                A[:, k] = shape_k / max(scale_k, eps)
                lam[k] = scale_k
        elif shape_per_component and not volume_per_component:  # EVE
            for k in range(G):
                A[:, k] = diag_W[:, k] / max(np.prod(diag_W[:, k]) ** (1.0 / d), eps)
            lam = float(np.sum(diag_W / A)) / (float(np.sum(n_k)) * d)
        elif (not shape_per_component) and volume_per_component:  # VEE
            for _ in range(50):
                lam = (diag_W / A[:, None]).sum(axis=0) / (n_k * d)
                newA = (diag_W / np.maximum(lam[None, :], eps)).sum(axis=1)
                newA = newA / max(np.prod(newA) ** (1.0 / d), eps)
                if float(np.max(np.abs(newA - A))) < 1e-12:
                    A = newA
                    break
                A = newA
            lam = (diag_W / A[:, None]).sum(axis=0) / (n_k * d)
        return diag_W

    def _objective(diag_W):
        if shape_per_component:
            inv_A = 1.0 / np.maximum(A, eps)
        else:
            inv_A = np.repeat(1.0 / np.maximum(A, eps)[:, None], G, axis=1)
        if volume_per_component:
            inv_lam = 1.0 / np.maximum(lam, eps)
            log_lam = np.log(np.maximum(lam, eps))
        else:
            inv_lam = np.full(G, 1.0 / max(lam, eps))
            log_lam = np.full(G, np.log(max(lam, eps)))
        # Negative log-likelihood, modulo constants.
        return float(np.sum(diag_W * inv_A * inv_lam[None, :]) + d * np.sum(n_k * log_lam))

    diag_W = _refresh_AL()
    trgt = _objective(diag_W)

    # ---- 2. Stiefel gradient descent with backtracking ----
    for _ in range(max_iter):
        # Build the gradient of the data-dependent objective wrt D.
        #   J(D) = Σ_k Σ_j (D^T W_k D)_jj / (λ_k A_kj)
        #   ∂J/∂D = 2 · Σ_k W_k · D · diag(1 / (λ_k A_k))
        grad = np.zeros((d, d))
        for k in range(G):
            A_k = A[:, k] if A.ndim == 2 else A
            lam_k = lam[k] if volume_per_component else lam
            inv = 1.0 / (max(lam_k, eps) * np.maximum(A_k, eps))
            grad += 2.0 * W[:, :, k] @ D * inv[None, :]
        # Project gradient onto tangent space of O(d) at D
        Dt_grad = D.T @ grad
        skew = 0.5 * (Dt_grad - Dt_grad.T)
        if float(np.linalg.norm(skew)) < tol:
            break

        # Backtracking line search along the geodesic D · expm(-α · skew)
        alpha = 1.0
        accepted = False
        for _ls in range(20):
            try:
                from scipy.linalg import expm

                D_new = D @ expm(-alpha * skew)
            except Exception:
                D_new = D - alpha * (D @ skew)
                # Re-orthonormalise
                Q, _ = np.linalg.qr(D_new)
                D_new = Q
            D_old = D
            D = D_new
            diag_W_new = _refresh_AL()
            trgt_new = _objective(diag_W_new)
            if trgt_new < trgt - 1e-14 * abs(trgt):
                trgt = trgt_new
                diag_W = diag_W_new
                accepted = True
                break
            else:
                D = D_old  # revert
                alpha *= 0.5
        if not accepted:
            break

    return D, A, lam


def _build_sigma_common_D(D, A, lam, G):
    d = D.shape[0]
    sigma = np.empty((d, d, G))
    for k in range(G):
        A_k = A[:, k] if A.ndim == 2 else A
        lam_k = lam[k] if np.ndim(lam) > 0 else lam
        sigma[:, :, k] = lam_k * D @ np.diag(A_k) @ D.T
    return sigma


def mstep_VEE(X, z, *, equal_pro: bool = False):
    n, d = X.shape
    n_k, n_total = _weights(z)
    mu = _means(X, z, n_k)
    W = _scatter_per_component(X, z, mu)
    D, A, lam = _flury_common_orientation(W, n_k, shape_per_component=False, volume_per_component=True)
    sigma = _build_sigma_common_D(D, A, lam, mu.shape[1])
    return {"pro": _proportions(n_k, n_total, equal_pro), "mu": mu, "sigma": sigma}


def mstep_EVE(X, z, *, equal_pro: bool = False):
    n, d = X.shape
    n_k, n_total = _weights(z)
    mu = _means(X, z, n_k)
    W = _scatter_per_component(X, z, mu)
    D, A, lam = _flury_common_orientation(W, n_k, shape_per_component=True, volume_per_component=False)
    sigma = _build_sigma_common_D(D, A, lam, mu.shape[1])
    return {"pro": _proportions(n_k, n_total, equal_pro), "mu": mu, "sigma": sigma}


def mstep_VVE(X, z, *, equal_pro: bool = False):
    n, d = X.shape
    n_k, n_total = _weights(z)
    mu = _means(X, z, n_k)
    W = _scatter_per_component(X, z, mu)
    D, A, lam = _flury_common_orientation(W, n_k, shape_per_component=True, volume_per_component=True)
    sigma = _build_sigma_common_D(D, A, lam, mu.shape[1])
    return {"pro": _proportions(n_k, n_total, equal_pro), "mu": mu, "sigma": sigma}


# ---------------------------------------------------------------- 1-D specials


def mstep_E(X, z, *, equal_pro: bool = False):
    """Univariate equal-variance — same as EII with d=1."""
    return mstep_EII(X, z, equal_pro=equal_pro)


def mstep_V(X, z, *, equal_pro: bool = False):
    """Univariate per-component variance — same as VII with d=1."""
    return mstep_VII(X, z, equal_pro=equal_pro)


# ---------------------------------------------------------------- registry


MSTEP_REGISTRY: dict[str, Callable] = {
    "E": mstep_E,
    "V": mstep_V,
    "EII": mstep_EII,
    "VII": mstep_VII,
    "EEI": mstep_EEI,
    "VEI": mstep_VEI,
    "EVI": mstep_EVI,
    "VVI": mstep_VVI,
    "EEE": mstep_EEE,
    "VEE": mstep_VEE,
    "EVE": mstep_EVE,
    "VVE": mstep_VVE,
    "EEV": mstep_EEV,
    "VEV": mstep_VEV,
    "EVV": mstep_EVV,
    "VVV": mstep_VVV,
}


def mstep(model_name: str, X: np.ndarray, z: np.ndarray, *, equal_pro: bool = False):
    """Dispatch to the per-model M-step and return ``(pro, mu, sigma)``."""
    try:
        fn = MSTEP_REGISTRY[model_name]
    except KeyError as e:
        raise ValueError(
            f"unknown modelName '{model_name}'. Valid: {sorted(MSTEP_REGISTRY)}"
        ) from e
    return fn(X, z, equal_pro=equal_pro)


EM_MODEL_NAMES_MULTI = (
    "EII", "VII", "EEI", "VEI", "EVI", "VVI",
    "EEE", "VEE", "EVE", "VVE",
    "EEV", "VEV", "EVV", "VVV",
)
EM_MODEL_NAMES_UNI = ("E", "V")
