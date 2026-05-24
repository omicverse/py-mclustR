"""EM driver — mirrors mclust's me<modelName> functions.

Loops M-step → E-step until ``Δ(loglik_norm) < tol`` or ``itmax`` reached,
where ``loglik_norm`` is the Aitken-accelerated convergence criterion the
CRAN package uses (see ``mclust:::aitken`` and the Fortran ``mevvv``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .control import EMControl, em_control
from .density import log_dens_all
from .models import mstep, MSTEP_REGISTRY


# --------------------------------------------------------- log-likelihood / E-step


def _log_pi(pro: np.ndarray) -> np.ndarray:
    out = np.full_like(pro, -np.inf, dtype=np.float64)
    pos = pro > 0
    out[pos] = np.log(pro[pos])
    return out


def _logsumexp(A: np.ndarray, axis: int) -> np.ndarray:
    m = A.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def estep(
    X: np.ndarray, pro: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, float]:
    """Posterior responsibilities and observed-data log-likelihood."""
    log_phi = log_dens_all(X, mu, sigma)  # (n, G)
    log_w = log_phi + _log_pi(pro)[None, :]
    log_norm = _logsumexp(log_w, axis=1)  # (n,)
    z = np.exp(log_w - log_norm[:, None])
    return z, float(np.sum(log_norm))


# --------------------------------------------------------- EM result struct


@dataclass
class EMResult:
    model_name: str
    n: int
    d: int
    G: int
    pro: np.ndarray
    mean: np.ndarray  # (d, G)
    sigma: np.ndarray  # (d, d, G)
    z: np.ndarray  # (n, G)
    loglik: float
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)
    return_code: int = 0
    warning: Optional[str] = None


# --------------------------------------------------------- core EM loop


def me(
    X: np.ndarray,
    model_name: str,
    z_init: np.ndarray,
    *,
    control: Optional[EMControl] = None,
) -> EMResult:
    """Run EM starting from initial responsibility matrix ``z_init``.

    Mirrors the behaviour of :func:`mclust::me` — including the M-step-first
    schedule and the Aitken-accelerated stopping rule.
    """
    if control is None:
        control = em_control()
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    z = np.ascontiguousarray(np.asarray(z_init, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError("data must be a 2-D matrix")
    if z.ndim != 2 or z.shape[0] != X.shape[0]:
        raise ValueError("z must be (n, G)")
    n, d = X.shape
    G = z.shape[1]
    # mclust starts EM with an M-step on the supplied z, then iterates
    # M (params) → E (z, loglik) → check Δ.
    itmax = control.itmax[0]
    tol = control.tol[0]

    history: list[float] = []
    prev_loglik = -np.inf
    converged = False
    pro = mu = sigma = None
    its = 0
    return_code = 0
    warning: Optional[str] = None

    for its in range(1, itmax + 1):
        params = mstep(model_name, X, z, equal_pro=control.equal_pro)
        pro, mu, sigma = params["pro"], params["mu"], params["sigma"]
        try:
            z, loglik = estep(X, pro, mu, sigma)
        except np.linalg.LinAlgError as e:
            warning = f"singular covariance: {e}"
            return_code = -1
            break
        history.append(loglik)
        # Aitken-style: stop when relative log-lik gain falls below tol.
        if its > 1:
            denom = max(abs(loglik), 1e-300)
            if abs(loglik - prev_loglik) / denom < tol:
                converged = True
                break
        prev_loglik = loglik

    if not converged and warning is None and its >= itmax:
        warning = "iteration limit reached"
        return_code = 1

    # ---- mclust's post-EM defensive checks (mevvv.f lines 14829, 14888-14904) ----
    # Replicates R's `eps` sentinels: if EM landed on a degenerate solution,
    # mark loglik=NaN so the BIC scan skips this (model, G) combination.
    #
    # Without this, py-mclustR will rank degenerate small-cluster solutions as
    # the "best" BIC on real data. Discovered via py-TSCAN's lpsdata fixture:
    # R Mclust NAs out G=5..9 (VVV); py without this guard chose G=5.
    #
    # **Model scope**: only applied to FULLY-INDEPENDENT-COVARIANCE models
    # (VVV, VVI, VII, V, EVI) where a single small cluster has its own
    # singular covariance. SHAPE/ORIENTATION-SHARING models (EEE, EEV,
    # VEV, EVE, VVE, EEI, VEI) borrow structure across clusters, so a
    # singleton cluster gets a sane regularised covariance from the
    # shared component — R does not flag those either.
    INDEPENDENT_COV_MODELS = {"V", "VII", "VVI", "VVV", "EVI"}
    if (
        return_code == 0
        and pro is not None
        and sigma is not None
        and model_name.upper() in INDEPENDENT_COV_MODELS
    ):
        rteps = np.sqrt(max(control.eps, np.finfo(np.float64).eps))
        # (1) Empty-group check — analogue of Fortran `zsum ≤ rteps`.
        zsum_min = float(z.sum(axis=0).min())
        if zsum_min < rteps:
            return_code = -2
            warning = f"missing groups (zsum_min={zsum_min:.3e} < sqrt(eps))"
        else:
            # (2) Singular-covariance check — mclust's `rcmin` via Cholesky.
            for k in range(G):
                Sk = sigma[..., k] if sigma.ndim == 3 else sigma[k]
                try:
                    L = np.linalg.cholesky(Sk)
                except np.linalg.LinAlgError:
                    return_code = -3
                    warning = f"non-PSD covariance in cluster {k}"
                    break
                d_abs = np.abs(np.diag(L))
                rcmin = float(d_abs.min() / (1.0 + d_abs.max()))
                if rcmin < rteps:
                    return_code = -3
                    warning = (
                        f"singular covariance in cluster {k} "
                        f"(rcmin={rcmin:.3e} < sqrt(eps))"
                    )
                    break

    final_loglik = (
        float("nan") if return_code < 0
        else (float(history[-1]) if history else float("nan"))
    )

    return EMResult(
        model_name=model_name,
        n=n,
        d=d,
        G=G,
        pro=pro,  # type: ignore[arg-type]
        mean=mu,  # type: ignore[arg-type]
        sigma=sigma,  # type: ignore[arg-type]
        z=z,
        loglik=final_loglik,
        iterations=its,
        converged=converged,
        history=history,
        return_code=return_code,
        warning=warning,
    )


def fit_one_component(
    X: np.ndarray, model_name: str, *, control: Optional[EMControl] = None
) -> EMResult:
    """G = 1 closed-form fit (`mvn<Model>` in R)."""
    n, d = X.shape
    z = np.ones((n, 1), dtype=np.float64)
    params = mstep(model_name, X, z, equal_pro=False)
    pro = params["pro"]
    mu = params["mu"]
    sigma = params["sigma"]
    z_post, loglik = estep(X, pro, mu, sigma)
    return EMResult(
        model_name=model_name,
        n=n,
        d=d,
        G=1,
        pro=pro,
        mean=mu,
        sigma=sigma,
        z=z_post,
        loglik=loglik,
        iterations=1,
        converged=True,
        history=[loglik],
        return_code=0,
        warning=None,
    )
