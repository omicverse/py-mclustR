"""Top-level :func:`Mclust` and :class:`MclustResult` — mirror of R's API.

Workflow (matches `mclust::Mclust`):

1. For each model_name × G:
   a) build initial responsibility ``z_init`` from the supplied
      hierarchical-clustering tree (or compute one);
   b) run EM (:func:`mclust_py.em.me`);
   c) record loglik + BIC.
2. Pick the (G, model_name) combination with the largest BIC.
3. Re-fit at the winning configuration (already in cache) and return a
   :class:`MclustResult` carrying parameters, classification, and the
   full BIC table — so users can inspect the model selection step exactly
   like in R.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .bic import bic as _bic
from .control import EMControl, em_control
from .em import EMResult, fit_one_component, me
from .hc import HCResult, hc, hclass, partition_to_z
from .models import EM_MODEL_NAMES_MULTI, EM_MODEL_NAMES_UNI


@dataclass
class MclustResult:
    """Output of :func:`Mclust`. Mirrors the slots of R's ``Mclust`` object."""

    data: np.ndarray
    n: int
    d: int
    G: int
    model_name: str
    pro: np.ndarray
    mean: np.ndarray
    sigma: np.ndarray
    z: np.ndarray
    classification: np.ndarray
    uncertainty: np.ndarray
    loglik: float
    df: int
    bic: float
    BIC: pd.DataFrame  # full G × model_name table
    iterations: int
    converged: bool
    em_result: EMResult
    history: list[float] = field(default_factory=list)

    def __repr__(self) -> str:  # mirrors R's print.Mclust
        return (
            f"Mclust({self.model_name}, G={self.G}) "
            f"loglik={self.loglik:.4f}  BIC={self.bic:.4f}"
        )


def _classification_from_z(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cls = np.argmax(z, axis=1) + 1  # 1-based, matching R
    unc = 1.0 - z[np.arange(z.shape[0]), np.argmax(z, axis=1)]
    return cls, unc


def _resolve_models(d: int, model_names: Optional[Sequence[str]]) -> list[str]:
    if model_names is not None:
        return [m for m in model_names]
    if d == 1:
        return list(EM_MODEL_NAMES_UNI)
    return list(EM_MODEL_NAMES_MULTI)


def _resolve_G(G: Optional[Iterable[int] | int]) -> list[int]:
    if G is None:
        return list(range(1, 10))  # R default: 1:9
    if isinstance(G, (int, np.integer)):
        return [int(G)]
    return [int(g) for g in G]


def Mclust(
    data: np.ndarray,
    G: Optional[Iterable[int] | int] = None,
    model_names: Optional[Sequence[str]] = None,
    *,
    z_init: Optional[np.ndarray] = None,
    initialization: Optional[HCResult | dict] = None,
    control: Optional[EMControl] = None,
    equal_pro: bool = False,
) -> MclustResult:
    """Fit Gaussian mixture models — pick best by BIC.

    Parameters
    ----------
    data : (n, d) array-like
    G : int | iterable[int] | None
        Numbers of components to try (default: ``1..9``).
    model_names : sequence[str] | None
        Restricted set of covariance models (default: all 14 multivariate
        / both univariate). Names follow mclust convention.
    z_init : (n, G_max) array | None
        Optional initial responsibility matrix to use for every fit
        (mostly used for R-parity tests).
    initialization : HCResult | dict | None
        Pre-computed hierarchical-clustering tree, or a dict like
        ``{"hcPairs": HCResult}``. If ``None``, a tree is built lazily
        the first time it is needed.
    control : EMControl | None
        EM control parameters; defaults to mclust's `emControl()`.
    equal_pro : bool
        Force equal mixing proportions (mclust's `control$equalPro`).
    """
    X = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    if control is None:
        control = em_control()
    if equal_pro:
        control.equal_pro = True

    Gs = _resolve_G(G)
    models = _resolve_models(d, model_names)

    # If an HC tree is provided as a dict, unwrap it.
    if isinstance(initialization, dict):
        initialization = initialization.get("hcPairs")

    hc_tree: Optional[HCResult] = (
        initialization if isinstance(initialization, HCResult) else None
    )

    # Lazy default tree only built when needed and only once.
    def _ensure_tree() -> HCResult:
        nonlocal hc_tree
        if hc_tree is None:
            hc_model = "VVV" if d > 1 else "V"
            hc_tree = hc(X, model_name=hc_model, use="SVD")
        return hc_tree

    bic_table = pd.DataFrame(
        index=[str(g) for g in Gs], columns=models, dtype=np.float64
    )
    fits: dict[tuple[int, str], EMResult] = {}

    for g in Gs:
        if g == 1:
            for m in models:
                try:
                    res = fit_one_component(X, m, control=control)
                    fits[(1, m)] = res
                    bic_table.loc[str(g), m] = _bic(
                        res.loglik, m, n, d, 1, equal_pro=equal_pro
                    )
                except Exception:
                    bic_table.loc[str(g), m] = np.nan
            continue
        # G ≥ 2 — need an init z.
        if z_init is not None:
            z = z_init[:, :g].copy() if z_init.shape[1] >= g else None
        else:
            z = None
        if z is None:
            tree = _ensure_tree()
            partition = hclass(tree, g)
            z = partition_to_z(partition, G=g)
        for m in models:
            try:
                res = me(X, m, z, control=control)
                fits[(g, m)] = res
                if not np.isfinite(res.loglik):
                    bic_table.loc[str(g), m] = np.nan
                else:
                    bic_table.loc[str(g), m] = _bic(
                        res.loglik, m, n, d, g, equal_pro=equal_pro
                    )
            except Exception:
                bic_table.loc[str(g), m] = np.nan

    # Pick best (G, model)
    flat = bic_table.stack(future_stack=True) if hasattr(pd.DataFrame, "stack") else bic_table.stack()
    flat = flat.dropna()
    if flat.empty:
        raise RuntimeError("no model converged on this data — try different G/modelNames")
    best_label = flat.idxmax()
    g_best, m_best = best_label
    g_best = int(g_best)
    res = fits[(g_best, m_best)]
    cls, unc = _classification_from_z(res.z)
    from .bic import n_mclust_params  # local import to avoid cycle look

    df = n_mclust_params(m_best, d, g_best, equal_pro=equal_pro)
    return MclustResult(
        data=X,
        n=n,
        d=d,
        G=g_best,
        model_name=m_best,
        pro=res.pro,
        mean=res.mean,
        sigma=res.sigma,
        z=res.z,
        classification=cls,
        uncertainty=unc,
        loglik=res.loglik,
        df=df,
        bic=float(bic_table.loc[str(g_best), m_best]),
        BIC=bic_table,
        iterations=res.iterations,
        converged=res.converged,
        em_result=res,
        history=res.history,
    )


def predict_mclust(
    result: MclustResult, newdata: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """E-step under fitted parameters — returns ``(z, classification)``."""
    from .em import estep

    Xn = np.ascontiguousarray(np.asarray(newdata, dtype=np.float64))
    if Xn.ndim == 1:
        Xn = Xn[:, None]
    z, _ = estep(Xn, result.pro, result.mean, result.sigma)
    cls, _ = _classification_from_z(z)
    return z, cls
