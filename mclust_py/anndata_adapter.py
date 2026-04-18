"""AnnData adapter — run :func:`Mclust` on a chosen embedding."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from .mclust import Mclust, MclustResult


def mclust_anndata(
    adata,
    use_rep: str = "X_pca",
    n_components: Optional[int] = None,
    G: Optional[Iterable[int] | int] = None,
    model_names: Optional[Sequence[str]] = None,
    *,
    key_added: str = "mclust",
    **mclust_kwargs,
) -> MclustResult:
    """Fit Mclust on an AnnData embedding and write results to ``adata``.

    Writes:
      - ``adata.obs[f"{key_added}"]`` — 1-based hard label
      - ``adata.obs[f"{key_added}_uncertainty"]``
      - ``adata.obsm[f"{key_added}_z"]`` — soft responsibilities
      - ``adata.uns[key_added]`` — fitted-parameter dict + BIC table
    """
    if use_rep == "X":
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    else:
        X = np.asarray(adata.obsm[use_rep])
    if n_components is not None:
        X = X[:, :n_components]
    fit = Mclust(X, G=G, model_names=model_names, **mclust_kwargs)
    adata.obs[key_added] = fit.classification.astype(int)
    adata.obs[f"{key_added}_uncertainty"] = fit.uncertainty
    adata.obsm[f"{key_added}_z"] = fit.z
    adata.uns[key_added] = {
        "model_name": fit.model_name,
        "G": fit.G,
        "loglik": fit.loglik,
        "bic": fit.bic,
        "df": fit.df,
        "BIC": fit.BIC,
        "pro": fit.pro,
        "mean": fit.mean,
        "sigma": fit.sigma,
        "use_rep": use_rep,
        "n_components": X.shape[1],
    }
    return fit
