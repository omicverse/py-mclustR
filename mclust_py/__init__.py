"""mclust_py — Python re-implementation of CRAN's `mclust`.

Mirrors the public surface of the R package for the core EM workflow:

>>> from mclust_py import Mclust
>>> import numpy as np, sklearn.datasets as ds
>>> X = ds.load_iris().data
>>> fit = Mclust(X, G=range(1, 6))
>>> fit.model_name, fit.G, fit.bic
('VEV', 2, ...)

The 14 covariance parameterizations of Banfield-Raftery / Celeux-Govaert
are all available; see :mod:`mclust_py.models` for the model letter
codes.
"""
from __future__ import annotations

from .control import EMControl, em_control
from .density import log_dens_all, log_dens_component
from .em import EMResult, estep, fit_one_component, me
from .bic import bic, n_mclust_params, n_var_params
from .hc import HCResult, hc, hclass, partition_to_z
from .models import (
    EM_MODEL_NAMES_MULTI,
    EM_MODEL_NAMES_UNI,
    MSTEP_REGISTRY,
    mstep,
)
from .mclust import Mclust, MclustResult, predict_mclust

# Optional AnnData adapter — only import if anndata is installed so the
# package stays usable without the scanpy stack.
try:
    from .anndata_adapter import mclust_anndata
except Exception:  # pragma: no cover — optional dependency
    mclust_anndata = None  # type: ignore[assignment]

__version__ = "0.1.0"

__all__ = [
    "Mclust",
    "MclustResult",
    "predict_mclust",
    "me",
    "estep",
    "EMResult",
    "fit_one_component",
    "mstep",
    "MSTEP_REGISTRY",
    "EM_MODEL_NAMES_MULTI",
    "EM_MODEL_NAMES_UNI",
    "EMControl",
    "em_control",
    "log_dens_all",
    "log_dens_component",
    "bic",
    "n_mclust_params",
    "n_var_params",
    "hc",
    "hclass",
    "partition_to_z",
    "HCResult",
    "mclust_anndata",
]
