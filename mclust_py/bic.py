"""Parameter counts and BIC formula — port of mclust's nMclustParams / bic."""
from __future__ import annotations

import math


def n_var_params(model_name: str, d: int, G: int) -> int:
    """Free covariance parameters under each model."""
    table = {
        "E": 1,
        "V": G,
        "EII": 1,
        "VII": G,
        "EEI": d,
        "VEI": G + (d - 1),
        "EVI": 1 + G * (d - 1),
        "VVI": G * d,
        "EEE": d * (d + 1) // 2,
        "EVE": 1 + G * (d - 1) + d * (d - 1) // 2,
        "VEE": G + (d - 1) + d * (d - 1) // 2,
        "VVE": G + G * (d - 1) + d * (d - 1) // 2,
        "EEV": 1 + (d - 1) + G * d * (d - 1) // 2,
        "VEV": G + (d - 1) + G * d * (d - 1) // 2,
        "EVV": 1 - G + G * d * (d + 1) // 2,
        "VVV": G * d * (d + 1) // 2,
    }
    if model_name not in table:
        raise ValueError(f"unknown model {model_name}")
    return table[model_name]


def n_mclust_params(
    model_name: str,
    d: int,
    G: int,
    *,
    noise: bool = False,
    equal_pro: bool = False,
) -> int:
    """Total free parameters: covariance + means + mixing weights (+ noise)."""
    if G == 0:
        if not noise:
            raise ValueError("undefined model: G=0 without noise component")
        return 1
    nparams = n_var_params(model_name, d, G) + G * d
    if not equal_pro:
        nparams += G - 1
    if noise:
        nparams += 2
    return nparams


def bic(loglik: float, model_name: str, n: int, d: int, G: int, **kw) -> float:
    """``2·loglik − k·log(n)`` — the *higher-is-better* mclust convention."""
    k = n_mclust_params(model_name, d, G, **kw)
    return 2.0 * loglik - k * math.log(n)
