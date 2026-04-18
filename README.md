# mclust-py

A **pure-Python re-implementation of CRAN [`mclust`](https://github.com/cran/mclust)** (Scrucca, Fop, Murphy, Raftery 2016) — Gaussian mixture model clustering with all 14 covariance parameterizations and BIC-driven model selection.

- AnnData-native — drop-in for the scanpy ecosystem
- No `rpy2`, no R install, no Fortran toolchain
- All 14 Banfield-Raftery / Celeux-Govaert parameterizations (EII, VII, EEI, VEI, EVI, VVI, EEE, VEE, EVE, VVE, EEV, VEV, EVV, VVV) plus the 1-D specials E / V
- Same Mclust() driver: BIC scan over (G × model), pick the maximum, return classification + soft responsibilities

> Same upstream-mirror pattern as [`monocle2-py`](https://github.com/omicverse/py-monocle2) and [`milor-py`](https://github.com/omicverse/py-miloR): the canonical implementation lives in [`omicverse`](https://github.com/Starlitnightly/omicverse); this repo is the standalone slice for users who want mclust without the full omicverse stack.

## Install

```bash
pip install mclust-py
```

## Quick-start

```python
import numpy as np
from sklearn import datasets
from mclust_py import Mclust

X = datasets.load_iris().data        # 150 × 4
fit = Mclust(X, G=range(1, 6))       # tries G = 1..5 across all 14 models
print(fit)                            # Mclust(VEV, G=2)  loglik=-215.83 BIC=-561.7
print(fit.BIC.iloc[:, :4].head())    # full BIC table

# Predict on new cells
z_new, cls_new = mclust_py.predict_mclust(fit, X_new)
```

Per-cell results are also written to AnnData via the convenience adapter:

```python
from mclust_py import mclust_anndata
fit = mclust_anndata(adata, use_rep="X_pca", n_components=10, G=range(2, 12))
# adata.obs['mclust']           — 1-based hard label
# adata.obs['mclust_uncertainty']
# adata.obsm['mclust_z']        — soft responsibilities
# adata.uns['mclust']           — BIC table + parameters
```

## Module map

| Module | What it covers |
|---|---|
| `mclust_py.mclust` | top-level `Mclust()` + `MclustResult` (BIC scan, model selection) |
| `mclust_py.em` | shared E-step + EM driver `me()` (Aitken stop, mirrors R's `meXXX`) |
| `mclust_py.models` | M-step for all 14 parameterizations (closed-form + Flury iterative) |
| `mclust_py.bic` | parameter counts and BIC formula |
| `mclust_py.hc` | Ward-on-SVD initial partition (`hc()` / `hclass()` / `partition_to_z()`) |
| `mclust_py.density` | per-component multivariate-normal log-density |
| `mclust_py.anndata_adapter` | `mclust_anndata()` for scanpy users |

## Notebooks

| Notebook | What it shows |
|---|---|
| [`examples/comparison_R_vs_python.ipynb`](examples/comparison_R_vs_python.ipynb) | Side-by-side comparison of CRAN `mclust 6.1.1` and `mclust-py` on 4 datasets × 14 covariance models × G ∈ {2,3,4,5}. Plots cluster scatters, BIC curves, classification confusion matrices, per-record loglik scatter, and per-model parity bars. Outputs are baked in so you can browse on GitHub without running anything. |

The notebook is built and executed by `examples/_build_notebooks.py`:

```bash
python examples/_build_notebooks.py
```

## R parity

`tests/r_parity_dump.R` runs CRAN `mclust::me()` on four canonical datasets (`blobs2`, `blobs5`, `iris`, `faithful`) for every `(modelName, G)` in `{14 models} × {2,3,4,5}` — 224 records total. `tests/r_parity_compare.py` then runs `mclust_py.me()` from the same `z_init` and reports correlations:

| Model | loglik rel-err | z corr | mean corr | sigma corr | classification match |
|------:|---------------:|-------:|----------:|-----------:|---------------------:|
| EII, VII, EEI, VEI, EVI, VVI, EEE, EEV, VEV, EVV, VVV | **~1e-15** (machine epsilon) | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| VEE | 6e-5 | 1.0000 | 1.0000 | 1.0000 | 0.999 |
| EVE | 5e-3 | 0.9751 | 0.9998 | 0.9843 | 0.975 |
| VVE | 3e-2 | 0.9351 | 0.9958 | 0.9198 | 0.939 |
| **overall (224 records)** | **3e-3** | **0.9935** | **0.9997** | **0.9931** | **0.9937** |

12 of 14 models hit bitwise parity with R. The two outliers (EVE / VVE) share a single orientation matrix `D` across components — the underlying optimisation problem (Browne–McNicholas 2013) has multiple local maxima, so different solvers can land on different stationary points. We use a Stiefel-manifold gradient descent (provably monotone) where R uses Browne–McNicholas MM; both solve the same problem and agree on simple data, but on hard data (e.g. iris with G ≥ 4) they may pick different local maxima.

For exact agreement on the EM step:

1. **Use the same `z_init`.** mclust's `hc()` is a Fortran routine that optimises the classification log-likelihood — Python's Ward-on-SVD approximates it well but is not bitwise identical. To force exact agreement, run R's `hc(data, modelName="VVV")` once and pass the resulting partition as `z_init` to `Mclust()`.
2. **Use the same control parameters.** Defaults match `emControl()` (`tol=1e-5`, `eps=.Machine$double.eps`).
3. **Use the same model_names.** Default in both is the full 14-model list for `d>1`, `c("E","V")` for `d=1`.

To regenerate the parity dump:

```bash
# in CMAP env (R + r-mclust)
Rscript tests/r_parity_dump.R

# then in omicdev env
python tests/r_parity_compare.py
pytest tests/test_r_parity.py -q
```

## Relationship to omicverse

This package is developed **upstream** in [`omicverse`](https://github.com/Starlitnightly/omicverse). If you already use omicverse, install it instead — `ov.utils.mclust_R(adata, ...)` and the upcoming `ov.utils.mclust_py(adata, ...)` cover the same surface plus the omicverse registry glue.

## Citation

If you use this package, please cite the original mclust paper:

> Scrucca L., Fop M., Murphy T.B., Raftery A.E. (2016). **mclust 5: clustering, classification and density estimation using Gaussian finite mixture models.** *The R Journal*, 8(1), 289–317.

and acknowledge omicverse / this repo for the Python port.

## License

GNU GPLv3 — matches both upstream `omicverse` and CRAN `mclust`.
