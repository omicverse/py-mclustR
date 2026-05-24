# Changelog

## v0.2.0 — 2026-05-24

### Native R-mclust parity fix

This release closes the **native-init parity** gap that v0.1.x left open: when
users called ``Mclust(X)`` without supplying a pre-computed R hclust ``z_init``,
the result could diverge from R Mclust on real data. The 251 existing
R-parity tests passed because they were always supplied with R's ``z_init``.

Discovered via the py-TSCAN port: on the canonical TSCAN ``lpsdata`` fixture
(131 cells × 5 PCs after PCA), R Mclust returned clusters of size 44/57/30
with loglik=−1727.97; py-mclustR v0.1.x returned 48/5/78 with loglik=−1628.23
— a degenerate small-cluster solution.

#### Root causes (two independent bugs)

1. **HC initialisation used scipy Ward linkage**, not mclust's
   model-based agglomerative VVV criterion (Fraley 1998 / Banfield-Raftery
   1993 log-determinant). The default-`use="SVD"` partition diverged from
   R's, so EM started from a different basin.

2. **EM lacked mclust's defensive aborts**. ``mevvv.f`` lines 14829 and
   14888-14904 abort with NA-loglik when a cluster's effective sample size
   or covariance condition number drops below ``sqrt(eps)``. v0.1.x's EM
   silently kept the degenerate result, so the BIC scan picked degenerate
   ``G`` values.

#### Changes

- **`mclust_py/hc.py`**: Added `_hcvvv_linkage` — Fraley 1998 agglomerative
  HC with the dual log-det/trace criterion from mclust's ``vvvtij``. The
  ``hc(model_name="VVV", use="SVD")`` path now matches R's ``hc()`` partition
  on canonical fixtures (ARI=1.0 on lpsdata PCA).

- **`mclust_py/em.py`**: After EM convergence, mclust's two defensive
  checks are now applied to **independent-covariance models** (V, VII, VVI,
  VVV, EVI). If a cluster's effective sample size or Cholesky reciprocal
  condition number drops below ``sqrt(eps)``, ``loglik`` is set to NaN and
  ``return_code = -2`` (empty group) or ``-3`` (singular covariance). The
  check is **not** applied to shape-sharing models (EEE, EEV, VEV, EVE,
  VVE, EEI, VEI), matching R's behaviour — those models borrow shape
  across components and don't trip on a singleton cluster.

#### Tests

- Added ``tests/test_native_init_parity.py`` (3 tests) — regression guard
  on the lpsdata fixture. All 254 tests pass (251 prior + 3 new).

#### Downstream

- **py-TSCAN's default-mode parity gate** now clears at Pearson=1.000000
  with the same cluster IDs as R Mclust. The previous ``xfail`` marker
  was removed.

#### Compatibility

API is fully backward-compatible. The only behaviour change is:

- `Mclust(X, G=2:9)` may now return a different best-G than v0.1.x on
  data where v0.1.x preferred degenerate-cluster solutions. The new
  pick matches R Mclust's choice.

## v0.1.0 — 2026-04-17

Initial release: pure-Python re-implementation of CRAN mclust 6.1.2 with
all 14 covariance parameterisations, BIC-driven model selection,
AnnData-native API. R-parity validated against z_init-supplied EM trajectories.
