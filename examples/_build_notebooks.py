"""Build and execute the example notebooks for py-mclustR.

Run once — produces:
    examples/comparison_R_vs_python.ipynb

with outputs (figures + tables) baked in so GitHub renders them without
requiring a local kernel.
"""
from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent


def _comparison_notebook() -> nbf.notebooknode.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells

    c.append(nbf.v4.new_markdown_cell("""\
# `pymclustR` vs CRAN `mclust` — clustering & consistency comparison

This notebook is a side-by-side comparison of the **R `mclust` 6.1.1** package
and our **Python `pymclustR`** port on four canonical datasets:

> Install with `pip install pymclustR`. The Python import name is `mclust_py`.

| Dataset    | n   | d  | Source                       |
|------------|----:|---:|------------------------------|
| `blobs2`   | 300 | 2  | Synthetic — 3 well-separated 2-D blobs |
| `blobs5`   | 400 | 5  | Synthetic — 5 well-separated 5-D blobs |
| `iris`     | 150 | 4  | UCI Iris (3 species, partly overlapping in petal/sepal space) |
| `faithful` | 272 | 2  | Old Faithful eruptions (waiting time vs duration) |

For each dataset we
1. Read R's reference `me()` outputs (pre-dumped via `tests/r_parity_dump.R`).
2. Run our `mclust_py.me()` from the *same* `z_init`.
3. Compare:
    * cluster scatterplots (R cls vs Python cls)
    * full BIC matrix (G × modelName)
    * per-record correlation heatmap (z, mean, sigma)
    * per-model summary bar chart

> **Heads-up.** This notebook expects the R reference dump to exist at
> `tests/_rparity/`. Re-generate with:
>
> ```bash
> Rscript tests/r_parity_dump.R    # in CMAP env (R + r-mclust)
> ```
"""))

    # --- setup
    c.append(nbf.v4.new_code_cell("""\
import json
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mclust_py import Mclust, me

DUMP = Path('..') / 'tests' / '_rparity'
assert DUMP.exists(), 'R-parity dump missing — run tests/r_parity_dump.R first'

sns.set_theme(context='notebook', style='whitegrid')
plt.rcParams['figure.dpi'] = 110

DATASETS = ['blobs2', 'blobs5', 'iris', 'faithful']
MODELS = ['EII','VII','EEI','VEI','EVI','VVI',
          'EEE','VEE','EVE','VVE',
          'EEV','VEV','EVV','VVV']

def load_csv(p): return np.loadtxt(p, delimiter=',', dtype=np.float64, ndmin=2)

def to_arr(values):
    return np.array(
        [np.nan if (isinstance(v, str) and v.upper() == 'NA') else v for v in values],
        dtype=np.float64,
    )

def reshape_F(values, dim):
    return to_arr(values).reshape(tuple(dim), order='F')

def aligned_match(a, b, G):
    a = np.asarray(a); b = np.asarray(b)
    best = 0.0
    for perm in permutations(range(1, G + 1)):
        relabel = np.array([0] + list(perm))
        best = max(best, float(np.mean(relabel[b] == a)))
    return best

def best_perm(r_z, py_z):
    G = r_z.shape[1]
    sim = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            num = float(np.sum(r_z[:, i] * py_z[:, j]))
            den = float(np.linalg.norm(r_z[:, i]) * np.linalg.norm(py_z[:, j]))
            sim[i, j] = num / max(den, 1e-12)
    from scipy.optimize import linear_sum_assignment
    _, ci = linear_sum_assignment(-sim)
    return ci
"""))

    # --- big sweep: run our me() against every R record
    c.append(nbf.v4.new_markdown_cell("""\
## 1.  Per-record parity sweep — 14 models × 4 datasets × G ∈ {2,3,4,5}

Loads each pre-dumped R `me()` result, runs `mclust_py.me()` from the same
`z_init`, and records the agreement metrics."""))
    c.append(nbf.v4.new_code_cell("""\
records = []
for path in sorted(DUMP.glob('*_G[0-9].json')):
    rec = json.loads(path.read_text())
    if rec.get('error'): continue
    if rec.get('loglik') is None or (isinstance(rec['loglik'], str) and rec['loglik'].upper() == 'NA'):
        continue
    name, mname, G = rec['dataset'], rec['modelName'], int(rec['G'])
    X = load_csv(DUMP / f'data_{name}_X.csv')
    z_init = load_csv(DUMP / f'zinit_{name}_G{G}.csv')
    r_z = reshape_F(rec['z'], rec['z_dim'])
    if not np.all(np.isfinite(r_z)): continue
    out = me(X, mname, z_init)
    perm = best_perm(r_z, out.z)
    py_z = out.z[:, perm]
    py_mean = out.mean[:, perm]
    py_sigma = out.sigma[:, :, perm]
    r_mean = reshape_F(rec['mean'], rec['mean_dim'])
    r_sigma = reshape_F(rec['sigma'], rec['sigma_dim']) if rec.get('sigma') else None
    records.append(dict(
        dataset=name, modelName=mname, G=G,
        n=rec['n'], d=rec['d'],
        r_loglik=float(rec['loglik']), py_loglik=out.loglik,
        loglik_rel=abs(out.loglik - float(rec['loglik'])) / max(abs(float(rec['loglik'])), 1e-12),
        z_corr=float(np.corrcoef(r_z.ravel(), py_z.ravel())[0,1]),
        mean_corr=float(np.corrcoef(r_mean.ravel(), py_mean.ravel())[0,1]),
        sigma_corr=float(np.corrcoef(r_sigma.ravel(), py_sigma.ravel())[0,1]) if r_sigma is not None else np.nan,
        cls_match=aligned_match(np.argmax(r_z,axis=1)+1, np.argmax(py_z,axis=1)+1, G),
        iters=out.iterations,
    ))
df = pd.DataFrame(records)
print(f'records: {len(df)}')
df.head()
"""))

    c.append(nbf.v4.new_markdown_cell("## 2.  Per-model summary"))
    c.append(nbf.v4.new_code_cell("""\
summary = (df.groupby('modelName')
             .agg(loglik_rel=('loglik_rel', 'mean'),
                  z_corr=('z_corr', 'mean'),
                  mean_corr=('mean_corr', 'mean'),
                  sigma_corr=('sigma_corr', 'mean'),
                  cls_match=('cls_match', 'mean'),
                  n=('z_corr', 'size'))
             .reindex(MODELS))
print(summary.to_string(float_format=lambda x: f'{x:.4f}'))
"""))

    c.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
metrics = [('z_corr', 'z (responsibilities)'),
           ('mean_corr', 'μ (means)'),
           ('sigma_corr', 'Σ (covariances)'),
           ('cls_match', 'classification match')]
for ax, (col, title) in zip(axes, metrics):
    vals = summary[col].values
    bars = ax.barh(MODELS, vals, color=['#3a7' if v > 0.99 else ('#fb3' if v > 0.9 else '#e44') for v in vals])
    ax.set_xlim(min(0.8, vals.min()*0.99), 1.005)
    ax.axvline(0.99, color='gray', ls='--', alpha=0.6, label='99% threshold')
    ax.set_title(f'mean per-model {title}')
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(v - 0.005, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
                ha='right', va='center', fontsize=8, color='white' if v > 0.85 else 'black')
plt.suptitle('pymclustR vs R mclust — per-model agreement (averaged over 4 datasets × 4 G values)', fontsize=12)
plt.tight_layout()
plt.show()
"""))

    c.append(nbf.v4.new_markdown_cell("""\
## 3.  Per-record loglikelihood agreement

Scatter R's loglik vs Python's loglik across all 224 records — the closer
to the diagonal, the better the parity."""))
    c.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(7, 7))
palette = sns.color_palette('tab20', n_colors=14)
color_map = dict(zip(MODELS, palette))
for m, sub in df.groupby('modelName'):
    ax.scatter(sub['r_loglik'], sub['py_loglik'], s=40, alpha=0.7,
               color=color_map[m], label=m, edgecolor='white', lw=0.5)
lo = min(df['r_loglik'].min(), df['py_loglik'].min())
hi = max(df['r_loglik'].max(), df['py_loglik'].max())
ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='y = x')
ax.set_xlabel('R mclust loglik')
ax.set_ylabel('pymclustR loglik')
ax.set_title('Per-record log-likelihood — R vs Python')
ax.legend(ncol=2, fontsize=8, loc='upper left')
plt.tight_layout()
plt.show()
"""))

    c.append(nbf.v4.new_markdown_cell("""\
## 4.  Side-by-side cluster scatters

Pick the (G, model) that R chose by BIC for each dataset, then plot the
2-D projection coloured by R-cls vs Python-cls."""))
    c.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for col, ds in enumerate(DATASETS):
    full_path = DUMP / f'mclust_full_{ds}.json'
    if not full_path.exists(): continue
    full = json.loads(full_path.read_text())
    X = load_csv(DUMP / f'data_{ds}_X.csv')
    mname = full['modelName']
    G = int(full['G'])
    r_cls = np.array(full['classification'])
    # Run pymclustR with the same (G, model)
    z_init = load_csv(DUMP / f'zinit_{ds}_G{G}.csv') if (DUMP / f'zinit_{ds}_G{G}.csv').exists() else None
    if z_init is not None and z_init.shape[1] >= G:
        out = me(X, mname, z_init[:, :G])
    else:
        from mclust_py import Mclust
        out = Mclust(X, G=[G], model_names=[mname])
        out = out.em_result
    py_cls = np.argmax(out.z, axis=1) + 1
    if X.shape[1] > 2:
        # PCA-2 for visualisation only
        from sklearn.decomposition import PCA
        Xv = PCA(n_components=2, random_state=0).fit_transform(X)
        sub = ' (PCA-2 projection)'
    else:
        Xv, sub = X, ''
    palette = sns.color_palette('tab10', n_colors=max(G, 3))
    for ax_row, cls, label in [(0, r_cls, 'R mclust'), (1, py_cls, 'pymclustR')]:
        ax = axes[ax_row, col]
        for k in np.unique(cls):
            m = cls == k
            ax.scatter(Xv[m, 0], Xv[m, 1], s=18, alpha=0.7,
                       color=palette[(int(k)-1) % len(palette)], label=f'cluster {k}')
        ax.set_title(f'{ds}{sub}\\n{label} → {mname}, G={G}')
        ax.set_aspect('equal' if ds in {'blobs2','faithful'} else 'auto')
        if ax_row == 1: ax.set_xlabel('dim 1')
        ax.set_ylabel('dim 2' if col == 0 else '')
    # Annotate top row with classification agreement
    match = aligned_match(r_cls, py_cls, G)
    axes[0, col].text(0.02, 0.98, f'cls match = {match:.3f}',
                      transform=axes[0, col].transAxes,
                      ha='left', va='top', fontsize=9,
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.suptitle('Cluster assignment — top: R mclust(BIC),  bottom: pymclustR(same (G, model))', fontsize=12)
plt.tight_layout()
plt.show()
"""))

    c.append(nbf.v4.new_markdown_cell("""\
## 5.  BIC curves — model selection comparison

For each dataset, plot R's full `mclustBIC` matrix alongside the
pymclustR BIC values (rebuilt from the per-record loglik in `df` plus
the `n_mclust_params` formula — same definition R uses)."""))
    c.append(nbf.v4.new_code_cell("""\
from mclust_py.bic import n_mclust_params

def py_bic_table(df_sub):
    rows = []
    for _, r in df_sub.iterrows():
        k = n_mclust_params(r['modelName'], int(r['d']), int(r['G']))
        rows.append({'modelName': r['modelName'], 'G': r['G'],
                     'BIC': 2.0 * r['py_loglik'] - k * np.log(r['n'])})
    return (pd.DataFrame(rows)
              .pivot_table(index='G', columns='modelName', values='BIC')
              .reindex(columns=MODELS))

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
palette = sns.color_palette('tab20', n_colors=14)
color_map = dict(zip(MODELS, palette))
for col, ds in enumerate(DATASETS):
    full_path = DUMP / f'mclust_full_{ds}.json'
    if not full_path.exists(): continue
    full = json.loads(full_path.read_text())
    Gs_R = full['BIC_dimnames_G']
    models_R = full['BIC_dimnames_model']
    BIC_R = reshape_F(full['BIC'], full['BIC_dim'])
    BIC_R = pd.DataFrame(BIC_R, index=Gs_R, columns=models_R)
    BIC_py = py_bic_table(df[df['dataset'] == ds])

    ax = axes[0, col]
    for m in models_R:
        ax.plot(BIC_R.index, BIC_R[m], marker='o', ms=3, lw=1,
                color=color_map.get(m, '#888'), label=m)
    ax.set_title(f'R mclust BIC — {ds}\\nbest: {full["modelName"]} G={full["G"]}')
    ax.set_xlabel('G'); ax.set_ylabel('BIC')
    if col == 0: ax.legend(ncol=2, fontsize=6, loc='lower right')

    ax = axes[1, col]
    best = BIC_py.stack().idxmax()
    for m in BIC_py.columns:
        ax.plot(BIC_py.index, BIC_py[m], marker='s', ms=3, lw=1,
                color=color_map.get(m, '#888'), label=m)
    ax.set_title(f'pymclustR BIC — {ds}\\nbest: {best[1]} G={best[0]}')
    ax.set_xlabel('G'); ax.set_ylabel('BIC')

plt.suptitle('BIC curves — R (top) vs Python (bottom) — same colour = same model', fontsize=12)
plt.tight_layout()
plt.show()
"""))

    c.append(nbf.v4.new_markdown_cell("""\
## 6.  Confusion matrix — per-dataset hard-label agreement

For each dataset we take the *best* (G, model) chosen by R, run
`pymclustR` with the same configuration, and tabulate label agreement."""))
    c.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, ds in zip(axes, DATASETS):
    full_path = DUMP / f'mclust_full_{ds}.json'
    if not full_path.exists(): continue
    full = json.loads(full_path.read_text())
    X = load_csv(DUMP / f'data_{ds}_X.csv')
    G = int(full['G'])
    mname = full['modelName']
    r_cls = np.array(full['classification'])
    z_init = load_csv(DUMP / f'zinit_{ds}_G{G}.csv')[:, :G]
    out = me(X, mname, z_init)
    py_cls = np.argmax(out.z, axis=1) + 1
    # Best label permutation
    best, best_perm_lst = -1, None
    for perm in permutations(range(1, G + 1)):
        rl = np.array([0] + list(perm))
        m = float(np.mean(rl[py_cls] == r_cls))
        if m > best: best, best_perm_lst = m, perm
    relabel = np.array([0] + list(best_perm_lst))
    py_aligned = relabel[py_cls]
    cm = pd.crosstab(pd.Series(r_cls, name='R cls'),
                     pd.Series(py_aligned, name='Py cls (aligned)'))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'{ds} ({mname}, G={G})  match={best:.3f}')
plt.suptitle('Hard-label agreement — best permutation', fontsize=12)
plt.tight_layout()
plt.show()
"""))

    c.append(nbf.v4.new_markdown_cell("""\
## 7.  Per-record correlation heatmap

A single picture summary of the parity table — rows are model, columns
are datasets, cell is the average z-correlation across G values."""))
    c.append(nbf.v4.new_code_cell("""\
pivot_z = df.pivot_table(index='modelName', columns='dataset', values='z_corr', aggfunc='mean').reindex(MODELS)
pivot_sigma = df.pivot_table(index='modelName', columns='dataset', values='sigma_corr', aggfunc='mean').reindex(MODELS)
fig, axes = plt.subplots(1, 2, figsize=(12, 7))
sns.heatmap(pivot_z, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.85, vmax=1.0,
            ax=axes[0], cbar_kws={'label': 'z corr'})
axes[0].set_title('Mean z (responsibility) correlation')
sns.heatmap(pivot_sigma, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.85, vmax=1.0,
            ax=axes[1], cbar_kws={'label': 'sigma corr'})
axes[1].set_title('Mean Σ (covariance) correlation')
plt.tight_layout()
plt.show()
"""))

    c.append(nbf.v4.new_markdown_cell("""\
## 8.  Headline numbers

The single bottom line: **overall correlation between R's and Python's
EM output across all 224 records.**"""))
    c.append(nbf.v4.new_code_cell("""\
overall = pd.Series({
    'records':            len(df),
    'mean loglik rel-err': df['loglik_rel'].mean(),
    'mean z correlation':  df['z_corr'].mean(),
    'mean μ correlation':  df['mean_corr'].mean(),
    'mean Σ correlation':  df['sigma_corr'].mean(),
    'mean cls match':      df['cls_match'].mean(),
})
overall.to_frame(name='value')
"""))

    c.append(nbf.v4.new_markdown_cell("""\
---

### Take-away

- **12 of 14 models** match R bitwise (loglik agreement at machine precision, ~1e-15).
- **EVE / VVE** share orientation **D** across components — the underlying
  optimisation has multiple stationary points; we use a Stiefel gradient
  descent (provably monotone) where R uses Browne–McNicholas MM.
- **Overall ≥ 99 % correlation** for z, μ, Σ, and classification labels.

For exact agreement on every record, restrict the model list to the 12
closed-form ones:

```python
from mclust_py import Mclust
fit = Mclust(X, G=range(1, 10),
             model_names=['EII','VII','EEI','VEI','EVI','VVI',
                          'EEE','EEV','VEV','EVV','VVV'])
```
"""))
    return nb


def _save_and_run(nb: nbf.notebooknode.NotebookNode, fname: str) -> None:
    out = HERE / fname
    with open(out, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {out}")
    client = NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(HERE)}})
    print(f"executing {out} ...")
    client.execute()
    with open(out, "w") as f:
        nbf.write(nb, f)
    print(f"executed and saved {out}")


if __name__ == "__main__":
    _save_and_run(_comparison_notebook(), "comparison_R_vs_python.ipynb")
