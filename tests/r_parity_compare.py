"""Compare every R-mclust me() record against mclust-py me() output.

Usage
-----
    python tests/r_parity_compare.py                   # summary only
    python tests/r_parity_compare.py --verbose         # per-record details
    python tests/r_parity_compare.py --models VVV VEV  # filter by model

Requires the JSON dump produced by ``Rscript tests/r_parity_dump.R``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DUMP = HERE / "_rparity"

# Allow running as a script or via pytest
sys.path.insert(0, str(HERE.parent))
from mclust_py import me


def _load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)


def _reshape_F(values, dim) -> np.ndarray:
    """R is column-major; rebuild the original shape using Fortran order.

    Coerces R's "NA" strings (jsonlite emits these as bare strings even
    inside numeric arrays) to NaN before stacking.
    """
    arr = np.array(
        [np.nan if (isinstance(v, str) and v.upper() == "NA") else v for v in values],
        dtype=np.float64,
    )
    return arr.reshape(tuple(dim), order="F")


def load_records():
    keys = sorted(DUMP.glob("*_G[0-9].json"))
    return [json.loads(k.read_text()) for k in keys]


def load_data(name: str) -> np.ndarray:
    return _load_csv(DUMP / f"data_{name}_X.csv")


def load_zinit(name: str, G: int) -> np.ndarray:
    return _load_csv(DUMP / f"zinit_{name}_G{G}.csv")


def _aligned_classification_match(cls_a, cls_b, G: int) -> float:
    """Best label-permutation accuracy — clusters are unordered."""
    from itertools import permutations

    cls_a = np.asarray(cls_a)
    cls_b = np.asarray(cls_b)
    best = 0.0
    for perm in permutations(range(1, G + 1)):
        relabel = np.array([0] + list(perm))
        new_b = relabel[cls_b]
        acc = float(np.mean(new_b == cls_a))
        if acc > best:
            best = acc
    return best


def compare_one(rec, *, verbose=False) -> dict:
    name = rec["dataset"]
    mname = rec["modelName"]
    G = int(rec["G"])
    X = load_data(name)
    z_init = load_zinit(name, G)
    out = me(X, mname, z_init)
    py_loglik = out.loglik

    def _num(x):
        if x is None or (isinstance(x, str) and x.upper() == "NA"):
            return float("nan")
        return float(x)

    r_loglik = _num(rec.get("loglik"))

    def _arr(values):
        return np.array(
            [np.nan if (isinstance(v, str) and v.upper() == "NA") else v for v in values],
            dtype=np.float64,
        )
    # R z is (n*G,) flattened col-major
    r_z = _reshape_F(rec["z"], rec["z_dim"])
    r_pro = _arr(rec["pro"])
    r_mean = _reshape_F(rec["mean"], rec["mean_dim"])
    r_sigma = (
        _reshape_F(rec["sigma"], rec["sigma_dim"]) if rec.get("sigma") is not None else None
    )
    if not np.all(np.isfinite(r_z)) or not np.all(np.isfinite(r_mean)):
        # R's me() returned NA — usually a singular-cov failure.  Skip.
        return {
            "dataset": name, "modelName": mname, "G": G,
            "n": rec["n"], "d": rec["d"],
            "r_loglik": r_loglik, "py_loglik": py_loglik,
            "loglik_rel_err": np.nan, "cls_match": np.nan,
            "z_corr": np.nan, "mean_corr": np.nan, "sigma_corr": np.nan,
            "iters": out.iterations, "converged": out.converged,
            "r_failed": True,
        }

    # log-likelihood
    if not np.isfinite(r_loglik) or not np.isfinite(py_loglik):
        loglik_rel = np.nan
    else:
        loglik_rel = abs(py_loglik - r_loglik) / max(abs(r_loglik), 1e-12)

    # classification
    py_cls = np.argmax(out.z, axis=1) + 1
    r_cls = np.argmax(r_z, axis=1) + 1
    cls_match = _aligned_classification_match(r_cls, py_cls, G)

    # z correlation per row → average Pearson
    # First permute py columns to best match R columns (via Hungarian on dot product).
    try:
        from scipy.optimize import linear_sum_assignment

        sim = np.zeros((G, G))
        for i in range(G):
            for j in range(G):
                # use cosine similarity; clamp NaN to 0
                a = r_z[:, i]
                b = out.z[:, j]
                num = float(np.sum(a * b))
                den = float(np.linalg.norm(a) * np.linalg.norm(b))
                sim[i, j] = num / max(den, 1e-12)
        ri, ci = linear_sum_assignment(-sim)
        py_z_perm = out.z[:, ci]
        py_mean_perm = out.mean[:, ci]
        py_sigma_perm = out.sigma[:, :, ci]
        py_pro_perm = out.pro[ci]
    except ImportError:
        py_z_perm = out.z
        py_mean_perm = out.mean
        py_sigma_perm = out.sigma
        py_pro_perm = out.pro

    z_corr = float(
        np.corrcoef(r_z.ravel(), py_z_perm.ravel())[0, 1]
    )
    mean_corr = float(np.corrcoef(r_mean.ravel(), py_mean_perm.ravel())[0, 1])
    sigma_corr = (
        float(np.corrcoef(r_sigma.ravel(), py_sigma_perm.ravel())[0, 1])
        if r_sigma is not None
        else float("nan")
    )

    info = {
        "dataset": name,
        "modelName": mname,
        "G": G,
        "n": rec["n"],
        "d": rec["d"],
        "r_loglik": r_loglik,
        "py_loglik": py_loglik,
        "loglik_rel_err": loglik_rel,
        "cls_match": cls_match,
        "z_corr": z_corr,
        "mean_corr": mean_corr,
        "sigma_corr": sigma_corr,
        "iters": out.iterations,
        "converged": out.converged,
    }
    if verbose:
        print(
            f"  {mname:>3}-G{G}  loglik R={r_loglik:.4f} py={py_loglik:.4f} "
            f"|Δ|/|·|={loglik_rel:.2e}  cls_match={cls_match:.3f} "
            f"z_corr={z_corr:.4f} mean_corr={mean_corr:.4f} "
            f"sigma_corr={sigma_corr:.4f} iters={out.iterations}"
        )
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()
    records = load_records()
    if args.models:
        records = [r for r in records if r["modelName"] in args.models]
    if args.datasets:
        records = [r for r in records if r["dataset"] in args.datasets]
    summary = []
    by_model: dict[str, list[dict]] = {}
    by_dataset: dict[str, list[dict]] = {}
    for rec in records:
        if rec.get("error"):
            continue
        info = compare_one(rec, verbose=args.verbose)
        summary.append(info)
        by_model.setdefault(info["modelName"], []).append(info)
        by_dataset.setdefault(info["dataset"], []).append(info)

    print("\n=== per-model summary (mean over (dataset, G)) ===")
    print(f"{'model':>5}  {'n':>3}  {'loglik_rel':>12}  {'cls_match':>10}  "
          f"{'z_corr':>8}  {'mean_corr':>10}  {'sigma_corr':>11}")
    for m, infos in sorted(by_model.items()):
        n = len(infos)
        loglik = np.nanmean([i["loglik_rel_err"] for i in infos])
        cls = np.nanmean([i["cls_match"] for i in infos])
        zc = np.nanmean([i["z_corr"] for i in infos])
        mc = np.nanmean([i["mean_corr"] for i in infos])
        sc = np.nanmean([i["sigma_corr"] for i in infos])
        print(f"{m:>5}  {n:>3}  {loglik:>12.2e}  {cls:>10.4f}  "
              f"{zc:>8.4f}  {mc:>10.4f}  {sc:>11.4f}")

    print("\n=== per-dataset summary ===")
    for ds, infos in sorted(by_dataset.items()):
        n = len(infos)
        loglik = np.nanmean([i["loglik_rel_err"] for i in infos])
        zc = np.nanmean([i["z_corr"] for i in infos])
        mc = np.nanmean([i["mean_corr"] for i in infos])
        sc = np.nanmean([i["sigma_corr"] for i in infos])
        print(f"  {ds:>10}  n={n:>3}  loglik_rel={loglik:.2e}  "
              f"z_corr={zc:.4f}  mean_corr={mc:.4f}  sigma_corr={sc:.4f}")

    # final pass/fail thresholds
    overall_z = np.nanmean([i["z_corr"] for i in summary])
    overall_mean = np.nanmean([i["mean_corr"] for i in summary])
    overall_sigma = np.nanmean([i["sigma_corr"] for i in summary])
    overall_cls = np.nanmean([i["cls_match"] for i in summary])
    print(
        f"\n=== overall ({len(summary)} records) ===\n"
        f"  z_corr     = {overall_z:.4f}\n"
        f"  mean_corr  = {overall_mean:.4f}\n"
        f"  sigma_corr = {overall_sigma:.4f}\n"
        f"  cls_match  = {overall_cls:.4f}"
    )


if __name__ == "__main__":
    main()
