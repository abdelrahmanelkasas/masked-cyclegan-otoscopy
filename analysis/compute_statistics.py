"""
Recompute every statistic reported in Tables 5-6 and Supplementary Table S2
from the raw per-fold result files (no GPU needed).

  * fold mean +/- sample SD (accuracy, macro-F1)
  * pooled out-of-fold accuracy with Wilson 95% CI
  * exact two-sided Wilcoxon signed-rank test on fold accuracies (zeros discarded)
  * cluster-robust 95% CI treating each test fold as the sampling unit
    (ratio estimator, sandwich variance, t with k-1 df; logit scale for accuracy)

Usage:  python analysis/compute_statistics.py
Output: results/statistics_summary.json  (+ printed tables)
"""
import os, json, itertools
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
SOURCES = {"chile": ["study_chile_perfold.json", "study_chile_perfold_efficientnet_b0-vit_b_16.json"],
           "ohio": ["study_ohio_perfold_resnet50-efficientnet_b0-vit_b_16.json"]}
# test-fold sizes produced by classification/grouped_split.grouped_kfold(seed=1234)
FOLD_N = {"chile": [139, 193, 148, 202, 198], "ohio": [91, 92, 91, 91, 89]}
CONDS = ("original", "masked", "ensemble")


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def exact_wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b); d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 1.0
    r = stats.rankdata(np.abs(d)); t_obs = r[d > 0].sum(); mu = r.sum() / 2
    hits = sum(abs(sum(r[i] for i in range(len(d)) if s[i]) - mu) >= abs(t_obs - mu) - 1e-9
               for s in itertools.product([0, 1], repeat=len(d)))
    return hits / 2 ** len(d)


def cluster_robust(k, n):
    """k, n: per-fold correct counts and sizes. Returns pooled p, CI (logit), design effect."""
    G = len(n); T = n.sum(); p = k.sum() / T; t = stats.t.ppf(0.975, G - 1)
    se = np.sqrt(G / (G - 1) * np.sum((k - p * n) ** 2)) / T
    sl = se / (p * (1 - p)); lg = np.log(p / (1 - p)); ilg = lambda x: 1 / (1 + np.exp(-x))
    return p, ilg(lg - t * sl), ilg(lg + t * sl), se ** 2 / (p * (1 - p) / T)


def diff_robust(kd, n):
    G = len(n); T = n.sum(); D = kd.sum() / T; t = stats.t.ppf(0.975, G - 1)
    se = np.sqrt(G / (G - 1) * np.sum((kd - D * n) ** 2)) / T
    return D, D - t * se, D + t * se


def main():
    out = {}
    for ds, files in SOURCES.items():
        res = {}
        for f in files:
            res.update(json.load(open(os.path.join(R, f)))["results"])
        n = np.array(FOLD_N[ds]); out[ds] = {}
        for m, r in res.items():
            k = {}; row = {}
            for c in CONDS:
                acc = np.array(r[c]["acc_folds"]); kc = np.round(acc * n)
                assert np.allclose(kc / n, acc), f"fold sizes do not match {ds}/{m}/{c}"
                k[c] = kc; lo, hi = wilson(kc.sum(), n.sum())
                p, clo, chi, deff = cluster_robust(kc, n)
                row[c] = {"mean": 100 * acc.mean(), "sd": 100 * acc.std(ddof=1),
                          "f1_mean": 100 * r[c]["f1_mean"], "f1_sd": 100 * r[c]["f1_std"] * np.sqrt(5 / 4),
                          "pooled": 100 * p, "wilson": [100 * lo, 100 * hi],
                          "cluster_robust": [100 * clo, 100 * chi], "design_effect": deff,
                          "folds": (100 * acc).round(1).tolist()}
            for c in ("masked", "ensemble"):
                D, dlo, dhi = diff_robust(k[c] - k["original"], n)
                row[c]["vs_original"] = {
                    "fold_mean_diff": 100 * float(np.mean(np.array(r[c]["acc_folds"]) - np.array(r["original"]["acc_folds"]))),
                    "wilcoxon_exact_p": exact_wilcoxon(r[c]["acc_folds"], r["original"]["acc_folds"]),
                    "pooled_diff": 100 * D, "pooled_diff_cluster_robust_ci": [100 * dlo, 100 * dhi]}
            out[ds][m] = row
            print(f"\n{ds} / {m}")
            for c in CONDS:
                x = row[c]; extra = ""
                if c != "original":
                    v = x["vs_original"]; extra = f"  d={v['fold_mean_diff']:+.1f}  p={v['wilcoxon_exact_p']:.3f}"
                print(f"  {c:9s} {x['mean']:.1f}±{x['sd']:.1f}  pooled {x['pooled']:.1f} "
                      f"({x['wilson'][0]:.1f}-{x['wilson'][1]:.1f})  F1 {x['f1_mean']:.1f}±{x['f1_sd']:.1f}{extra}")
    json.dump(out, open(os.path.join(R, "statistics_summary.json"), "w"), indent=1, default=float)
    print("\n[saved] results/statistics_summary.json")


if __name__ == "__main__":
    main()
