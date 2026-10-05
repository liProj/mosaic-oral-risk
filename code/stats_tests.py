"""Paired corrected resampled t-tests and Holm adjustment.

The estimated variance of a paired mean difference is s^2 * (1/n + rho),
where rho is mean test size / mean training size. Training sets overlap in
both repeated cross-validation and leave-one-year-out validation. The correction
is approximate, especially when only a few year-held-out folds are available.
"""
import numpy as np
import pandas as pd
from scipy import stats

from metrics import HIGHER_BETTER, TARGET


def nadeau_bengio(diff, n_train, n_test):
    """Corrected resampled t-test on paired per-fold differences."""
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = diff.mean()
    var = diff.var(ddof=1)
    if var <= 0:
        return mean, np.inf * np.sign(mean) if mean else 0.0, 0.0 if mean else 1.0
    rho = float(np.mean(n_test) / np.mean(n_train))
    se = np.sqrt(var * (1.0 / n + rho))
    t = mean / se
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return mean, t, p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    run = 0.0
    for i, j in enumerate(order):
        adj = (m - i) * p[j]
        run = max(run, adj)
        out[j] = min(1.0, run)
    return out


def signed_improvement(metric, ours, theirs):
    """Per-fold improvement, oriented so that positive always means 'ours is better'."""
    if metric in TARGET:
        tgt = TARGET[metric]
        return np.abs(theirs - tgt) - np.abs(ours - tgt)
    return (ours - theirs) if HIGHER_BETTER.get(metric, True) else (theirs - ours)


METRICS = ["strat_auc", "auc", "ap", "ap_lift", "ipa", "brier", "loglik", "r2_ll",
           "c_index", "nb_1pct", "nb_3pct", "nb_5pct", "cal_slope", "cal_intercept",
           "oe_ratio", "ece"]


def compare(raw, ours="mosaic", reference="cox_multivar", metrics_=None, protocols=None):
    """Paired comparison of `ours` vs `reference` for every metric x dataset x protocol."""
    metrics_ = metrics_ or METRICS
    rows = []
    for (ds, proto), g in raw.groupby(["dataset", "protocol"]):
        if protocols and proto not in protocols:
            continue
        a = g[g.model == ours].set_index("fold")
        b = g[g.model == reference].set_index("fold")
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        a, b = a.loc[common], b.loc[common]
        # the Nadeau-Bengio correction needs the per-fold train/test sizes; the training set is
        # whatever is left of the subcohort after the test fold is removed
        n_test = a.n_test.values
        n_train = np.maximum(1, _subcohort_n(ds) - n_test)
        for mt in metrics_:
            if mt not in a.columns:
                continue
            d = signed_improvement(mt, a[mt].values, b[mt].values)
            mean, t, p = nadeau_bengio(d, n_train, n_test)
            rows.append(dict(dataset=ds, protocol=proto, metric=mt,
                             ours=float(np.nanmean(a[mt].values)),
                             reference=float(np.nanmean(b[mt].values)),
                             ours_sd=float(np.nanstd(a[mt].values, ddof=1)),
                             reference_sd=float(np.nanstd(b[mt].values, ddof=1)),
                             improvement=mean, t=t, p=p, n_folds=len(common),
                             wins=int(np.nansum(d > 0)), losses=int(np.nansum(d < 0))))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p.values)
        out["verdict"] = np.where(out.improvement > 0,
                                  np.where(out.p_holm < 0.05, "better (sig)", "better"),
                                  np.where(out.p_holm < 0.05, "worse (sig)", "worse"))
    return out


_SUB_N = {"caries": 11888, "gingivitis": 11984, "combined": 11851}


def _subcohort_n(ds):
    return _SUB_N.get(ds, 12049)


if __name__ == "__main__":
    import sys
    from config import JD
    raw = pd.read_csv(f"{JD}/results/experiments/main_raw.csv")
    ref = sys.argv[1] if len(sys.argv) > 1 else "cox_multivar"
    out = compare(raw, "mosaic", ref)
    out.to_csv(f"{JD}/results/experiments/tests_vs_{ref}.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwins {int((out.improvement > 0).sum())}/{len(out)}; "
          f"significant wins {int(((out.improvement > 0) & (out.p_holm < 0.05)).sum())}, "
          f"significant losses {int(((out.improvement < 0) & (out.p_holm < 0.05)).sum())}")
