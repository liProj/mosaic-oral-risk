"""Evaluate predicted probabilities for single-window outcomes.

Follow-up-stratified AUC pools within-quintile rankings by pair count, reducing
between-window ranking effects while retaining within-stratum duration variation.
The function breslow_H0 is an empirical time-scale construction, not a standard
event-time Breslow estimate. Harrell concordance on window endpoints is retained
only as a descriptive comparator metric, not onset-time concordance.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score


def risk_from_linpred(eta, T, H0):
    """Window risk of a proportional-hazards model: 1 - exp(-H0(T) * exp(eta))."""
    return 1.0 - np.exp(-np.asarray(H0(T)) * np.exp(np.asarray(eta)))


def breslow_H0(T_tr, y_tr, eta_tr):
    """Event-count-matched scale with monotone decile adjustment.

    The historical function name is retained for compatibility. Exact event
    times are unavailable; this is not an event-time Breslow estimator.
    """
    T_tr = np.asarray(T_tr, float)
    y_tr = np.asarray(y_tr, float)
    w = np.exp(np.asarray(eta_tr, float))
    # scale: solve sum(1 - exp(-lam*T*w)) = sum(y) by bisection (monotone in lam)
    lo, hi = 1e-8, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.sum(1.0 - np.exp(-mid * T_tr * w)) < y_tr.sum():
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)

    # shape: per-decile multiplicative correction, isotonised so H0 stays non-decreasing
    qs = np.unique(np.quantile(T_tr, np.linspace(0, 1, 11)))
    if len(qs) < 3:
        return lambda t: lam * np.asarray(t, float)
    idx = np.clip(np.searchsorted(qs, T_tr, side="right") - 1, 0, len(qs) - 2)
    grid, cum = [0.0], [0.0]
    for k in range(len(qs) - 1):
        m = idx == k
        width = qs[k + 1] - qs[k]
        if m.sum() < 25 or width <= 0:
            inc = lam * width
        else:
            # local rate so that expected == observed inside the decile
            obs = y_tr[m].sum()
            exp_ = np.sum(1.0 - np.exp(-lam * T_tr[m] * w[m]))
            inc = lam * width * (obs + 0.5) / (exp_ + 0.5)
        grid.append(qs[k + 1])
        cum.append(cum[-1] + max(inc, 1e-12))
    grid, cum = np.array(grid), np.array(cum)

    def H0(t):
        t = np.asarray(t, float)
        inside = np.interp(np.clip(t, 0.0, grid[-1]), grid, cum)
        # beyond the training range, extend at the global rate rather than flattening
        return inside + np.where(t > grid[-1], lam * (t - grid[-1]), 0.0)

    return H0


def harrell_c(T, y, risk):
    """Concordance of `risk` against (T, y) with right-censoring semantics.

    Reported for comparability with the reference Cox analysis only; see the module docstring.
    """
    T = np.asarray(T, float)
    y = np.asarray(y, int)
    r = np.asarray(risk, float)
    order = np.argsort(T)
    T, y, r = T[order], y[order], r[order]
    conc = disc = tied = 0.0
    ev = np.where(y == 1)[0]
    for i in ev:
        later = T > T[i]
        n = later.sum()
        if n == 0:
            continue
        conc += np.sum(r[later] < r[i])
        disc += np.sum(r[later] > r[i])
        tied += np.sum(r[later] == r[i])
    tot = conc + disc + tied
    return float((conc + 0.5 * tied) / tot) if tot else np.nan


def strat_auc(p, y, T, n_strata=5):
    """AUC computed inside quintiles of follow-up length, pooled by pair count."""
    p, y, T = np.asarray(p, float), np.asarray(y, int), np.asarray(T, float)
    qs = np.quantile(T, np.linspace(0, 1, n_strata + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    num = den = 0.0
    for k in range(n_strata):
        m = (T > qs[k]) & (T <= qs[k + 1])
        if m.sum() < 10 or y[m].sum() == 0 or y[m].sum() == m.sum():
            continue
        npairs = y[m].sum() * (m.sum() - y[m].sum())
        num += npairs * roc_auc_score(y[m], p[m])
        den += npairs
    return float(num / den) if den else np.nan


def calibration(p, y):
    """Calibration-in-the-large / slope on the logit scale, plus O/E and ECE."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    lp = logit(p)
    import statsmodels.api as sm
    try:
        res = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        inter, slope = float(res.params[0]), float(res.params[1])
    except Exception:
        inter, slope = np.nan, np.nan
    # ECE over deciles of predicted risk
    q = np.quantile(p, np.linspace(0, 1, 11))
    q[0], q[-1] = -np.inf, np.inf
    ece = 0.0
    for k in range(10):
        m = (p > q[k]) & (p <= q[k + 1])
        if m.sum():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return dict(cal_intercept=inter, cal_slope=slope,
                oe_ratio=float(y.sum() / max(p.sum(), 1e-9)), ece=float(ece))


def net_benefit(p, y, threshold):
    """Decision-curve net benefit at a risk threshold (per worker screened)."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    flag = p >= threshold
    n = len(y)
    tp = float(np.sum(flag & (y == 1)))
    fp = float(np.sum(flag & (y == 0)))
    w = threshold / (1.0 - threshold)
    return (tp - w * fp) / n


def evaluate(p, y, T, eta=None, prefix=""):
    """All metrics for one model on one fold."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, int)
    T = np.asarray(T, float)
    if eta is None:
        eta = logit(p)
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    brier0 = float(np.mean((base - y) ** 2))
    # binomial log-likelihood, and its scaled version against the no-covariate model
    ll = float(np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))
    ll0 = float(base * np.log(base) + (1 - base) * np.log1p(-base))
    out = {
        "auc": float(roc_auc_score(y, p)),
        "strat_auc": strat_auc(p, y, T),
        "ap": float(average_precision_score(y, p)),
        "ap_lift": float(average_precision_score(y, p) / base),
        "brier": brier,
        "ipa": float(1.0 - brier / brier0),
        "loglik": ll,
        "r2_ll": float(1.0 - ll / ll0),
        "c_index": harrell_c(T, y, eta),
        "nb_1pct": net_benefit(p, y, 0.01),
        "nb_3pct": net_benefit(p, y, 0.03),
        "nb_5pct": net_benefit(p, y, 0.05),
    }
    out.update(calibration(p, y))
    return {f"{prefix}{k}": v for k, v in out.items()}


HIGHER_BETTER = {"auc": True, "strat_auc": True, "ap": True, "ap_lift": True, "brier": False,
                 "ipa": True, "loglik": True, "r2_ll": True, "c_index": True,
                 "nb_1pct": True, "nb_3pct": True, "nb_5pct": True, "ece": False}
# calibration slope / intercept / O-E are "closer to target is better", handled separately
TARGET = {"cal_slope": 1.0, "cal_intercept": 0.0, "oe_ratio": 1.0}
