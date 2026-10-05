"""Supplementary analyses that the figures need and the main sweep does not produce.

  task_weights.csv       the meta-learned auxiliary weights, averaged over folds
  hazard_profile.csv     MOSAIC's discrete-time cumulative risk by predicted-risk group
  conformal_coverage.csv empirical coverage of residual-based bands, by follow-up
  importance.csv         permutation importance of each covariate for MOSAIC
  learning_curve.csv     stratified AUC against the share of the training fold used, with and
                         without auxiliary supervision -- to examine data-efficiency differences
  featureset_raw.csv     the three nested feature sets, for the reference model, the strongest
                         baseline and MOSAIC

Usage: python extras.py [weights|hazard|conformal|importance|curve|featureset|all]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines
import features
import metrics
from experiment import load
from mosaic import GRID
from mosaic_stack import MosaicStack

from config import JD
OUT = f"{JD}/results/experiments"
N_FOLD = 5


def _folds(y, seed=0, k=N_FOLD):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


# ---------------------------------------------------------------------------- task weights
def task_weights(dataset="caries"):
    Xs, y, T, AY, AM, m, aux_cols, _ = load(dataset)
    lams = []
    for tr, te in _folds(y):
        mod = MosaicStack(seed=0).fit(Xs.iloc[tr], y[tr], T[tr], AY[tr], AM[tr])
        lams.append(np.asarray(mod.lam_, float))
    L = np.mean(lams, axis=0)
    sd = np.std(lams, axis=0)
    rows = [dict(task=c, weight=float(L[i + 1]), weight_sd=float(sd[i + 1]),
                 events=int(np.nansum(AY[:, i] == 1)))
            for i, c in enumerate(aux_cols)]
    out = pd.DataFrame(rows).sort_values("weight", ascending=False)
    out.to_csv(f"{OUT}/task_weights.csv", index=False)
    print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------- hazard & conformal
def hazard_and_conformal(dataset="caries"):
    Xs, y, T, AY, AM, m, _, _ = load(dataset)
    folds = _folds(y)
    p_all = np.full(len(y), np.nan)
    width = np.full(len(y), np.nan)
    haz_rows = []
    for tr, te in folds:
        mod = MosaicStack(seed=0).fit(Xs.iloc[tr], y[tr], T[tr], AY[tr], AM[tr])
        p_all[te] = mod.predict_risk(Xs.iloc[te], T[te])
        width[te] = mod.conformal_width(T[te])
        # cumulative risk profile: evaluate each test worker at every grid boundary
        labs = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        # qcut on an ndarray returns a Categorical; comparing it gives a plain ndarray, which
        # has no .values, so keep the group labels as a numpy array of strings
        grp = np.asarray(pd.qcut(mod.net_.linpred(Xs.iloc[te]), 4, labels=labs).astype(str))
        for t_end in GRID[1:]:
            # The curve is drawn from the interval-censored head itself, not the stacked risk.
            # 1 - prod_k (1-h_k)^{o_ik} is monotone in the horizon by construction, whereas the
            # logistic stacker can put a negative weight on a base learner and so is not: the
            # stacked risk can dip as the horizon grows, which is meaningless as a cumulative
            # risk.  This panel is about the time structure the head fits, so the head is what
            # it should show; the stacked model is what every reported metric uses.
            r = mod.net_.predict_risk(Xs.iloc[te], np.full(len(te), t_end))
            for g in labs:
                sel = grp == g
                if sel.sum():
                    haz_rows.append(dict(risk_group=g, t_end=float(t_end),
                                         cum_risk=float(r[sel].mean())))
    hz = (pd.DataFrame(haz_rows).groupby(["risk_group", "t_end"], observed=True)
          .cum_risk.mean().reset_index())
    hz.to_csv(f"{OUT}/hazard_profile.csv", index=False)

    res = np.abs(y - p_all)
    q = np.quantile(T, np.linspace(0, 1, 6))
    q[0], q[-1] = -np.inf, np.inf
    rows = []
    for k in range(5):
        sel = (T > q[k]) & (T <= q[k + 1]) & np.isfinite(width)
        rows.append(dict(quintile=k + 1, n=int(sel.sum()),
                         median_followup=float(np.median(T[sel])),
                         mean_width=float(np.nanmean(width[sel])),
                         coverage=float(np.mean(res[sel] <= width[sel]))))
    cov = pd.DataFrame(rows)
    cov.to_csv(f"{OUT}/conformal_coverage.csv", index=False)
    print(hz.head(8).to_string(index=False))
    print(cov.to_string(index=False))
    return hz, cov


# ------------------------------------------------------------------------------ importance
def importance(dataset="caries", n_repeat=5):
    Xs, y, T, AY, AM, m, _, _ = load(dataset)
    folds = _folds(y)
    rng = np.random.default_rng(0)
    rows = []
    for fi, (tr, te) in enumerate(folds[:3]):
        mod = MosaicStack(seed=0).fit(Xs.iloc[tr], y[tr], T[tr], AY[tr], AM[tr])
        Xte = Xs.iloc[te].reset_index(drop=True)
        base = metrics.strat_auc(mod.predict_risk(Xte, T[te]), y[te], T[te])
        for col in Xs.columns:
            drops = []
            for _ in range(n_repeat):
                Xp = Xte.copy()
                Xp[col] = Xp[col].values[rng.permutation(len(Xp))]
                drops.append(base - metrics.strat_auc(mod.predict_risk(Xp, T[te]),
                                                      y[te], T[te]))
            rows.append(dict(fold=fi, feature=col, drop_strat_auc=float(np.mean(drops))))
        print(f"  fold {fi} done (base stratified AUC {base:.4f})", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/importance.csv", index=False)
    print(out.groupby("feature").drop_strat_auc.mean().sort_values(ascending=False)
          .head(15).to_string())
    return out


# --------------------------------------------------------------------------- learning curve
def learning_curve(dataset="caries", fracs=(0.15, 0.3, 0.5, 0.75, 1.0)):
    Xs, y, T, AY, AM, m, _, _ = load(dataset)
    folds = _folds(y)
    rng = np.random.default_rng(0)
    rows = []
    variants = {"mosaic": dict(), "A1_no_aux": dict(aux=False, use_aux_features=False)}
    for frac in fracs:
        for fi, (tr, te) in enumerate(folds[:3]):
            # subsample the training fold, keeping the event rate
            pos = tr[y[tr] == 1]
            neg = tr[y[tr] == 0]
            sub = np.concatenate([rng.permutation(pos)[:max(10, int(frac * len(pos)))],
                                  rng.permutation(neg)[:max(50, int(frac * len(neg)))]])
            for name, kw in variants.items():
                mod = MosaicStack(seed=0, **kw).fit(Xs.iloc[sub], y[sub], T[sub],
                                                    AY[sub], AM[sub])
                p = mod.predict_risk(Xs.iloc[te], T[te])
                rows.append(dict(model=name, frac=frac, fold=fi, n_train=len(sub),
                                 events_train=int(y[sub].sum()),
                                 strat_auc=metrics.strat_auc(p, y[te], T[te])))
            print(f"  frac {frac} fold {fi} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/learning_curve.csv", index=False)
    print(out.groupby(["model", "frac"]).strat_auc.mean().unstack().round(4).to_string())
    return out


# ------------------------------------------------------------------------------ featuresets
def featureset(dataset="caries"):
    rows = []
    for fs in ("paper", "base", "traj"):
        Xs, y, T, AY, AM, m, _, _ = load(dataset, fs)
        folds = _folds(y)
        for name, M, needs_aux in [("cox_multivar", baselines.MultivarCox, False),
                                   ("lightgbm", baselines.GBM, False),
                                   ("mosaic", MosaicStack, True)]:
            for fi, (tr, te) in enumerate(folds):
                mod = M()
                if needs_aux:
                    mod.fit(Xs.iloc[tr], y[tr], T[tr], AY[tr], AM[tr])
                else:
                    mod.fit(Xs.iloc[tr], y[tr], T[tr])
                p = mod.predict_risk(Xs.iloc[te], T[te])
                rows.append(dict(featureset=fs, model=name, fold=fi,
                                 n_features=Xs.shape[1],
                                 strat_auc=metrics.strat_auc(p, y[te], T[te]),
                                 **{k: v for k, v in
                                    metrics.evaluate(p, y[te], T[te]).items()
                                    if k in ("auc", "ap_lift", "ipa", "ece")}))
            print(f"  {fs}/{name} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/featureset_raw.csv", index=False)
    print(out.groupby(["featureset", "model"]).strat_auc.mean().unstack().round(4).to_string())
    return out


def no_year(dataset="caries"):
    """Does the calendar-year covariate itself drive the era-transfer ranking?

    `base_year` is a legitimate covariate -- a deployed service knows when it examined someone
    -- and it is given to every model equally.  But under leave-one-year-out its test value is
    never seen in training, so a reviewer is right to ask whether the transfer comparison is
    really about the models or about how each one extrapolates an unseen level.  This refits
    the three headline methods under both protocols with the column removed.
    """
    rows = []
    for drop in (False, True):
        Xs, y, T, AY, AM, m, _, _ = load(dataset)
        if drop and "base_year" in Xs.columns:
            Xs = Xs.drop(columns=["base_year"])
        G = m.base_year.values
        for proto in ("cv", "loyo"):
            fl = list(_folds(y)) if proto == "cv" else [
                (np.where(m.base_year.values != yv)[0], np.where(m.base_year.values == yv)[0])
                for yv in sorted(set(m.base_year.dropna()))
                if (m.base_year.values == yv).sum() >= 100 and y[m.base_year.values == yv].sum() >= 5]
            for name, M, needs_aux in [("cox_multivar", baselines.MultivarCox, False),
                                       ("lightgbm", baselines.GBM, False),
                                       ("mosaic", MosaicStack, True)]:
                for fi, (tr, te) in enumerate(fl):
                    mod = M()
                    if needs_aux:
                        mod.fit(Xs.iloc[tr], y[tr], T[tr], AY[tr], AM[tr], G[tr])
                    else:
                        mod.fit(Xs.iloc[tr], y[tr], T[tr])
                    p = mod.predict_risk(Xs.iloc[te], T[te])
                    rows.append(dict(drop_base_year=drop, protocol=proto, model=name, fold=fi,
                                     strat_auc=metrics.strat_auc(p, y[te], T[te])))
                print(f"  drop={drop} {proto}/{name} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/no_year.csv", index=False)
    print(out.groupby(["protocol", "model", "drop_base_year"]).strat_auc.mean()
          .unstack().round(4).to_string())
    return out


FUNCS = {"weights": task_weights, "noyear": no_year, "hazard": hazard_and_conformal, "importance": importance,
         "curve": learning_curve, "featureset": featureset}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, fn in FUNCS.items():
        if what in ("all", k):
            print(f"=== {k}", flush=True)
            fn()
