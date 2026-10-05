"""Evaluate MOSAIC and baselines across three oral outcomes.

cv: repeated stratified five-fold validation (three repeats).
loyo: leave one baseline year out; training may include later years.
split: train on baseline years <= 2017 and test on years >= 2018.

Usage: python experiment.py [main|ablation|all] [base|paper|traj]
The main analysis uses base; ablations use caries under cv and loyo.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines
import features
import metrics
from mosaic_stack import ICBoost, MosaicStack

from config import JD
OUT = f"{JD}/results/experiments"

DATASETS = {"caries": ("free_caries", "ev_caries", "caries_incidencia"),
            "gingivitis": ("free_ging", "ev_ging", "gingivitis_incidencia"),
            "combined": ("free_both", "ev_comb", None)}

N_REPEAT, N_FOLD = 3, 5


def load(dataset, featureset="base"):
    X, meta, blocks = features.build(featureset)
    free, ev, own = DATASETS[dataset]
    sub = (meta[free] == 1).values
    Xs = X[sub].reset_index(drop=True)
    m = meta[sub].reset_index(drop=True)
    y = m[ev].values.astype(int)
    T = m.py.values.astype(float)
    # auxiliary tasks: every co-recorded incident outcome except the primary's own components
    drop = {"caries_incidencia", "gingivitis_incidencia"} if dataset == "combined" else {own}
    aux_cols = [c for c in meta.columns
                if c.endswith(("_incidencia", "_final")) and c not in drop]
    AY = m[aux_cols].values.astype(float)
    AM = (~np.isnan(AY)).astype(float)
    AY = np.nan_to_num(AY)
    return Xs, y, T, AY, AM, m, aux_cols, blocks


def folds(protocol, y, m, seed=0):
    """Yield (name, train_idx, test_idx)."""
    if protocol == "cv":
        for r in range(N_REPEAT):
            skf = StratifiedKFold(N_FOLD, shuffle=True, random_state=1000 + r)
            for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
                yield f"r{r}f{k}", tr, te
    elif protocol == "loyo":
        yr = m.base_year.values
        for year in sorted(set(yr[~pd.isna(yr)])):
            te = np.where(yr == year)[0]
            tr = np.where(yr != year)[0]
            if len(te) < 100 or y[te].sum() < 5:
                continue
            yield f"y{int(year)}", tr, te
    elif protocol == "split":
        yr = m.base_year.values
        tr = np.where(yr <= 2017)[0]
        te = np.where(yr >= 2018)[0]
        yield "le2017_ge2018", tr, te
    else:
        raise ValueError(protocol)


def _mosaic_variants():
    """The proposed method plus its ablation grid: one entry per design decision."""
    return {
        "MOSAIC": dict(),
        # --- multi-task supervision and how it is weighted ---
        "A1_no_aux": dict(aux=False, use_aux_features=False),
        "A2_aux_equal_weights": dict(equal_task_weights=True),
        "A3_aux_budget_equal": dict(meta_weights=False),
        "A4_aux_uncertainty_weights": dict(meta_weights=False, uncertainty_weights=True),
        # --- the likelihood ---
        "A5_binary_head": dict(head="binary"),
        "A6_cox_head": dict(head="cox"),
        # --- the trunk ---
        "A7_mlp_trunk": dict(trunk="mlp"),
        "A8_attn_trunk": dict(trunk="attn"),
        # --- regularisation, ensembling, distillation ---
        "A9_no_ema": dict(ema=False),
        "A10_single_model": dict(n_models=1, distil=False),
        "A11_no_distil": dict(distil=False),
        # --- the second base learner and the stack ---
        "A12_no_boost": dict(use_boost=False),
        "A13_no_aux_features": dict(use_aux_features=False),
        "A14_no_stack": dict(stack=False),
        "A15_no_calibration": dict(calibrate=False),
        # --- era invariance ---
        "A16_no_rex": dict(rex_weight=0.0),
    }


def run_one(model_fn, Xs, y, T, AY, AM, tr, te, needs_aux, groups=None):
    mod = model_fn()
    if needs_aux:
        mod.fit(Xs.iloc[tr], y[tr], T[tr], AY[tr], AM[tr],
                None if groups is None else groups[tr])
    else:
        mod.fit(Xs.iloc[tr], y[tr], T[tr])
    p = mod.predict_risk(Xs.iloc[te], T[te])
    eta = mod.linpred(Xs.iloc[te])
    return p, eta, mod


def main(which="main", datasets=tuple(DATASETS), protocols=("cv", "loyo", "split"),
         featureset="base"):
    os.makedirs(OUT, exist_ok=True)
    rows, oof = [], []
    if which == "main":
        models = {M.name: (M, False) for M in baselines.ALL}
        models["icboost"] = (ICBoost, False)
        models["mosaic"] = (MosaicStack, True)
    else:
        models = {k: ((lambda kw=v: MosaicStack(**kw)), True)
                  for k, v in _mosaic_variants().items()}

    for ds in datasets:
        Xs, y, T, AY, AM, m, aux_cols, blocks = load(ds, featureset)
        print(f"\n=== {ds}: n={len(y)} events={y.sum()} "
              f"({100 * y.mean():.2f}%) feats={Xs.shape[1]} aux={len(aux_cols)}", flush=True)
        for proto in protocols:
            fl = list(folds(proto, y, m))
            for name, (M, needs_aux) in models.items():
                t0 = time.time()
                for fname, tr, te in fl:
                    try:
                        p, eta, mod = run_one(M, Xs, y, T, AY, AM, tr, te, needs_aux,
                                             groups=m.base_year.values)
                    except Exception as e:
                        print(f"  !! {name} {proto}/{fname} failed: {type(e).__name__}: {e}",
                              flush=True)
                        continue
                    r = metrics.evaluate(p, y[te], T[te], eta=eta)
                    r.update(model=name, dataset=ds, protocol=proto, fold=fname,
                             n_test=len(te), events_test=int(y[te].sum()))
                    # keep the per-worker predictions: the ROC, precision-recall, calibration
                    # and decision-curve figures are all built from them, and recomputing them
                    # later would mean retraining everything
                    oof.append(pd.DataFrame(dict(
                        model=name, dataset=ds, protocol=proto, fold=fname,
                        wid=m.wid.values[te], y=y[te], T=T[te], p=p)))
                    lam = np.atleast_1d(np.asarray(getattr(mod, "lam_", np.nan), float))
                    r["aux_weight_mean"] = (float(np.mean(lam[1:])) if lam.size > 1
                                            else float("nan"))
                    rows.append(r)
                done = [x for x in rows if x["model"] == name and x["protocol"] == proto
                        and x["dataset"] == ds]
                if done:
                    sa = np.mean([x["strat_auc"] for x in done])
                    print(f"  {proto:6s} {name:24s} strat_auc {sa:.4f}  "
                          f"({time.time() - t0:.0f}s, {len(done)} folds)", flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/{which}_raw.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/{which}_raw.csv", index=False)
    if oof:
        o = pd.concat(oof, ignore_index=True)
        o.to_csv(f"{OUT}/{'oof_predictions' if which == 'main' else which + '_oof'}.csv",
                 index=False)
        print(f"wrote per-worker predictions ({len(o)} rows)")
    print(f"\nwrote {OUT}/{which}_raw.csv  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "main"
    fs = sys.argv[2] if len(sys.argv) > 2 else "base"
    # The ablation grid is run on the primary outcome only, and only under the two protocols
    # that carry a claim: within-era cross-validation and transfer to an unseen recording era.
    # Seventeen variants across three datasets and three protocols would be about four times
    # the compute for no extra conclusion.
    ABL = dict(datasets=("caries",), protocols=("cv", "loyo"))
    if what == "all":
        main("main", featureset=fs)
        main("ablation", featureset=fs, **ABL)
    elif what == "ablation":
        main("ablation", featureset=fs, **ABL)
    else:
        main(what, featureset=fs)
