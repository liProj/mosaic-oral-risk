"""Combine the MOSAIC neural head, LightGBM, and ICBoost through internal fusion.

Auxiliary occurrence flags and terminal states supervise neural representations.
Their predicted probabilities enter the tree branches. Auxiliary and tree
cross-fitting use different partitions; see docs/method_notes.md for internal
validation dependencies. Outer test subjects remain isolated.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import icboost
from metrics import breslow_H0
from mosaic import Mosaic


class ICBoost:
    """XGBoost using window-likelihood gradients and Fisher-scoring curvature.

    A capped leaf update stabilizes scoring steps. Early stopping and subsequent
    logistic calibration use the same internal validation split.
    """
    name = "icboost"

    def __init__(self, n_estimators=800, learning_rate=0.05, max_depth=4,
                 min_child_weight=1.0, subsample=0.8, colsample=0.8, reg_lambda=5.0,
                 max_delta_step=0.7, val_frac=0.2, seed=0):
        self.params = dict(eta=learning_rate, max_depth=max_depth,
                           min_child_weight=min_child_weight, subsample=subsample,
                           colsample_bytree=colsample, reg_lambda=reg_lambda,
                           max_delta_step=max_delta_step, base_score=0.0,
                           tree_method="hist", seed=seed, disable_default_eval_metric=1)
        self.n_estimators = n_estimators
        self.val_frac = val_frac
        self.seed = seed

    def fit(self, X, y, T):
        import xgboost as xgb
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        T = np.asarray(T, float)
        self.H0_ = breslow_H0(T, y, np.zeros(len(y)))        # intercept-only, then held fixed

        rng = np.random.default_rng(self.seed + 5)
        idx = rng.permutation(len(y))
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        va = np.concatenate([pos[:max(5, int(self.val_frac * len(pos)))],
                             neg[:max(5, int(self.val_frac * len(neg)))]])
        tr = np.setdiff1d(idx, va)
        H0tr = np.asarray(self.H0_(T[tr]))
        H0va = np.asarray(self.H0_(T[va]))

        dtr = xgb.DMatrix(X[tr], label=y[tr])
        dva = xgb.DMatrix(X[va], label=y[va])
        self.bst_ = xgb.train(
            self.params, dtr, num_boost_round=self.n_estimators,
            obj=icboost.make_objective_xgb(H0tr, y[tr]),
            custom_metric=icboost.make_eval_xgb(H0va, y[va]),
            evals=[(dva, "val")], early_stopping_rounds=60, verbose_eval=False)
        p = np.clip(icboost.risk(self._raw(X[va]), H0va), 1e-6, 1 - 1e-6)
        self.cal_ = LogisticRegression(max_iter=1000).fit(logit(p).reshape(-1, 1), y[va])
        return self

    def _raw(self, X):
        import xgboost as xgb
        return self.bst_.predict(xgb.DMatrix(np.asarray(X, float)),
                                 iteration_range=(0, self.bst_.best_iteration + 1),
                                 output_margin=True)

    def linpred(self, X):
        return self._raw(X)

    def predict_risk(self, X, T):
        raw = icboost.risk(self._raw(X), np.asarray(self.H0_(np.asarray(T, float))))
        z = logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        return self.cal_.predict_proba(z)[:, 1]


class MosaicStack:
    """The full proposed method."""
    name = "mosaic"
    is_ph = False

    def __init__(self, n_inner=3, seed=0, aux=True, meta_weights=True, head="interval",
                 trunk="fm", ema=True, n_models=3, distil=True, calibrate=True,
                 uncertainty_weights=False, equal_task_weights=False, aux_budget=0.5,
                 use_boost=True, use_aux_features=True, stack=True, inner_models=1,
                 rex_weight=1.0, boost="both", **mosaic_kw):
        self.n_inner = n_inner
        self.seed = seed
        self.use_boost = use_boost
        self.use_aux_features = use_aux_features
        self.stack = stack
        self.inner_models = inner_models
        self.boost = boost
        self.calibrate = calibrate
        self.net_kw = dict(aux=aux, meta_weights=meta_weights, head=head, trunk=trunk,
                           ema=ema, n_models=n_models, distil=distil,
                           uncertainty_weights=uncertainty_weights,
                           equal_task_weights=equal_task_weights, aux_budget=aux_budget,
                           rex_weight=rex_weight, calibrate=False, **mosaic_kw)

    def _make_boosters(self):
        """The boosted learners in the stack.

        `gbm`     LightGBM on the window flag with the log-odds of the offset risk as an init
                  score -- a logistic surrogate for the current-status likelihood, exact in the
                  small-risk limit.
        `icboost` XGBoost on that likelihood written out exactly (icboost.py).

        The default keeps both.  They score within 0.01 stratified AUC of each other but are
        not the same model: different likelihood, different library, different split
        heuristics, so the stack has something to combine.  `boost="gbm"` / `"icboost"` keep
        one of them, which is what the ablations use to report what each contributes.
        """
        import baselines
        if self.boost == "icboost":
            return [ICBoost(seed=self.seed)]
        if self.boost == "gbm":
            return [baselines.GBM(seed=self.seed)]
        return [baselines.GBM(seed=self.seed), ICBoost(seed=self.seed)]

    # ------------------------------------------------------------------ helpers
    def _aux_risk(self, net, X, T):
        """M1's predicted risk for every task, used as features for M2."""
        import torch
        from mosaic import DEVICE, window_logrisk
        Z, M = net._prep(X)
        occ = torch.as_tensor(net._occ(T), device=DEVICE)
        Zt = torch.as_tensor(Z, device=DEVICE)
        Mt = torch.as_tensor(M, device=DEVICE)
        with torch.no_grad():
            nets = ([net.student_] if getattr(net, "student_", None) is not None
                    else net.models_)
            ps = []
            for n in nets:
                lp, _ = window_logrisk(n(Zt, Mt), occ)
                ps.append(lp.exp())
            p = torch.stack(ps).mean(0)
        return logit(np.clip(p.cpu().numpy().astype(float), 1e-6, 1 - 1e-6))

    def _boost_features(self, X, aux_lr):
        if not self.use_aux_features or aux_lr is None:
            return np.asarray(X, float)
        return np.column_stack([np.asarray(X, float), aux_lr[:, 1:]])

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, T, aux_y=None, aux_mask=None, groups=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = np.asarray(y, int)
        T = np.asarray(T, float)
        n = len(y)

        # ---- inner cross-fitting: out-of-fold M1 risk and auxiliary risks
        oof_p1 = np.full(n, np.nan)
        oof_aux = None
        skf = StratifiedKFold(self.n_inner, shuffle=True, random_state=self.seed + 31)
        inner_kw = dict(self.net_kw)
        inner_kw.update(n_models=self.inner_models, distil=False)
        for itr, ite in skf.split(X, y):
            net = Mosaic(seed=self.seed, **inner_kw)
            net.fit(X.iloc[itr], y[itr], T[itr],
                    None if aux_y is None else aux_y[itr],
                    None if aux_mask is None else aux_mask[itr],
                    None if groups is None else np.asarray(groups)[itr])
            oof_p1[ite] = net.predict_risk(X.iloc[ite], T[ite])
            a = self._aux_risk(net, X.iloc[ite], T[ite])
            if oof_aux is None:
                oof_aux = np.full((n, a.shape[1]), np.nan)
            oof_aux[ite] = a

        # ---- final M1 on all training data (full ensemble + distillation)
        self.net_ = Mosaic(seed=self.seed, **self.net_kw)
        self.net_.fit(X, y, T, aux_y, aux_mask, groups)
        self.lam_ = getattr(self.net_, "lam_", np.array([1.0]))

        # ---- M2 on the covariates plus the out-of-fold auxiliary risks
        if self.use_boost:
            A = self._boost_features(X, oof_aux)
            protos = self._make_boosters()
            self.boosters_ = [b.fit(A, y, T) for b in protos]
            oof_p2 = [np.full(n, np.nan) for _ in protos]
            bfolds = list(StratifiedKFold(self.n_inner, shuffle=True,
                                          random_state=self.seed + 77).split(A, y))
            for itr, ite in bfolds:
                for j, _ in enumerate(protos):
                    b = self._make_boosters()[j].fit(A[itr], y[itr], T[itr])
                    oof_p2[j][ite] = b.predict_risk(A[ite], T[ite])
        else:
            self.boosters_ = []
            oof_p2 = []

        # ---- stacker on the inner out-of-fold predictions
        cols = [oof_p1] + list(oof_p2)
        Zs = np.column_stack([logit(np.clip(c, 1e-6, 1 - 1e-6)) for c in cols])
        good = np.isfinite(Zs).all(1)
        if self.stack and Zs.shape[1] > 1:
            self.stacker_ = LogisticRegression(max_iter=2000).fit(Zs[good], y[good])
        else:
            self.stacker_ = None
            # with one learner, or stacking switched off, average the logits
            self.stack_w_ = np.ones(Zs.shape[1]) / Zs.shape[1]
        if self.calibrate:
            self._fit_calibration(Zs[good], y[good])
        else:
            self.temp_ = (1.0, 0.0)
        self._fit_conformal(Zs[good], y[good], T[good])
        return self

    def _combine(self, Zs):
        if self.stacker_ is not None:
            return self.stacker_.decision_function(Zs)
        return Zs @ self.stack_w_

    def _fit_calibration(self, Zs, y):
        from scipy.optimize import minimize
        z = self._combine(Zs)

        def nll(th):
            q = np.clip(expit(th[0] * z + th[1]), 1e-9, 1 - 1e-9)
            return -np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))

        r = minimize(nll, np.array([1.0, 0.0]), method="Nelder-Mead")
        self.temp_ = (float(r.x[0]), float(r.x[1]))

    def _fit_conformal(self, Zs, y, T):
        a, b = self.temp_
        p = expit(a * self._combine(Zs) + b)
        res = np.abs(y - p)
        self.cq_ = np.quantile(T, np.linspace(0, 1, 6))
        self.cq_[0], self.cq_[-1] = -np.inf, np.inf
        self.cw_ = []
        for k in range(5):
            m = (T > self.cq_[k]) & (T <= self.cq_[k + 1])
            self.cw_.append(float(np.quantile(res[m], 0.9)) if m.sum() > 20 else np.nan)

    # ------------------------------------------------------------------ predict
    def _stack_inputs(self, X, T):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        p1 = self.net_.predict_risk(X, T)
        cols = [p1]
        if self.boosters_:
            A = self._boost_features(X, self._aux_risk(self.net_, X, T))
            cols += [b.predict_risk(A, T) for b in self.boosters_]
        return np.column_stack([logit(np.clip(c, 1e-6, 1 - 1e-6)) for c in cols])

    def predict_risk(self, X, T):
        a, b = self.temp_
        return expit(a * self._combine(self._stack_inputs(X, np.asarray(T, float))) + b)

    def linpred(self, X):
        """A follow-up-free score: the combined risk at the cohort's median window length."""
        T = np.full(len(X), 2.16)
        return self._combine(self._stack_inputs(X, T))

    def conformal_width(self, T):
        T = np.asarray(T, float)
        k = np.clip(np.searchsorted(self.cq_, T, side="right") - 1, 0, 4)
        return np.asarray(self.cw_, float)[k]
