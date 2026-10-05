"""Competitor models for the jcm-15-07650 reproduction.

Every model exposes the same interface:

    m = Model(**kwargs); m.fit(X_tr, y_tr, T_tr); p = m.predict_risk(X_te, T_te)

where y is the single-window interval-censored event flag and T the window length in years.
`predict_risk` must return P(event detected inside (0, T]).  Models of proportional-hazards
form return a linear predictor from `linpred` and get their window risk through the shared
baseline cumulative hazard in metrics.breslow_H0, so the follow-up length is handled
identically for all of them.

REFERENCE is the model the reference paper's analysis corresponds to: a Cox proportional
hazards fit on (person-years, event flag).  The article itself reports only one crude
univariate HR at a time and no model at all, so CrudeCox reproduces that literally (the
strongest single crude predictor) and MultivarCox is the fair multivariable upgrade of it,
which is what every comparison in the paper uses as the reference method.
"""
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metrics import breslow_H0, risk_from_linpred

warnings.filterwarnings("ignore")


class _Base:
    name = "base"
    is_ph = True

    def fit(self, X, y, T):
        raise NotImplementedError

    def linpred(self, X):
        raise NotImplementedError

    def _fit_H0(self, X, y, T):
        self.H0_ = breslow_H0(T, y, self.linpred(X))

    def predict_risk(self, X, T):
        return risk_from_linpred(self.linpred(X), T, self.H0_)


def _prep(X, imputer=None, scaler=None, fit=False):
    """Median-impute then standardise; missingness indicators for columns that have any."""
    Xv = X.values if hasattr(X, "values") else X
    if fit:
        imputer = SimpleImputer(strategy="median").fit(Xv)
        scaler = StandardScaler().fit(imputer.transform(Xv))
    Z = scaler.transform(imputer.transform(Xv))
    return Z, imputer, scaler


class Rate(_Base):
    """No-covariate reference: a single incidence rate, i.e. the article's headline number."""
    name = "rate_only"

    def fit(self, X, y, T):
        self.lam_ = y.sum() / np.sum(T)
        return self

    def linpred(self, X):
        return np.zeros(len(X))

    def predict_risk(self, X, T):
        return 1.0 - np.exp(-self.lam_ * np.asarray(T, float))


class CrudeCox(_Base):
    """The reference article's own analysis: the single best crude univariate Cox predictor.

    The article fits one univariate Cox per characteristic and never combines them, so the
    best a reader could do with what is published is to use the strongest single one.  Which
    variable that is is chosen on the training fold only.
    """
    name = "cox_crude"

    def fit(self, X, y, T):
        X = pd.DataFrame(X)
        best, best_ll = None, -np.inf
        for c in X.columns:
            d = pd.DataFrame({"v": X[c].values, "py": T, "e": y}).dropna()
            if d.v.nunique() < 2:
                continue
            try:
                cf = CoxPHFitter().fit(d, duration_col="py", event_col="e")
            except Exception:
                continue
            if cf.log_likelihood_ > best_ll:
                best_ll, best = cf.log_likelihood_, (c, float(cf.params_.iloc[0]),
                                                     float(d.v.median()))
        self.col_, self.beta_, self.med_ = best
        self._fit_H0(X, y, T)
        return self

    def linpred(self, X):
        v = pd.DataFrame(X)[self.col_].fillna(self.med_).values
        return self.beta_ * v


class MultivarCox(_Base):
    """Multivariable Cox on (person-years, event): the reference method for all comparisons."""
    name = "cox_multivar"

    def __init__(self, penalizer=0.1):
        self.penalizer = penalizer

    def fit(self, X, y, T):
        Z, self.imp_, self.sc_ = _prep(X, fit=True)
        d = pd.DataFrame(Z, columns=[f"v{i}" for i in range(Z.shape[1])])
        d["py"], d["e"] = T, y
        self.cf_ = CoxPHFitter(penalizer=self.penalizer).fit(d, duration_col="py",
                                                             event_col="e")
        self.cols_ = [c for c in d.columns if c.startswith("v")]
        self._fit_H0(X, y, T)
        return self

    def linpred(self, X):
        Z, _, _ = _prep(X, self.imp_, self.sc_)
        return (Z * self.cf_.params_[self.cols_].values).sum(axis=1)


class ClogLog(_Base):
    """Correctly specified parametric baseline: binomial cloglog with a log-H0(T) offset.

    This is the statistically right linear model for single-window interval-censored data, so
    it isolates how much of any gain comes from merely fixing the likelihood rather than from
    flexible learning.
    """
    name = "cloglog"
    is_ph = True

    def fit(self, X, y, T):
        import statsmodels.api as sm
        Z, self.imp_, self.sc_ = _prep(X, fit=True)
        T = np.asarray(T, float)
        off = np.log(np.clip(T, 1e-6, None))
        A = sm.add_constant(Z)
        self.res_ = sm.GLM(y, A, family=sm.families.Binomial(sm.families.links.CLogLog()),
                           offset=off).fit(maxiter=200)
        self.beta_ = np.asarray(self.res_.params, float)
        return self

    def linpred(self, X):
        Z, _, _ = _prep(X, self.imp_, self.sc_)
        return self.beta_[0] + Z @ self.beta_[1:]

    def predict_risk(self, X, T):
        eta = self.linpred(X) + np.log(np.clip(np.asarray(T, float), 1e-6, None))
        return 1.0 - np.exp(-np.exp(np.clip(eta, -30, 30)))


class LogitOffset(_Base):
    """Logistic regression on the window flag with log(T) as an offset-like feature."""
    name = "logit"

    def fit(self, X, y, T):
        Z, self.imp_, self.sc_ = _prep(X, fit=True)
        A = np.column_stack([Z, np.log(np.clip(T, 1e-6, None))])
        self.m_ = LogisticRegression(max_iter=2000, C=1.0).fit(A, y)
        return self

    def linpred(self, X):
        Z, _, _ = _prep(X, self.imp_, self.sc_)
        return Z @ self.m_.coef_[0][:-1]

    def predict_risk(self, X, T):
        Z, _, _ = _prep(X, self.imp_, self.sc_)
        A = np.column_stack([Z, np.log(np.clip(np.asarray(T, float), 1e-6, None))])
        return self.m_.predict_proba(A)[:, 1]


class GBM(_Base):
    """LightGBM on the window flag with log(T) as an offset via init_score."""
    name = "lightgbm"

    def __init__(self, n_estimators=400, learning_rate=0.03, num_leaves=15,
                 min_child_samples=60, subsample=0.8, colsample=0.8, reg_lambda=5.0,
                 seed=0):
        self.kw = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                       num_leaves=num_leaves, min_child_samples=min_child_samples,
                       subsample=subsample, subsample_freq=1, colsample_bytree=colsample,
                       reg_lambda=reg_lambda, random_state=seed, verbose=-1)

    def fit(self, X, y, T):
        import lightgbm as lgb
        self.lam_ = y.sum() / np.sum(T)
        init = np.log(np.clip(1.0 - np.exp(-self.lam_ * T), 1e-9, 1 - 1e-9))
        init = init - np.log1p(-np.exp(init))                     # logit of the offset risk
        self.m_ = lgb.LGBMClassifier(**self.kw)
        self.m_.fit(np.asarray(X, float), y, init_score=init)
        return self

    def _init(self, T):
        p0 = np.clip(1.0 - np.exp(-self.lam_ * np.asarray(T, float)), 1e-9, 1 - 1e-9)
        return np.log(p0) - np.log1p(-p0)

    def linpred(self, X):
        return self.m_.predict(np.asarray(X, float), raw_score=True)

    def predict_risk(self, X, T):
        return expit(self.linpred(X) + self._init(T))


class XGB(GBM):
    name = "xgboost"

    def __init__(self, n_estimators=400, learning_rate=0.03, max_depth=4, seed=0):
        self.kw = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                       max_depth=max_depth, subsample=0.8, colsample_bytree=0.8,
                       reg_lambda=5.0, min_child_weight=20, random_state=seed,
                       eval_metric="logloss", tree_method="hist")

    def fit(self, X, y, T):
        import xgboost as xgb
        self.lam_ = y.sum() / np.sum(T)
        self.m_ = xgb.XGBClassifier(**self.kw)
        self.m_.fit(np.asarray(X, float), y, base_margin=self._init(T))
        return self

    def linpred(self, X):
        import xgboost as xgb
        dm = xgb.DMatrix(np.asarray(X, float))
        return self.m_.get_booster().predict(dm, output_margin=True)


class RF(_Base):
    """Random forest on the window flag, with follow-up length supplied as a feature.

    Stands in for a random survival forest, which cannot be fitted here: scikit-survival has
    no aarch64 wheel and, more fundamentally, an RSF needs event times, which this database
    does not record.
    """
    name = "random_forest"

    def __init__(self, n_estimators=500, min_samples_leaf=20, seed=0):
        self.kw = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                       n_jobs=4, random_state=seed, class_weight=None)

    def fit(self, X, y, T):
        Z, self.imp_, self.sc_ = _prep(X, fit=True)
        A = np.column_stack([Z, np.log(np.clip(T, 1e-6, None))])
        self.m_ = RandomForestClassifier(**self.kw).fit(A, y)
        return self

    def linpred(self, X):
        from scipy.special import logit
        Z, _, _ = _prep(X, self.imp_, self.sc_)
        A = np.column_stack([Z, np.zeros(len(Z))])
        return logit(np.clip(self.m_.predict_proba(A)[:, 1], 1e-6, 1 - 1e-6))

    def predict_risk(self, X, T):
        Z, _, _ = _prep(X, self.imp_, self.sc_)
        A = np.column_stack([Z, np.log(np.clip(np.asarray(T, float), 1e-6, None))])
        return self.m_.predict_proba(A)[:, 1]


REFERENCE = MultivarCox
ALL = [Rate, CrudeCox, MultivarCox, ClogLog, LogitOffset, GBM, XGB, RF]
