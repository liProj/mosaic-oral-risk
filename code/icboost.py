"""Window-likelihood gradients and Fisher-scoring curvature for ICBoost.

For u = H0(T) * exp(score), the modeled window risk is 1 - exp(-u).
The Bernoulli window log-likelihood is y*log(1-exp(-u)) - (1-y)*u.
The objective returns its exact negative gradient and positive Fisher curvature,
not the observed Hessian. The released ICBoost branch uses XGBoost.
Interpreting window probabilities as onset risks requires recording assumptions.
"""
import numpy as np


def _u(score, H0T):
    return np.clip(H0T * np.exp(np.clip(score, -20, 20)), 1e-10, 60.0)


def make_objective(H0T, y):
    """Return a LightGBM objective closure for fixed per-row baseline hazard H0(T_i).

    Signature note: LightGBM's scikit-learn API calls a custom objective as
    ``objective(y_true, y_pred)``, with y_pred the current raw score -- NOT ``(score, dataset)``
    as the native Booster API does.  Getting this backwards is silent: the labels are then read
    as the score, the gradient is constant in the model, nothing improves after the first round
    and early stopping reports ``best_iteration_ = 1`` while the model still produces plausible
    looking predictions.
    """
    H0T = np.asarray(H0T, float)
    y = np.asarray(y, float)

    def obj(y_true, y_pred):
        score = np.asarray(y_pred, float).ravel()
        u = _u(score, H0T)
        em1 = np.expm1(u)                               # e^u - 1, accurate for small u
        # exact gradient of the negative log-likelihood
        dl = np.where(y > 0.5, u / np.maximum(em1, 1e-12), -u)
        grad = -dl
        # Positive Fisher-scoring curvature; this is not the observed Hessian.
        p = -np.expm1(-u)
        dpdf = u * np.exp(-u)
        hess = np.maximum(dpdf ** 2 / np.maximum(p * (1.0 - p), 1e-12), 1e-6)
        return grad, hess

    return obj


def grad_hess(score, H0T, y):
    """Exact gradient and Fisher curvature of the negative log-likelihood at `score`."""
    u = _u(score, H0T)
    em1 = np.expm1(u)
    dl = np.where(y > 0.5, u / np.maximum(em1, 1e-12), -u)
    p = -np.expm1(-u)
    dpdf = u * np.exp(-u)
    hess = np.maximum(dpdf ** 2 / np.maximum(p * (1.0 - p), 1e-12), 1e-6)
    return -dl, hess


def make_objective_xgb(H0T, y):
    """Window objective for the native XGBoost (predt, DMatrix) interface."""
    H0T = np.asarray(H0T, float)
    y = np.asarray(y, float)

    def obj(predt, dmat):
        return grad_hess(np.asarray(predt, float).ravel(), H0T, y)

    return obj


def make_eval_xgb(H0T, y, name="ic_nll"):
    H0T = np.asarray(H0T, float)
    y = np.asarray(y, float)

    def ev(predt, dmat):
        u = _u(np.asarray(predt, float).ravel(), H0T)
        p = -np.expm1(-u)
        ll = np.where(y > 0.5, np.log(np.clip(p, 1e-12, 1.0)), -u)
        return name, float(-ll.mean())

    return ev


def make_eval(H0T, y):
    """Mean negative log-likelihood, for early stopping (sklearn-API signature)."""
    H0T = np.asarray(H0T, float)
    y = np.asarray(y, float)

    def ev(y_true, y_pred):
        score = np.asarray(y_pred, float).ravel()
        u = _u(score, H0T)
        p = -np.expm1(-u)
        ll = np.where(y > 0.5, np.log(np.clip(p, 1e-12, 1.0)), -u)
        return "ic_nll", float(-ll.mean()), False

    return ev


def risk(score, H0T):
    return -np.expm1(-_u(score, H0T))


def check_derivatives(tol=2e-4):
    """Check what the objective actually claims.

    1. the gradient is the exact derivative of the negative log-likelihood (finite differences);
    2. the curvature returned is the Fisher information of the complementary-log-log model,
       not the observed Hessian -- this is deliberate, see make_objective;
    3. the curvature is finite and positive on the checked domain.
    """
    rng = np.random.default_rng(0)
    H0T = rng.uniform(0.01, 0.6, 400)
    y = (rng.random(400) < 0.3).astype(float)
    f = rng.normal(0, 1.0, 400)
    g, h = make_objective(H0T, y)(y, f)

    def nll(fv):
        u = _u(fv, H0T)
        p = -np.expm1(-u)
        return -np.where(y > 0.5, np.log(np.clip(p, 1e-12, 1.0)), -u)

    eps = 1e-5
    g_fd = (nll(f + eps) - nll(f - eps)) / (2 * eps)
    eg = float(np.max(np.abs(g - g_fd)))

    u = _u(f, H0T)
    p = -np.expm1(-u)
    fisher = (u * np.exp(-u)) ** 2 / (p * (1.0 - p))
    eh = float(np.max(np.abs(h - fisher) / np.maximum(fisher, 1e-12)))
    pos = bool((h > 0).all())

    ok = eg < tol and eh < 1e-8 and pos
    print(f"gradient max abs error vs finite differences {eg:.2e}; "
          f"curvature max relative error vs Fisher information {eh:.2e}; "
          f"positive-definite {pos} -> {'OK' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if check_derivatives() else 1)
