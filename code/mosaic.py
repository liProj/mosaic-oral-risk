"""MOSAIC -- Multi-task Occupational Surveillance with Adaptive Interval-censored Calibration.

The method this project proposes for the task the reference paper (jcm-15-07650) defines.  It is
built around what is actually true of the released database, which the reference analysis ignores:

 1. INTERVAL-CENSORED LIKELIHOOD.  The database records only whether an event was detected
    somewhere inside each worker's own window (0, T_i], never when.  MOSAIC therefore models a
    discrete-time hazard on a fixed grid and evaluates the exact current-status likelihood
        P(y_i = 1) = 1 - prod_k (1 - h_k(x_i))^{o_ik},
    where o_ik in [0,1] is the fraction of interval k covered by (0, T_i].  The reference
    analysis instead feeds T_i to a Cox partial likelihood as if it were an event time, which it
    demonstrably is not (ablation `cox_head` measures what that costs).

 2. MULTI-TASK BORROWING.  The same 12,049 workers carry 17 other incident outcomes recorded by
    the same examinations.  With only 470 caries and 149 gingivitis events, the primary task is
    data-starved while the auxiliaries are not (ametropia alone has 8,271 events).  MOSAIC puts
    a hazard head on every outcome over a shared trunk.

 3. META-LEARNED TASK WEIGHTS.  Auxiliary tasks help only if they pull the trunk in a direction
    the primary task also wants.  MOSAIC estimates that directly: every `meta_every` epochs it
    measures the cosine alignment between each auxiliary task's training gradient and the
    primary task's *validation* gradient, and moves the task weights along it.  A task whose
    gradient is orthogonal or opposed decays towards zero on its own.

 4. NATIVE MISSINGNESS.  Half the cohort has no cholesterol or triglyceride measurement.  Each
    feature gets a learned "missing" embedding instead of being median-imputed, so missingness
    is modelled rather than erased.

 5. FEATURE-TOKENISED ATTENTION TRUNK.  Each covariate becomes a token; a small transformer
    encoder over the tokens plus a CLS readout gives the trunk, which is the current standard
    for small tabular problems and lets the model find the interactions a Cox model cannot.

 6. EMA SELF-DISTILLATION.  An exponential-moving-average teacher supplies a consistency target
    on the hazards, which is what makes a 470-event problem trainable without collapsing.

 7. CALIBRATED, CONFORMAL OUTPUT.  Predictions are ensembled over seeds, distilled back into a
    single student, temperature-calibrated on out-of-fold predictions, and equipped with
    Mondrian conformal risk bands stratified by follow-up length.

All seven are separately ablated in experiment.py.
"""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# the machine is shared with another long GPU job; keep our footprint small
torch.set_num_threads(2)

# interval grid in years; the last interval absorbs everything beyond it
GRID = np.array([0.0, 1.0, 1.75, 2.25, 3.0, 4.0, 5.5, 9.0])


def occupancy(T, grid=GRID):
    """o[i, k] = fraction of interval k covered by (0, T_i]."""
    T = np.asarray(T, float)[:, None]
    lo, hi = grid[None, :-1], grid[None, 1:]
    return np.clip((np.minimum(T, hi) - lo) / (hi - lo), 0.0, 1.0)


class FeatureTokenizer(nn.Module):
    """One token per covariate, with a learned embedding for 'this value is missing'."""

    def __init__(self, n_feat, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_feat, d))
        self.miss = nn.Parameter(torch.randn(n_feat, d) * 0.1)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.1)

    def forward(self, x, m):
        # x: (B, F) standardised with missing set to 0; m: (B, F) 1 where missing
        tok = x.unsqueeze(-1) * self.w + self.b
        tok = torch.where(m.unsqueeze(-1).bool(), self.miss.expand_as(tok), tok)
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)


class Trunk(nn.Module):
    """Three interchangeable trunks, selected by `kind`; `fm` is the default.

    fm    missingness-aware feature embeddings plus an explicit second-order interaction term
          (the factorisation-machine identity 0.5*[(sum e_j)^2 - sum e_j^2]), then an MLP.
          Gives pairwise interactions at O(F*d) cost instead of attention's O(F^2*d), which
          matters here: the attention trunk costs ~1.96 s per batch of 1024 on this machine's
          CPU against ~20 ms for this one -- a factor of 100 -- and with 470 events the extra
          capacity is not what the problem is short of.
    mlp   the same embeddings summed, with no interaction term; isolates what the second-order
          term is worth.
    attn  feature-tokenised transformer encoder with a CLS readout, kept as an ablation.
    """

    def __init__(self, n_feat, d=32, depth=2, heads=4, dropout=0.1, kind="fm"):
        super().__init__()
        self.kind = kind
        self.tok = FeatureTokenizer(n_feat, d)
        if kind == "attn":
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=dropout,
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, depth)
            self.out_dim = d
        else:
            extra = 2 * d if kind == "fm" else d
            self.mlp = nn.Sequential(nn.Linear(2 * n_feat + extra, 96), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(96, 48), nn.ReLU())
            self.out_dim = 48

    def forward(self, x, m):
        if self.kind == "attn":
            return self.enc(self.tok(x, m))[:, 0]
        e = self.tok(x, m)[:, 1:]                       # (B, F, d); drop the CLS token
        s = e.sum(1)
        parts = [x, m, s]
        if self.kind == "fm":
            parts.append(0.5 * (s ** 2 - (e ** 2).sum(1)))
        return self.mlp(torch.cat(parts, dim=1))


class MosaicNet(nn.Module):
    def __init__(self, n_feat, n_task, n_interval, d=32, depth=2, dropout=0.1, kind="fm"):
        super().__init__()
        self.trunk = Trunk(n_feat, d=d, depth=depth, dropout=dropout, kind=kind)
        # one hazard head per task; the baseline hazard level is a free per-interval bias
        self.head = nn.Linear(self.trunk.out_dim, n_task * n_interval)
        self.n_task, self.n_interval = n_task, n_interval
        # The head must NOT be zero-initialised.  With W = 0 the gradient reaching the trunk is
        # W^T dL/dlogit = 0, so the trunk never learns and the model collapses onto the
        # per-interval bias, i.e. the marginal incidence rate.  A small random head keeps the
        # initial hazards near the base rate (bias -4) while still passing gradient down.
        nn.init.normal_(self.head.weight, std=0.05)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x, m):
        z = self.trunk(x, m)
        return self.head(z).view(-1, self.n_task, self.n_interval)     # logit hazards


def window_logrisk(logit_h, occ):
    """log P(event in window) and log P(no event), from per-interval logit hazards.

    S = prod_k (1 - h_k)^{o_k}, so log S = sum_k o_k * log(1 - h_k) = -sum_k o_k * softplus(z_k).
    """
    log_s = -(occ.unsqueeze(1) * F.softplus(logit_h)).sum(-1)          # (B, n_task)
    log_p = torch.log(-torch.expm1(log_s.clamp(max=-1e-7)))
    return log_p, log_s


def cox_partial_loss(eta, T, y):
    """Breslow partial likelihood of a Cox fit on (T, y) -- the reference analysis's own
    likelihood, which treats T as the event time even though it is only a window length."""
    order = torch.argsort(T, descending=True)
    eta, y = eta[order], y[order]
    log_cum = torch.logcumsumexp(eta, dim=0)        # risk set = subjects with T >= T_i
    ev = y > 0.5
    if ev.sum() < 1:
        return eta.sum() * 0.0
    return -((eta[ev] - log_cum[ev]).sum() / ev.sum())


def all_task_losses(logit_h, occ, Y, K):
    """Interval-censored Bernoulli loss for every task at once.

    window_logrisk already evaluates all tasks in one pass, so the per-task losses are read off
    the same tensors.  Calling task_loss() in a Python loop recomputed that pass once per task,
    which dominated the runtime when there are 18 of them.
    """
    log_p, log_s = window_logrisk(logit_h, occ)
    per = -(Y * log_p + (1.0 - Y) * log_s) * K
    return per.sum(0) / K.sum(0).clamp(min=1.0), per      # (n_task,), (B, n_task)


def task_loss(logit_h, occ, y, mask, idx):
    """Interval-censored Bernoulli loss for one task."""
    log_p, log_s = window_logrisk(logit_h, occ)
    lp, ls = log_p[:, idx], log_s[:, idx]
    per = -(y * lp + (1 - y) * ls)
    per = per * mask
    return per.sum() / mask.sum().clamp(min=1.0)


class Mosaic:
    """Fit/predict wrapper matching the baselines.py interface."""
    name = "mosaic"
    is_ph = False

    def __init__(self, aux=True, meta_weights=True, head="interval", trunk="fm",
                 ema=True, n_models=3, distil=True, calibrate=True, uncertainty_weights=False,
                 epochs=150, lr=3e-3, wd=1e-4, batch=2048, d=32, depth=1, dropout=0.1,
                 meta_every=15, meta_lr=0.5, meta_temp=1.0, aux_budget=0.5,
                 equal_task_weights=False, ema_weight=0.1, rex_weight=1.0, seed=0, val_frac=0.2,
                 verbose=False):
        self.__dict__.update(locals())
        del self.self

    def _occ(self, T):
        """Interval occupancy, or a constant window for the `binary` head ablation."""
        if self.head == "binary":
            # the model is given no information about how long the worker was observed, which
            # is what a naive classifier on the event flag does
            return np.ones((len(np.asarray(T)), len(GRID) - 1), np.float32) / (len(GRID) - 1)
        return occupancy(T).astype(np.float32)

    # ---------------------------------------------------------------- data plumbing
    def _prep_fit(self, X):
        Xv = np.asarray(X, float)
        self.med_ = np.nanmedian(Xv, axis=0)
        self.med_ = np.where(np.isnan(self.med_), 0.0, self.med_)
        Z = np.where(np.isnan(Xv), self.med_, Xv)
        self.mu_, self.sd_ = Z.mean(0), Z.std(0) + 1e-6
        return self._prep(X)

    def _prep(self, X):
        Xv = np.asarray(X, float)
        M = np.isnan(Xv).astype(np.float32)
        Z = np.where(np.isnan(Xv), self.med_, Xv)
        Z = (Z - self.mu_) / self.sd_
        Z = np.where(M > 0, 0.0, Z)                 # value is carried by the missing embedding
        return Z.astype(np.float32), M

    # ---------------------------------------------------------------- training
    def fit(self, X, y, T, aux_y=None, aux_mask=None, groups=None):
        """Fit the multi-task interval-censored model.

        aux_y   (n, n_aux) auxiliary binary outcomes
        aux_mask(n, n_aux) 1 where the auxiliary outcome was observed
        groups  (n,) environment label, here the year of the baseline examination.  Used by the
                risk-extrapolation penalty: the released database has a recording era whose
                caries rate is 3-10x every other year, so a model is free to score well by
                learning which era a worker belongs to.  Penalising the VARIANCE of the
                per-era loss (V-REx) pushes the representation towards one that works equally
                well in every era, which is what transfer to an unseen year needs.
        """
        Z, M = self._prep_fit(X)
        y = np.asarray(y, float)
        occ = self._occ(T)
        if aux_y is None or not self.aux:
            aux_y = np.zeros((len(y), 0), np.float32)
            aux_mask = np.zeros((len(y), 0), np.float32)
        Y = np.column_stack([y, aux_y]).astype(np.float32)
        K = np.column_stack([np.ones(len(y), np.float32), aux_mask]).astype(np.float32)
        self.n_task_ = Y.shape[1]

        rng = np.random.default_rng(self.seed)
        n = len(y)
        # stratified train/validation split; validation drives the meta-weights and early stop
        idx = rng.permutation(n)
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        nv_p = max(5, int(self.val_frac * len(pos)))
        nv_n = max(5, int(self.val_frac * len(neg)))
        val = np.concatenate([pos[:nv_p], neg[:nv_n]])
        tr = np.setdiff1d(idx, val)

        tens = lambda a: torch.as_tensor(a, device=DEVICE)
        Zt, Mt, Yt, Kt, Ot = map(tens, (Z, M, Y, K, occ))
        Tt = tens(np.asarray(T, np.float32))
        self.T_tr_, self.y_tr_, self.tr_idx_ = np.asarray(T, float), y, tr
        if groups is None or self.rex_weight <= 0:
            Gt, self.group_vals_ = None, None
        else:
            g = pd.Series(groups).fillna(-1).values
            self.group_vals_ = np.unique(g)
            Gt = tens(np.searchsorted(self.group_vals_, g).astype(np.int64))
        n_groups = 1 if self.group_vals_ is None else len(self.group_vals_)

        self.models_ = []
        n_models = self.n_models if self.n_models and self.n_models > 1 else 1
        for mi in range(n_models):
            torch.manual_seed(self.seed * 1000 + mi)
            net = MosaicNet(Z.shape[1], self.n_task_, len(GRID) - 1, d=self.d,
                            depth=self.depth, dropout=self.dropout,
                            kind=self.trunk).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.lr, total_steps=self.epochs, pct_start=0.3)
            ema = copy.deepcopy(net) if self.ema else None
            if ema is not None:
                for p in ema.parameters():
                    p.requires_grad_(False)
            # Task weights.  The primary task keeps unit weight and the auxiliaries SHARE a
            # fixed budget `aux_budget`, which the meta-learner then reallocates among them.
            # Giving every task weight 1 is catastrophic here and not a straw man: the
            # auxiliaries outnumber the primary task 16 to 1 and some are two orders of
            # magnitude more prevalent (ametropia has 8,271 events against 470), so their
            # gradients capture the shared trunk.  Measured on one fold, naive equal weighting
            # drops stratified AUC from 0.752 to 0.516 -- worse than no auxiliaries at all.
            n_aux = max(self.n_task_ - 1, 1)
            lam = torch.full((self.n_task_,), self.aux_budget / n_aux, device=DEVICE)
            lam[0] = 1.0
            if self.equal_task_weights:
                lam = torch.ones(self.n_task_, device=DEVICE)
            log_sig = torch.zeros(self.n_task_, device=DEVICE, requires_grad=True)
            opt_sig = torch.optim.Adam([log_sig], lr=1e-2)

            best, best_state, bad = np.inf, None, 0
            for ep in range(self.epochs):
                net.train()
                perm = torch.as_tensor(rng.permutation(tr), device=DEVICE)
                for s in range(0, len(perm), self.batch):
                    b = perm[s:s + self.batch]
                    zh = net(Zt[b], Mt[b])
                    losses, per_sample = all_task_losses(zh, Ot[b], Yt[b], Kt[b])
                    if self.head == "cox":
                        losses = torch.cat([
                            cox_partial_loss(zh[:, 0].mean(-1), Tt[b], Yt[b, 0]).reshape(1),
                            losses[1:]])
                    if self.uncertainty_weights:
                        # Kendall-style homoscedastic uncertainty weighting (ablation A3)
                        tot = (torch.exp(-log_sig) * losses + 0.5 * log_sig).sum()
                    else:
                        tot = (lam * losses).sum()
                    if Gt is not None:
                        # V-REx: variance of the primary loss across the eras present in the
                        # batch.  Computed by scatter-add over the per-sample losses already
                        # available, not by looping over groups in Python -- the loop version
                        # re-evaluated the window risk once per group and made training ten
                        # times slower (19.5 s against 2.1 s for 30 epochs).
                        gb = Gt[b]
                        cnt = torch.zeros(n_groups, device=DEVICE).index_add_(
                            0, gb, torch.ones_like(gb, dtype=torch.float32))
                        tot_g = torch.zeros(n_groups, device=DEVICE).index_add_(
                            0, gb, per_sample[:, 0])
                        # masked variance computed entirely in tensor ops: a Python-side
                        # branch on `keep.sum()` forces a GPU->CPU sync on every batch, which
                        # is expensive while the device is shared with another job
                        keep = (cnt >= 64).float()
                        n_keep = keep.sum()
                        gl = tot_g / cnt.clamp(min=1.0)
                        mu = (gl * keep).sum() / n_keep.clamp(min=1.0)
                        var = (((gl - mu) ** 2) * keep).sum() / (n_keep - 1).clamp(min=1.0)
                        tot = tot + self.rex_weight * var * (n_keep >= 2).float()
                    if ema is not None:
                        with torch.no_grad():
                            zt = ema(Zt[b], Mt[b])
                        # consistency on the LOGIT hazards.  On the probability scale the
                        # hazards sit around e^-4 ~ 0.018, so an MSE there is ~3e-4 with a
                        # gradient small enough to be a no-op; in logit space the term is
                        # scaled like the rest of the objective.
                        tot = tot + self.ema_weight * F.mse_loss(zh, zt)
                    opt.zero_grad(set_to_none=True)
                    if self.uncertainty_weights:
                        opt_sig.zero_grad(set_to_none=True)
                    tot.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                    if self.uncertainty_weights:
                        opt_sig.step()
                    if ema is not None:
                        with torch.no_grad():
                            for pe, pn in zip(ema.parameters(), net.parameters()):
                                pe.mul_(0.99).add_(pn, alpha=0.01)
                            for be, bn in zip(ema.buffers(), net.buffers()):
                                be.copy_(bn)
                sched.step()

                # `equal_task_weights` is the naive-multi-task ablation, so the meta-learner
                # must not quietly reallocate its weights back into the budget
                if (self.aux and self.meta_weights and not self.equal_task_weights
                        and not self.uncertainty_weights and self.n_task_ > 1
                        and (ep + 1) % self.meta_every == 0):
                    lam = self._meta_update(net, lam, Zt, Mt, Yt, Kt, Ot, tr, val)

                # early stopping on the primary validation loss
                net.eval()
                with torch.no_grad():
                    zh = net(Zt[val], Mt[val])
                    vl = float(cox_partial_loss(zh[:, 0].mean(-1), Tt[val], Yt[val, 0])
                               if self.head == "cox"
                               else task_loss(zh, Ot[val], Yt[val, 0], Kt[val, 0], 0))
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_state = copy.deepcopy((ema or net).state_dict())
                else:
                    bad += 1
                    if bad >= 40:
                        break
            final = MosaicNet(Z.shape[1], self.n_task_, len(GRID) - 1, d=self.d,
                              depth=self.depth, dropout=self.dropout,
                              kind=self.trunk).to(DEVICE)
            final.load_state_dict(best_state)
            final.eval()
            self.models_.append(final)
            self.lam_ = lam.detach().cpu().numpy()

        if self.distil and len(self.models_) > 1:
            self._distil(Zt, Mt, Ot, tr, val, Z.shape[1])
        if self.head == "cox":
            from metrics import breslow_H0
            Xtr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
            self.H0_ = breslow_H0(np.asarray(T, float)[tr], y[tr], self._eta_cox(Xtr))
        if self.calibrate:
            self._fit_calibration(X, y, T, val)
        else:
            self.temp_ = (1.0, 0.0)
        self._fit_conformal(X, y, T, val)
        return self

    def _meta_update(self, net, lam, Zt, Mt, Yt, Kt, Ot, tr, val):
        """Move auxiliary weights along cos(grad_aux_train, grad_primary_val).

        The alignment is measured on the SHARED TRUNK only.  Each task owns a disjoint block of
        rows in the output head, so including the head dilutes every cosine towards zero (it
        measured ~3e-4 here) and the weights never move.  Task weights act on the primary task
        only through the shared representation, so the trunk is also the theoretically correct
        place to measure the alignment.

        The cosines are then standardised across tasks before the multiplicative update, which
        makes the step scale-free: what matters is which auxiliaries agree with the primary task
        *relative to the others*, not the absolute cosine, which is small for any single task.
        """
        params = [p for p in net.trunk.parameters() if p.requires_grad]

        def flat_grad(loss):
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return torch.cat([(torch.zeros_like(p) if gi is None else gi).reshape(-1)
                              for gi, p in zip(g, params)])

        net.eval()
        zh_v = net(Zt[val], Mt[val])
        g_val = flat_grad(task_loss(zh_v, Ot[val], Yt[val, 0], Kt[val, 0], 0))
        g_val = g_val / (g_val.norm() + 1e-12)
        sel = torch.as_tensor(tr[:2048], device=DEVICE)
        zh_t = net(Zt[sel], Mt[sel])
        cos = np.full(self.n_task_, np.nan)
        for t in range(1, self.n_task_):
            if Kt[sel, t].sum() < 10:
                continue
            gt = flat_grad(task_loss(zh_t, Ot[sel], Yt[sel, t], Kt[sel, t], t))
            cos[t] = float(torch.dot(gt / (gt.norm() + 1e-12), g_val))
        net.zero_grad(set_to_none=True)
        net.train()

        # Reallocate the auxiliary budget by a softmax over standardised alignments: the
        # allocation is relative, so a task only needs to agree with the primary task *more
        # than the others* to gain weight, and a persistently misaligned task decays towards
        # zero.  Standardising first makes the step scale-free, which matters because any one
        # cosine is small (order 1e-2 on the trunk).
        new = torch.zeros_like(lam)
        new[0] = 1.0
        ok = ~np.isnan(cos)
        ok[0] = False
        if ok.sum() >= 2:
            c = cos[ok]
            z = (c - c.mean()) / (c.std() + 1e-12)
            w = np.exp(z / max(self.meta_temp, 1e-6))
            w = w / w.sum() * self.aux_budget
            # move only part of the way, so the allocation is smoothed over meta steps
            prev = lam.detach().cpu().numpy()[ok]
            w = (1 - self.meta_lr) * prev + self.meta_lr * w
            for t, v in zip(np.where(ok)[0], w):
                new[t] = float(v)
        else:
            new = lam.clone()
        self.cos_last_ = cos
        return new

    def _distil(self, Zt, Mt, Ot, tr, val, n_feat):
        """Distil the ensemble's hazards into one student of the same architecture."""
        with torch.no_grad():
            # average the ensemble in logit space and distil there, for the same reason the
            # EMA consistency term does: probability-space MSE on hazards near 0.02 has
            # vanishing gradients and the student ends up worse than one of its teachers
            # (measured: stratified AUC 0.692 distilled against 0.752 for a single model).
            teach = torch.stack([m(Zt, Mt) for m in self.models_]).mean(0)
        stu = MosaicNet(n_feat, self.n_task_, len(GRID) - 1, d=self.d, depth=self.depth,
                        dropout=self.dropout, kind=self.trunk).to(DEVICE)
        opt = torch.optim.AdamW(stu.parameters(), lr=self.lr, weight_decay=self.wd)
        rng = np.random.default_rng(self.seed + 7)
        best, best_state, bad = np.inf, None, 0
        for ep in range(120):
            stu.train()
            perm = torch.as_tensor(rng.permutation(tr), device=DEVICE)
            for s in range(0, len(perm), self.batch):
                b = perm[s:s + self.batch]
                loss = F.mse_loss(stu(Zt[b], Mt[b]), teach[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            stu.eval()
            with torch.no_grad():
                vl = float(F.mse_loss(stu(Zt[val], Mt[val]), teach[val]))
            if vl < best - 1e-7:
                best, bad, best_state = vl, 0, copy.deepcopy(stu.state_dict())
            else:
                bad += 1
                if bad >= 20:
                    break
        stu.load_state_dict(best_state)
        stu.eval()
        self.student_ = stu

    # ---------------------------------------------------------------- prediction
    def _eta_cox(self, X):
        Z, M = self._prep(X)
        Zt = torch.as_tensor(Z, device=DEVICE)
        Mt = torch.as_tensor(M, device=DEVICE)
        with torch.no_grad():
            nets = ([self.student_] if getattr(self, "student_", None) is not None
                    else self.models_)
            return torch.stack([n(Zt, Mt)[:, 0].mean(-1) for n in nets]
                               ).mean(0).cpu().numpy().astype(float)

    def _raw_risk(self, X, T, use_student=True):
        if self.head == "cox":
            from metrics import risk_from_linpred
            return risk_from_linpred(self._eta_cox(X), T, self.H0_)
        Z, M = self._prep(X)
        occ = torch.as_tensor(self._occ(T), device=DEVICE)
        Zt = torch.as_tensor(Z, device=DEVICE)
        Mt = torch.as_tensor(M, device=DEVICE)
        with torch.no_grad():
            if use_student and getattr(self, "student_", None) is not None:
                lp, _ = window_logrisk(self.student_(Zt, Mt), occ)
                p = lp[:, 0].exp()
            else:
                ps = []
                for m in self.models_:
                    lp, _ = window_logrisk(m(Zt, Mt), occ)
                    ps.append(lp[:, 0].exp())
                p = torch.stack(ps).mean(0)
        return p.cpu().numpy().astype(float)

    def _fit_calibration(self, X, y, T, val):
        """Platt scaling on the held-out split (never on the folds used for reporting)."""
        from scipy.optimize import minimize
        from scipy.special import expit, logit
        p = np.clip(self._raw_risk(X.iloc[val] if hasattr(X, "iloc") else X[val], T[val]),
                    1e-6, 1 - 1e-6)
        z = logit(p)
        yv = np.asarray(y, float)[val]

        def nll(th):
            q = np.clip(expit(th[0] * z + th[1]), 1e-9, 1 - 1e-9)
            return -np.mean(yv * np.log(q) + (1 - yv) * np.log1p(-q))

        r = minimize(nll, np.array([1.0, 0.0]), method="Nelder-Mead")
        self.temp_ = (float(r.x[0]), float(r.x[1]))

    def _fit_conformal(self, X, y, T, val):
        """Mondrian conformal quantiles of |y - p| by follow-up quintile."""
        p = self.predict_risk(X.iloc[val] if hasattr(X, "iloc") else X[val], T[val])
        res = np.abs(np.asarray(y, float)[val] - p)
        tv = np.asarray(T, float)[val]
        self.cq_ = np.quantile(tv, np.linspace(0, 1, 6))
        self.cq_[0], self.cq_[-1] = -np.inf, np.inf
        self.cw_ = []
        for k in range(5):
            m = (tv > self.cq_[k]) & (tv <= self.cq_[k + 1])
            self.cw_.append(float(np.quantile(res[m], 0.9)) if m.sum() > 20 else np.nan)

    def predict_risk(self, X, T):
        from scipy.special import expit, logit
        p = np.clip(self._raw_risk(X, T), 1e-6, 1 - 1e-6)
        a, b = getattr(self, "temp_", (1.0, 0.0))
        return expit(a * logit(p) + b)

    def linpred(self, X):
        """A follow-up-free risk score: the window risk at the median window length."""
        from scipy.special import logit
        T = np.full(len(X), 2.16)
        return logit(np.clip(self._raw_risk(X, T), 1e-9, 1 - 1e-9))

    def conformal_width(self, T):
        T = np.asarray(T, float)
        k = np.clip(np.searchsorted(self.cq_, T, side="right") - 1, 0, 4)
        return np.asarray(self.cw_, float)[k]
