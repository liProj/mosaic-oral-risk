"""Construct three nested feature sets for oral-risk experiments.

paper: 18 encoded covariates from the reference analysis.
base: 52 baseline features, used in the main comparisons.
traj: 114 features including follow-up summaries, used only for retrospective
sensitivity analysis; these summaries are unavailable at baseline.

Window duration is passed separately from X. Its parameterization differs across
the baseline and window-supervised models; see baselines.py and mosaic_stack.py.
"""
import os

import numpy as np
import pandas as pd

from config import JD, DTA

PAPER_NUM = ["edad1", "pas1", "pad1", "cintura1", "imc1", "glucosa1", "colesterol1",
             "trigliceridos1"]
PAPER_BIN = ["sexo", "noche", "fuma", "alcohol"]
PAPER_CAT = {"occ": [0, 1, 2, 3], "expo": [0, 1, 2, 3]}

EXTRA_NUM = ["talla_cm1", "peso1", "hemoglobina1", "cadera1", "tiempomeses1"]
EXTRA_BIN = ["familiar_dm", "familiar_hta", "tiempo_sentado", "HTA_autorreporte",
             "diabetes_autorreporte", "Hipertiroidismo", "Hipotiroidismo",
             "cardiovascular_inicial", "estres1", "depresion1", "ansiedad1",
             "diabetes_inicial", "hipertension_medida_1", "hipertension_inicial",
             "Obesidad_IMC1", "cinturaATPIII_1", "lumbalgia1_2",
             "ametropia_prevalencia", "onicomicosis_prevalencia", "pterigion_prevalencia",
             "estereopsis_prevalencia", "discromatopsia_prevalencia"]
EXTRA_NUM2 = ["audiometria1", "espirometria1", "hipoacusia_neurosensorial1", "imccat1",
              "lumbalgia1", "diabetes1"]

WAVE_VARS = ["peso", "pas", "pad", "cintura", "glucosa", "colesterol", "trigliceridos",
             "hemoglobina"]


_RAW_CACHE = {}


def _raw_extra():
    """Columns of the Stata file that data_prep.py does not carry over.

    Cached: the Stata file is 107 MB and build() needs this twice per call, which dominated
    the start-up cost of every experiment worker.
    """
    if "extra" in _RAW_CACHE:
        return _RAW_CACHE["extra"]
    cols = EXTRA_BIN + EXTRA_NUM2 + ["fecha1"]
    raw = pd.read_stata(DTA, convert_categoricals=False)
    f = raw[raw.TiempoSeguimiento.notna()].reset_index(drop=True)
    out = f[[c for c in cols if c in f.columns]].copy()
    out["wid"] = np.arange(len(out))
    out["base_year"] = pd.to_datetime(out.fecha1, format="%d/%m/%Y",
                                      errors="coerce").dt.year
    _RAW_CACHE["extra"] = out.drop(columns=["fecha1"])
    return _RAW_CACHE["extra"]


def trajectory_features(waves):
    """Per-worker summaries of the repeated examinations."""
    w = waves.sort_values(["wid", "wave"]).copy()
    out = w.groupby("wid").agg(n_visit=("wave", "count"),
                               span=("t_years", "max")).reset_index()
    out["mean_gap"] = out.span / (out.n_visit - 1).clip(lower=1)
    for v in WAVE_VARS:
        d = w.dropna(subset=[v]).copy()
        g = d.groupby("wid")
        agg = g.agg(**{f"{v}_mean": (v, "mean"), f"{v}_sd": (v, "std"),
                       f"{v}_n": (v, "count"), f"{v}_last": (v, "last"),
                       f"{v}_first": (v, "first")})
        # closed-form least-squares slope against time in years, over visits that carry both a
        # measurement and a parseable date; undefined (NaN) with fewer than two distinct times
        dd = d.dropna(subset=["t_years"])
        x, y = dd.t_years, dd[v]
        s = pd.DataFrame({"n": 1.0, "x": x, "y": y, "xy": x * y, "xx": x * x},
                         index=dd.index).groupby(dd.wid).sum()
        den = s.n * s.xx - s.x ** 2
        agg = agg.join(((s.n * s.xy - s.x * s.y) / den.where(den.abs() > 1e-9)
                        ).rename(f"{v}_slope"))
        agg[f"{v}_delta"] = agg[f"{v}_last"] - agg[f"{v}_first"]
        out = out.merge(agg.reset_index(), on="wid", how="left")
    # smoking / alcohol change across waves
    for v in ["fuma", "alcohol"]:
        g = w.dropna(subset=[v]).groupby("wid")[v]
        out = out.merge(pd.DataFrame({f"{v}_ever": g.max(), f"{v}_mean": g.mean()}
                                     ).reset_index(), on="wid", how="left")
    return out


def build(featureset="traj"):
    """Return (X, meta, blocks) where meta carries wid, person-time and every outcome.

    Memoised on disk: the feature matrices are deterministic, and every experiment worker
    rebuilds the same three of them.
    """
    cache = f"{JD}/results/features_{featureset}.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    coh = pd.read_pickle(f"{JD}/results/cohort.pkl")
    cols, blocks = [], {}

    def take(df, names, block):
        got = [c for c in names if c in df.columns]
        blocks.setdefault(block, []).extend(got)
        return df[got]

    parts = [take(coh, PAPER_NUM + PAPER_BIN, "paper")]
    for c, levels in PAPER_CAT.items():
        dm = pd.get_dummies(coh[c].astype(int), prefix=c).reindex(
            columns=[f"{c}_{l}" for l in levels], fill_value=0).astype(float)
        dm = dm.drop(columns=[dm.columns[0]])          # reference level dropped
        blocks.setdefault("paper", []).extend(dm.columns)
        parts.append(dm)

    if featureset in ("base", "traj"):
        parts.append(take(coh, EXTRA_NUM, "base"))
        ex = _raw_extra()
        assert (ex.wid.values == coh.wid.values).all()
        parts.append(take(ex, EXTRA_BIN + EXTRA_NUM2 + ["base_year"], "base"))
    if featureset == "traj":
        tr = trajectory_features(pd.read_pickle(f"{JD}/results/waves.pkl"))
        tr = tr.set_index("wid").reindex(coh.wid.values).reset_index(drop=True)
        tr = tr.drop(columns=[c for c in ("span",) if c in tr.columns])
        parts.append(take(tr, [c for c in tr.columns if c != "wid"], "traj"))

    X = pd.concat(parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.astype(float)
    meta = coh[["wid", "py", "years", "n_visits",
                "free_caries", "free_ging", "free_both",
                "ev_caries", "ev_ging", "ev_comb"]].copy()
    meta["base_year"] = _raw_extra().base_year.values
    aux = [c for c in coh.columns if c.endswith(("_incidencia", "_final"))]
    for c in aux:
        meta[c] = coh[c].values
    blocks = {k: [c for c in v if c in X.columns] for k, v in blocks.items()}
    pd.to_pickle((X, meta, blocks), cache)
    return X, meta, blocks


if __name__ == "__main__":
    for fs in ("paper", "base", "traj"):
        X, meta, blocks = build(fs)
        print(f"{fs:6s} -> {X.shape[1]:3d} features, "
              f"missing {100 * X.isna().mean().mean():.1f}% of cells, "
              f"blocks { {k: len(v) for k, v in blocks.items()} }")
    X, meta, blocks = build("traj")
    X.to_pickle(f"{JD}/results/X_traj.pkl")
    meta.to_pickle(f"{JD}/results/meta.pkl")
    print("\nfeature list:")
    for b, v in blocks.items():
        print(f"  {b}: {v}")
