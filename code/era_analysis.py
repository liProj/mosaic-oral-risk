"""Estimate outcome rates and covariate associations across baseline years.

Compares crude Cox associations with year-adjusted, regularized estimates. Both
year adjustment and regularization change between fits. These are descriptive
associations and do not identify causes of calendar-year differences.
"""
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2

from config import JD
OUT = f"{JD}/results/era"

SUB = {"caries": ("free_caries", "ev_caries"),
       "gingivitis": ("free_ging", "ev_ging"),
       "combined": ("free_both", "ev_comb")}

CATS = [("Type of occupation", "occ", {0: "Office", 1: "Manual/Physical",
                                       2: "Customer service/Social services",
                                       3: "Healthcare professional"}),
        ("Years of exposure", "expo", {0: "<1 year", 1: "1 to <5 years",
                                       2: "5 to <10 years", 3: "≥10 years"}),
        ("Sex", "sexo", {0: "Female", 1: "Male"}),
        ("Smoker", "fuma", {0: "No", 1: "Yes"}),
        ("Alcohol consumption", "alcohol", {0: "No", 1: "Yes"}),
        ("Night work", "noche", {0: "No", 1: "Yes"})]

CONT = [("Age in years", "edad1"), ("Abdominal waist", "cintura1"),
        ("Systolic blood pressure", "pas1"), ("Diastolic blood pressure", "pad1"),
        ("Glucose", "glucosa1"), ("Triglycerides", "trigliceridos1"),
        ("Total cholesterol", "colesterol1")]


def with_year(coh):
    raw = pd.read_stata(f"{JD}/data/Base Ocupacional.dta", convert_categoricals=False)
    f = raw[raw.TiempoSeguimiento.notna()].reset_index(drop=True)
    coh = coh.copy()
    coh["base_year"] = pd.to_datetime(f.fecha1, format="%d/%m/%Y",
                                      errors="coerce").dt.year.values
    return coh


def rate_by_year(coh):
    rows = []
    for ds, (free, ev) in SUB.items():
        s = coh[coh[free] == 1]
        for yr, g in s.groupby("base_year"):
            e = int(g[ev].sum())
            py = float(g.py.sum())
            r = 1000 * e / py if py else np.nan
            lo = 1000 * chi2.ppf(0.025, 2 * e) / 2 / py if e > 0 else 0.0
            hi = 1000 * chi2.ppf(0.975, 2 * e + 2) / 2 / py if py else np.nan
            rows.append(dict(dataset=ds, year=int(yr), n=len(g), events=e,
                             py=round(py, 1), rate=round(r, 2),
                             lo=round(lo, 2), hi=round(hi, 2),
                             share_of_events=round(100 * e / int(s[ev].sum()), 1)))
    return pd.DataFrame(rows)


def era_composition(coh):
    """How the era lines up with the characteristics the article stratifies on."""
    s = coh[coh.free_caries == 1].copy()
    s["y2015"] = (s.base_year == 2015).astype(int)
    rows = []
    for blk, col, labs in CATS:
        for k, nm in labs.items():
            g = s[s[col] == k]
            rows.append(dict(block=blk, label=nm, n=len(g),
                             pct_2015_baseline=round(100 * g.y2015.mean(), 1)))
    return pd.DataFrame(rows)


def _fit_adjusted(d, yd, keep_col):
    """Cox adjusted for baseline-year dummies, made robust to thin year strata.

    Subsetting to the workers who have a given laboratory measurement can leave a baseline year
    with no events or no variation, which makes the design singular; those year columns are
    dropped and the penalty is raised until the fit converges.
    """
    yd = yd.loc[:, yd.std() > 0]
    ev = d.e.values.astype(bool)
    keep = [c for c in yd.columns if yd[c].values[ev].sum() >= 3 and yd[c].sum() >= 20]
    yd = yd[keep]
    for pen in (0.05, 0.2, 1.0, 5.0):
        try:
            return CoxPHFitter(penalizer=pen).fit(pd.concat([d, yd], axis=1), "py", "e")
        except Exception:
            continue
    # last resort: adjust for the single era indicator rather than all years
    yr2015 = yd.get("yr_2015")
    cols = pd.DataFrame({"yr_2015": yr2015}) if yr2015 is not None else pd.DataFrame(index=d.index)
    return CoxPHFitter(penalizer=1.0).fit(pd.concat([d, cols], axis=1), "py", "e")


def crude_vs_adjusted(coh):
    """Every contrast the article prints, crude and adjusted for baseline year."""
    rows = []
    for ds, (free, ev) in SUB.items():
        s = coh[coh[free] == 1].copy().reset_index(drop=True)
        s["e"] = s[ev].astype(int)
        yd = pd.get_dummies(s.base_year.fillna(0).astype(int), prefix="yr").astype(float)
        yd = yd.drop(columns=[yd.columns[0]])

        for blk, col, labs in CATS:
            dm = pd.get_dummies(s[col].astype(int), prefix="g").astype(float)
            ref = sorted(labs)[0]
            dm = dm.drop(columns=[f"g_{ref}"])
            crude = CoxPHFitter().fit(pd.concat([dm, s[["py", "e"]]], axis=1), "py", "e")
            adj = _fit_adjusted(pd.concat([dm, s[["py", "e"]]], axis=1), yd, None)
            for k, nm in labs.items():
                if k == ref:
                    continue
                c = f"g_{k}"
                rows.append(dict(
                    dataset=ds, block=blk, label=nm,
                    crude_hr=float(np.exp(crude.params_[c])),
                    crude_lo=float(np.exp(crude.confidence_intervals_.loc[c].iloc[0])),
                    crude_hi=float(np.exp(crude.confidence_intervals_.loc[c].iloc[1])),
                    adj_hr=float(np.exp(adj.params_[c])),
                    adj_lo=float(np.exp(adj.confidence_intervals_.loc[c].iloc[0])),
                    adj_hi=float(np.exp(adj.confidence_intervals_.loc[c].iloc[1]))))
        for blk, col in CONT:
            d = s[[col, "py", "e"]].dropna().copy()
            d[col] = d[col] / 5.0
            crude = CoxPHFitter().fit(d, "py", "e")
            adj = _fit_adjusted(d, yd.loc[d.index], col)
            rows.append(dict(
                dataset=ds, block=blk, label="per 5-unit increase",
                crude_hr=float(np.exp(crude.params_[col])),
                crude_lo=float(np.exp(crude.confidence_intervals_.loc[col].iloc[0])),
                crude_hi=float(np.exp(crude.confidence_intervals_.loc[col].iloc[1])),
                adj_hr=float(np.exp(adj.params_[col])),
                adj_lo=float(np.exp(adj.confidence_intervals_.loc[col].iloc[0])),
                adj_hi=float(np.exp(adj.confidence_intervals_.loc[col].iloc[1]))))
    out = pd.DataFrame(rows)
    out["pct_change"] = (100 * (out.adj_hr - out.crude_hr) / out.crude_hr).round(1)
    out["attenuated_to_null"] = ((out.crude_lo > 1) | (out.crude_hi < 1)) & \
                                ((out.adj_lo <= 1) & (out.adj_hi >= 1))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    coh = with_year(pd.read_pickle(f"{JD}/results/cohort.pkl"))
    ry = rate_by_year(coh)
    ry.to_csv(f"{OUT}/rate_by_year.csv", index=False)
    ec = era_composition(coh)
    ec.to_csv(f"{OUT}/era_composition.csv", index=False)
    cv = crude_vs_adjusted(coh)
    cv.to_csv(f"{OUT}/crude_vs_adjusted.csv", index=False)

    print("--- caries incidence per 1000 py by baseline year")
    print(ry[ry.dataset == "caries"][["year", "n", "events", "rate", "lo", "hi",
                                      "share_of_events"]].to_string(index=False))
    print("\n--- share of the cohort with a 2015 baseline, by tenure band")
    print(ec[ec.block == "Years of exposure"].to_string(index=False))
    print("\n--- crude vs baseline-year-adjusted hazard ratios (caries)")
    c = cv[(cv.dataset == "caries")]
    print(c[["block", "label", "crude_hr", "adj_hr", "pct_change",
             "attenuated_to_null"]].round(3).to_string(index=False))
    n_att = int(cv.attenuated_to_null.sum())
    print(f"\ncontrasts that lose their significance once baseline year is held fixed: "
          f"{n_att}/{len(cv)}")


if __name__ == "__main__":
    main()
