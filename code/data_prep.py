"""Build the occupational cohort and visit-level tables for MOSAIC.

The released follow-up field records the first-to-last observation span in 30-day
months. Dental occurrence flags supervise a single observation window. Auxiliary
fields include both occurrence flags and terminal health states. verify_structure
checks cohort counts and date-span consistency against the source dataset.
"""
import numpy as np
import pandas as pd

from config import JD
from config import DTA

# time-varying blocks that exist across the repeated examinations
WAVE_VARS = ["peso", "pas", "pad", "cintura", "cadera", "glucosa", "colesterol",
             "trigliceridos", "hemoglobina"]
# the reference paper's baseline covariate set
BASE_NUM = ["edad1", "pas1", "pad1", "cintura1", "imc1", "glucosa1", "colesterol1",
            "trigliceridos1"]
BASE_CAT = ["sexo", "occ", "noche", "expo", "fuma", "alcohol"]
# Auxiliary occurrence flags and terminal health states
OUTCOMES = ["caries_incidencia", "gingivitis_incidencia", "ametropia_incidencia",
            "onicomicosis_incidencia", "pterigion_incidencia", "estereopsis_incidencia",
            "discromatopsia_incidencia", "depresion_final", "estres_final", "ansiedad_final",
            "acv_final", "infarto_final", "cardiovascular_final", "lumbalgia_final",
            "PatronObstructivo_final", "PatronRestrictivo_final", "audiometria_final"]


def load_raw():
    return pd.read_stata(DTA, convert_categoricals=False)


def visit_dates(df):
    """Parsed visit dates for waves 1..7 as a DataFrame."""
    out = {}
    for w in range(1, 8):
        c = f"fecha{w}"
        out[w] = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
    return pd.DataFrame(out, index=df.index)


def build_cohort(raw):
    """The 12,049 eligible workers with the reference paper's derived variables."""
    f = raw[raw.TiempoSeguimiento.notna()].copy()
    f = f.reset_index(drop=True)
    f["wid"] = np.arange(len(f))

    f["py"] = f.TiempoSeguimiento * 30.0 / 365.25          # person-years, paper's convention
    f["years"] = f.py                                       # observation window length
    f["occ"] = f.tipo_de_ocupacion.replace({4: 2}).astype(int)
    f["noche"] = f.TrabajoNocturno.astype(int)
    f["fuma"] = f.antec_fumar1.astype(int)
    f["alcohol"] = f.consumo_alcohol1.astype(int)
    # job tenure at baseline, in the article's four bands
    f["expo"] = np.digitize(f.tiempomeses1, [12, 60, 120])
    f["n_visits"] = visit_dates(f).notna().sum(axis=1)

    # the three subcohort masks and their event flags
    f["free_caries"] = (f.caries_prevalencia == 0).astype(int)
    f["free_ging"] = (f.gingivitis_prevalencia == 0).astype(int)
    f["free_both"] = ((f.caries_prevalencia == 0) & (f.gingivitis_prevalencia == 0)).astype(int)
    f["ev_caries"] = (f.caries_incidencia == 1).astype(int)
    f["ev_ging"] = (f.gingivitis_incidencia == 1).astype(int)
    f["ev_comb"] = ((f.caries_incidencia == 1) | (f.gingivitis_incidencia == 1)).astype(int)
    return f


def build_waves(f):
    """Long format: one row per worker-visit with the time-varying measurements."""
    dates = visit_dates(f)
    rows = []
    for w in range(1, 8):
        cols = {}
        for v in WAVE_VARS:
            c = f"{v}{w}"
            cols[v] = f[c] if c in f.columns else np.nan
        # smoking / alcohol are re-asked at each wave under inconsistent names
        for nm, cands in (("fuma", [f"antec_fumar{w}"]),
                          ("alcohol", [f"antec_alcohol{w}", f"consumo_alcohol{w}",
                                       f"antec_alcoho{w}"])):
            col = next((c for c in cands if c in f.columns), None)
            cols[nm] = f[col] if col else np.nan
        blk = pd.DataFrame(cols)
        blk["wid"] = f.wid.values
        blk["wave"] = w
        blk["date"] = dates[w].values
        rows.append(blk)
    long = pd.concat(rows, ignore_index=True)
    base = long[long.wave == 1].set_index("wid")["date"]
    long["t_years"] = (long.date - long.wid.map(base)).dt.days / 365.25
    # a visit row is informative if it has a date or any measurement
    meas = WAVE_VARS + ["fuma", "alcohol"]
    long["observed"] = long.date.notna() | long[meas].notna().any(axis=1)
    return long[long.observed].drop(columns="observed").reset_index(drop=True)


def verify_structure(raw, f):
    """Assert the facts the whole analysis rests on; raises if the released file ever changes."""
    checks = {}
    checks["n_raw"] = (len(raw), 79660)
    checks["n_cohort"] = (len(f), 12049)
    checks["n_excluded"] = (len(raw) - len(f), 67611)
    checks["free_caries"] = (int(f.free_caries.sum()), 11888)
    checks["free_ging"] = (int(f.free_ging.sum()), 11984)
    checks["free_both"] = (int(f.free_both.sum()), 11851)
    checks["ev_caries_sub"] = (int(f.loc[f.free_caries == 1, "ev_caries"].sum()), 470)
    checks["ev_ging_sub"] = (int(f.loc[f.free_ging == 1, "ev_ging"].sum()), 149)
    checks["ev_comb_sub"] = (int(f.loc[f.free_both == 1, "ev_comb"].sum()), 542)
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"cohort construction no longer matches the article: {bad}")

    # person-time == first-to-last visit span, including for cases
    dates = visit_dates(f)
    first = dates[1]
    last = dates.ffill(axis=1).iloc[:, -1]
    span = (last - first).dt.days
    dev = (span - f.TiempoSeguimiento * 30).abs()
    n_ok = int((dev <= 0.5).sum())
    cases = (f.ev_caries == 1) | (f.ev_ging == 1)
    dev_cases = float(np.nanmax(dev[cases].values))
    if dev_cases > 0.5:
        raise SystemExit(f"cases are censored at detection after all (max dev {dev_cases})")
    return dict(span_matches=n_ok, n=len(f), max_dev_cases=dev_cases)


def main():
    raw = load_raw()
    f = build_cohort(raw)
    info = verify_structure(raw, f)
    print(f"structure verified: {info['span_matches']}/{info['n']} spans match "
          f"TiempoSeguimiento*30; max deviation over all cases = {info['max_dev_cases']:.2f} d")

    keep = (["wid", "py", "years", "n_visits", "TiempoSeguimiento", "tiempomeses1",
             "talla_cm1", "peso1", "hemoglobina1", "cadera1", "familiar_dm", "familiar_hta",
             "tiempo_sentado", "caries_prevalencia", "gingivitis_prevalencia",
             "free_caries", "free_ging", "free_both", "ev_caries", "ev_ging", "ev_comb"]
            + BASE_NUM + BASE_CAT + OUTCOMES)
    keep = list(dict.fromkeys(keep))
    coh = f[keep].copy()
    coh.to_pickle(f"{JD}/results/cohort.pkl")
    print(f"cohort.pkl: {coh.shape[0]} x {coh.shape[1]}")

    waves = build_waves(f)
    waves.to_pickle(f"{JD}/results/waves.pkl")
    print(f"waves.pkl: {waves.shape[0]} worker-visits, "
          f"{waves.groupby('wid').size().mean():.2f} per worker")
    print(waves.groupby("wave").size().to_dict())


if __name__ == "__main__":
    main()
