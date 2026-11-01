"""
evaluation.py -- performance, calibration, selective prediction and the two main-text tables

Paper:    Table 3 (AUROC, paired loss, AUPRC lift, calibration slope; single models and deep ensembles),
          Table 4 (nine uncertainty measures x three levels x three families), Results RQ2b
Origin:   performance/metrics.py, calibration.py, auprc.py, ensemble_performance.py, levels/selective_prediction.py,
          levels/level3_ensembles_mc.py, tables_figures/table3.py of the full package, merged; the Table-4 section
          is the cell computation of tables_figures/figure4_levels.py without the drawing, with the measure names
          and the rounding rule of tables_figures/supplementary_tables.py
Reads:    config.MODELS checkpoints (S6), config.SIGNALS/scores_*.npz and mc_mi_*.npz (patient-level, stay local),
          the aggregate CSVs of config.SIGNALS and config.REPORT
Writes:   config.REPORT/leistung_natural.csv, reliability_bins_natural.csv, ebene3_natural.csv, kurven_natural.npz,
          kalibrierung_kennzahlen.csv, auprc_natural.csv, ebene3_e14.csv;
          config.APPENDIX/table_2_ensemble_rows.csv/.md, table_3.csv/.md, table_3_cells.csv/.md (= Table 4)
Run:      python evaluation.py --evaluate       (S6 performance / calibration and S7 level 3)
          python evaluation.py --performance    (AUPRC, calibration slope, ensemble performance)
          python evaluation.py --levels         (level 3 of the deep ensembles and the Monte Carlo mutual information)
          python evaluation.py --tables         (Table 3, Table 4)
          or through run_pipeline.py --evaluate / --performance / --levels / --tables / --all
"""
from __future__ import annotations

import argparse
import csv
import numpy as np
import os
import pandas as pd
import re
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import common as C  # noqa: E402
from models import architectures as A  # noqa: E402


# ==========================================================================================
# section from the full package: performance/metrics.py
# ------------------------------------------------------------------------------------------
# metrics.py -- discrimination and calibration of the deterministic models: AUROC, Brier,
#               ECE raw and after temperature scaling, reliability bins
#
# Paper:    Table 3 (AUROC, ECE, temperature), Figure 1 and SF5 (reliability bins)
# Origin:   src/rerun_pipeline.py v5.7 / 2026-08-20, _logits(), _temperatur(), _ece(),
#           s6_leistung() (stage S6)
# Reads:    config.MODELS/modell_ihm_<family>_seed<n>.pt of the deterministic families;
#           data via common.lade_natural / common.lade_eicu
# Writes:   config.REPORT/leistung_natural.csv (familie, seed, domaene, auroc, brier, ece_roh,
#           ece_temperatur, temperatur), config.REPORT/reliability_bins_natural.csv (familie,
#           domaene, skala, seed, bin, kante_lo, kante_hi, mean_pred, obs_rate, n)
# Run:      python performance/metrics.py   (or run_pipeline.py --evaluate)
#
# Temperature: T > 0 by minimising the BCE on the validation part of the split (LBFGS on
# log T). ECE: decile edges of the probabilities pooled over the seeds, per family and
# domain; edges 0 and 1 forced; the same edges for the raw and the temperature-scaled
# probabilities.
# ==========================================================================================

def _logits(modell, X, batch=1024):
    modell.eval()
    raus = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            raus.append(modell(X[i:i + batch].to(A.GERAET)).cpu())
    return torch.cat(raus)


def _temperatur(logits_val, y_val):
    """T > 0 by NLL minimisation on the validation part (calibration is measured and
    corrected; labels are never redefined)."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)
    lv, yv = logits_val.detach(), y_val.detach()

    def schritt():
        opt.zero_grad()
        v = F.binary_cross_entropy_with_logits(lv / torch.exp(logT), yv)
        v.backward()
        return v
    opt.step(schritt)
    return float(torch.exp(logT).item())


def _ece(p, y, kanten):
    idx = np.clip(np.searchsorted(kanten, p, side="right") - 1, 0,
                  len(kanten) - 2)
    e, N = 0.0, len(p)
    for b in range(len(kanten) - 1):
        m = idx == b
        if m.any():
            e += m.sum() / N * abs(y[m].mean() - p[m].mean())
    return float(e)


def s6_leistung(daten, eicu):
    """Performance and calibration per deterministic family, seed and domain."""
    from sklearn.metrics import roc_auc_score
    dom_X = {"in_domain": daten["Xte"], "transfer": eicu["X"]}
    dom_y = {"in_domain": daten["yte"].numpy().astype(int),
             "transfer": eicu["y"].astype(int)}
    z_l, z_r = [], []
    for familie, (arch, variante, _) in C.FAMILIEN.items():
        if variante != "det":
            continue
        roh, ska, temps = {d: {} for d in dom_X}, {d: {} for d in dom_X}, {}
        for seed in C.SEEDS:
            ck = C._finde(os.path.join(C.ORDNER["modelle"],
                                       f"modell_{C.TASK}_{familie}_seed{seed}.pt"))
            if ck is None:
                continue
            c = torch.load(ck, map_location="cpu", weights_only=False)
            C.seed_alles(seed)
            m = C.baue_modell_natural(c["best_hp"], familie)
            m.load_state_dict(c["state_dict"])
            Tw = _temperatur(_logits(m, daten["Xval"]), daten["yval"])
            temps[seed] = Tw
            for dom in dom_X:
                lg = _logits(m, dom_X[dom])
                roh[dom][seed] = torch.sigmoid(lg).numpy()
                ska[dom][seed] = torch.sigmoid(lg / Tw).numpy()
            del m
        if not temps:
            continue
        for dom in dom_X:
            y = dom_y[dom]
            pool = np.concatenate(list(roh[dom].values()))
            kanten = np.unique(np.quantile(pool, np.linspace(0, 1, 11)))
            kanten[0], kanten[-1] = 0.0, 1.0
            for seed in roh[dom]:
                pr, ps = roh[dom][seed], ska[dom][seed]
                z_l.append([familie, seed, dom,
                            round(roc_auc_score(y, pr), 5),
                            round(float(np.mean((pr - y) ** 2)), 5),
                            round(_ece(pr, y, kanten), 5),
                            round(_ece(ps, y, kanten), 5),
                            round(temps[seed], 4)])
                for skala, pp in (("roh", pr), ("temperatur", ps)):
                    idx = np.clip(np.searchsorted(kanten, pp, side="right")
                                  - 1, 0, len(kanten) - 2)
                    for b in range(len(kanten) - 1):
                        msk = idx == b
                        if msk.any():
                            z_r.append([familie, dom, skala, seed, b,
                                        round(float(kanten[b]), 5),
                                        round(float(kanten[b + 1]), 5),
                                        round(float(pp[msk].mean()), 5),
                                        round(float(y[msk].mean()), 5),
                                        int(msk.sum())])
        C.log(f"[S6] {familie}: performance / calibration over "
              f"{len(temps)} seeds (decile edges per family x domain, "
              "pooled over seeds).")
    for name, kopf, zeilen in (
            ("leistung_natural.csv",
             ["familie", "seed", "domaene", "auroc", "brier", "ece_roh",
              "ece_temperatur", "temperatur"], z_l),
            ("reliability_bins_natural.csv",
             ["familie", "domaene", "skala", "seed", "bin", "kante_lo",
              "kante_hi", "mean_pred", "obs_rate", "n"], z_r)):
        ziel = C._sicher(C.ORDNER["report"], name)
        with open(ziel, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(kopf)
            w.writerows(zeilen)
        C.log(f"[S6] written: {ziel} ({len(zeilen)} rows)")


def main_metrics(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    config.ensure_run_folders()
    daten = C.lade_natural()
    s6_leistung(daten, C.lade_eicu(daten))



# ==========================================================================================
# section from the full package: performance/calibration.py
# ------------------------------------------------------------------------------------------
# calibration.py -- patient-level calibration (slope, intercept, CITL) and the decile table
#
# Paper:    calibration slope / intercept in Results 1 and Table 3, over- and under-estimation
#           by decile (reliability curves, Figure 1 and SF5); imported by supporting/best_epoch.py
# Origin:   src/kalibrierung_kennzahlen.py, SKRIPT_STAND_CAL
#           "v1.1 / 2026-08-24 (A2 als interne Konsistenz deklariert)"
# Reads:    config.SIGNALS/scores_<family>_seed<seed>.npz   (p_<domain>, y_<domain>, patient-level)
#           config.REPORT/leistung_natural.csv               (brier, temperatur, ece_roh, ece_temperatur)
# Writes:   config.REPORT/kalibrierung_kennzahlen.csv        (regression quantities, long format)
#           config.REPORT/kalibrierung_dezile.csv            (decile detail, pooled over seeds)
# Checks:   A1  Brier recomputed from p and y matches the column `brier` of leistung_natural.csv
#               (tolerance 6e-6, calibrated to the five-decimal storage precision)
#           A2  internal consistency of the temperature scaling: dividing the logit by T must
#               multiply the slope by T and leave the intercept unchanged (holds for any T; it
#               tests logit arithmetic, scaling and regression against each other, not T itself)
#           A3  IRLS converges (change < 1e-10 within 100 iterations)
# Run:      python performance/calibration.py
#
# What is computed (per family, seed, domain and scale "roh" / "temperatur")
#   slope, intercept   logistic regression y ~ 1 + logit(p) by IRLS without regularisation;
#                      slope = 1 and intercept = 0 are ideal, slope < 1 means predictions that
#                      are spread too wide (under-estimation at the bottom, over-estimation at the top)
#   citl               calibration-in-the-large: intercept of the offset model with slope fixed at 1
#   pred_mittel, obs_mittel, ece_dezil, n
#   decile table       per decile of the pooled raw predictions: n, mean prediction, observed rate,
#                      ratio -- the numerical basis of the reliability curves
#
# Diagnostic (reported, never aborts): the decile ECE computed here is set against `ece_roh` and
# `ece_temperatur` of leistung_natural.csv. A deviation is expected because the pipeline may set
# its bin edges differently (edges from the RAW deciles for both scales).
#
# Only aggregates are written; patient-level files are read locally.
# ==========================================================================================

SKRIPT_STAND_CAL = "package 1.0 / 2026-10-08 (port of kalibrierung_kennzahlen.py v1.1)"
SEEDS = config.SEEDS
DOMS = ("in_domain", "transfer")
EPS = 1e-7
ANKER_TOLERANZ = 6e-6
T9 = 2.262          # t quantile, 97.5 %, 9 degrees of freedom (ten seeds)
N_BINS = 10


def abbruch(t):
    raise SystemExit("ABORT (fail-fast): " + t)


def _sicher(name):
    """Output path inside config.REPORT (guards against path escapes)."""
    config.REPORT.mkdir(parents=True, exist_ok=True)
    ziel = (config.REPORT / name).resolve()
    if config.REPORT.resolve() not in ziel.parents:
        abbruch("output path outside of " + str(config.REPORT))
    return ziel


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def irls(X, y, offset=None, max_iter=100):
    """Logistic regression by iteratively reweighted least squares.
    No regularisation term, so that the slope stays unbiased."""
    n, k = X.shape
    beta = np.zeros(k)
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    for _ in range(max_iter):
        eta = X @ beta + off
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        z = eta - off + (y - mu) / w
        A = X.T @ (w[:, None] * X)
        b = X.T @ (w * z)
        try:
            neu = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            abbruch("A3: IRLS singular (degenerate predictions?).")
        if np.max(np.abs(neu - beta)) < 1e-10:
            return neu
        beta = neu
    abbruch("A3: IRLS did not converge.")


def kalibrierung(p, y):
    """Returns (slope, intercept) of the free model and CITL (slope fixed at 1)."""
    L = logit(p)
    X = np.column_stack([np.ones_like(L), L])
    b = irls(X, y)
    citl = irls(np.ones((len(L), 1)), y, offset=L)[0]
    return float(b[1]), float(b[0]), float(citl)


def ece_dezil(p, y, kanten):
    """Decile ECE with the given bin edges (quantiles of the predictions)."""
    idx = np.clip(np.digitize(p, kanten[1:-1]), 0, N_BINS - 1)
    e = 0.0
    for b in range(N_BINS):
        m = idx == b
        if m.sum():
            e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(e)


def skaliere(p, T):
    """Temperature scaling of probabilities: sigmoid(logit(p) / T)."""
    return 1.0 / (1.0 + np.exp(-logit(p) / float(T)))


def mki(w):
    """Mean and half-width of the 95 % t-interval over the seeds."""
    w = np.asarray(w, float)
    m = float(w.mean())
    return m, T9 * float(w.std(ddof=1)) / np.sqrt(len(w))


def main_calibration(argv=None):
    print("=" * 78)
    print("CALIBRATION METRICS  (patient level)")
    print(f"  script version: {SKRIPT_STAND_CAL}")
    print("=" * 78)
    pfad_le = config.REPORT / "leistung_natural.csv"
    if not pfad_le.exists():
        abbruch(f"{pfad_le} is missing (run performance/metrics.py first).")
    le = pd.read_csv(pfad_le)

    fam = {}
    for pf in config.SIGNALS.glob("scores_*_seed*.npz"):
        m = re.match(r"scores_(.+)_seed(\d+)\.npz$", pf.name)
        if m:
            fam.setdefault(m.group(1), []).append(int(m.group(2)))
    # families of the package with the complete set of seeds (alphabetical, as in the original run)
    familien = sorted(f for f, s in fam.items()
                      if sorted(s) == SEEDS and f in config.ALL_FAMILIES)
    if not familien:
        abbruch("no family with the complete set of 10 seeds found.")
    print(f"  families: {', '.join(familien)}")

    zeilen, dez_zeilen = [], []
    for familie in familien:
        pool = {dom: {"p": [], "y": []} for dom in DOMS}
        for seed in SEEDS:
            z = np.load(config.SIGNALS / f"scores_{familie}_seed{seed}.npz")
            for dom in DOMS:
                p = np.asarray(z[f"p_{dom}"], float).ravel()
                y = np.asarray(z[f"y_{dom}"], float).ravel()
                r = le[(le.familie == familie) & (le.seed == seed)
                       & (le.domaene == dom)]
                if len(r) != 1:
                    abbruch(f"leistung: {len(r)} rows for "
                            f"{familie}/{seed}/{dom}.")
                d_ = abs(float(((p - y) ** 2).mean()) - float(r.brier.iloc[0]))
                if d_ > ANKER_TOLERANZ:
                    abbruch(f"A1: Brier {familie}/{seed}/{dom} deviates by "
                            f"{d_:.2e} -- not the same predictions.")
                T = float(r.temperatur.iloc[0])
                pool[dom]["p"].append(p)
                pool[dom]["y"].append(y)

                for skala, pp in (("roh", p), ("temperatur", skaliere(p, T))):
                    s_, i_, c_ = kalibrierung(pp, y)
                    kanten = np.quantile(pp, np.linspace(0, 1, N_BINS + 1))
                    zeilen.append({
                        "familie": familie, "seed": seed, "domaene": dom,
                        "skala": skala, "n": len(p), "temperatur": round(T, 4),
                        "slope": round(s_, 5), "intercept": round(i_, 5),
                        "citl": round(c_, 5),
                        "pred_mittel": round(float(pp.mean()), 5),
                        "obs_mittel": round(float(y.mean()), 5),
                        "ece_dezil": round(ece_dezil(pp, y, kanten), 5)})
                # A2: identity of the temperature scaling
                s_roh = zeilen[-2]["slope"]
                s_tmp = zeilen[-1]["slope"]
                if abs(s_tmp - s_roh * T) > 1e-4 * max(1.0, abs(s_tmp)):
                    abbruch(f"A2: {familie}/{seed}/{dom}: slope_temp "
                            f"{s_tmp:.5f} != slope_roh*T "
                            f"{s_roh * T:.5f} -- scaling or regression "
                            "is faulty.")
                if abs(zeilen[-1]["intercept"] - zeilen[-2]["intercept"]) > 1e-3:
                    abbruch(f"A2: {familie}/{seed}/{dom}: intercept changes "
                            "under scaling.")

        # decile detail on the raw predictions pooled over the seeds
        for dom in DOMS:
            p = np.concatenate(pool[dom]["p"])
            y = np.concatenate(pool[dom]["y"])
            kanten = np.quantile(p, np.linspace(0, 1, N_BINS + 1))
            idx = np.clip(np.digitize(p, kanten[1:-1]), 0, N_BINS - 1)
            for b in range(N_BINS):
                m = idx == b
                if not m.sum():
                    continue
                pm, om = float(p[m].mean()), float(y[m].mean())
                dez_zeilen.append({
                    "familie": familie, "domaene": dom, "dezil": b + 1,
                    "n": int(m.sum()), "kante_lo": round(float(kanten[b]), 6),
                    "kante_hi": round(float(kanten[b + 1]), 6),
                    "pred_mittel": round(pm, 6), "obs_rate": round(om, 6),
                    "verhaeltnis": round(om / pm, 4) if pm > 0 else np.nan})
        print(f"  [OK] {familie}  (A1-A3 passed)")

    df = pd.DataFrame(zeilen)
    dz = pd.DataFrame(dez_zeilen)
    df.to_csv(_sicher("kalibrierung_kennzahlen.csv"), index=False)
    dz.to_csv(_sicher("kalibrierung_dezile.csv"), index=False)
    print(f"\n  written: kalibrierung_kennzahlen.csv ({len(df)} rows)")
    print(f"           kalibrierung_dezile.csv ({len(dz)} rows)")

    print("\n" + "-" * 78)
    print("SLOPE and INTERCEPT, raw, mean +- 95 % interval over 10 seeds")
    print(f"{'Family':<12}{'Domain':<11}{'Slope':>18}{'Intercept':>18}"
          f"{'CITL':>10}{'ECE':>9}")
    for familie in familien:
        for dom in DOMS:
            g = df[(df.familie == familie) & (df.domaene == dom)
                   & (df.skala == "roh")]
            s, hs = mki(g.slope)
            i_, hi = mki(g.intercept)
            print(f"{familie:<12}{dom:<11}{s:>10.4f} ±{hs:>6.4f}"
                  f"{i_:>10.4f} ±{hi:>6.4f}{g.citl.mean():>10.4f}"
                  f"{g.ece_dezil.mean():>9.4f}")

    print("\n" + "-" * 78)
    print("OVER- and UNDER-ESTIMATION: lowest 3 deciles and top decile "
          "(raw, pooled)")
    print(f"{'Family':<12}{'Domain':<11}{'low pred':>12}{'low obs':>11}"
          f"{'factor':>8}{'top pred':>11}{'top obs':>10}{'factor':>8}")
    for familie in familien:
        for dom in DOMS:
            g = dz[(dz.familie == familie) & (dz.domaene == dom)]
            lo = g[g.dezil <= 3]
            hi = g[g.dezil == N_BINS]
            lp = (lo.pred_mittel * lo.n).sum() / lo.n.sum()
            lo_ = (lo.obs_rate * lo.n).sum() / lo.n.sum()
            hp = float(hi.pred_mittel.iloc[0])
            ho = float(hi.obs_rate.iloc[0])
            print(f"{familie:<12}{dom:<11}{lp:>12.4f}{lo_:>11.4f}"
                  f"{lo_ / lp:>7.1f}x{hp:>11.4f}{ho:>10.4f}{ho / hp:>7.2f}x")

    print("\n" + "-" * 78)
    print("DIAGNOSTIC: own decile ECE against the stored values "
          "(a deviation is expected)")
    print(f"{'Family':<12}{'Domain':<11}{'raw own':>11}{'raw stored':>11}"
          f"{'diff':>9}{'temp own':>12}{'temp stored':>12}{'diff':>9}")
    for familie in familien:
        for dom in DOMS:
            g = df[(df.familie == familie) & (df.domaene == dom)]
            gr = le[(le.familie == familie) & (le.domaene == dom)]
            er = g[g.skala == "roh"].ece_dezil.mean()
            et = g[g.skala == "temperatur"].ece_dezil.mean()
            sr, st = gr.ece_roh.mean(), gr.ece_temperatur.mean()
            print(f"{familie:<12}{dom:<11}{er:>11.4f}{sr:>11.4f}{er - sr:>+9.4f}"
                  f"{et:>12.4f}{st:>12.4f}{et - st:>+9.4f}")

    print("\n  Slope 1 and intercept 0 are ideal. Slope < 1 means predictions")
    print("  spread too wide: under-estimation at the bottom, over-estimation")
    print("  at the top. CITL measures location alone (slope fixed at 1).")
    print("  The decile table is the numerical basis of the reliability curves.")
    print("=" * 78)



# ==========================================================================================
# section from the full package: performance/auprc.py
# ------------------------------------------------------------------------------------------
# auprc.py -- AUPRC (average precision), prevalence and lift on the MIMIC test set and eICU
#
# Paper:    AUPRC and AUPRC lift columns of Table 3 (performance under domain shift)
# Origin:   src/auprc_natural.py, SKRIPT_STAND_AUPRC "v1 / 2026-08-22 (AUPRC fuer Tabelle 3, post hoc)"
# Reads:    config.SIGNALS/scores_<family>_seed<seed>.npz   (p_<domain>, y_<domain>, patient-level)
#           config.REPORT/leistung_natural.csv               (brier, auroc)
# Writes:   config.REPORT/auprc_natural.csv                  (one row per family, seed, domain)
# Checks:   Brier recomputed from the same p and y must match the column `brier` of
#           leistung_natural.csv (tolerance 6e-6, calibrated to the five-decimal storage
#           precision; same logic as cross_entropy.py)
# Run:      python performance/auprc.py
#
# What is computed (per family, seed, domain)
#   auprc        average precision (sklearn average_precision_score, step-wise area under the
#                precision-recall curve)
#   praevalenz   share of positive cases = AUPRC baseline of a random ranking of the same cohort.
#                Raw AUPRC values are comparable across cohorts only together with this baseline.
#   auprc_lift   auprc / praevalenz (factor over chance, comparable across cohorts, analogous to
#                the skill of the cross-entropy)
#
# Only aggregates are written; patient-level files are read locally.
# ==========================================================================================

SKRIPT_STAND_AUPRC = "package 1.0 / 2026-10-08 (port of auprc_natural.py v1)"








def main_auprc(argv=None):
    print("=" * 78)
    print("AUPRC  MIMIC test set and eICU  (Table 3)")
    print(f"  script version: {SKRIPT_STAND_AUPRC}")
    print("=" * 78)
    pfad_le = config.REPORT / "leistung_natural.csv"
    if not pfad_le.exists():
        abbruch(f"{pfad_le} is missing (run performance/metrics.py first).")
    le = pd.read_csv(pfad_le)

    fam = {}
    for pf in config.SIGNALS.glob("scores_*_seed*.npz"):
        m = re.match(r"scores_(.+)_seed(\d+)\.npz$", pf.name)
        if m:
            fam.setdefault(m.group(1), []).append(int(m.group(2)))
    # families of the package with the complete set of seeds (alphabetical, as in the original run)
    familien = sorted(f for f, s in fam.items()
                      if sorted(s) == SEEDS and f in config.ALL_FAMILIES)
    if not familien:
        abbruch("no family with the complete set of 10 seeds found.")
    print(f"  families: {', '.join(familien)}")

    zeilen = []
    for familie in familien:
        for seed in SEEDS:
            z = np.load(config.SIGNALS / f"scores_{familie}_seed{seed}.npz")
            for dom in DOMS:
                p = np.asarray(z[f"p_{dom}"], float).ravel()
                y = np.asarray(z[f"y_{dom}"], float).ravel()
                r = le[(le.familie == familie) & (le.seed == seed)
                       & (le.domaene == dom)]
                if len(r) != 1:
                    abbruch(f"leistung: {len(r)} rows for "
                            f"{familie}/{seed}/{dom}.")
                d_ = abs(float(((p - y) ** 2).mean()) - float(r.brier.iloc[0]))
                if d_ > ANKER_TOLERANZ:
                    abbruch(f"check: Brier {familie}/{seed}/{dom} deviates by "
                            f"{d_:.2e} -- not the same predictions.")
                ap = float(average_precision_score(y, p))
                prev = float(y.mean())
                zeilen.append({"familie": familie, "seed": seed,
                               "domaene": dom, "n": len(p),
                               "praevalenz": round(prev, 5),
                               "auprc": round(ap, 5),
                               "auprc_lift": round(ap / prev, 4),
                               "auroc": float(r.auroc.iloc[0])})
        print(f"  [OK] {familie}  (Brier check passed)")

    df = pd.DataFrame(zeilen)
    df.to_csv(_sicher("auprc_natural.csv"), index=False)
    print(f"\n  written: auprc_natural.csv ({len(df)} rows)")
    print("\n" + "-" * 78)
    print("AUPRC, mean +- 95 % interval over 10 seeds")
    print(f"{'Family':<11}{'Domain':<11}{'AUPRC':>17}{'Baseline':>12}"
          f"{'Lift':>8}{'AUROC':>9}")
    for familie in familien:
        for dom in DOMS:
            g = df[(df.familie == familie) & (df.domaene == dom)]
            m, h = mki(g.auprc)
            print(f"{familie:<11}{dom:<11}{m:>10.4f} ±{h:>5.4f}"
                  f"{g.praevalenz.mean():>12.4f}{g.auprc_lift.mean():>8.2f}"
                  f"{g.auroc.mean():>9.4f}")
    print("\n  Baseline = prevalence of the cohort (AUPRC of a random ranking).")
    print("  Compare raw AUPRC across cohorts only together with the baseline;")
    print("  the lift (AUPRC / prevalence) is the comparable quantity.")
    print("=" * 78)



# ==========================================================================================
# section from the full package: performance/ensemble_performance.py
# ------------------------------------------------------------------------------------------
# ensemble_performance.py -- deep-ensemble rows of Table 3: AUROC, AUPRC lift, slope, ECE
#
# Paper:    Table 3 ensemble rows (mean-probability ensemble of the ten seed models per base
#           family) and the ensemble transfer-loss sentence
# Origin:   src/ensemble_leistung_tabelle2.py, SKRIPT_STAND_ENS
#           "v1 / 2026-09-21 (Ensemble-Mittel: AUROC, Lift, Steigung, ECE; Jackknife-Spanne)"
# Reads:    config.SIGNALS/scores_<family>_seed<seed>.npz   (p_<domain>, y_<domain>, patient-level)
#           config.REPORT/mi_ensemble_natural.csv            (uncertainty/deep_ensembles.py; A2)
#           config.REPORT/leistung_natural.csv               (performance/metrics.py; A3)
# Writes:   config.APPENDIX/table_2_ensemble_rows.csv / .md  (file names of the original run kept)
# Checks:   A1  y identical across the seeds (same ordering of the test set)
#           A2  ensemble AUROC per family / cohort reproduces mi_ensemble_natural.csv
#               (ebene 'leistung', groesse 'p_quer_leistung', wert_voll) within 1e-4
#           A3  mean single-model AUROC (control) reproduces leistung_natural.csv within 1e-4
#           If A2 or A3 fails nothing is written (consistency with the run's own files).
# Run:      python performance/ensemble_performance.py
#
# Per base family (GRU, LSTM, Transformer) the ensemble mean p_quer of the ten seed models is
# formed and, per cohort: auroc, auprc, lift = AUPRC / prevalence, slope and intercept of the
# logistic recalibration (IRLS without regularisation), ece_roh (deciles of the ensemble
# prediction) and the transfer loss (in-domain minus transfer) for AUROC and lift. In addition the
# jackknife range (min-max over the ten nine-member ensembles) for AUROC and slope.
# ==========================================================================================

SKRIPT_STAND_ENS = "package 1.0 / 2026-10-08 (port of ensemble_leistung_tabelle2.py v1)"
FAM = [(config.LABELS[f], f) for f in config.BASE_FAMILIES]   # (GRU, GRU_det), (LSTM, LSTM_det), (Transformer, T1_det)
DOMS_PAIRS = [("in_domain", "in-domain"), ("transfer", "transfer")]

pruefungen = []




def pruefe(k, text, ok, detail=""):
    pruefungen.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {k}  {text}" + (f"   ({detail})" if detail else ""))


def logit_ens(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def irls_ens(z, y, iters=50):
    """Logistic recalibration y ~ 1 + z by Newton / IRLS steps; returns (intercept, slope)."""
    X = np.column_stack([np.ones_like(z), z])
    beta = np.zeros(2)
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        w = mu * (1 - mu)
        step = np.linalg.solve((X * w[:, None]).T @ X + 1e-10 * np.eye(2), X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def ece_dezile(p, y):
    """Decile ECE with edges from the quantiles of p."""
    kanten = np.quantile(p, np.linspace(0, 1, 11))
    ix = np.clip(np.digitize(p, kanten[1:-1]), 0, 9)
    return float(sum((ix == b).mean() * abs(y[ix == b].mean() - p[ix == b].mean())
                     for b in range(10) if (ix == b).any()))


def kennzahlen(p, y):
    """AUROC, AUPRC, lift, slope, intercept and decile ECE of one probability vector."""
    pi = y.mean()
    a, b = irls_ens(logit_ens(p), y)
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "lift": float(average_precision_score(y, p)) / pi, "slope": b, "intercept": a,
            "ece_roh": ece_dezile(p, y)}


def main_ensemble_performance(argv=None):
    print("=" * 78)
    print("TABLE 3  ensemble rows from the member predictions")
    print(f"  script version: {SKRIPT_STAND_ENS}")
    print("=" * 78)
    config.APPENDIX.mkdir(parents=True, exist_ok=True)
    ens = pd.read_csv(config.REPORT / "mi_ensemble_natural.csv", dtype={"schwelle": str})
    le = pd.read_csv(config.REPORT / "leistung_natural.csv")
    zeilen = []
    a2, a3 = 0.0, 0.0
    for name, fam in FAM:
        P, Y = {d: [] for d, _ in DOMS_PAIRS}, {d: None for d, _ in DOMS_PAIRS}
        for seed in SEEDS:
            z = np.load(config.SIGNALS / f"scores_{fam}_seed{seed}.npz")
            for dom, _ in DOMS_PAIRS:
                p = np.asarray(z[f"p_{dom}"], float).ravel()
                y = np.asarray(z[f"y_{dom}"], float).ravel()
                if Y[dom] is not None and not np.array_equal(y, Y[dom]):
                    abbruch(f"A1: {fam}/{dom}: y differs between seeds.")
                Y[dom] = y
                P[dom].append(p)
        zeile = {"familie": name}
        for dom, dname in DOMS_PAIRS:
            Pm = np.stack(P[dom])
            k = kennzahlen(Pm.mean(0), Y[dom])
            # jackknife: ten nine-member ensembles
            jk = [kennzahlen(np.delete(Pm, i, 0).mean(0), Y[dom]) for i in range(len(SEEDS))]
            for kk, v in k.items():
                zeile[f"{kk}_{dom}"] = v
            zeile[f"auroc_jack_{dom}"] = f"{min(j['auroc'] for j in jk):.3f}-{max(j['auroc'] for j in jk):.3f}"
            zeile[f"slope_jack_{dom}"] = f"{min(j['slope'] for j in jk):.3f}-{max(j['slope'] for j in jk):.3f}"
            r = ens[(ens.familie == fam) & (ens.ebene == "leistung") & (ens.domaene == dom)]
            if len(r) != 1:
                abbruch(f"A2: mi_ensemble leistung {fam}/{dom}: {len(r)} rows.")
            a2 = max(a2, abs(float(r.wert_voll.iloc[0]) - k["auroc"]))
            einzel = float(le[(le.familie == fam) & (le.domaene == dom)].auroc.mean())
            a3 = max(a3, abs(einzel - float(np.mean([roc_auc_score(Y[dom], p) for p in P[dom]]))))
            zeile[f"einzel_auroc_{dom}"] = einzel
        zeile["auroc_verlust"] = zeile["auroc_in_domain"] - zeile["auroc_transfer"]
        zeile["lift_in_domain_minus_transfer"] = zeile["lift_in_domain"] - zeile["lift_transfer"]
        zeilen.append(zeile)
        print(f"  {name:<12} AUROC {zeile['auroc_in_domain']:.3f} -> {zeile['auroc_transfer']:.3f} "
              f"(loss {zeile['auroc_verlust']:.3f})  lift {zeile['lift_in_domain']:.2f} -> "
              f"{zeile['lift_transfer']:.2f}  slope {zeile['slope_in_domain']:.2f} -> "
              f"{zeile['slope_transfer']:.2f}  ECE {zeile['ece_roh_in_domain']:.3f} -> "
              f"{zeile['ece_roh_transfer']:.3f}")
    pruefe("A1", "y identical across seeds in both cohorts", True)
    pruefe("A2", "ensemble AUROC reproduces mi_ensemble_natural.csv (< 1e-4)", a2 < 1e-4, f"{a2:.2e}")
    pruefe("A3", "single-model mean reproduces leistung_natural.csv (< 1e-4)", a3 < 1e-4, f"{a3:.2e}")
    if not all(pruefungen):
        print("\nRESULT: consistency checks failed, NOTHING written.")
        raise SystemExit(1)
    df = pd.DataFrame(zeilen)
    df.to_csv(config.APPENDIX / "table_2_ensemble_rows.csv", index=False)
    with open(config.APPENDIX / "table_2_ensemble_rows.md", "w", encoding="utf-8") as f:
        f.write("| Model | AUROC, in-domain | AUROC, transfer | Paired loss | AUPRC lift, in-domain / transfer | Slope, in-domain | Slope, transfer |\n|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(f"| Deep ensemble of the {r.familie} | {r.auroc_in_domain:.3f} | {r.auroc_transfer:.3f} | "
                    f"{r.auroc_verlust:.3f} | {r.lift_in_domain:.2f} / {r.lift_transfer:.2f} | "
                    f"{r.slope_in_domain:.2f} | {r.slope_transfer:.2f} |\n")
        f.write("\nEnsemble = mean of the ten seed models' probabilities (6); one ensemble per family, "
                "no seed interval. Jackknife ranges (nine-member ensembles) in the CSV.\n")
    print(f"  written: {config.APPENDIX.name}/table_2_ensemble_rows.csv / .md")
    print("=" * 78)



# ==========================================================================================
# section from the full package: levels/selective_prediction.py
# ------------------------------------------------------------------------------------------
# selective_prediction.py -- level 3: risk-coverage curves, AURC, E-AURC, risk at 80 % coverage
#
# Paper:    Results RQ2b; Table 4 (level 3); Figure 3 and SF6 (risk-coverage curves)
# Origin:   src/rerun_pipeline.py v5.7 / 2026-08-20, COVERAGE_GITTER, _risk_coverage(),
#           s7_ebene3() (stage S7)
# Reads:    config.SIGNALS/scores_<family>_seed<n>.npz (patient-level scores of stage S5, with the
#           Monte Carlo logit variance mc_epistemisch_<dom> added by mc_dropout.py)
# Writes:   config.REPORT/ebene3_natural.csv (arch, familie, seed, signal, domaene, schwelle,
#           aurc, eaurc, risiko80), config.REPORT/kurven_natural.npz (risk-coverage curves on
#           the 96-point coverage grid, key <family>__<signal>__<domain>, plus 'coverage')
# Checks:   every scores file must carry mc_epistemisch_<dom> for both cohorts (otherwise the level-3
#           rows of the logit variance would be silently missing): run mc_dropout.py first
# Run:      python evaluation.py --evaluate   (or run_pipeline.py --evaluate); no raw data needed
#
# Convention: patients are accepted in ascending order of the uncertainty score; the risk is
# the error rate of the accepted patients (error = prediction at the primary threshold 0.1288
# != label). AURC is the exact mean over all ranks; E-AURC = AURC minus the AURC of the oracle
# ordering; risk at 80 % coverage is read off the curve.
# ==========================================================================================

COVERAGE_GITTER = np.round(np.linspace(0.05, 1.0, 96), 6)


def _risk_coverage(score, fehler):
    """Accept in ascending order of uncertainty; risk = error rate of the accepted.
    AURC exact over all ranks; E-AURC = AURC - AURC_optimal.
    Returns (aurc, eaurc, risk at 80 % coverage, curve on COVERAGE_GITTER as float32)."""
    o = np.argsort(score, kind="stable")
    f = fehler[o].astype(np.float64)
    kum = np.cumsum(f) / np.arange(1, len(f) + 1)
    aurc = float(kum.mean())
    f_opt = np.sort(fehler.astype(np.float64))
    aurc_opt = float((np.cumsum(f_opt) / np.arange(1, len(f) + 1)).mean())
    ks = np.maximum((COVERAGE_GITTER * len(f)).astype(int), 1) - 1
    kurve = kum[ks]
    r80 = float(kum[max(int(0.8 * len(f)) - 1, 0)])
    return aurc, aurc - aurc_opt, r80, kurve.astype(np.float32)


def s7_ebene3():
    """Level 3 for every measure stored in the scores files, per family, seed and domain."""
    z3, kurven = [], {}
    prim = C.SCHWELLEN[0][1]
    for familie, (arch, variante, _) in C.FAMILIEN.items():
        if variante != "det":
            continue
        for seed in C.SEEDS:
            zs = C._finde(os.path.join(C.ORDNER["signale"],
                                       f"scores_{familie}_seed{seed}.npz"))
            if zs is None:
                continue
            S_ = np.load(zs)
            fehlt = [d for d in ("in_domain", "transfer")
                     if f"mc_epistemisch_{d}" not in S_.files]
            if fehlt:
                C.abbruch(f"{os.path.basename(zs)} carries no Monte Carlo logit variance "
                          f"for {fehlt} -- run mc_dropout.py (run_pipeline.py --mc) "
                          "before --evaluate.")
            for dom in ("in_domain", "transfer"):
                y = S_[f"y_{dom}"].astype(int)
                fehler = ((S_[f"p_{dom}"] > prim).astype(int) != y).astype(int)
                sig = {"dichte": S_[f"dichte_{dom}"],
                       "softmax_entropie": S_[f"entropie_{dom}"]}
                for v_ in ("rmd", "gmm_klasse", "gda_l2"):
                    k_ = f"{v_}_{dom}"
                    if k_ in S_.files:
                        sig[v_] = S_[k_]
                sig["mc_epistemisch"] = S_[f"mc_epistemisch_{dom}"]   # Monte Carlo logit variance
                for name, sc in sig.items():
                    aurc, eaurc, r80, kurve = _risk_coverage(
                        np.asarray(sc, float), fehler)
                    z3.append([arch, familie, seed, name, dom, "0.1288",
                               round(aurc, 5), round(eaurc, 5),
                               round(r80, 5)])
                    kurven.setdefault(
                        f"{familie}|{name}|{dom}", []).append(kurve)
            C.log(f"[S7] {familie}/seed {seed}: level 3 computed.")
    if not z3:
        C.abbruch("S7: no scores found -- run --signals first.")
    ziel = C._sicher(C.ORDNER["report"], "ebene3_natural.csv")
    with open(ziel, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arch", "familie", "seed", "signal", "domaene",
                    "schwelle", "aurc", "eaurc", "risiko80"])
        w.writerows(z3)
    C.log(f"[S7] written: {ziel} ({len(z3)} rows)")
    npz = {k.replace("|", "__"): np.stack(v) for k, v in kurven.items()}
    npz["coverage"] = COVERAGE_GITTER.astype(np.float32)
    np.savez_compressed(C._sicher(C.ORDNER["report"], "kurven_natural.npz"),
                        **npz)
    C.log(f"[S7] written: kurven_natural.npz ({len(kurven)} curve families "
          f"x {len(COVERAGE_GITTER)} grid points)")


def main_selective(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    config.ensure_run_folders()
    s7_ebene3()



# ==========================================================================================
# section from the full package: levels/level3_ensembles_mc.py
# ------------------------------------------------------------------------------------------
# level3_ensembles_mc.py -- level 3 (selective prediction) for deep ensembles and MC-dropout mutual information
#
# Paper:    supplementary level-3 rows for deep ensembles and MC-dropout MI (Table S5, Figure 4 ensemble rows,
#           input of levels/deferral_breakdown.py and levels/entropy_deciles.py)
# Origin:   src/ebene3_e14.py  v1 / 2026-08-24 (E14-E3, S7-Spiegel mit Reproduktionsanker)
# Reads:    SIGNALS/scores_<family>_seed<n>.npz, SIGNALS/mc_mi_<family>_seed<n>.npz (patient-level, stay local),
#           REPORT/ebene3_natural.csv
# Writes:   REPORT/ebene3_e14.csv (aggregates only)
# Checks:   A1 cohort sizes per domain (7,298 / 66,934) and identical labels across seeds;
#           A2 the risk-coverage metric mirrored here reproduces the softmax_entropie rows of ebene3_natural.csv
#              for every family, seed and domain to 6e-6 (storage rounding, 5 decimals);
#           A3 MI identity U_total = U_aleatoric + U_epistemic to 1e-9, U_epistemic >= -1e-9
# Run:      python levels/level3_ensembles_mc.py
#
# What it computes (descriptive, declared post hoc after the level-1/2 results were known):
#   (a) ENSEMBLE, every reported family with ten complete seeds, both domains: predictor p_bar = mean of the ten
#       member probabilities; error label e = 1[1(p_bar > threshold) != y]; measures U_total = H(p_bar), U_aleatoric,
#       U_epistemic (mutual information); metrics AURC, E-AURC, risk at 80 % coverage. Variability: leave-one-member-
#       out jackknife (ten ensembles of nine), reported as a range, NOT a confidence interval.
#   (b) MC-MI, families GRU/LSTM/T1, per seed: measures from mc_mi_*.npz (probability-scale decomposition of the
#       Monte Carlo passes of the same model, mc_dropout.py), error label of the deterministic prediction of the
#       same family and seed (pipeline convention), paired contrasts against the softmax entropy from
#       ebene3_natural.csv.
#
# Metric convention: risk_coverage mirrors _risk_coverage of the pipeline verbatim: stable ascending sort by the
# measure, accepted from the bottom; risk = error rate of the accepted; AURC = mean of the cumulative error rate over
# ALL ranks; E-AURC = AURC minus oracle AURC (errors sorted ascending); risk@80 = cumulative rate at rank
# floor(0.8 n) - 1.
# ==========================================================================================

SKRIPT_STAND_L3 = "package port of ebene3_e14.py v1 / 2026-08-24"
SEEDS_L3 = list(config.SEEDS)
N_SOLL = {"in_domain": 7298, "transfer": 66934}
SCHWELLE = config.THRESHOLD
MC_FAMILIEN = tuple(config.BASE_FAMILIES)        # GRU_det, LSTM_det, T1_det
EPS_L3 = 1e-12




def _sicher_l3(name):
    """Output path inside config.REPORT (guards against path escapes)."""
    config.REPORT.mkdir(parents=True, exist_ok=True)
    ziel = (config.REPORT / name).resolve()
    if config.REPORT.resolve() not in ziel.parents:
        abbruch("output path outside of " + str(config.REPORT))
    return ziel


def H(p):
    """Binary entropy in nats, probabilities clipped to [EPS_L3, 1 - EPS_L3]."""
    p = np.clip(np.asarray(p, float), EPS_L3, 1 - EPS_L3)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def risk_coverage(score, fehler):
    """Mirror of the pipeline's _risk_coverage (without the curve grid): returns (AURC, E-AURC, risk@80)."""
    o = np.argsort(score, kind="stable")
    f = fehler[o].astype(np.float64)
    kum = np.cumsum(f) / np.arange(1, len(f) + 1)
    aurc = float(kum.mean())
    f_opt = np.sort(fehler.astype(np.float64))
    aurc_opt = float((np.cumsum(f_opt) / np.arange(1, len(f) + 1)).mean())
    r80 = float(kum[max(int(config.COVERAGE * len(f)) - 1, 0)])
    return aurc, aurc - aurc_opt, r80


def mi_groessen(P):
    """Ensemble decomposition from a (members, n) probability matrix: p_bar, U_total, U_aleatoric, U_epistemic."""
    p_quer = P.mean(0)
    u_ges = H(p_quer)
    u_alea = H(P).mean(0)
    u_epi = u_ges - u_alea
    if np.max(np.abs(u_ges - (u_alea + u_epi))) > 1e-9 \
            or np.min(u_epi) < -1e-9:
        abbruch("A3: MI identity / non-negativity violated.")
    return p_quer, u_ges, u_alea, u_epi


def main_level3(argv=None):
    ap = argparse.ArgumentParser(description="Level 3 (AURC, E-AURC, risk@80) for deep ensembles and MC-dropout MI; "
                                             "writes REPORT/ebene3_e14.csv.")
    ap.parse_args(argv)

    print("=" * 78)
    print("LEVEL 3 FOR DEEP ENSEMBLES AND MC-MI")
    print(f"  Script version: {SKRIPT_STAND_L3}")
    print("=" * 78)

    pfad_e3 = config.REPORT / "ebene3_natural.csv"
    if not pfad_e3.exists():
        abbruch(f"{pfad_e3} missing (run the pipeline's evaluate stage first).")
    e3 = pd.read_csv(pfad_e3)

    fam = {}
    for pf in config.SIGNALS.glob("scores_*_seed*.npz"):
        m = re.match(r"scores_(.+)_seed(\d+)\.npz$", pf.name)
        if m:
            fam.setdefault(m.group(1), []).append(int(m.group(2)))
    familien = sorted(f for f, s in fam.items()
                      if sorted(s) == SEEDS_L3 and f in config.REPORTED_FAMILIES)
    if not familien:
        abbruch("no reported family with ten complete seeds found.")
    print(f"  Families: {', '.join(familien)}\n")

    zeilen = []
    for familie in familien:
        P = {dom: [] for dom in DOMS}
        Y = {}
        ENT = {dom: [] for dom in DOMS}
        for seed in SEEDS_L3:
            z = np.load(config.SIGNALS / f"scores_{familie}_seed{seed}.npz")
            for dom in DOMS:
                p = np.asarray(z[f"p_{dom}"], float).ravel()
                y = np.asarray(z[f"y_{dom}"], float).ravel()
                if len(p) != N_SOLL[dom]:
                    abbruch(f"A1: {familie}/{seed}/{dom}: {len(p)} rows.")
                if dom in Y:
                    if not np.array_equal(Y[dom], y):
                        abbruch(f"A1: y differs ({familie}/{seed}/{dom}).")
                else:
                    Y[dom] = y
                P[dom].append(p)
                ENT[dom].append(np.asarray(z[f"entropie_{dom}"],
                                           float).ravel())

        # A2: mirror of the pipeline's level 3 against ebene3_natural.csv (entropy rows, per seed)
        for dom in DOMS:
            for i, seed in enumerate(SEEDS_L3):
                fe = ((P[dom][i] > SCHWELLE).astype(int)
                      != Y[dom].astype(int)).astype(int)
                aurc, eaurc, r80 = risk_coverage(ENT[dom][i], fe)
                r = e3[(e3.familie == familie) & (e3.seed == seed)
                       & (e3.signal == "softmax_entropie")
                       & (e3.domaene == dom)]
                if len(r) != 1:
                    abbruch(f"A2: ebene3_natural has no unique row for "
                            f"{familie}/{seed}/{dom}.")
                for name, ist, soll in (("aurc", aurc, r.aurc),
                                        ("eaurc", eaurc, r.eaurc),
                                        ("risiko80", r80, r.risiko80)):
                    if abs(ist - float(soll.iloc[0])) > 6e-6:
                        abbruch(f"A2: {name} {familie}/{seed}/{dom}: "
                                f"{ist:.5f} vs {float(soll.iloc[0]):.5f} -- "
                                "metric convention differs.")

        # (a) ensemble: full plus jackknife
        P_ = {dom: np.stack(P[dom]) for dom in DOMS}
        mengen = [("voll", list(range(10)))] + \
                 [(f"jack_ohne_{SEEDS_L3[i]}", [j for j in range(10) if j != i])
                  for i in range(10)]
        for name, idx in mengen:
            for dom in DOMS:
                p_quer, u_ges, u_alea, u_epi = mi_groessen(P_[dom][idx])
                fe = ((p_quer > SCHWELLE).astype(int)
                      != Y[dom].astype(int)).astype(int)
                for sig, werte in (("mi_gesamt", u_ges),
                                   ("mi_aleatorisch", u_alea),
                                   ("mi_epistemisch", u_epi)):
                    aurc, eaurc, r80 = risk_coverage(werte, fe)
                    zeilen.append(["ensemble", familie, name, sig, dom,
                                   SCHWELLE, round(aurc, 5), round(eaurc, 5),
                                   round(r80, 5),
                                   round(float(fe.mean()), 5)])
        print(f"  [OK] {familie}  (A1-A3 passed, ensemble computed)")

    # (b) MC-MI against the deterministic label, paired per seed
    for familie in MC_FAMILIEN:
        if familie not in familien:
            continue
        fehlend = [s for s in SEEDS_L3
                   if not (config.SIGNALS / f"mc_mi_{familie}_seed{s}.npz").exists()]
        if fehlend:
            print(f"  [!] {familie}: mc_mi missing for seeds {fehlend}, "
                  "MC-MI part skipped.")
            continue
        for seed in SEEDS_L3:
            zmi = np.load(config.SIGNALS / f"mc_mi_{familie}_seed{seed}.npz")
            zsc = np.load(config.SIGNALS / f"scores_{familie}_seed{seed}.npz")
            for dom in DOMS:
                p_det = np.asarray(zsc[f"p_{dom}"], float).ravel()
                y = np.asarray(zsc[f"y_{dom}"], float).ravel()
                fe = ((p_det > SCHWELLE).astype(int)
                      != y.astype(int)).astype(int)
                for sig, schluessel in (("mi_gesamt", "mi_ges"),
                                        ("mi_aleatorisch", "mi_alea"),
                                        ("mi_epistemisch", "mi_epi")):
                    werte = np.asarray(zmi[f"{schluessel}_{dom}"],
                                       float).ravel()
                    aurc, eaurc, r80 = risk_coverage(werte, fe)
                    zeilen.append(["mc_mi", familie, str(seed), sig, dom,
                                   SCHWELLE, round(aurc, 5), round(eaurc, 5),
                                   round(r80, 5), round(float(fe.mean()), 5)])
        print(f"  [OK] {familie}  (MC-MI level 3 computed)")

    df = pd.DataFrame(zeilen, columns=["verfahren", "familie", "mitglied",
                                       "signal", "domaene", "schwelle",
                                       "aurc", "eaurc", "risiko80",
                                       "fehlerrate_voll"])
    df.to_csv(_sicher_l3("ebene3_e14.csv"), index=False)
    print(f"\n  written: ebene3_e14.csv ({len(df)} rows)")

    print("\n" + "-" * 78)
    print("TRANSFER: ensemble (H(p_bar) as measure) against single-model entropy")
    print(f"{'Family':<12}{'E-AURC ens.':>13}{'jack range':>20}"
          f"{'E-AURC det':>12}{'R@80 ens.':>11}{'R@80 det':>10}"
          f"{'error ens.':>12}{'det':>7}")
    for familie in familien:
        g = df[(df.verfahren == "ensemble") & (df.familie == familie)
               & (df.domaene == "transfer") & (df.signal == "mi_gesamt")]
        voll = g[g.mitglied == "voll"].iloc[0]
        jack = g[g.mitglied != "voll"]
        d = e3[(e3.familie == familie) & (e3.signal == "softmax_entropie")
               & (e3.domaene == "transfer")]
        print(f"{familie:<12}{voll.eaurc:>13.4f}"
              f"   [{jack.eaurc.min():.4f}; {jack.eaurc.max():.4f}] "
              f"{d.eaurc.mean():>12.4f}{voll.risiko80:>11.4f}"
              f"{d.risiko80.mean():>10.4f}{voll.fehlerrate_voll:>12.4f}"
              f"{'':>7}")
    print("\n  (the deterministic error rate at full coverage is not a column of ebene3_natural;\n"
          "  the mc_mi rows of ebene3_e14.csv carry it as fehlerrate_voll.)")

    print("\nTRANSFER: MC-MI against entropy, paired per seed "
          "(E-AURC difference, negative = MI better)")
    for familie in MC_FAMILIEN:
        g = df[(df.verfahren == "mc_mi") & (df.familie == familie)
               & (df.domaene == "transfer")]
        if not len(g):
            continue
        ent = e3[(e3.familie == familie) & (e3.signal == "softmax_entropie")
                 & (e3.domaene == "transfer")].sort_values("seed")
        for sig in ("mi_gesamt", "mi_epistemisch"):
            s_ = g[g.signal == sig].copy()
            s_["seed"] = s_.mitglied.astype(int)
            s_ = s_.sort_values("seed")
            d = np.asarray(s_.eaurc) - np.asarray(ent.eaurc)
            h = T9 * d.std(ddof=1) / np.sqrt(len(d))
            print(f"  {familie:<12}{sig:<16} E-AURC {s_.eaurc.mean():.4f}  "
                  f"diff {d.mean():+.4f} [{d.mean() - h:+.4f}; "
                  f"{d.mean() + h:+.4f}]")
    print("\n  Metric convention: mirror of the pipeline's level 3, check A2 passed per family/seed/domain.")
    print("  Jackknife ranges are correlated re-uses of the same members, not confidence intervals.")
    print("=" * 78)



# ==========================================================================================
# section from the full package: tables_figures/table3.py
# ------------------------------------------------------------------------------------------
# table3.py -- Table 3 of the manuscript: discrimination and calibration in-domain and after transfer
#
# Paper:    Table 3 (GRU, LSTM, Transformer and their deep ensembles); the passage "Model
#           performance after transfer" quotes the same cells
# Origin:   src/npj_code_package/recompute/manuscript_numbers.py (written 2026-10-07), its Table 3
#           construction only; the manuscript-revision text output of that script is not ported
# Reads:    config.REPORT/leistung_natural.csv       (AUROC per family, seed and cohort)
#           config.REPORT/auprc_natural.csv          (AUPRC lift)
#           config.REPORT/kalibrierung_kennzahlen.csv (calibration slope, raw scale)
#           config.APPENDIX/table_2_ensemble_rows.csv (deep-ensemble rows from
#               performance/ensemble_performance.py; optional, the ensemble rows are left out
#               when it is missing)
# Writes:   config.APPENDIX/table_3.md, config.APPENDIX/table_3.csv
# Checks:   none beyond the arithmetic; a cell without data is written as "n/a" instead of a value
# Run:      python tables_figures/table3.py
#
# Cells: means over the ten seeds; the paired AUROC loss (in-domain minus transfer, per seed)
# carries a 95 % t-interval across the seeds. AUROC and loss with three decimals, lift and slope
# with two, rounded by Python's format (as in the original script).
# ==========================================================================================

BASE = [(config.LABELS[f], f) for f in config.BASE_FAMILIES]  # ("GRU", "GRU_det"), ("LSTM", "LSTM_det"), ("Transformer", "T1_det")
ENSEMBLE_NAME = {"GRU": "GRU", "LSTM": "LSTM", "Transformer": "transformer"}   # "Deep ensemble of the ..."
KOPF = ["Model", "AUROC, in-domain", "AUROC, transfer", "Paired loss [seed interval]",
        "AUPRC lift, in-domain / transfer", "Slope, in-domain", "Slope, transfer"]


def interval(v):
    """(mean, lower, upper) with a 95 % t-interval across the seeds; nan for no data."""
    v = np.asarray(pd.to_numeric(pd.Series(v), errors="coerce").dropna(), float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    if len(v) == 1:
        return (float(v[0]),) * 3
    h = T9 * v.std(ddof=1) / np.sqrt(len(v))
    return (float(v.mean()), float(v.mean() - h), float(v.mean() + h))


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def table_rows():
    """The rows of Table 3 as lists of formatted cells (the base families, then the ensembles)."""
    leist = pd.read_csv(config.REPORT / "leistung_natural.csv")
    auprc = pd.read_csv(config.REPORT / "auprc_natural.csv")
    kal = pd.read_csv(config.REPORT / "kalibrierung_kennzahlen.csv")
    p_ens = config.APPENDIX / "table_2_ensemble_rows.csv"
    ensr = pd.read_csv(p_ens) if p_ens.exists() else None

    rows = []
    for name, fam in BASE:
        r = {}
        for dom in ("in_domain", "transfer"):
            g = leist[(leist.familie == fam) & (leist.domaene == dom)]
            r[f"auroc_{dom}"] = interval(g.auroc)
            ga = auprc[(auprc.familie == fam) & (auprc.domaene == dom)]
            r[f"lift_{dom}"] = interval(ga.auprc_lift)
            gk = kal[(kal.familie == fam) & (kal.domaene == dom) & (kal.skala == "roh")]
            r[f"slope_{dom}"] = interval(gk.slope)
        w = leist[leist.familie == fam].pivot_table(index="seed", columns="domaene", values="auroc")
        r["loss"] = interval(w["in_domain"] - w["transfer"])
        rows.append([name, f(r["auroc_in_domain"][0]), f(r["auroc_transfer"][0]),
                     f"{f(r['loss'][0])} [{f(r['loss'][1])}; {f(r['loss'][2])}]",
                     f"{f(r['lift_in_domain'][0], 2)} / {f(r['lift_transfer'][0], 2)}",
                     f(r["slope_in_domain"][0], 2), f(r["slope_transfer"][0], 2)])
    if ensr is None:
        print(f"  [!] {p_ens.name} not found; the deep-ensemble rows are left out.")
    else:
        for _, r in ensr.iterrows():
            nm = ENSEMBLE_NAME.get(r.familie, r.familie)
            rows.append([f"Deep ensemble of the {nm}", f"{float(r.auroc_in_domain):.3f}",
                         f"{float(r.auroc_transfer):.3f}", f"{float(r.auroc_verlust):.3f}",
                         f"{float(r.lift_in_domain):.2f} / {float(r.lift_transfer):.2f}",
                         f"{float(r.slope_in_domain):.2f}", f"{float(r.slope_transfer):.2f}"])
    return rows


def main_table3(argv=None):
    ap = argparse.ArgumentParser(
        description="Write Table 3 (discrimination and calibration in-domain and after transfer) "
                    "as table_3.md and table_3.csv into config.APPENDIX.")
    ap.parse_args(argv)
    rows = table_rows()
    config.APPENDIX.mkdir(parents=True, exist_ok=True)
    md = ["# Table 3. Mortality discrimination and calibration in-domain and after external transfer", "",
          "| " + " | ".join(KOPF) + " |", "|" + "---|" * len(KOPF)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "Means over ten training seeds; paired loss = AUROC in-domain minus AUROC transfer per seed, "
           "with the 95 % t-interval across the seeds. Deep ensembles: mean-probability ensemble of the ten "
           "seed models (performance/ensemble_performance.py), no seed interval."]
    with open(config.APPENDIX / "table_3.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(config.APPENDIX / "table_3.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(KOPF)
        for r in rows:
            w.writerow(r)
    print(f"written: {config.APPENDIX / 'table_3.md'} and table_3.csv ({len(rows)} rows)")
    return 0



# ==========================================================================================
# section: Table 4 cells (from tables_figures/figure4_levels.py of the full package, without the
# drawing; measure names, the half-up rounding rule and the ALEA criterion from
# tables_figures/supplementary_tables.py v2.0 / 2026-10-06)
# ------------------------------------------------------------------------------------------
# Paper:    Table 4 (nine uncertainty measures x three levels x three base families; the cells are the
#           means over ten seeds, ensembles one value, rounded once to three decimals)
# Reads:    config.SIGNALS/ddu_seeds_natural.csv, mc_seeds_natural.csv; config.REPORT/mc_mi_seeds_natural.csv,
#           mi_ensemble_natural.csv, ebene3_natural.csv, ebene3_e14.csv
# Writes:   config.APPENDIX/table_3_cells.csv and .md (file names of the original run; this is Table 4)
# Checks:   A1 ten seeds per measure, family and level (ensembles: one full row plus the jackknife range),
#           all 81 cells present; A3 aleatoric part within ALEA_TOL of the total uncertainty (MC dropout
#           and deep ensemble); the former axis check A4 belongs to the figure and is not part of the table
# ==========================================================================================
from decimal import Decimal, ROUND_HALF_UP  # noqa: E402

ALEA_TOL = 0.001   # ALEA check: |aleatoric - total| per cell (criterion set 2026-10-06 before level 3 was inspected)
# Internal name -> display name in the table (the order of the rows is ZEILEN_T4 below).
SIGNALNAMEN = {
    "Predictive entropy": "Single model: predictive entropy",
    "MC dropout: predictive entropy": "MC dropout: predictive entropy (U_total)",
    "MC dropout: aleatoric part": "MC dropout: aleatoric part (U_alea)",
    "MC-dropout MI (epistemic)": "MC dropout: mutual information (MI)",
    "MC dropout: logit variance": "MC dropout: logit variance (epistemic)",
    "Deep ensemble (total)": "Deep ensemble: predictive entropy (U_total)",
    "Deep ensemble: aleatoric part": "Deep ensemble: aleatoric part (U_alea)",
    "Deep ensemble: mutual information": "Deep ensemble: mutual information (MI)",
    "Mahalanobis density (class-specific)": "Density: class-conditional Gaussian",
}


def _dez(x):
    """The value as a Decimal in its shortest representation: 0.76247 stays 0.76247 and does
    not become the binary fraction 0.762469999..."""
    return Decimal(repr(float(x)))


def intervall(w):
    """Mean and 95 % t-interval across the seeds. The mean is formed exactly in decimal
    arithmetic (the aggregates carry five decimals) so that the rounding in runde() does not
    depend on binary residues. The difference to numpy.mean is of order 1e-16."""
    w = np.asarray(w, float)
    h = float(T9 * w.std(ddof=1) / np.sqrt(len(w)))
    m = float(sum(_dez(v) for v in w) / Decimal(len(w)))
    return m, m - h, m + h


def runde(x, nk=3):
    """Half-up rounding to nk decimals on the decimal representation. Applied ONCE to the
    unrounded value -- never re-round an already rounded cell (0.76247 -> 0.7625 -> 0.763
    would be wrong; the correct cell is 0.762)."""
    x = float(x)
    if not np.isfinite(x):
        return "nan"
    q = _dez(x).quantize(Decimal(1).scaleb(-nk), rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)                      # no "-0.000"
    return f"{q:.{nk}f}"


SIG_T4 = str(config.SIGNALS)
REP_T4 = str(config.REPORT)
APP_T4 = str(config.APPENDIX)
# (row label, arch in the seed files, family name in the ensemble and level-3 files)
ARCH_T4 = [("GRU", "GRU", "GRU_det"), ("LSTM", "LSTM", "LSTM_det"), ("Transformer", "T1", "T1_det")]
if [f for _, _, f in ARCH_T4] != list(config.BASE_FAMILIES):
    raise SystemExit("evaluation.py (Table 4): ARCH_T4 and config.BASE_FAMILIES name different families.")
TAU_T4 = str(config.THRESHOLD)            # "0.1288"; the seed files store the threshold as text
# internal names, from bottom to top as in the figure of the full package (the table lists them top-down)
ZEILEN_T4 = ["Mahalanobis density (class-specific)", "MC dropout: logit variance", "Deep ensemble: mutual information",
             "Deep ensemble: aleatoric part", "Deep ensemble (total)", "MC-dropout MI (epistemic)",
             "MC dropout: aleatoric part", "MC dropout: predictive entropy", "Predictive entropy"]
EBENEN_T4 = ["E1", "E2", "E3"]
# A3: pairs (total, aleatoric) of the entropy decomposition
ALEA_PAARE_T4 = [("MC dropout", "MC dropout: predictive entropy", "MC dropout: aleatoric part"),
                 ("Deep ensemble", "Deep ensemble (total)", "Deep ensemble: aleatoric part")]
pruefungen_t4 = []


def pruefe_t4(k, text, ok, detail=""):
    pruefungen_t4.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {k}  {text}" + (f"   ({detail})" if detail else ""))


def tabellenname(sname):
    """Display name as plain text (as in Supplementary Table 6 of the full package)."""
    return SIGNALNAMEN[sname]


def lies_csv_t4(name, ordner):
    """Read one aggregate CSV from its stage folder; the text columns schwelle, fit_menge, kovarianz
    and mitglied get '' for NaN (schwelle stays text)."""
    p = os.path.join(ordner, name)
    if not os.path.exists(p):
        abbruch(f"{p} is missing.")
    d = pd.read_csv(p, dtype={"schwelle": str})
    for c in ("schwelle", "fit_menge", "kovarianz", "mitglied"):
        if c in d.columns:
            d[c] = d[c].fillna("").astype(str)
    return d


def nur_hauptdichte(g):
    """Main variant of the density: class-specific covariance; 'tied' rows out."""
    if "kovarianz" in g.columns:
        k = g.kovarianz.astype(str).str.lower()
        g2 = g[~k.str.contains("tied")]
        if len(g2) != 10 and len(g2) > 0:
            g2 = g2[(k.loc[g2.index] == "") | k.loc[g2.index].str.contains("klass|class|voll|full")]
        return g2
    return g


def seeds10(g, was):
    if len(g) != 10:
        extra = ""
        if "kovarianz" in g.columns:
            extra = f"; kovarianz values: {sorted(g.kovarianz.astype(str).unique())}"
        abbruch(f"A1: {was}: {len(g)} instead of 10 seeds{extra}.")
    return intervall(g.wert if "wert" in g.columns else g.eaurc)


def main_table4(argv=None):
    """Table 4: the cell values from the aggregate files, written as table_3_cells.csv / .md."""
    print("=" * 78)
    print("TABLE 4  Measures across the three levels, three families (cells)")
    print("=" * 78)
    os.makedirs(APP_T4, exist_ok=True)
    dd = lies_csv_t4("ddu_seeds_natural.csv", SIG_T4)
    mm = lies_csv_t4("mc_mi_seeds_natural.csv", REP_T4)
    mc = lies_csv_t4("mc_seeds_natural.csv", SIG_T4)
    en = lies_csv_t4("mi_ensemble_natural.csv", REP_T4)
    e3 = lies_csv_t4("ebene3_natural.csv", REP_T4)
    e14 = lies_csv_t4("ebene3_e14.csv", REP_T4)

    def e14_diagnose(was):
        sp = [c for c in ("verfahren", "familie", "mitglied", "signal", "domaene") if c in e14.columns]
        print(f"  [DIAGNOSIS] {was} -- columns: {list(e14.columns)}")
        print("  combinations (verfahren, familie, signal, domaene), first 40:")
        print(e14[sp].drop_duplicates().head(40).to_string())
        abbruch("A1: ebene3_e14 filter is empty, see the diagnosis above.")

    D = {}  # D[(family label, signal, level)] = (m, lo, hi)
    for anz, kurz, fam in ARCH_T4:
        # entropy and density: levels 1/2 from ddu_seeds, level 3 from ebene3_natural
        for sname, sig in (("Predictive entropy", "softmax_entropie"),
                           ("Mahalanobis density (class-specific)", "dichte")):
            g = dd[(dd.arch == kurz) & (dd.signal == sig) & (dd.ebene == "ebene1_ood") & (dd.schwelle == "")]
            if sig == "dichte":
                g = nur_hauptdichte(g[(g.fit_menge == "voll_natuerlich")])
            D[(anz, sname, "E1")] = seeds10(g, f"{sig} E1 {kurz}")
            g = dd[(dd.arch == kurz) & (dd.signal == sig) & (dd.ebene == "ebene2_fehler")
                   & (dd.domaene == "transfer") & (dd.schwelle == TAU_T4)]
            if sig == "dichte":
                g = nur_hauptdichte(g[(g.fit_menge == "voll_natuerlich")])
            D[(anz, sname, "E2")] = seeds10(g, f"{sig} E2 {kurz}")
            g = e3[(e3.familie == fam) & (e3.signal == sig) & (e3.domaene == "transfer")]
            D[(anz, sname, "E3")] = seeds10(g, f"{sig} E3 {fam}")
        # MC dropout total / aleatoric / MI: levels 1/2 from mc_mi_seeds (variante mittel),
        # level 3 from ebene3_e14 (verfahren mc_mi)
        for sname, sig in (("MC dropout: predictive entropy", "mi_gesamt"),
                           ("MC dropout: aleatoric part", "mi_aleatorisch"),
                           ("MC-dropout MI (epistemic)", "mi_epistemisch")):
            g = mm[(mm.arch == kurz) & (mm.variante == "mittel") & (mm.signal == sig) & (mm.ebene == "ebene1_ood")]
            D[(anz, sname, "E1")] = seeds10(g, f"{sig} E1 {kurz}")
            g = mm[(mm.arch == kurz) & (mm.variante == "mittel") & (mm.signal == sig)
                   & (mm.ebene == "ebene2_fehler") & (mm.domaene == "transfer") & (mm.schwelle == TAU_T4)]
            D[(anz, sname, "E2")] = seeds10(g, f"{sig} E2 {kurz}")
            g = e14[(e14.verfahren == "mc_mi") & (e14.familie == fam) & (e14.signal == sig)
                    & (e14.domaene == "transfer")]
            if "mitglied" in g.columns:
                g = g[~g.mitglied.str.startswith("jack")]      # MC: one row per seed
            if len(g) == 0:
                e14_diagnose(f"mc_mi {sig} E3 {fam}")
            D[(anz, sname, "E3")] = seeds10(g, f"mc_mi {sig} E3 {fam}")
        # logit variance of the Monte Carlo passes: levels 1/2 from mc_seeds, level 3 from ebene3_natural
        g = mc[(mc.arch == kurz) & (mc.signal == "mc_epistemisch") & (mc.ebene == "ebene1_ood")]
        D[(anz, "MC dropout: logit variance", "E1")] = seeds10(g, f"mc_epistemisch E1 {kurz}")
        g = mc[(mc.arch == kurz) & (mc.signal == "mc_epistemisch") & (mc.ebene == "ebene2_fehler")
               & (mc.domaene == "transfer") & (mc.schwelle == TAU_T4)]
        D[(anz, "MC dropout: logit variance", "E2")] = seeds10(g, f"mc_epistemisch E2 {kurz}")
        g = e3[(e3.familie == fam) & (e3.signal == "mc_epistemisch") & (e3.domaene == "transfer")]
        D[(anz, "MC dropout: logit variance", "E3")] = seeds10(g, f"mc_epistemisch E3 {fam}")
        # ensembles: one value plus the jackknife range
        for sname, sig in (("Deep ensemble (total)", "mi_gesamt"),
                           ("Deep ensemble: aleatoric part", "mi_aleatorisch"),
                           ("Deep ensemble: mutual information", "mi_epistemisch")):
            g = en[(en.familie == fam) & (en.groesse == sig) & (en.ebene == "ebene1_ood")]
            if len(g) != 1:
                abbruch(f"A1: ensemble {sig} E1 {fam}: {len(g)} rows.")
            r = g.iloc[0]; D[(anz, sname, "E1")] = (float(r.wert_voll), float(r.jack_min), float(r.jack_max))
            g = en[(en.familie == fam) & (en.groesse == sig) & (en.ebene == "ebene2_fehler")
                   & (en.domaene == "transfer") & (en.schwelle == TAU_T4)]
            if len(g) != 1:
                abbruch(f"A1: ensemble {sig} E2 {fam}: {len(g)} rows.")
            r = g.iloc[0]; D[(anz, sname, "E2")] = (float(r.wert_voll), float(r.jack_min), float(r.jack_max))
            g = e14[(e14.verfahren == "ensemble") & (e14.familie == fam) & (e14.signal == sig)
                    & (e14.domaene == "transfer")]
            voll = g[g.mitglied == "voll"] if "mitglied" in g.columns else g
            if len(voll) == 0:
                e14_diagnose(f"ensemble {sig} E3 {fam}")
            if len(voll) != 1:
                abbruch(f"A1: ensemble {sig} E3 {fam}: {len(voll)} full-ensemble rows.")
            jack = g[g.mitglied.str.startswith("jack")] if "mitglied" in g.columns else g.iloc[0:0]
            m = float(voll.eaurc.iloc[0])
            lo, hi = (float(jack.eaurc.min()), float(jack.eaurc.max())) if len(jack) else (m, m)
            D[(anz, sname, "E3")] = (m, lo, hi)
        print(f"  {anz:<12} read")

    erwartet = {(a, z, eb) for a, _, _ in ARCH_T4 for z in ZEILEN_T4 for eb in EBENEN_T4}
    if set(D) != erwartet:
        abbruch(f"A1: cells incomplete or superfluous: missing {sorted(erwartet - set(D))[:3]}, "
                f"surplus {sorted(set(D) - erwartet)[:3]}.")

    # A3: aleatoric part against the total uncertainty, per family and level
    for kurzname, n_ges, n_alea in ALEA_PAARE_T4:
        for eb in EBENEN_T4:
            abw = {a: abs(D[(a, n_alea, eb)][0] - D[(a, n_ges, eb)][0]) for a, _, _ in ARCH_T4}
            wo = max(abw, key=abw.get)
            pruefe_t4(f"A3_{eb}_{kurzname}", f"aleatoric = total within {ALEA_TOL}",
                      abw[wo] <= ALEA_TOL, f"largest deviation {abw[wo]:.5f} at {wo}")
    # extract of every cell
    print("\n  [EXTRACT] cells (GRU / LSTM / Transformer), three decimals, rounded once:")
    for sname in ZEILEN_T4[::-1]:
        for eb in EBENEN_T4:
            print(f"    {tabellenname(sname):<46} {eb}: "
                  + " / ".join(runde(D[(a, sname, eb)][0], 3) for a, _, _ in ARCH_T4))

    # the cells as a file (written in any case)
    rows = []
    for sname in ZEILEN_T4[::-1]:
        row = {"signal": tabellenname(sname)}
        for eb in EBENEN_T4:
            row[eb] = " / ".join(runde(D[(a, sname, eb)][0], 3) for a, _, _ in ARCH_T4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(APP_T4, "table_3_cells.csv"), index=False)
    with open(os.path.join(APP_T4, "table_3_cells.md"), "w", encoding="utf-8") as f:
        f.write("| Signal | Cohort separation | Error detection | Deferral (excess area) |\n|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['signal']} | {r['E1']} | {r['E2']} | {r['E3']} |\n")
        f.write("\nCells: GRU / LSTM / Transformer, means over ten seeds (ensembles: one value); "
                "rounded once, from the unrounded mean, to three decimals.\n")
    print(f"  written: {os.path.join(APP_T4, 'table_3_cells.csv')} / .md")
    n_fail = sum(1 for p in pruefungen_t4 if not p)
    if n_fail:
        print(f"\nRESULT: {n_fail} of {len(pruefungen_t4)} structural checks FAIL; the table was written nonetheless.")
        return 1
    print(f"\nRESULT: all {len(pruefungen_t4)} checks PASS.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Performance (S6), level 3 (S7), AUPRC / calibration / ensembles, Tables 3 and 4.")
    ap.add_argument("--evaluate", action="store_true", help="S6 performance and calibration, S7 level 3 (needs the checkpoints and scores)")
    ap.add_argument("--performance", action="store_true", help="AUPRC and lift, calibration slope, ensemble performance (Table 3 inputs)")
    ap.add_argument("--levels", action="store_true", help="level 3 of the deep ensembles and the Monte Carlo mutual information (ebene3_e14.csv)")
    ap.add_argument("--tables", action="store_true", help="Table 3 and Table 4 from the aggregate files")
    a = ap.parse_args(argv)
    if not (a.evaluate or a.performance or a.levels or a.tables):
        ap.print_help()
        return 0
    config.ensure_run_folders()
    if a.evaluate:
        daten = C.lade_natural()
        eicu = C.lade_eicu(daten)
        s6_leistung(daten, eicu)
        s7_ebene3()
    for flag, steps in ((a.performance, (main_auprc, main_calibration, main_ensemble_performance)),
                        (a.levels, (main_level3,)), (a.tables, (main_table3, main_table4))):
        if flag:
            for step in steps:
                rc = step([])
                if rc not in (None, 0):
                    return int(rc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
