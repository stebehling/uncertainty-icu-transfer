"""
ensembles.py (deep_ensembles.py of the full package) -- mutual-information decomposition of the deep ensembles (ten seeds)

Paper:    deep-ensemble rows of Figure 4 / the level tables (mi_gesamt, mi_aleatorisch,
          mi_epistemisch at level 1 and 2), Table S8 threshold sensitivity (tau = 0.5 rows),
          ensemble performance baseline read by performance/ensemble_performance.py
Origin:   src/mi_ensemble_natural.py, SKRIPT_STAND "v1 / 2026-08-22 (E14a, Ensemble-MI)"
Reads:    config.SIGNALS/scores_<family>_seed<seed>.npz   (p_<domain>, y_<domain>, patient-level)
          config.REPORT/leistung_natural.csv               (auroc)
Writes:   config.REPORT/mi_ensemble_natural.csv            (long format, all numbers)
Checks:   A1  y identical across the ten seeds of a family (same test set)
          A2  per seed: AUROC from p / y reproduces the column `auroc` of leistung_natural.csv
              at the five-decimal storage precision (6e-6)
          A3  identity U_ges = U_alea + U_epi within 1e-9; U_epi >= -1e-9 everywhere
Run:      python uncertainty/deep_ensembles.py

For every deterministic family the ten seeds form a deep ensemble (Lakshminarayanan et al.
2017: random initialisation plus data shuffling; both are controlled by the seed). From the
STORED p per seed the following is computed per patient:
    p_quer  = mean of the 10 member probabilities
    U_ges   = H(p_quer)                      (predictive entropy, nats)
    U_alea  = mean of H(p_m)                 (expected entropy)
    U_epi   = U_ges - U_alea                 (mutual information, >= 0)
Evaluation:
    level 1: AUROC eICU vs MIMIC per quantity (eICU = positive class)
    level 2: AUROC against the ensemble error label e = 1[1(p_quer > t) != y],
             t = config.THRESHOLD (primary) and config.THRESHOLD_SENSITIVITY
    (E)    : AUROC / AUPRC of the ensemble mean p_quer as a baseline next to the mean of the
             single models (rows 'p_quer_leistung': wert_voll = AUROC, jack_min = AUPRC,
             jack_max = prevalence)
Variability: leave-one-member-out jackknife (10 nine-member ensembles), reported as a range
and explicitly NOT a confidence interval. The verdict column follows the declared grammar
(kohortenerhoeht = mean >= 0.55 and whole range above 0.5; invertiert = whole range below 0.5;
abwesend otherwise).

Only aggregates are written; patient-level files are read locally.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

SKRIPT_STAND = "package 1.0 / 2026-10-08 (port of mi_ensemble_natural.py v1)"
SEEDS = config.SEEDS
DOMS = ("in_domain", "transfer")
SCHWELLEN = (config.THRESHOLD, config.THRESHOLD_SENSITIVITY)   # (0.1288, 0.5)
EPS = 1e-12
SCHRANKE = 0.55     # declared bound for the verdict "kohortenerhoeht"


def abbruch(t):
    raise SystemExit("ABORT (fail-fast): " + t)


def _sicher(name):
    """Output path inside config.REPORT (guards against path escapes)."""
    config.REPORT.mkdir(parents=True, exist_ok=True)
    ziel = (config.REPORT / name).resolve()
    if config.REPORT.resolve() not in ziel.parents:
        abbruch("output path outside of " + str(config.REPORT))
    return ziel


def H(p):
    """Binary entropy in nats."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def ood_auroc(score_id, score_ood):
    """Level 1: AUROC of the score for eICU (positive class) against MIMIC."""
    y = np.concatenate([np.zeros(len(score_id)), np.ones(len(score_ood))])
    return float(roc_auc_score(y, np.concatenate([score_id, score_ood])))


def fehler_auroc(score, e):
    """Level 2: AUROC of the score against the error label; None if e is constant."""
    if len(np.unique(e)) < 2:
        return None
    return float(roc_auc_score(e, score))


def mi_groessen(P):
    """P: (M, N) member probabilities -> (p_quer, u_ges, u_alea, u_epi)."""
    p_quer = P.mean(0)
    u_ges = H(p_quer)
    u_alea = H(P).mean(0)
    u_epi = u_ges - u_alea
    return p_quer, u_ges, u_alea, u_epi


def urteil(werte):
    """Verdict on the level-1 AUROC values (full ensemble plus jackknife members)."""
    lo, hi, m = min(werte), max(werte), float(np.mean(werte))
    if m >= SCHRANKE and lo > 0.5:
        return "kohortenerhoeht"
    if hi < 0.5:
        return "invertiert"
    return "abwesend"


def main():
    print("=" * 78)
    print("DEEP ENSEMBLES  mutual-information decomposition")
    print(f"  script version: {SKRIPT_STAND}")
    print("=" * 78)
    le = pd.read_csv(config.REPORT / "leistung_natural.csv")
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
    print(f"  families: {', '.join(familien)}\n")

    zeilen = []
    for familie in familien:
        P = {dom: [] for dom in DOMS}
        Y = {}
        for seed in SEEDS:
            z = np.load(config.SIGNALS / f"scores_{familie}_seed{seed}.npz")
            for dom in DOMS:
                p = np.asarray(z[f"p_{dom}"], float).ravel()
                y = np.asarray(z[f"y_{dom}"], float).ravel()
                if dom in Y:
                    if not np.array_equal(Y[dom], y):
                        abbruch(f"A1: y differs ({familie}/seed{seed}/{dom}).")
                else:
                    Y[dom] = y
                r = le[(le.familie == familie) & (le.seed == seed)
                       & (le.domaene == dom)]
                if len(r) != 1:
                    abbruch(f"leistung: {len(r)} rows for "
                            f"{familie}/{seed}/{dom}.")
                d_ = abs(float(roc_auc_score(y, p)) - float(r.auroc.iloc[0]))
                if d_ > 6e-6:
                    abbruch(f"A2: AUROC {familie}/{seed}/{dom} deviates by "
                            f"{d_:.2e} -- not the same predictions.")
                P[dom].append(p)
        P = {dom: np.stack(P[dom]) for dom in DOMS}          # (10, N)

        # full ensemble plus jackknife (index sets)
        mengen = [("voll", list(range(10)))] + \
                 [(f"jack_ohne_{SEEDS[i]}", [j for j in range(10) if j != i])
                  for i in range(10)]
        G = {}
        for name, idx in mengen:
            G[name] = {}
            for dom in DOMS:
                p_quer, u_ges, u_alea, u_epi = mi_groessen(P[dom][idx])
                if np.max(np.abs(u_ges - (u_alea + u_epi))) > 1e-9 \
                        or np.min(u_epi) < -1e-9:
                    abbruch(f"A3: identity / non-negativity violated "
                            f"({familie}/{name}/{dom}).")
                G[name][dom] = {"p": p_quer, "ges": u_ges, "alea": u_alea,
                                "epi": u_epi}

        # level 1 per quantity: full ensemble plus jackknife values
        for kurz, lang in (("epi", "mi_epistemisch"), ("ges", "mi_gesamt"),
                           ("alea", "mi_aleatorisch")):
            e1 = {name: ood_auroc(G[name]["in_domain"][kurz],
                                  G[name]["transfer"][kurz])
                  for name, _ in mengen}
            jack = [v for n, v in e1.items() if n != "voll"]
            zeilen.append({"familie": familie, "groesse": lang,
                           "ebene": "ebene1_ood", "domaene": "mimic_vs_eicu",
                           "schwelle": "", "wert_voll": round(e1["voll"], 5),
                           "jack_min": round(min(jack), 5),
                           "jack_max": round(max(jack), 5),
                           "urteil": urteil(jack + [e1["voll"]])})
        # level 2 against the ensemble error label
        for dom in DOMS:
            for t in SCHWELLEN:
                e = ((G["voll"][dom]["p"] > t).astype(int)
                     != Y[dom].astype(int)).astype(int)
                for kurz, lang in (("epi", "mi_epistemisch"),
                                   ("ges", "mi_gesamt"),
                                   ("alea", "mi_aleatorisch")):
                    a = fehler_auroc(G["voll"][dom][kurz], e)
                    jack = [fehler_auroc(G[n][dom][kurz],
                                         ((G[n][dom]["p"] > t).astype(int)
                                          != Y[dom].astype(int)).astype(int))
                            for n, _ in mengen if n != "voll"]
                    zeilen.append({"familie": familie, "groesse": lang,
                                   "ebene": "ebene2_fehler", "domaene": dom,
                                   "schwelle": str(t),
                                   "wert_voll": round(a, 5) if a else "",
                                   "jack_min": round(min(jack), 5),
                                   "jack_max": round(max(jack), 5),
                                   "urteil": ""})
        # (E) performance baseline of the ensemble mean
        for dom in DOMS:
            p_quer = G["voll"][dom]["p"]
            einzel = le[(le.familie == familie)
                        & (le.domaene == dom)].auroc.mean()
            zeilen.append({"familie": familie, "groesse": "p_quer_leistung",
                           "ebene": "leistung", "domaene": dom, "schwelle": "",
                           "wert_voll": round(float(
                               roc_auc_score(Y[dom], p_quer)), 5),
                           "jack_min": round(float(average_precision_score(
                               Y[dom], p_quer)), 5),
                           "jack_max": round(float(Y[dom].mean()), 5),
                           "urteil": f"einzel_auroc_mittel={einzel:.5f}"})
        print(f"  [OK] {familie}  (A1-A3 passed)")

    df = pd.DataFrame(zeilen)
    df.to_csv(_sicher("mi_ensemble_natural.csv"), index=False)
    print(f"\n  written: mi_ensemble_natural.csv ({len(df)} rows)")
    print("  (rows 'p_quer_leistung': wert_voll=AUROC, jack_min=AUPRC, "
          "jack_max=prevalence)")

    print("\n" + "-" * 78)
    print("PRIMARY ENDPOINT: level-1 AUROC of the mutual information")
    print(f"{'Family':<12}{'MI (full)':>10}{'Jackknife range':>22}{'Verdict':>18}")
    for familie in familien:
        r = df[(df.familie == familie) & (df.groesse == "mi_epistemisch")
               & (df.ebene == "ebene1_ood")].iloc[0]
        print(f"{familie:<12}{r.wert_voll:>10.4f}"
              f"   [{r.jack_min:.4f}; {r.jack_max:.4f}]  "
              f"{r.urteil:>18}")
    print("\n  Verdict: 'kohortenerhoeht' = mean "
          f">= {SCHRANKE}\n  AND range entirely above 0.5. The range is a "
          "leave-one-member-out\n  jackknife (correlated reuse of the same members), "
          "NOT a confidence interval.")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Mutual-information decomposition of the ten-seed deep ensembles "
                    "(levels 1 and 2, jackknife range, ensemble performance baseline); "
                    "writes mi_ensemble_natural.csv to config.REPORT.")
    ap.parse_args()
    main()
