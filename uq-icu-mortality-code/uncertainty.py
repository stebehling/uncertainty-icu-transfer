"""
uncertainty.py -- representation density, prediction caches and the level-1 / level-2 measures

Paper:    Methods (representation density; uncertainty measures); Results RQ1 and RQ2a; Table 4
Origin:   uncertainty/density.py, uncertainty/caches.py and uncertainty/signals.py of the full package, merged
Reads:    config.MODELS checkpoints, config.DATASET_MIMIC, config.DATASET_EICU (through common)
Writes:   config.CACHES/p_ihm_*.npz, features_*.npz (patient-level, stay local);
          config.SIGNALS/scores_*.npz (patient-level), ddu_seeds_natural.csv
          (the Monte Carlo measures and mc_seeds_natural.csv come from mc_dropout.py)
Run:      python uncertainty.py --caches      (GPU, minutes)
          python uncertainty.py --signals     (CPU, about 25 min for the three base families)
          or through run_pipeline.py --caches / --signals / --all
"""
from __future__ import annotations

import argparse
import csv
import numpy as np
import os
import sys
import torch
from pathlib import Path
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401
import common as C  # noqa: E402
from models import architectures as A  # noqa: E402


# ==========================================================================================
# section from the full package: uncertainty/density.py
# ------------------------------------------------------------------------------------------
# density.py -- class-conditional Gaussian density on the penultimate features, entropy, AUROCs
#
# Paper:    Methods (representation density, DDU; level 1 and level 2 AUROCs)
# Origin:   src/ddu_ohne_masken.py (library part only: EPS, _t_quantil, kennzahlen,
#           extrahiere_features, _gauss_logpdf, fit_gda, neg_log_dichte, ood_auroc,
#           binaere_entropie, fehler_auroc); signal_funktionen() from src/rerun_pipeline.py v5.7
#           (lines 261-299, now a plain lookup without the AST fallback)
# Reads:    nothing
# Writes:   nothing
# Run:      library module; used by uncertainty/caches.py, uncertainty/signals.py,
#           run_pipeline.py (input check) and the supporting analyses
#
# Functions:
#   extrahiere_features  forward-pre-hook on modell.kopf, eval mode, batch 512 -> float64 features
#   fit_gda              class-conditional Gaussians with empirical log-priors; tied=True shares
#                        the covariance (Mahalanobis variant), tied=False fits one per class
#   neg_log_dichte       -log q(z) marginalised over the classes (logsumexp); Cholesky with
#                        jitter 1e-6, increased tenfold up to six times if not positive definite
#   binaere_entropie     entropy of a Bernoulli prediction (clipped at EPS)
#   ood_auroc            AUROC in-domain (0) vs transfer (1) of a score
#   fehler_auroc         AUROC of a score against the error label (None if one class only)
#   kennzahlen           mean, SD, 95 % t-interval over seeds
# ==========================================================================================

EPS = 1e-7


def _t_quantil(n):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1))
    except Exception:
        tab = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
               8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}
        return tab.get(n, 1.96)


def kennzahlen(werte):
    """(mean, sd, ci_low, ci_high, n, interval excludes zero) over the given values."""
    a = np.asarray([w for w in werte if w is not None], dtype=float)
    n = len(a)
    if n < 2:
        return (float(a.mean()) if n else float("nan"),
                float("nan"), float("nan"), float("nan"), n, False)
    mittel, std = float(a.mean()), float(a.std(ddof=1))
    halb = _t_quantil(n) * std / np.sqrt(n)
    lo, hi = mittel - halb, mittel + halb
    return mittel, std, lo, hi, n, bool(lo > 0 or hi < 0)


# ==========================================================================
# computational core (taken verbatim from the frozen original)
# ==========================================================================
def extrahiere_features(modell, X, batch=512):
    """Captures the input of modell.kopf through a forward-pre-hook. eval mode =>
    one deterministic pass (dropout off), as DDU requires."""
    modell.eval()
    speicher = {}

    def vorhaken(_modul, eingang):
        speicher["h"] = eingang[0].detach()

    handle = modell.kopf.register_forward_pre_hook(vorhaken)
    feats = []
    try:
        with torch.no_grad():
            for i in range(0, len(X), batch):
                modell(X[i:i + batch].to(A.GERAET))
                feats.append(speicher["h"].cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(feats, axis=0).astype(np.float64)


def _gauss_logpdf(Z, mu, cov, jitter):
    d = Z.shape[1]
    diff = Z - mu
    j = jitter
    for _ in range(6):
        try:
            cf = cho_factor(cov + j * np.eye(d), lower=True)
            logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
            sol = cho_solve(cf, diff.T).T
            quad = np.einsum("nd,nd->n", diff, sol)
            return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)
        except np.linalg.LinAlgError:
            j *= 10.0
    raise SystemExit("ABORT (fail-fast): covariance not positive definite even with increased jitter.")


def fit_gda(F, y, tied=False, jitter=1e-6):
    klassen = np.unique(y)
    n, d = F.shape
    mus = [F[y == c].mean(axis=0) for c in klassen]
    logpriors = [np.log(len(F[y == c]) / n) for c in klassen]
    if tied:
        pooled = np.zeros((d, d))
        for c in klassen:
            diff = F[y == c] - F[y == c].mean(axis=0)
            pooled += diff.T @ diff
        covs = [pooled / n for _ in klassen]
    else:
        covs = [np.cov(F[y == c], rowvar=False) for c in klassen]
    return {"klassen": klassen, "mus": mus, "covs": covs,
            "logpriors": logpriors, "jitter": jitter}


def neg_log_dichte(F, gda):
    terme = [lp + _gauss_logpdf(F, mu, cov, gda["jitter"])
             for mu, cov, lp in zip(gda["mus"], gda["covs"], gda["logpriors"])]
    return -logsumexp(np.stack(terme, axis=1), axis=1)


def ood_auroc(score_id, score_ood):
    y = np.concatenate([np.zeros(len(score_id)), np.ones(len(score_ood))])
    s = np.concatenate([score_id, score_ood])
    return roc_auc_score(y, s)


def binaere_entropie(p, eps=EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def fehler_auroc(score, e):
    """AUROC of the score against the error label (1 = wrong). None if only one
    class is present."""
    if score is None or len(np.unique(e)) < 2:
        return None
    return roc_auc_score(e, score)


# ==========================================================================
_SIGNAL_NAMEN = ["extrahiere_features", "_gauss_logpdf", "fit_gda",
                 "neg_log_dichte", "ood_auroc", "binaere_entropie",
                 "fehler_auroc", "kennzahlen", "_t_quantil"]


def signal_funktionen():
    """The density / entropy / AUROC functions as a name -> function dict plus their
    source, in the form the stage modules and supporting analyses expect."""
    fkt = {n: globals()[n] for n in _SIGNAL_NAMEN}
    return fkt, "uncertainty/density.py"



# ==========================================================================================
# section from the full package: uncertainty/caches.py
# ------------------------------------------------------------------------------------------
# caches.py -- predicted probabilities and penultimate features of every deterministic model
#
# Paper:    Methods (predictions on the MIMIC test set and the eICU cohort; features for the
#           representation density)
# Origin:   src/rerun_pipeline.py v5.7 / 2026-08-20, caches() (stage S4)
# Reads:    config.MODELS/modell_ihm_<family>_seed<n>.pt of the deterministic families;
#           data via common.lade_natural / common.lade_eicu
# Writes:   config.CACHES/p_ihm_<family>_det_seed<n>_<domain>.npz (keys p, y),
#           config.CACHES/features_<family>_seed<n>.npz (keys fit, test, y_fit, eicu; float16),
#           config.CACHES/manifest_caches.json   -- all patient-level, stay inside RUNS
# Checks:   checkpoint input_size == 25; existing files are kept (resume)
# Run:      python uncertainty/caches.py   (or run_pipeline.py --caches)
# ==========================================================================================

def caches(daten, eicu):
    """Sigmoid probabilities of one deterministic pass per domain and the features at the
    head input (fit = training part of the split, test, eicu)."""
    fkt, quelle = signal_funktionen()
    dom_X = {"in_domain": daten["Xte"], "transfer": eicu["X"] if eicu else None}
    dom_y = {"in_domain": daten["yte"].numpy(),
             "transfer": eicu["y"] if eicu else None}
    for familie, (arch, variante, _) in C.FAMILIEN.items():
        if variante != "det":
            continue
        for seed in C.SEEDS:
            ck = C._finde(os.path.join(C.ORDNER["modelle"],
                                       f"modell_{C.TASK}_{familie}_seed{seed}.pt"))
            if ck is None:
                continue
            c = torch.load(ck, map_location="cpu", weights_only=False)
            if c["best_hp"].get("input_size") != 25:
                C.abbruch(f"{ck}: input_size != 25.")
            C.seed_alles(seed)
            modell = C.baue_modell_natural(c["best_hp"], familie)
            modell.load_state_dict(c["state_dict"])
            modell.eval()
            fertig = True
            for dom in ("in_domain", "transfer"):
                if dom_X[dom] is None:
                    fertig = False
                    continue
                ziel = C._sicher(C.ORDNER["caches"],
                                 f"p_{C.TASK}_{familie}_det_seed{seed}_{dom}.npz")
                if not os.path.exists(ziel):
                    p = torch.sigmoid(A._det_vorhersage(
                        modell, dom_X[dom])).numpy()
                    np.savez_compressed(ziel, p=p, y=dom_y[dom])
            zf = C._sicher(C.ORDNER["caches"],
                           f"features_{familie}_seed{seed}.npz")
            if not os.path.exists(zf):
                Ffit = fkt["extrahiere_features"](modell, daten["Xtr"])
                Fte = fkt["extrahiere_features"](modell, daten["Xte"])
                bl = {"fit": Ffit.astype(np.float16),
                      "test": Fte.astype(np.float16),
                      "y_fit": daten["ytr"].numpy().astype(np.int8)}
                if eicu:
                    bl["eicu"] = fkt["extrahiere_features"](
                        modell, eicu["X"]).astype(np.float16)
                np.savez_compressed(zf, **bl)
            C.log(f"[S4] {familie}/seed {seed}: caches "
                  f"{'complete' if fertig and eicu else 'without eICU part'}.")
            del modell
    C.manifest_schreiben(C.ORDNER["caches"], "manifest_caches.json",
                         {"signal_quelle": quelle,
                          "hinweis": "features float16; fit = training part of the split."},
                         skript="uncertainty/caches.py")


def main_caches(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    config.ensure_run_folders()
    daten = C.lade_natural()
    caches(daten, C.lade_eicu(daten))



# ==========================================================================================
# section from the full package: uncertainty/signals.py
# ------------------------------------------------------------------------------------------
# signals.py -- uncertainty measures per seed: level 1 (cohort separation) and level 2 (error
#               detection) for density, density variants and entropy
#
# Paper:    Results RQ1 / RQ2a; Table 4; Supplementary tables of the density variants
# Origin:   src/rerun_pipeline.py v5.7 / 2026-08-20, signale(), fit_varianten(), _l2norm(),
#           kernreport() (stage S5); the Monte Carlo block of the original stage (passes of the
#           separately trained MC-dropout twins) was replaced by mc_dropout.py (DECISIONS.md D9)
# Reads:    config.CACHES/features_<family>_seed<n>.npz and p_ihm_<family>_det_seed<n>_<dom>.npz
# Writes:   config.SIGNALS/ddu_seeds_natural.csv (columns arch, seed, fit_menge, kovarianz,
#           ebene, signal, domaene, schwelle, wert),
#           config.SIGNALS/scores_<family>_seed<n>.npz (patient-level scores, stay inside RUNS;
#           mc_dropout.py adds the Monte Carlo logit variance mc_epistemisch_<dom> afterwards),
#           config.SIGNALS/manifest_signale.json
# Checks:   an existing CSV must be a bit-identical subset of the new rows (superset watchdog)
# Run:      python uncertainty.py --signals   (or run_pipeline.py --signals); CPU
#
# Measures: dichte (class-conditional Gaussian, fit sets voll_natuerlich = training part of the
# split and mukhoti_balanciert = class-balanced subset, covariances klassenspezifisch and
# tied_mahalanobis), rmd (relative density score: class density minus background density),
# gmm_klasse (per-class Gaussian mixture, 3 components), gda_l2 (density on L2-normalised
# features), softmax_entropie. Level-2 error labels at the six thresholds of common.SCHWELLEN.
# ==========================================================================================

# ---------------------- density variants
GMM_SEED = 42
VARIANTEN_SIGNALE = ("rmd", "gmm_klasse", "gda_l2")


def _l2norm(F):
    F = np.asarray(F, dtype=np.float64)
    return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)


def fit_varianten(Ffit, y_fit, fkt, nur=None):
    """Background Gaussian for the relative density score, per-class GMMs (3 components),
    L2-GDA -- all on the given fitting set. Returns dict measure -> score_fn(F) -> ndarray;
    `nur` restricts the variants that are fitted."""
    from sklearn.mixture import GaussianMixture
    Ffit = np.asarray(Ffit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=int)
    gda = fkt["fit_gda"](Ffit, y_fit, tied=False)
    mu_bg = Ffit.mean(axis=0)
    zent = Ffit - mu_bg
    cov_bg = zent.T @ zent / len(Ffit) + 1e-6 * np.eye(Ffit.shape[1])
    gmms, logpri = {}, {}
    n_ges = len(y_fit)
    for c in ((0, 1) if (nur is None or "gmm_klasse" in nur) else ()):
        Fc = Ffit[y_fit == c]
        gmms[c] = GaussianMixture(
            n_components=3, covariance_type="full", reg_covar=1e-6,
            max_iter=200, n_init=1, init_params="kmeans",
            random_state=GMM_SEED).fit(Fc)
        logpri[c] = float(np.log(len(Fc) / n_ges))
    gda_l2 = (fkt["fit_gda"](_l2norm(Ffit), y_fit, tied=False)
              if (nur is None or "gda_l2" in nur) else None)

    def s_rmd(F):
        F = np.asarray(F, dtype=np.float64)
        nll_k = fkt["neg_log_dichte"](F, gda)
        d = F - mu_bg
        L = np.linalg.cholesky(cov_bg)
        from scipy.linalg import cho_solve
        maha = np.einsum("ij,ij->i", d, cho_solve((L, True), d.T).T)
        logdet = 2.0 * float(np.log(np.diag(L)).sum())
        k = F.shape[1]
        nll_bg = 0.5 * (maha + logdet + k * np.log(2 * np.pi))
        return nll_k - nll_bg

    def s_gmm(F):
        F = np.asarray(F, dtype=np.float64)
        from scipy.special import logsumexp
        lg = np.stack([logpri[c] + gmms[c].score_samples(F)
                       for c in (0, 1)], axis=1)
        return -logsumexp(lg, axis=1)

    def s_l2(F):
        return fkt["neg_log_dichte"](_l2norm(F), gda_l2)

    alle = {"rmd": s_rmd, "gmm_klasse": s_gmm, "gda_l2": s_l2}
    return alle if nur is None else {k: v for k, v in alle.items()
                                     if k in nur}


# ---------------------------------------------------------------- stage S5
def signale(daten, eicu):
    """Level-1 and level-2 AUROCs of every measure per deterministic family and seed."""
    fkt, quelle = signal_funktionen()
    y_fit = daten["ytr"].numpy().astype(int)
    pos = np.flatnonzero(y_fit == 1)
    neg = np.flatnonzero(y_fit == 0)
    rng = np.random.default_rng(C.BAL_SEED)
    bal_idx = np.sort(np.concatenate([pos, rng.choice(neg, size=len(pos),
                                                      replace=False)]))
    fit_mengen = {"voll_natuerlich": np.arange(len(y_fit)),
                  "mukhoti_balanciert": bal_idx}
    zeilen_ddu = []
    for familie, (arch, variante, _) in C.FAMILIEN.items():
        if variante != "det":
            continue
        for seed in C.SEEDS:
            zf = C._finde(os.path.join(C.ORDNER["caches"],
                                       f"features_{familie}_seed{seed}.npz"))
            if zf is None:
                continue
            F_ = np.load(zf)
            if "eicu" not in F_.files:
                C.log(f"[S5] {familie}/seed {seed}: no eICU features -- skipped.")
                continue
            Fd = {"in_domain": F_["test"].astype(np.float64),
                  "transfer": F_["eicu"].astype(np.float64)}
            Ffit = F_["fit"].astype(np.float64)
            etikett, p_dom = {}, {}
            for dom in ("in_domain", "transfer"):
                pc = C._finde(os.path.join(
                    C.ORDNER["caches"],
                    f"p_{C.TASK}_{familie}_det_seed{seed}_{dom}.npz"))
                if pc is None:
                    C.abbruch(f"det cache missing: {familie}/seed{seed}/{dom}.")
                dnp = np.load(pc)
                p_dom[dom] = np.asarray(dnp["p"], float).ravel()
                yv = np.asarray(dnp["y"], float).ravel()
                etikett[dom] = {s_name: ((p_dom[dom] > t).astype(int)
                                         != yv.astype(int)).astype(int)
                                for s_name, t in C.SCHWELLEN}
            for fitname, idx in fit_mengen.items():
                for kovname, tied in (("klassenspezifisch", False),
                                      ("tied_mahalanobis", True)):
                    gda = fkt["fit_gda"](Ffit[idx], y_fit[idx], tied=tied)
                    s = {d_: fkt["neg_log_dichte"](Fd[d_], gda)
                         for d_ in Fd}
                    zeilen_ddu.append([arch, seed, fitname, kovname,
                                       "ebene1_ood", "dichte",
                                       "mimic_vs_eicu", "",
                                       round(fkt["ood_auroc"](
                                           s["in_domain"], s["transfer"]), 5)])
                    for dom in Fd:
                        for s_name, _t in C.SCHWELLEN:
                            a = fkt["fehler_auroc"](s[dom],
                                                    etikett[dom][s_name])
                            zeilen_ddu.append([arch, seed, fitname, kovname,
                                               "ebene2_fehler", "dichte", dom,
                                               s_name,
                                               round(a, 5) if a else ""])
            # density variants: all on the primary fitting set; on the balanced
            # fitting set only the relative density score (sensitivity)
            var_scores = {}
            for fitname, idx in fit_mengen.items():
                nur = None if fitname == "voll_natuerlich" else ("rmd",)
                fn_ = fit_varianten(Ffit[idx], y_fit[idx], fkt, nur=nur)
                for sig, f_ in fn_.items():
                    var_scores[(fitname, sig)] = {d_: f_(Fd[d_]) for d_ in Fd}
            for (fitname, sig), s in var_scores.items():
                zeilen_ddu.append([arch, seed, fitname, "",
                                   "ebene1_ood", sig, "mimic_vs_eicu", "",
                                   round(fkt["ood_auroc"](
                                       s["in_domain"], s["transfer"]), 5)])
                for dom in Fd:
                    for s_name, _t in C.SCHWELLEN:
                        a_ = fkt["fehler_auroc"](s[dom], etikett[dom][s_name])
                        zeilen_ddu.append([arch, seed, fitname, "",
                                           "ebene2_fehler", sig, dom, s_name,
                                           round(a_, 5) if a_ else ""])
            ent = {d_: fkt["binaere_entropie"](p_dom[d_]) for d_ in p_dom}
            zeilen_ddu.append([arch, seed, "", "", "ebene1_ood",
                               "softmax_entropie", "mimic_vs_eicu", "",
                               round(fkt["ood_auroc"](ent["in_domain"],
                                                      ent["transfer"]), 5)])
            for dom in ent:
                for s_name, _t in C.SCHWELLEN:
                    a = fkt["fehler_auroc"](ent[dom], etikett[dom][s_name])
                    zeilen_ddu.append([arch, seed, "", "", "ebene2_fehler",
                                       "softmax_entropie", dom, s_name,
                                       round(a, 5) if a else ""])
            zs = C._sicher(C.ORDNER["signale"],
                           f"scores_{familie}_seed{seed}.npz")
            if not os.path.exists(zs):
                bl = {}
                for dom in Fd:
                    gda_p = fkt["fit_gda"](Ffit, y_fit, tied=False)
                    bl[f"dichte_{dom}"] = fkt["neg_log_dichte"](
                        Fd[dom], gda_p).astype(np.float32)
                    for sig in VARIANTEN_SIGNALE:
                        bl[f"{sig}_{dom}"] = var_scores[
                            ("voll_natuerlich", sig)][dom].astype(np.float32)
                    bl[f"entropie_{dom}"] = ent[dom].astype(np.float32)
                    bl[f"p_{dom}"] = p_dom[dom].astype(np.float32)
                    bl[f"y_{dom}"] = np.load(C._finde(os.path.join(
                        C.ORDNER["caches"],
                        f"p_{C.TASK}_{familie}_det_seed{seed}_{dom}.npz"))
                        )["y"].astype(np.int8)
                np.savez_compressed(zs, **bl)
            C.log(f"[S5] {familie}/seed {seed}: measures computed.")
    if not zeilen_ddu:
        C.abbruch("S5: no measures computed (features / caches missing?).")
    for name, kopf, zeilen in (
            ("ddu_seeds_natural.csv",
             ["arch", "seed", "fit_menge", "kovarianz", "ebene", "signal",
              "domaene", "schwelle", "wert"], zeilen_ddu),):
        ziel = C._sicher(C.ORDNER["signale"], name)
        if os.path.exists(ziel):
            with open(ziel, newline="", encoding="utf-8") as f:
                alte = {tuple(r[:-1]): r[-1] for r in list(csv.reader(f))[1:]}
            neue = {tuple(str(x) for x in z[:-1]): str(z[-1]) for z in zeilen}
            verletzt = [k for k, v in alte.items()
                        if k in neue and neue[k] != v]
            fehlend = [k for k in alte if k not in neue]
            if verletzt or fehlend:
                C.abbruch(f"{name}: superset watchdog -- {len(verletzt)} "
                          f"differing / {len(fehlend)} missing old rows. "
                          "Frozen measures would be changed.")
            C.log(f"[S5] {name}: {len(alte)} old rows contained bit-identically "
                  "(superset ok), file is extended.")
        with open(ziel, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(kopf)
            w.writerows(zeilen)
        C.log(f"[S5] written: {ziel} ({len(zeilen)} rows)")
    C.manifest_schreiben(C.ORDNER["signale"], "manifest_signale.json",
                         {"signal_quelle": quelle,
                          "schwellen": dict(C.SCHWELLEN),
                          "fit_mengen": {k: int(len(v))
                                         for k, v in fit_mengen.items()},
                          "bal_seed": C.BAL_SEED,
                          "hinweis": "Monte Carlo measures: mc_dropout.py (manifest_mc_dropout.json)"},
                         skript="uncertainty/signals.py")
    kernreport(zeilen_ddu, fkt)


def kernreport(zeilen_ddu, fkt):
    """Console summary: mean [95 % t-interval] over seeds at the primary threshold."""
    print("\n" + "=" * 78)
    print("CORE REPORT  (mean [95 % t-CI] over seeds; primary threshold 0.1288)")
    print("=" * 78)
    def zeige(zeilen, sig_idx, wert_idx, filt):
        gruppen = {}
        for z in zeilen:
            if not filt(z) or z[wert_idx] == "":
                continue
            gruppen.setdefault((z[0], z[sig_idx]), []).append(float(z[wert_idx]))
        for (arch, sig), v in sorted(gruppen.items()):
            m, sd, lo, hi, n, _ = fkt["kennzahlen"](v)
            print(f"  {arch:<11}{sig:<20}{m:>8.4f}  [{lo:.4f}, {hi:.4f}]  n={n}")
    print("-- level 1 (cohort separation), primary density + entropy --")
    zeige(zeilen_ddu, 5, 8, lambda z: z[4] == "ebene1_ood" and
          (z[5] == "softmax_entropie" or (z[2] == "voll_natuerlich"
                                          and z[3] == "klassenspezifisch")))
    print("-- level 2 transfer, threshold 0.1288 --")
    zeige(zeilen_ddu, 5, 8, lambda z: z[4] == "ebene2_fehler"
          and z[6] == "transfer" and z[7] == "0.1288"
          and (z[5] == "softmax_entropie" or (z[2] == "voll_natuerlich"
                                              and z[3] == "klassenspezifisch")))
    print("   (Monte Carlo measures: mc_dropout.py)")
    print("=" * 78)


def main_signals(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    config.ensure_run_folders()
    daten = C.lade_natural()
    signale(daten, C.lade_eicu(daten))



def main(argv=None):
    ap = argparse.ArgumentParser(description="Prediction caches (S4) and level-1 / level-2 measures (S5).")
    ap.add_argument("--caches", action="store_true")
    ap.add_argument("--signals", action="store_true")
    a = ap.parse_args(argv)
    if not (a.caches or a.signals):
        ap.print_help()
        return 0
    config.ensure_run_folders()
    daten = C.lade_natural()
    eicu = C.lade_eicu(daten)
    if a.caches:
        caches(daten, eicu)
    if a.signals:
        signale(daten, eicu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
