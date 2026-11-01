"""
mc_dropout.py -- Monte Carlo dropout of the trained models: logit variance, predictive entropy, aleatoric part
                 and mutual information from 30 stochastic passes per model and cohort

Paper:    the Monte Carlo rows of Table 4 (levels 1 and 2 here; level 3 through evaluation.py, which reads the
          logit variance from the scores files and the MI quantities from mc_mi_*.npz) and the Monte Carlo
          sentences of RQ1 / RQ2a / RQ2b
Origin:   the Monte Carlo stage of DECISIONS.md D9 (levels/mc_base_stage.py of the review tooling, v1.2 /
          2026-10-11; pass loop of levels/mc_det_dropout.py v1.0 / 2026-10-09), which replaced the separately
          trained MC-dropout twins with a heteroscedastic head of the earlier draft (src/mc_mi_stufe.py)
Method:   every trained deterministic model (GRU, LSTM, Transformer; seeds 42-51) runs T = 30 times per cohort
          with every dropout layer that was active in training switched on (config.MC_MASKS = "full": the
          nn.Dropout modules and the inter-layer dropout of the recurrent encoder); the masks are drawn after
          torch.manual_seed(config.MC_SEED) immediately before the passes of each cohort; batch 512, no
          shuffling (models/architectures.py, mc_passes). Probability scale, as for the deep ensembles:
          p_t = sigmoid(logit_t), U_total = H(mean_t p_t), U_alea = mean_t H(p_t), MI = U_total - U_alea.
          Logit variance = variance of the 30 logits per patient (ddof 0). Errors = the deterministic
          prediction of the SAME model at the threshold versus the outcome (own-model pairing).
Reads:    config.MODELS/modell_ihm_<family>_seed<n>.pt            (the trained deterministic models)
          config.SIGNALS/scores_<family>_seed<n>.npz             (p_<dom>, y_<dom> and the stage-S5 measures)
          config.CACHES/p_ihm_<family>_det_seed<n>_<dom>.npz     (A4: cache and scores carry the same p and y)
          data via common.lade_natural / common.lade_eicu
Writes:   config.SIGNALS/mc_mi_<family>_seed<n>.npz     (patient-level, stays inside RUNS: mi_ges / mi_alea /
                                                         mi_epi and v_epi per cohort, float32)
          config.SIGNALS/scores_<family>_seed<n>.npz    (rewritten in place: mc_epistemisch_<dom> = logit variance
                                                         added, every other key bit-identical; stale mc_* keys of
                                                         an earlier run are dropped)
          config.SIGNALS/mc_seeds_natural.csv           (arch, seed, ebene, signal = mc_epistemisch, domaene,
                                                         schwelle, wert; levels 1 and 2 at the six thresholds)
          config.REPORT/mc_mi_seeds_natural.csv         (arch, seed, variante = mittel, ebene, signal, domaene,
                                                         schwelle, wert; levels 1 and 2 at 0.1288 and 0.5)
          config.SIGNALS/manifest_mc_dropout.json
Checks:   A1  n per cohort 7,298 / 66,934; checkpoint, scores file and caches present
          A2  one deterministic pass of the loaded checkpoint reproduces the stored p_<dom> of the scores file
              within 1e-5 -- checkpoint and inputs are those of stage S5
          A3  U_total = U_alea + MI within 1e-9 and MI >= -1e-9 (models/architectures.py, mc_signale)
          A4  cache p / y equal the scores p / y (one label convention for every stage)
          A5  the rewritten scores file keeps every non-mc key array-identical; if the two CSVs already exist,
              every existing row must be reproduced bit-identically (five decimals) before a file is overwritten
          A7  no BatchNorm module; the modules in train mode during the passes are listed in the manifest
          Resume: when the manifest of this script version and every output exist, the stage is skipped.
Run:      python mc_dropout.py     (after run_pipeline.py --signals and before --evaluate; GPU, about 40 min for
                                    3 families x 10 seeds x 2 cohorts x 30 passes)
          or through run_pipeline.py --mc / --all

Console: per family the mean and 95 % interval (t9 = 2.262) of the level-1 MI over the ten seeds and the verdict
by the declared grammar: kohortenerhoeht (cohort-elevated) = mean >= 0.55 and the whole interval above 0.5;
invertiert (inverted) = whole interval below 0.5; abwesend (absent) otherwise.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import common as C  # noqa: E402
from models import architectures as A  # noqa: E402

SKRIPT_STAND = "short version 1.1 / 2026-10-26 (Monte Carlo dropout of the trained models; DECISIONS.md D9)"
FAMILIEN = tuple(config.BASE_FAMILIES)            # ("GRU_det", "LSTM_det", "T1_det")
SEEDS = list(config.SEEDS)
DOMS = ("in_domain", "transfer")
N_SOLL = {"in_domain": 7298, "transfer": 66934}
MI_SCHWELLEN = (config.THRESHOLD, config.THRESHOLD_SENSITIVITY)   # (0.1288, 0.5)
TOL_DET = 1e-5
T9 = 2.262
SCHRANKE = 0.55     # declared bound for the verdict "kohortenerhoeht"
URTEIL = {"kohortenerhoeht": "cohort-elevated", "invertiert": "inverted", "abwesend": "absent"}
KOPF_MI = ["arch", "seed", "variante", "ebene", "signal", "domaene", "schwelle", "wert"]
KOPF_MC = ["arch", "seed", "ebene", "signal", "domaene", "schwelle", "wert"]
MANIFEST = "manifest_mc_dropout.json"


def abbruch(t):
    raise SystemExit("ABORT (fail-fast): " + t)


def ood_auroc(a, b):
    """Level 1: AUROC of the score for eICU (positive class) against MIMIC."""
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    return float(roc_auc_score(y, np.concatenate([a, b])))


def fehler_auroc(s, e):
    """Level 2: AUROC of the score against the error label; None if e is constant."""
    return None if len(np.unique(e)) < 2 else float(roc_auc_score(e, s))


def r5(x):
    """Five decimals; '' when the AUROC is undefined (convention of the result files)."""
    return round(x, 5) if x else ""


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def lade_npz(pfad):
    """Every array of an npz file as a dict; the handle is closed before returning (an open
    handle blocks the in-place replacement of the file on Windows)."""
    with np.load(pfad) as z:
        return {k: z[k] for k in z.files}


def schreibe_npz(pfad, arrays):
    """Write compressed to a temporary file next to the target, then replace the target."""
    tmp = pfad + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, pfad)


def superset_csv(ziel, zeilen):
    """A5: an existing file must be reproduced bit-identically (its rows are a subset of the new rows)."""
    if not os.path.exists(ziel):
        return
    with open(ziel, newline="", encoding="utf-8") as f:
        alte = {tuple(r[:-1]): r[-1] for r in list(csv.reader(f))[1:]}
    neue = {tuple(str(x) for x in z[:-1]): str(z[-1]) for z in zeilen}
    verletzt = [k for k, v in alte.items() if k in neue and neue[k] != v]
    fehlend = [k for k in alte if k not in neue]
    if verletzt or fehlend:
        abbruch(f"A5: {os.path.basename(ziel)}: {len(verletzt)} differing / {len(fehlend)} missing old rows "
                "-- the existing file would be changed (a file of an earlier run or method? remove it by hand).")
    print(f"  [OK] A5  {os.path.basename(ziel)}: {len(alte)} existing rows reproduced bit-identically.")


def fertig():
    """Resume: the manifest of this script version and every output exist."""
    p_man = os.path.join(C.ORDNER["signale"], MANIFEST)
    if not os.path.exists(p_man):
        return False
    man = json.load(io.open(p_man, encoding="utf-8"))
    if man.get("skript_stand") != SKRIPT_STAND:
        return False
    if not (os.path.exists(os.path.join(C.ORDNER["signale"], "mc_seeds_natural.csv"))
            and os.path.exists(os.path.join(C.ORDNER["report"], "mc_mi_seeds_natural.csv"))):
        return False
    for familie in FAMILIEN:
        for seed in SEEDS:
            zs = os.path.join(C.ORDNER["signale"], f"scores_{familie}_seed{seed}.npz")
            if not (os.path.exists(zs) and os.path.exists(
                    os.path.join(C.ORDNER["signale"], f"mc_mi_{familie}_seed{seed}.npz"))):
                return False
            with np.load(zs) as z:
                if any(f"mc_epistemisch_{d}" not in z.files for d in DOMS):
                    return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Monte Carlo dropout of the trained models (30 passes per model and cohort): logit variance "
                    "-> scores_*.npz and mc_seeds_natural.csv; predictive entropy, aleatoric part and mutual "
                    "information -> mc_mi_<family>_seed<n>.npz and mc_mi_seeds_natural.csv.")
    ap.add_argument("--force", action="store_true", help="recompute even when every output of this version exists")
    a = ap.parse_args(argv)
    print("=" * 78)
    print("MONTE CARLO DROPOUT OF THE TRAINED MODELS  (30 passes per model and cohort)")
    print(f"  script version: {SKRIPT_STAND}")
    print(f"  device: {A.GERAET} | torch {torch.__version__} | T = {A.T_MC} passes | masks '{A.MC_MASKS}' | "
          f"mask seed {C.MC_ZIEHUNGS_SEED} per cohort")
    print("=" * 78)
    config.ensure_run_folders()
    if fertig() and not a.force:
        print("  every output of this version present -- stage skipped (resume; --force recomputes).")
        return 0

    daten = C.lade_natural()
    eicu = C.lade_eicu(daten, pflicht=True)
    X = {"in_domain": daten["Xte"], "transfer": eicu["X"]}
    for dom in DOMS:
        if len(X[dom]) != N_SOLL[dom]:
            abbruch(f"A1: {dom} has {len(X[dom])} rows instead of {N_SOLL[dom]}.")

    zeilen_mi, zeilen_mc, module = [], [], {}
    for familie in FAMILIEN:
        arch = familie.replace("_det", "")            # GRU | LSTM | T1 (the 'arch' column of the result files)
        e1_epi = []
        for seed in SEEDS:
            ck = os.path.join(C.ORDNER["modelle"], f"modell_{C.TASK}_{familie}_seed{seed}.pt")
            zs = os.path.join(C.ORDNER["signale"], f"scores_{familie}_seed{seed}.npz")
            if not (os.path.exists(ck) and os.path.exists(zs)):
                abbruch(f"A1: {ck} or {zs} is missing (run --seeds / --signals first).")
            cm = torch.load(ck, map_location="cpu", weights_only=False)
            C.seed_alles(seed)
            m = C.baue_modell_natural(cm["best_hp"], familie)
            m.load_state_dict(cm["state_dict"])
            sc = lade_npz(zs)
            p_det, y = {}, {}
            for dom in DOMS:
                pc = os.path.join(C.ORDNER["caches"], f"p_{C.TASK}_{familie}_det_seed{seed}_{dom}.npz")
                if not os.path.exists(pc):
                    abbruch(f"A1: cache {pc} is missing (run --caches first).")
                cache = lade_npz(pc)
                p_c = np.asarray(cache["p"], float).ravel()
                y_c = np.asarray(cache["y"], float).ravel().astype(int)
                p_s = np.asarray(sc[f"p_{dom}"], float).ravel()
                y_s = np.asarray(sc[f"y_{dom}"], float).ravel().astype(int)
                if len(p_s) != N_SOLL[dom] or not np.array_equal(p_c, p_s) or not np.array_equal(y_c, y_s):
                    abbruch(f"A4: {familie}/seed {seed}/{dom}: cache and scores file differ in p or y.")
                p_det[dom], y[dom] = p_s, y_s
            # A2: the loaded checkpoint reproduces the stored predictions
            for dom in DOMS:
                p_hat = 1.0 / (1.0 + np.exp(-A._det_vorhersage(m, X[dom]).double().numpy()))
                abw = float(np.max(np.abs(p_hat.astype(np.float32).astype(float) - p_det[dom])))
                if abw > TOL_DET:
                    abbruch(f"A2: {familie}/seed {seed}/{dom}: the deterministic pass deviates {abw:.2e} from "
                            "the stored p -- not the checkpoint or the inputs of stage S5 (torch version / device?).")
            # A7: module inventory under the masks (once per family)
            if familie not in module:
                A._dropout_an(m)
                inv = [{"module": n_, "type": type(m_).__name__,
                        "p": float(getattr(m_, "p", getattr(m_, "dropout", np.nan)))}
                       for n_, m_ in m.named_modules() if m_.training and not list(m_.children())]
                rnn = getattr(m, "rnn", None)
                n_bn = sum(isinstance(m_, nn.modules.batchnorm._BatchNorm) for m_ in m.modules())
                m.eval()
                if n_bn:
                    abbruch(f"A7: {familie} has BatchNorm modules; the passes would change their statistics.")
                module[familie] = {"dropout_hp": float(cm["best_hp"]["dropout"]),
                                   "num_layers": int(cm["best_hp"]["num_layers"]),
                                   "modules_in_train_mode": inv,
                                   "rnn": None if rnn is None else {"type": type(rnn).__name__,
                                                                     "dropout": float(rnn.dropout),
                                                                     "num_layers": int(rnn.num_layers)}}
            # the passes and the measures
            S = {}
            for dom in DOMS:
                torch.manual_seed(C.MC_ZIEHUNGS_SEED)       # convention of the reported run: seed right before the passes
                M = A.mc_passes(m, X[dom])                   # (30, N) logits, float64
                if M.shape != (A.T_MC, N_SOLL[dom]):
                    abbruch(f"A1: {familie}/seed {seed}/{dom}: passes have shape {M.shape}.")
                S[dom] = A.mc_signale(M)                     # A3 inside
            # rows of mc_mi_seeds_natural.csv: levels 1 and 2 of U_total / U_alea / MI at 0.1288 and 0.5
            for sig, key in (("mi_gesamt", "u_total"), ("mi_aleatorisch", "u_alea"), ("mi_epistemisch", "mi")):
                zeilen_mi.append([arch, seed, "mittel", "ebene1_ood", sig, "mimic_vs_eicu", "",
                                  round(ood_auroc(S["in_domain"][key], S["transfer"][key]), 5)])
                if sig == "mi_epistemisch":
                    e1_epi.append(zeilen_mi[-1][-1])
                for dom in DOMS:
                    for t in MI_SCHWELLEN:
                        e = ((p_det[dom] > t).astype(int) != y[dom]).astype(int)
                        zeilen_mi.append([arch, seed, "mittel", "ebene2_fehler", sig, dom, str(t),
                                          r5(fehler_auroc(S[dom][key], e))])
            # rows of mc_seeds_natural.csv: levels 1 and 2 of the logit variance at the six thresholds
            zeilen_mc.append([arch, seed, "ebene1_ood", "mc_epistemisch", "mimic_vs_eicu", "",
                              round(ood_auroc(S["in_domain"]["logit_var"], S["transfer"]["logit_var"]), 5)])
            for dom in DOMS:
                for s_name, t in C.SCHWELLEN:
                    e = ((p_det[dom] > t).astype(int) != y[dom]).astype(int)
                    zeilen_mc.append([arch, seed, "ebene2_fehler", "mc_epistemisch", dom, s_name,
                                      r5(fehler_auroc(S[dom]["logit_var"], e))])
            # patient-level outputs: mc_mi_*.npz and the logit variance in the scores file
            bl = {}
            for dom in DOMS:
                for key, nm in (("u_total", "ges"), ("u_alea", "alea"), ("mi", "epi")):
                    bl[f"mi_{nm}_{dom}"] = S[dom][key].astype(np.float32)
                bl[f"v_epi_{dom}"] = S[dom]["logit_var"].astype(np.float32)
            schreibe_npz(C._sicher(C.ORDNER["signale"], f"mc_mi_{familie}_seed{seed}.npz"), bl)
            neu = {k: v for k, v in sc.items() if not k.startswith("mc_")}
            for dom in DOMS:
                neu[f"mc_epistemisch_{dom}"] = S[dom]["logit_var"].astype(np.float32)
            schreibe_npz(zs, neu)
            chk = lade_npz(zs)
            behalten = [k for k in sc if not k.startswith("mc_")]
            if sorted(chk) != sorted(behalten + [f"mc_epistemisch_{d}" for d in DOMS]) \
                    or any(not np.array_equal(chk[k], sc[k]) for k in behalten):
                abbruch(f"A5: the rewritten {os.path.basename(zs)} does not preserve the non-mc keys.")
            tr = S["transfer"]
            e_tr = ((p_det["transfer"] > MI_SCHWELLEN[0]).astype(int) != y["transfer"]).astype(int)
            print(f"  [OK] {familie}/seed {seed}  (A1-A5, A7 passed)  level 1 MI "
                  f"{ood_auroc(S['in_domain']['mi'], tr['mi']):.4f} | level 2 (transfer, {MI_SCHWELLEN[0]}) "
                  f"U_total {f4(fehler_auroc(tr['u_total'], e_tr))} MI {f4(fehler_auroc(tr['mi'], e_tr))} "
                  f"logit variance {f4(fehler_auroc(tr['logit_var'], e_tr))}", flush=True)
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        mw = float(np.mean(e1_epi))
        h = T9 * float(np.std(e1_epi, ddof=1)) / np.sqrt(len(e1_epi))
        lo, hi = mw - h, mw + h
        u = ("kohortenerhoeht" if (mw >= SCHRANKE and lo > 0.5) else ("invertiert" if hi < 0.5 else "abwesend"))
        print(f"  ==> {familie}: level-1 MI {mw:.4f} [{lo:.4f}; {hi:.4f}] -> {URTEIL[u]} [{u}]")

    ziel_mi = C._sicher(C.ORDNER["report"], "mc_mi_seeds_natural.csv")
    ziel_mc = C._sicher(C.ORDNER["signale"], "mc_seeds_natural.csv")
    superset_csv(ziel_mi, zeilen_mi)
    superset_csv(ziel_mc, zeilen_mc)
    pd.DataFrame(zeilen_mi, columns=KOPF_MI).to_csv(ziel_mi, index=False)
    with open(ziel_mc, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(KOPF_MC)
        w.writerows(zeilen_mc)
    C.manifest_schreiben(C.ORDNER["signale"], MANIFEST,
                         {"skript_stand": SKRIPT_STAND,
                          "method": "Monte Carlo dropout of the trained deterministic models (DECISIONS.md D9)",
                          "families": list(FAMILIEN), "seeds": SEEDS, "T": int(A.T_MC), "masks": A.MC_MASKS,
                          "mask_seed": int(C.MC_ZIEHUNGS_SEED), "device": str(A.GERAET), "torch": torch.__version__,
                          "thresholds_mc_seeds": [s for s, _ in C.SCHWELLEN],
                          "thresholds_mc_mi_seeds": [str(t) for t in MI_SCHWELLEN],
                          "tolerances": {"A2_det_pass": TOL_DET, "A3_decomposition": 1e-9},
                          "dropout_modules": module},
                         skript="mc_dropout.py")
    print(f"\n  written: {ziel_mi}  ({len(zeilen_mi)} rows)")
    print(f"           {ziel_mc}  ({len(zeilen_mc)} rows)")
    print(f"           {len(FAMILIEN) * len(SEEDS)} mc_mi_*.npz written, {len(FAMILIEN) * len(SEEDS)} scores_*.npz "
          "rewritten (patient-level, stay inside RUNS; data use agreement)")
    print("  Verdict by the declared grammar (bound 0.55 against 0.5); interval = 95 % over the 10 paired")
    print("  training seeds (t9 = 2.262), not a confidence interval in the sampling sense.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
