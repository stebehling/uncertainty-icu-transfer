"""
common.py -- shared helpers of the pipeline stages: run constants, family registry, data
             loading, model factory, training core, logging

Paper:    Methods (cohorts and frozen split, model families, training protocol)
Origin:   src/rerun_pipeline.py v5.7 / 2026-08-20 (constants, seed_alles, lade_natural,
          lade_eicu, baue_modell_natural, val_bce, trainiere_mit_val, hp_fuer, log,
          manifest_schreiben, _sicher)
Reads:    config.DATASET_MIMIC, config.DATASET_EICU, config.SPLIT/split_v2.csv and
          manifest_split_v2.json, config.SEARCH/beste_hp_<family>.json
Writes:   config.LOGS/<date>.log (log), JSON manifests (manifest_schreiben),
Checks:   split SHA-256 against the manifest, 23,472 / 5,878 training / validation sequences,
          prevalences against the manifest, one-hot block sum, input_size 25, eICU variable
          order == MIMIC, eICU shape (66934, 48, 13), finite values, eICU prevalence 0.105-0.115
Run:      library module (imported by run_pipeline.py and the stage modules)

Family registry: name -> (arch for the result tables, variant, build type)
  build type "L"     recurrent model of models/architectures.py (GRU, LSTM)
  build type "T"     TransformerNatural (T1)
  variant            always "det": the three deterministic families are the only models that are
                     trained; the Monte Carlo dropout signals are sampled from these models
                     (mc_dropout.py). The field is kept because the file names of the original run
                     carry it (p_ihm_<family>_det_seed<n>_<domain>.npz).
Short version: the constrained families (spectral normalisation, residual block) and the
separately trained MC-dropout twins of earlier drafts are not included; every family has its own
hyperparameter search.
"""
from __future__ import annotations

import csv  # noqa: F401  (re-exported for the stage modules' convenience)
import datetime
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from models import architectures as A  # noqa: E402

# ------------------------------------------------------------------ constants
PIPELINE_STAND = "short version 1.1 / 2026-10-26 (port of rerun_pipeline.py v5.7; three base families; Monte Carlo dropout of the trained models)"
TASK = "ihm"
SEEDS = list(config.SEEDS)
MC_ZIEHUNGS_SEED = config.MC_SEED   # seed of the MC-dropout masks, separate from the training seed
FAMILIEN = {                        # name -> (arch_for_cache, variant, build type)
    "GRU_det": ("GRU", "det", "L"),
    "LSTM_det": ("LSTM", "det", "L"),
    "T1_det": ("T1", "det", "T"),
}
if set(FAMILIEN) != set(config.ALL_FAMILIES):
    raise SystemExit(f"common.FAMILIEN {sorted(FAMILIEN)} != config.ALL_FAMILIES "
                     f"{sorted(config.ALL_FAMILIES)}.")
SCHWELLEN = [("0.1288", 0.1288), ("0.5", 0.5),      # primary threshold, sensitivity
             ("0.021", 0.021), ("0.060", 0.060),    # robustness grid
             ("0.090", 0.090), ("0.182", 0.182)]
if SCHWELLEN[0][1] != config.THRESHOLD or SCHWELLEN[1][1] != config.THRESHOLD_SENSITIVITY:
    raise SystemExit("common.SCHWELLEN does not match config.THRESHOLD / THRESHOLD_SENSITIVITY.")
N_TRIALS = config.N_TRIALS
BAL_SEED = 42                      # deterministic draw of the class-balanced fitting set
ORDNER = {"split": config.SPLIT, "suchen": config.SEARCH, "modelle": config.MODELS,
          "caches": config.CACHES, "signale": config.SIGNALS, "report": config.REPORT,
          "logs": config.LOGS}


def abbruch(t):
    raise SystemExit("ABORT (fail-fast): " + t)


def _sicher(unterordner, dateiname):
    """Output path inside config.RUNS (anything else aborts); creates the folder."""
    ziel = os.path.abspath(os.path.join(unterordner, dateiname))
    basis = os.path.abspath(config.RUNS)
    if os.path.commonpath([ziel, basis]) != basis:
        abbruch(f"{ziel} lies outside {config.RUNS}.")
    os.makedirs(os.path.dirname(ziel), exist_ok=True)
    return ziel


def _finde(pfad):
    """The path as str if it exists, else None (paths come from config, no search)."""
    p = str(pfad)
    return p if os.path.exists(p) else None


def sha(b):
    return hashlib.sha256(b).hexdigest()


def manifest_schreiben(unterordner, name, inhalt, skript="run_pipeline.py"):
    inhalt = dict(inhalt)
    inhalt["erzeugt_am"] = datetime.datetime.now().isoformat(timespec="seconds")
    inhalt["skript"] = skript
    inhalt["pipeline_stand"] = PIPELINE_STAND
    with io.open(_sicher(unterordner, name), "w", encoding="utf-8") as f:
        json.dump(inhalt, f, ensure_ascii=False, indent=1)


def log(text):
    print(text)
    with io.open(_sicher(ORDNER["logs"],
                         datetime.date.today().isoformat() + ".log"),
                 "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ----------------------------------------------- seeds and determinism
def seed_alles(seed):
    """First action of every trial / run, BEFORE any instantiation."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.initial_seed() != seed:
        abbruch(f"torch.initial_seed()={torch.initial_seed()} != {seed}.")


# ---------------------------------------------------------------------- data
def lade_natural():
    """dataset.npz + frozen split -> train / val / test tensors (25 columns)."""
    p_npz = _finde(config.DATASET_MIMIC) or abbruch(f"{config.DATASET_MIMIC} missing.")
    p_spl = _finde(os.path.join(ORDNER["split"], "split_v2.csv")) \
        or abbruch(f"{ORDNER['split']}/split_v2.csv missing -- run data/split.py first.")
    p_man = _finde(os.path.join(ORDNER["split"], "manifest_split_v2.json")) \
        or abbruch("manifest_split_v2.json missing.")
    d = np.load(p_npz, allow_pickle=True)
    spl = pd.read_csv(p_spl)
    man = json.load(io.open(p_man, encoding="utf-8"))
    kanon = "".join(f"{int(s)},{t}\n" for s, t in
                    zip(spl.stay_id, spl.teil)).encode()
    if sha(kanon) != man["zuordnung_sha256"]:
        abbruch("split SHA-256 differs from the manifest -- frozen split changed?")
    ids = np.asarray(d["stay_ids_train"], dtype=np.int64)
    X13 = np.asarray(d["X_train"], dtype=np.float32)
    y = np.asarray(d[f"y_{TASK}_train"], dtype=np.float32)
    mw, std = d["mittelwert"], d["std"]
    teil = spl.set_index("stay_id").teil.reindex(ids)
    if teil.isna().any():
        abbruch("split does not cover stay_ids_train completely.")
    ist_val = (teil == "val").to_numpy()
    X25 = A.one_hot_gcs(X13, mw, std)
    if X25.shape[2] != 25:
        abbruch(f"one_hot_gcs returns {X25.shape[2]} instead of 25 columns.")
    oh = X25[:, :, 12:25].sum(axis=2)
    if not np.allclose(oh, 1.0, atol=1e-5):
        abbruch("one-hot block sum != 1 (MIMIC).")
    Xte = A.one_hot_gcs(np.asarray(d["X_test"], dtype=np.float32), mw, std)
    yte = np.asarray(d[f"y_{TASK}_test"], dtype=np.float32)
    n_tr, n_va = int((~ist_val).sum()), int(ist_val.sum())
    if (n_tr, n_va) != (23472, 5878):
        abbruch(f"split sizes {n_tr}/{n_va} instead of 23,472/5,878.")
    pt, pv = float(y[~ist_val].mean()), float(y[ist_val].mean())
    if (round(pt, 4), round(pv, 4)) != (round(man["praevalenz_stays"]["train"], 4),
                                        round(man["praevalenz_stays"]["val"], 4)):
        abbruch(f"prevalences {pt:.4f}/{pv:.4f} != manifest.")
    to = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    print(f"  [OK] data: train {n_tr:,} / val {n_va:,} / test {len(Xte):,}"
          f" | prevalence {pt:.4f}/{pv:.4f} | input_size 25 | split SHA ok")
    return {"Xtr": to(X25[~ist_val]), "ytr": to(y[~ist_val]),
            "Xval": to(X25[ist_val]), "yval": to(y[ist_val]),
            "Xte": to(Xte), "yte": to(yte), "input_size": 25,
            "mw": mw, "std": std}


def lade_eicu(daten, pflicht=True):
    """config.DATASET_EICU -> X_eicu (13 raw variables) -> MIMIC training statistics ->
    one-hot -> 25 columns. The measurement-mask channels of the file are never loaded."""
    p = _finde(config.DATASET_EICU)
    if p is None:
        if pflicht:
            abbruch(f"{config.DATASET_EICU} missing.")
        print("  [SKIP] eICU: tensor not present -- transfer part skipped.")
        return None
    e = np.load(p, allow_pickle=True)
    var_m = [str(v) for v in np.load(_finde(config.DATASET_MIMIC),
                                     allow_pickle=True)["variablen"]]
    var_e = [str(v) for v in e["variablen"]]
    if var_m != var_e:
        abbruch("variable order eICU != MIMIC.")
    Xr = np.asarray(e["X_eicu"], dtype=np.float32)
    if Xr.shape != (66934, 48, 13):
        abbruch(f"eICU shape {Xr.shape} instead of (66934, 48, 13).")
    Xs = (Xr - daten["mw"]) / daten["std"]
    X25 = A.one_hot_gcs(Xs.astype(np.float32), daten["mw"], daten["std"])
    if not np.isfinite(X25).all():
        abbruch("eICU: non-finite values after standardisation.")
    oh = X25[:, :, 12:25].sum(axis=2)
    if not np.allclose(oh, 1.0, atol=1e-5):
        abbruch("one-hot block sum != 1 (eICU).")
    y = np.asarray(e[f"y_{TASK}_eicu"], dtype=np.float32).ravel()
    if not (0.105 <= y.mean() <= 0.115):
        abbruch(f"eICU prevalence {y.mean():.4f} outside 0.105-0.115.")
    print(f"  [OK] eICU: n {len(y):,} | prevalence {y.mean():.4f} | "
          "variable order ok | mask channels never loaded")
    return {"X": torch.tensor(X25, dtype=torch.float32), "y": y}


# ------------------------------------------------------------- model factory
def baue_modell_natural(hp, familie):
    """Builds the model of a family from its hyperparameters (see the registry above)."""
    arch, _variante, bauart = FAMILIEN[familie]
    if bauart == "L":
        return A.baue_modell(dict(hp, arch=arch))
    m = A.TransformerNatural(hp["input_size"], hp["d_model"], hp["nhead"],
                             hp["num_layers"], hp["dim_feedforward"],
                             hp["dropout"])
    return m.to(A.GERAET)


# ------------------------------------------------------------------- training
def val_bce(modell, X, y, batch=1024):
    """Validation criterion: ONE deterministic pass (dropout off), BCE with logits."""
    modell.eval()
    s, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(A.GERAET)
            yb = y[i:i + batch].to(A.GERAET)
            s += float(F.binary_cross_entropy_with_logits(
                modell(xb), yb, reduction="sum"))
            n += len(yb)
    return s / n


def trainiere_mit_val(modell, daten, hp, familie, trial=None):
    """Training core: batches(), Adam(lr), BCE with logits, fixed hp['num_epochs'],
    NO early stopping; per epoch one validation BCE (measurement only) and, inside a
    hyperparameter search, the Optuna report / pruning decision.
    Returns (model, validation history, seconds per epoch)."""
    import optuna
    if familie not in FAMILIEN:
        abbruch(f"unknown family {familie}.")
    verlust = A.verlust_fabrik(TASK)
    opt = torch.optim.Adam(modell.parameters(), lr=hp["learning_rate"])
    Xtr, ytr = daten["Xtr"], daten["ytr"]
    verlauf, t0 = [], time.time()
    for ep in range(hp["num_epochs"]):
        modell.train()
        for xb, yb in A.batches(Xtr, ytr, hp["batch_size"]):
            xb, yb = xb.to(A.GERAET), yb.to(A.GERAET)
            opt.zero_grad()
            Lv = verlust(modell(xb), yb)
            Lv.backward()
            opt.step()
        vb = val_bce(modell, daten["Xval"], daten["yval"])
        verlauf.append(round(vb, 6))
        if trial is not None:
            trial.report(vb, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return modell, verlauf, (time.time() - t0) / hp["num_epochs"]


# ------------------------------------------------------------ hyperparameters
def hp_fuer(familie):
    """The family's own search result (config.SEARCH/beste_hp_<family>.json)."""
    p_hp = _finde(os.path.join(ORDNER["suchen"], f"beste_hp_{familie}.json")) \
        or abbruch(f"beste_hp_{familie}.json missing -- run --search {familie} first.")
    return json.load(io.open(p_hp, encoding="utf-8"))["hp"]
