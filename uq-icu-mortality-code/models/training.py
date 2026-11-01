"""
training.py -- hyperparameter search and the ten training seeds of every family

Paper:    Methods (hyperparameter optimisation; ten independently seeded models per family)
Origin:   models/hyperparameter_search.py and models/train_seeds.py of the full package, merged; the
          spectral-norm bookkeeping of the constrained arms is not part of the short version
Reads:    data via common.lade_natural; config.SEARCH/beste_hp_<family>.json
Writes:   config.SEARCH/<family>.db, beste_hp_<family>.json, manifest_suche_<family>.json;
          config.MODELS/modell_ihm_<family>_seed<n>.pt (never overwritten)
Run:      python models/training.py --search GRU_det      (Optuna, 30 trials; GPU, hours)
          python models/training.py --seeds GRU_det       (ten seeds; GPU, minutes to an hour)
          or through run_pipeline.py --search / --seeds / --all
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402,F401
import common as C  # noqa: E402
from models import architectures as A  # noqa: E402


# ==========================================================================================
# section from the full package: models/hyperparameter_search.py
# ------------------------------------------------------------------------------------------
# hyperparameter_search.py -- Optuna search (TPE, 30 trials, median pruning) per model family
#
# Paper:    Methods (hyperparameter optimisation; search spaces of the recurrent and the
#           Transformer families)
# Origin:   src/rerun_pipeline.py v5.7 / 2026-08-20, hp_natural() and suche() (stage S2)
# Reads:    data via common.lade_natural (dataset.npz, frozen split)
# Writes:   config.SEARCH/<family>.db (Optuna SQLite study, resumable),
#           config.SEARCH/beste_hp_<family>.json, config.SEARCH/manifest_suche_<family>.json
# Checks:   an existing beste_hp_<family>.json must carry the same HPs (frozen search)
# Run:      python models/hyperparameter_search.py --family GRU_det
#           (or run_pipeline.py --search GRU_det); GPU, hours per family
#
# Search space (shared): num_layers 1-4, learning_rate 1e-5..1e-2 (log), num_epochs 10-40,
# batch_size {32, 64, 128}, dropout 0.2-0.8; recurrent: hidden_size 32-256; Transformer:
# d_model {64, 128, 256}, nhead {2, 4, 8}, dim_feedforward 64-512 step 64. Objective: validation
# BCE of one deterministic pass after the last epoch (minimised). Trial seed = 10000 + trial number.
# ==========================================================================================

def hp_natural(trial, familie):
    """Shared search space for all families; architecture-specific part by build type."""
    _, _, bauart = C.FAMILIEN[familie]
    hp = {
        "input_size": 25,
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2,
                                             log=True),
        "num_epochs": trial.suggest_int("num_epochs", 10, 40),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.2, 0.8),
    }
    if bauart == "L":
        hp["hidden_size"] = trial.suggest_int("hidden_size", 32, 256)
    else:
        hp["d_model"] = trial.suggest_categorical("d_model", [64, 128, 256])
        hp["nhead"] = trial.suggest_categorical("nhead", [2, 4, 8])
        hp["dim_feedforward"] = trial.suggest_int("dim_feedforward", 64, 512,
                                                  step=64)
    return hp


def suche(familie, daten):
    """One Optuna study per family (resumable through the SQLite storage)."""
    import optuna
    if familie not in C.FAMILIEN:
        C.abbruch(f"unknown family {familie}.")
    db = C._sicher(C.ORDNER["suchen"], f"{familie}.db")
    study = optuna.create_study(
        study_name=familie, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=A.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                           n_warmup_steps=6),
        storage=f"sqlite:///{db}", load_if_exists=True)
    fertig = len([t for t in study.trials
                  if t.state.is_finished()])
    rest = C.N_TRIALS - fertig
    C.log(f"[S2] {familie}: {fertig} trials present, {max(rest, 0)} open "
          f"(budget {C.N_TRIALS}).")

    def ziel(trial):
        C.seed_alles(10_000 + trial.number)
        hp = hp_natural(trial, familie)
        modell = C.baue_modell_natural(hp, familie)
        _, verlauf, _ = C.trainiere_mit_val(modell, daten, hp, familie, trial)
        return verlauf[-1]

    if rest > 0:
        study.optimize(ziel, n_trials=rest)
    beste = dict(study.best_trial.params)
    beste["input_size"] = 25
    ziel_hp = C._sicher(C.ORDNER["suchen"], f"beste_hp_{familie}.json")
    inhalt = {"familie": familie, "val_bce": study.best_value,
              "n_trials": len(study.trials), "hp": beste}
    if os.path.exists(ziel_hp):
        alt = json.load(io.open(ziel_hp, encoding="utf-8"))
        if alt.get("hp") != beste:
            C.abbruch(f"{ziel_hp} exists with different HPs -- the frozen "
                      "search would be changed.")
        C.log(f"[S2] {familie}: beste_hp unchanged (resume).")
    else:
        with io.open(ziel_hp, "w", encoding="utf-8") as f:
            json.dump(inhalt, f, ensure_ascii=False, indent=1)
        C.manifest_schreiben(C.ORDNER["suchen"], f"manifest_suche_{familie}.json",
                             {**inhalt, "sampler": f"TPE(seed={A.SEED})",
                              "pruner": "MedianPruner(10, 6)",
                              "kriterium": "validation BCE, one deterministic pass"},
                             skript="models/hyperparameter_search.py")
    C.log(f"[S2] {familie}: best validation BCE {study.best_value:.5f} | HPs -> "
          f"{ziel_hp}")


def main_search(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, metavar="FAMILY",
                    choices=list(C.FAMILIEN),
                    help="family to search")
    a = ap.parse_args(argv)
    config.ensure_run_folders()
    daten = C.lade_natural()
    suche(a.family, daten)



# ==========================================================================================
# section from the full package: models/train_seeds.py
# ------------------------------------------------------------------------------------------
# train_seeds.py -- final training of the ten seeds (42-51) of one model family
#
# Paper:    Methods (ten independently seeded models per family; deterministic mode)
# Origin:   src/rerun_pipeline.py v5.7 / 2026-08-20, finale_seeds() and norm_zeilen() (stage S3)
# Reads:    data via common.lade_natural; config.SEARCH/beste_hp_<family>.json (common.hp_fuer)
# Writes:   config.MODELS/modell_ihm_<family>_seed<n>.pt (state_dict, best_hp, validation
#           history, seconds per epoch, MC draw seed) -- never overwritten;
#           config.SIGNALS/spektralnormen.csv for the constrained arms (raw and effective
#           spectral norm of every normalised encoder matrix)
# Checks:   an existing checkpoint is skipped, never overwritten
# Run:      python models/train_seeds.py --family GRU_det   (or run_pipeline.py --seeds GRU_det);
#           GPU, hours per family
# ==========================================================================================

def finale_seeds(familie, daten):
    """Trains the ten seeds of a family with its (own or inherited) hyperparameters."""
    hp = C.hp_fuer(familie)
    arch, variante, bauart = C.FAMILIEN[familie]
    for seed in C.SEEDS:
        ziel = C._sicher(C.ORDNER["modelle"],
                         f"modell_{C.TASK}_{familie}_seed{seed}.pt")
        if os.path.exists(ziel):
            C.log(f"[S3] {familie}/seed {seed}: checkpoint present -- "
                  "skipped (never overwritten).")
            continue
        C.seed_alles(seed)
        modell = C.baue_modell_natural(hp, familie)
        modell, verlauf, s_je_ep = C.trainiere_mit_val(modell, daten, hp, familie)
        torch.save({"state_dict": modell.state_dict(),
                    "best_hp": dict(hp, arch=arch), "familie": familie,
                    "seed": seed, "val_verlauf": verlauf,
                    "gewaehlte_epoche": hp["num_epochs"],
                    "sekunden_je_epoche": round(s_je_ep, 2),
                    "mc_ziehungs_seed": C.MC_ZIEHUNGS_SEED}, ziel)
        C.log(f"[S3] {familie}/seed {seed}: final validation BCE {verlauf[-1]:.5f} "
              f"({s_je_ep:.1f}s/epoch) -> {os.path.basename(ziel)}")
        del modell
        if torch.cuda.is_available():
            torch.cuda.empty_cache()




def main_seeds(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, metavar="FAMILY", choices=list(C.FAMILIEN),
                    help="family whose ten seeds are trained")
    a = ap.parse_args(argv)
    config.ensure_run_folders()
    daten = C.lade_natural()
    finale_seeds(a.family, daten)



def main(argv=None):
    ap = argparse.ArgumentParser(description="Hyperparameter search or final seeds of one family.")
    ap.add_argument("--search", metavar="FAMILY", choices=list(C.FAMILIEN))
    ap.add_argument("--seeds", metavar="FAMILY", choices=list(C.FAMILIEN))
    a = ap.parse_args(argv)
    if not (a.search or a.seeds):
        ap.print_help()
        return 0
    config.ensure_run_folders()
    daten = C.lade_natural()
    if a.search:
        suche(a.search, daten)
    if a.seeds:
        finale_seeds(a.seeds, daten)
    return 0


if __name__ == "__main__":
    sys.exit(main())
