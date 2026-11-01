"""
config.py -- paths, seeds and constants shared by every script of the package (short version:
             main-text results only, the three base families; Monte Carlo dropout by sampling the
             trained models themselves)

Paper:    all results (the single place where paths and run constants are defined)
Origin:   new for the package; replaces src/config.py, src/pfade_rohdaten.py and the
          "parent of src" working-directory logic of the original scripts
Reads:    environment variables UQ_ROOT, MIMIC_DIR, EICU_DIR (all optional)
Writes:   nothing
Checks:   none (ensure_run_folders() creates the run folders; describe() reports which paths exist)
Run:      python config.py      (prints the resolved paths and whether they exist)

Layout of the working directory ROOT (default: the parent of the package folder):
    dataset.npz                      MIMIC-IV tensor, labels, split pool  (data/mimic_dataset.py)
    eicu_dataset_harmonized.npz      eICU-CRD tensor                      (data/prepare_eicu.py)
    rerun_natural/                   every run output, in the stage folders listed below
The stage folder names are those of the original run, so that the shipped results in
results/ and a new run can be compared file by file (checks.py compare).
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("UQ_ROOT", PACKAGE_DIR.parent)).resolve()

# raw data (PhysioNet, credentialed access); only data/ and supporting/strict_cohort.py read them
MIMIC_DIR = Path(os.environ.get("MIMIC_DIR", ROOT / "physionet" / "mimiciv" / "3.1"))
EICU_DIR = Path(os.environ.get("EICU_DIR", ROOT / "physionet" / "eicu-crd" / "2.0"))

# prepared tensors
DATASET_MIMIC = ROOT / "dataset.npz"
DATASET_EICU = ROOT / "eicu_dataset_harmonized.npz"
EICU_MANIFEST = ROOT / "eicu_dataset_harmonized.manifest.json"

# run outputs (names of the original run kept)
RUNS = ROOT / "rerun_natural"
SPLIT = RUNS / "01_split"          # split_v2.csv, manifest_split_v2.json
SEARCH = RUNS / "02_suchen"        # Optuna studies and beste_hp_<family>.json
MODELS = RUNS / "03_modelle"       # modell_ihm_<family>_seed<n>.pt
CACHES = RUNS / "04_vorhersagen"   # p_ihm_*.npz, features_*.npz (patient-level, stays local)
SIGNALS = RUNS / "05_signale"      # scores_*.npz, mc_mi_*.npz (patient-level), ddu_seeds_natural.csv, mc_seeds_natural.csv
REPORT = RUNS / "06_report"        # leistung_natural.csv, ebene3_natural.csv, ... (aggregates)
FIGURES = RUNS / "07_abbildungen"
APPENDIX = RUNS / "08_appendix"    # supplementary tables and supporting analyses (aggregates)
LOGS = RUNS / "logs"

# study constants
SEEDS = list(range(42, 52))        # ten training seeds per family
MC_SEED = 42                       # seed of the Monte Carlo dropout masks (set per cohort, right before the passes)
MC_PASSES = 30
MC_MASKS = "full"                  # every dropout layer that was active in training is sampled: the nn.Dropout
                                   # modules and the inter-layer dropout of the recurrent encoder (see
                                   # models/architectures.py, mc_passes)
THRESHOLD = 0.1288                 # classification threshold = mortality rate of the training pool
THRESHOLD_SENSITIVITY = 0.5
COVERAGE = 0.8                     # risk at 80 % coverage
N_TRIALS = 30                      # Optuna trials per family

# model families reported in the manuscript (the sigmaReparam transformer is not part of the package).
# The Monte Carlo dropout measures come from these same models (dropout sampled at prediction time);
# no separately trained Monte Carlo family exists.
BASE_FAMILIES = ["GRU_det", "LSTM_det", "T1_det"]
CONSTRAINED_FAMILIES = []          # the constrained variants (SN, SN/res) are not part of the short version
REPORTED_FAMILIES = BASE_FAMILIES + CONSTRAINED_FAMILIES          # the three base families
ALL_FAMILIES = list(REPORTED_FAMILIES)                             # everything that is trained
LABELS = {"GRU_det": "GRU", "LSTM_det": "LSTM", "T1_det": "Transformer"}
ARCH_OF = {f: f.split("_")[0] for f in ALL_FAMILIES}               # GRU | LSTM | T1

# glossary of the German identifiers kept in code and output columns (see README)
GLOSSARY = {
    "familie": "model family", "arch": "architecture", "seed": "training seed", "domaene": "cohort (in_domain | transfer)",
    "ebene1_ood": "level 1, cohort separation", "ebene2_fehler": "level 2, error detection", "ebene3": "level 3, selective prediction",
    "schwelle": "classification threshold", "wert": "value", "signal": "uncertainty measure",
    "softmax_entropie": "predictive entropy", "dichte": "representation density (class-conditional Gaussian)",
    "mc_epistemisch": "logit variance across the Monte Carlo dropout passes of the trained model", "mi_epistemisch": "mutual information",
    "mi_gesamt": "total predictive uncertainty of the Monte Carlo / ensemble mean", "mi_aleatorisch": "aleatoric part",
    "fit_menge": "fitting set of the density (voll_natuerlich = whole training pool, mukhoti_balanciert = class-balanced subset)",
    "kovarianz": "covariance (klassenspezifisch = per class, tied = shared)", "steigung": "calibration slope",
    "achsenabschnitt": "calibration intercept", "ece": "expected calibration error", "temperatur": "temperature",
    "aurc": "area under the risk-coverage curve", "eaurc": "excess AURC", "risiko80": "risk at 80 % coverage",
    "fehlerrate": "misclassification rate", "praevalenz": "prevalence", "ce_roh": "cross-entropy, raw",
    "skill_roh": "skill relative to the prevalence baseline", "gruppe": "decile group", "anteil_fehler": "share of all misclassifications",
}


def ensure_run_folders():
    """Create the run folders (idempotent); called by every script that writes."""
    for p in (RUNS, SPLIT, SEARCH, MODELS, CACHES, SIGNALS, REPORT, FIGURES, APPENDIX, LOGS):
        p.mkdir(parents=True, exist_ok=True)


def describe():
    """One line per path, for --check-inputs and for the README's quick check."""
    rows = [("ROOT", ROOT), ("MIMIC_DIR", MIMIC_DIR), ("EICU_DIR", EICU_DIR), ("DATASET_MIMIC", DATASET_MIMIC),
            ("DATASET_EICU", DATASET_EICU), ("RUNS", RUNS)]
    return "\n".join(f"  {k:14s} {v}  {'(exists)' if Path(v).exists() else '(missing)'}" for k, v in rows)


if __name__ == "__main__":
    print(describe())
