"""
run_pipeline.py -- the whole study in one command: data, hyperparameters, seeds, caches, signals, evaluation, tables

Paper:    Tables 1-4 and the RQ1 / RQ2a / RQ2b values of the main text (threshold 0.1288)
Origin:   run_pipeline.py of the full package (input checks, stages S2-S7), extended by the data stage
          and the analysis stages so that one command runs everything in the correct order
Reads:    the raw PhysioNet tables (--data only); config.DATASET_MIMIC, config.DATASET_EICU, config.SPLIT
Writes:   everything below config.RUNS (01_split ... 08_appendix, logs)
Checks:   --check-inputs = seeding determinism, split SHA / sizes / prevalences, eICU variable order /
          shape / finiteness / prevalence, density hook on an untrained model
Run:      python run_pipeline.py --data          raw tables -> dataset.npz, eICU tensor, stay map, split, Tables 1-2
          python run_pipeline.py --all           check inputs, search (skipped when beste_hp_<family>.json
                                                 exists -- the shipped results/02_suchen files are used
                                                 automatically), seeds, caches, signals, Monte Carlo
                                                 passes, evaluation, ensembles, performance, level 3,
                                                 Tables 3-4
          python run_pipeline.py --data --all    both, in order
          single stages: --check-inputs, --search FAMILY, --seeds FAMILY, --caches, --signals, --mc,
          --evaluate, --ensembles, --performance, --levels, --tables (they combine and run in pipeline order)

Families (common.FAMILIEN): GRU_det LSTM_det T1_det. Seeds 42-51. The Monte Carlo dropout signals are
sampled from these trained models (mc_dropout.py; masks drawn with seed config.MC_SEED per cohort); the
stage runs after --signals and before --evaluate, because level 3 reads the logit variance it adds to the
scores files. Existing outputs are skipped (resume); checkpoints are never overwritten.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import common as C  # noqa: E402
from models import architectures as A  # noqa: E402
from models.training import suche, finale_seeds  # noqa: E402
import uncertainty as U  # noqa: E402
import evaluation as E  # noqa: E402
import mc_dropout  # noqa: E402
import ensembles  # noqa: E402
from data import mimic_dataset, prepare_eicu, cohort_tables  # noqa: E402


# ------------------------------------------------------------- input checks
def check_inputs():
    """Sanity checks before any training: seeding, data, eICU, density hook.
    Returns (daten, eicu) for the following stages."""
    print("=" * 78)
    print("INPUT CHECKS")
    print(f"Package: {C.PIPELINE_STAND}")
    print(f"Device: {A.GERAET} | torch {torch.__version__}")
    print(config.describe())
    print("=" * 78)

    # 1 seeds: same seed -> bit-identical initial weights and first permutation
    def init_hash(seed):
        C.seed_alles(seed)
        m = C.baue_modell_natural({"input_size": 25, "hidden_size": 32,
                                   "num_layers": 1, "dropout": 0.3}, "GRU_det")
        h = C.sha(b"".join(p.detach().cpu().numpy().tobytes()
                           for p in m.parameters()))
        perm = torch.randperm(1000)[:5].tolist()
        return h, perm
    h1, p1_ = init_hash(42)
    h2, p2_ = init_hash(42)
    h3, p3_ = init_hash(43)
    if h1 != h2 or p1_ != p2_:
        C.abbruch("seeds: the same seed does not give a bit-identical initialisation.")
    if h1 == h3 or p1_ == p3_:
        C.abbruch("seeds: different seeds give identical initialisations.")
    print("  [OK] seeds: same -> bit-identical, different -> different "
          f"(permutation probe {p1_} vs {p3_})")

    daten = C.lade_natural()                                 # 2 split and MIMIC tensor
    eicu = C.lade_eicu(daten, pflicht=True)                  # 3 eICU tensor

    fkt, quelle = U.signal_funktionen()                      # 4 hook / density pass-through
    C.seed_alles(42)
    m = C.baue_modell_natural({"input_size": 25, "hidden_size": 32,
                               "num_layers": 1, "dropout": 0.3}, "GRU_det")
    Fp = fkt["extrahiere_features"](m, daten["Xtr"][:256])
    yk = daten["ytr"][:256].numpy().astype(int)
    for tied in (False, True):
        g = fkt["fit_gda"](Fp, yk, tied=tied)
        s = fkt["neg_log_dichte"](Fp, g)
        if not np.isfinite(s).all():
            C.abbruch("density: non-finite density scores.")
    _ = fkt["ood_auroc"](s[:100], s[100:200])
    print(f"  [OK] density hook / GDA pass-through (source: {quelle})")
    return daten, eicu


# -------------------------------------------------------------------- data
def data_stage():
    """Raw PhysioNet tables -> dataset.npz, eICU tensor, stay/patient map, frozen split, Tables 1-2.
    Every product that already exists is kept."""
    print("=" * 78 + "\nDATA\n" + "=" * 78)
    if config.DATASET_MIMIC.exists():
        C.log(f"[DATA] {config.DATASET_MIMIC.name} present -- kept.")
    else:
        mimic_dataset.build_dataset()
    if config.DATASET_EICU.exists():
        C.log(f"[DATA] {config.DATASET_EICU.name} present -- kept.")
    else:
        rc = prepare_eicu.main([])
        if rc not in (None, 0):
            C.abbruch(f"prepare_eicu returned {rc}.")
    karte = config.ROOT / "stay_subject_map.csv"
    if karte.exists():
        C.log("[DATA] stay_subject_map.csv present -- kept.")
    else:
        mimic_dataset.exportiere_stay_subject_map()
    if (config.SPLIT / "split_v2.csv").exists():
        C.log("[DATA] split_v2.csv present -- kept (common.lade_natural verifies its SHA).")
    else:
        mimic_dataset.main_split()
    if (config.APPENDIX / "table_1_cohort_flow.md").exists():
        C.log("[DATA] Tables 1-2 present -- kept.")
    else:
        cohort_tables.main()
    C.log("[DATA] finished.")


# -------------------------------------------------------------- evaluation
def evaluate(daten, eicu):
    """S6 (performance / calibration) + S7 (level 3) from the stored checkpoints and
    scores. Watchdog: every deterministic family with checkpoints MUST have its scores
    files, and every scores file MUST carry the Monte Carlo logit variance of
    mc_dropout.py, otherwise S7 would be silently incomplete."""
    fehlend, ohne_mc = [], []
    for familie, (arch, variante, _) in C.FAMILIEN.items():
        if variante != "det":
            continue
        for seed in C.SEEDS:
            ck = C._finde(os.path.join(C.ORDNER["modelle"],
                                       f"modell_{C.TASK}_{familie}_seed{seed}.pt"))
            zs = C._finde(os.path.join(C.ORDNER["signale"],
                                       f"scores_{familie}_seed{seed}.npz"))
            if ck is not None and zs is None:
                fehlend.append(f"{familie}/seed{seed}")
            elif zs is not None:
                with np.load(zs) as S_:
                    if any(f"mc_epistemisch_{d}" not in S_.files
                           for d in ("in_domain", "transfer")):
                        ohne_mc.append(f"{familie}/seed{seed}")
    if fehlend:
        C.abbruch(f"--evaluate: {len(fehlend)} scores files missing "
                  f"(e.g. {fehlend[:3]}). Run --signals first.")
    if ohne_mc:
        C.abbruch(f"--evaluate: {len(ohne_mc)} scores files without the Monte Carlo "
                  f"logit variance (e.g. {ohne_mc[:3]}). Run --mc first.")
    C.log("[EVALUATE] scores files complete (Monte Carlo logit variance present).")
    E.s6_leistung(daten, eicu)
    E.s7_ebene3()
    C.log("[EVALUATE] S6 and S7 finished.")


def _step(name, fn, *args):
    """Run one analysis step; a non-zero return code aborts the pipeline."""
    C.log(f"[{name}] start")
    rc = fn(*args)
    if rc not in (None, 0):
        C.abbruch(f"{name} returned {rc}.")
    C.log(f"[{name}] finished")


def hyperparameters_present(fam):
    """beste_hp_<family>.json in the run folder; copied from results/02_suchen if only shipped there."""
    ziel = config.SEARCH / f"beste_hp_{fam}.json"
    if ziel.exists():
        return True
    quelle = config.PACKAGE_DIR / "results" / "02_suchen" / f"beste_hp_{fam}.json"
    if quelle.exists():
        config.SEARCH.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(quelle, ziel)
        C.log(f"[S2] {fam}: hyperparameters taken from the shipped results/02_suchen (search skipped).")
        return True
    return False


# ------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(
        description="The whole study in pipeline order. Flags combine; without a flag nothing is computed.")
    ap.add_argument("--data", action="store_true", help="raw tables -> tensors, split, Tables 1-2 (needs MIMIC_DIR, EICU_DIR)")
    ap.add_argument("--check-inputs", action="store_true", help="sanity checks (no training, nothing written)")
    ap.add_argument("--search", metavar="FAMILY", help="Optuna study of one family (resumable)")
    ap.add_argument("--seeds", metavar="FAMILY", help="train the ten final seeds of one family")
    ap.add_argument("--caches", action="store_true", help="probabilities and penultimate features of every deterministic seed")
    ap.add_argument("--signals", action="store_true", help="levels 1 and 2 of density and entropy -> ddu_seeds_natural.csv, scores_*.npz")
    ap.add_argument("--mc", action="store_true", help="30 Monte Carlo dropout passes of every trained model: logit variance, predictive "
                                                      "entropy, aleatoric part, mutual information (GPU, about 40 min; before --evaluate)")
    ap.add_argument("--evaluate", action="store_true", help="performance / calibration (S6) and level 3 (S7)")
    ap.add_argument("--ensembles", action="store_true", help="deep ensembles of the ten seeds: levels 1 and 2")
    ap.add_argument("--performance", action="store_true", help="AUPRC and lift, calibration slope, ensemble performance")
    ap.add_argument("--levels", action="store_true", help="level 3 of the deep ensembles and the Monte Carlo mutual information")
    ap.add_argument("--tables", action="store_true", help="Table 3 and Table 4 from the aggregate files")
    ap.add_argument("--all", action="store_true",
                    help="check inputs, search every family whose hyperparameters are missing, train all seeds, "
                         "caches, signals, mc, evaluate, ensembles, performance, levels, tables")
    a = ap.parse_args()
    flags = [a.data, a.check_inputs, a.search, a.seeds, a.caches, a.signals, a.evaluate, a.mc, a.ensembles,
             a.performance, a.levels, a.tables, a.all]
    if not any(flags):
        ap.print_help()
        return
    config.ensure_run_folders()
    if a.data:
        data_stage()

    if a.all:
        daten, eicu = check_inputs()
        for fam in C.FAMILIEN:
            if hyperparameters_present(fam):
                C.log(f"[S2] {fam}: beste_hp present -- search skipped.")
            else:
                suche(fam, daten)
        for fam in C.FAMILIEN:
            finale_seeds(fam, daten)
        U.caches(daten, eicu)
        U.signale(daten, eicu)
        _step("MC", mc_dropout.main, [])
        evaluate(daten, eicu)
        _step("ENSEMBLES", ensembles.main)
        for name, fn in (("AUPRC", E.main_auprc), ("CALIBRATION", E.main_calibration),
                         ("ENSEMBLE PERFORMANCE", E.main_ensemble_performance)):
            _step(name, fn, [])
        _step("LEVEL 3 ENSEMBLES / MC", E.main_level3, [])
        _step("TABLE 3", E.main_table3, [])
        _step("TABLE 4", E.main_table4, [])
        C.log("[ALL] every stage finished.")
        return

    daten, eicu = None, None
    if a.check_inputs:
        daten, eicu = check_inputs()
    if a.search:
        if a.search not in C.FAMILIEN:
            C.abbruch(f"unknown family {a.search}.")
        if daten is None:
            daten = C.lade_natural()
        suche(a.search, daten)
    if a.seeds:
        if a.seeds not in C.FAMILIEN:
            C.abbruch(f"unknown family {a.seeds}.")
        if daten is None:
            daten = C.lade_natural()
        finale_seeds(a.seeds, daten)
    if a.caches or a.signals or a.evaluate:
        if daten is None:
            daten = C.lade_natural()
        if eicu is None:
            eicu = C.lade_eicu(daten)
    if a.caches:
        U.caches(daten, eicu)
    if a.signals:
        U.signale(daten, eicu)
    if a.mc:
        _step("MC", mc_dropout.main, [])
    if a.evaluate:
        evaluate(daten, eicu)
    if a.ensembles:
        _step("ENSEMBLES", ensembles.main)
    if a.performance:
        for name, fn in (("AUPRC", E.main_auprc), ("CALIBRATION", E.main_calibration),
                         ("ENSEMBLE PERFORMANCE", E.main_ensemble_performance)):
            _step(name, fn, [])
    if a.levels:
        _step("LEVEL 3 ENSEMBLES / MC", E.main_level3, [])
    if a.tables:
        _step("TABLE 3", E.main_table3, [])
        _step("TABLE 4", E.main_table4, [])


if __name__ == "__main__":
    main()
