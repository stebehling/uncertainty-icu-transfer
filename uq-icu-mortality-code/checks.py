"""
checks.py -- smoke test on synthetic data (no PhysioNet data) and the comparison of a run with results/

Paper:    none (installation check and acceptance test of a new run)
Origin:   checks/smoke_test.py and checks/compare_with_results.py of the full package, merged
Run:      python checks.py smoke [--cpu]                    (under a minute; prints PASS / FAIL per check)
          python checks.py compare [--run DIR] [--tolerance 0] [--allow GLOB]
                                                            (compares <run>/rerun_natural with results/)
"""
from __future__ import annotations

import os
import sys

if "--cpu" in sys.argv:                        # must happen before torch is imported
    # "-1" (no valid device) rather than "": with an empty string torch still reports CUDA as
    # available and then fails inside cuDNN's initialisation when the first RNN is built
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import fnmatch
import glob
import io
import numpy as np
import numpy as np  # noqa: E402
import pandas as pd
import re
import torch  # noqa: E402
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import common as C  # noqa: E402
from models import architectures as A  # noqa: E402
import uncertainty as D  # noqa: E402
from uncertainty import fit_varianten, _l2norm  # noqa: E402
from evaluation import _logits, _temperatur, _ece, kalibrierung, _risk_coverage, COVERAGE_GITTER  # noqa: E402


# ==========================================================================================
# section from the full package: checks/smoke_test.py
# ------------------------------------------------------------------------------------------
# smoke_test.py -- exercise the core functions of the package on synthetic data (no PhysioNet data needed)
#
# Paper:    none (installation check); the real cohorts are needed for every reported number
# Origin:   new for the package
# Reads:    nothing
# Writes:   nothing (console only)
# Checks:   S1 one-hot encoding of the GCS block (13 raw variables -> 25 model inputs)
#           S2 every model family builds and runs a forward pass (one logit per patient)
#           S3 seeding: the same seed gives bit-identical initial weights, a different seed does not
#           S4 training loop: two epochs of the GRU and one of the Transformer family; finite validation
#              BCE; repeating the GRU training with the same seed reproduces its validation history bit
#              for bit
#           S5 Monte Carlo dropout of the trained model: the passes differ (dropout active, inter-layer
#              dropout of the recurrent encoder included), the decomposition U_total = U_alea + MI holds,
#              MI and the logit variance are non-negative, the same mask seed reproduces the passes, the
#              model is back in eval mode afterwards
#           S6 representation density: feature extraction, class-conditional Gaussian (per-class and
#              shared covariance), relative density score, per-class GMM, L2-normalised density; finite
#              scores and level-1 AUROCs in [0, 1]
#           S7 performance: temperature by LBFGS, decile ECE, IRLS calibration slope / intercept / CITL,
#              binary entropy
#           S8 selective prediction: risk-coverage curve on the 96-point grid, AURC, E-AURC >= 0, risk at 80 %
# Run:      python checks/smoke_test.py [--cpu]      (under a minute; --cpu hides the GPU)
# Exit:     0 all checks pass, 1 otherwise; on some Windows machines a GPU run ends with 0xC0000409 after
#           printing its verdict (CUDA library unload at process exit, see README "Platform notes")
#
# The synthetic cohort has no clinical meaning: 48 hourly steps of 13 standard-normal variables, the
# last one treated as the GCS, and an outcome drawn from a logistic function of two of them. The
# size gates of common.lade_natural / lade_eicu (the real cohort sizes) are deliberately not touched,
# so this test does not run the stage scripts; it calls the functions they are built from.
# ==========================================================================================




N_TR, N_VAL, N_TE, N_OOD = 384, 96, 128, 128
HP_RNN = {"input_size": 25, "hidden_size": 16, "num_layers": 2, "dropout": 0.2,
          "batch_size": 32, "learning_rate": 1e-3, "num_epochs": 2}
HP_T = {"input_size": 25, "d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32,
        "dropout": 0.1, "batch_size": 32, "learning_rate": 1e-3, "num_epochs": 1}

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def synthetic(n, rng, shift=0.0):
    """Standardised 13-variable tensor (n, 48, 13) and a logistic outcome."""
    X = rng.normal(size=(n, 48, 13)).astype(np.float32) + np.float32(shift)
    eta = 2.0 * X[:, :, 0].mean(axis=1) + 0.5 * X[:, -1, 1] - 1.2
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(np.float32)
    return X, y


def main_smoke(argv=None):
    ap = argparse.ArgumentParser(description="Smoke test of the package on synthetic data.")
    ap.add_argument("--cpu", action="store_true", help="hide the GPU (set before torch is imported)")
    ap.parse_args(argv)
    print(f"device {A.GERAET} | torch {torch.__version__} | numpy {np.__version__}")
    rng = np.random.default_rng(0)
    mw, std = np.zeros(13), np.ones(13)
    mw[A.GCS_INDEX], std[A.GCS_INDEX] = 11.0, 3.0          # de-standardised GCS ~ N(11, 3) -> clipped to 3..15
    Xtr13, ytr = synthetic(N_TR, rng)
    Xva13, yva = synthetic(N_VAL, rng)
    Xte13, yte = synthetic(N_TE, rng)
    Xood13, _ = synthetic(N_OOD, rng, shift=1.5)

    # S1 one-hot encoding ------------------------------------------------------------
    print("S1 one-hot encoding")
    X25 = A.one_hot_gcs(Xtr13, mw, std)
    check("13 variables -> 25 inputs", X25.shape == (N_TR, 48, 25), str(X25.shape))
    check("one-hot block sums to 1 in every hour", np.allclose(X25[:, :, 12:25].sum(axis=2), 1.0))
    check("continuous block unchanged", np.array_equal(X25[:, :, :12], np.delete(Xtr13, A.GCS_INDEX, axis=2)))
    to = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)  # noqa: E731
    daten = {"Xtr": to(X25), "ytr": to(ytr), "Xval": to(A.one_hot_gcs(Xva13, mw, std)), "yval": to(yva),
             "Xte": to(A.one_hot_gcs(Xte13, mw, std)), "yte": to(yte), "input_size": 25, "mw": mw, "std": std}
    Xood = to(A.one_hot_gcs(Xood13, mw, std))

    # S2 model factory -----------------------------------------------------------------
    print("S2 model factory and forward pass")
    modelle = {}
    for fam in C.FAMILIEN:
        hp = HP_T if fam.startswith("T1") else HP_RNN
        C.seed_alles(42)
        try:
            m = C.baue_modell_natural(hp, fam)
            m.eval()
            with torch.no_grad():
                out = m(daten["Xte"][:8].to(A.GERAET))
            ok = torch.is_tensor(out) and out.shape == (8,)
            n_par = sum(p.numel() for p in m.parameters())
            check(f"{fam}: builds, forward pass gives the expected output", ok, f"{n_par:,} parameters")
            modelle[fam] = m
        except SystemExit as e:
            check(f"{fam}: builds", False, str(e))

    # S3 seeding -----------------------------------------------------------------------
    print("S3 seeding")

    def init_hash(seed):
        C.seed_alles(seed)
        m = C.baue_modell_natural(HP_RNN, "GRU_det")
        return C.sha(b"".join(p.detach().cpu().numpy().tobytes() for p in m.parameters()))
    h42a, h42b, h43 = init_hash(42), init_hash(42), init_hash(43)
    check("same seed -> bit-identical initial weights", h42a == h42b)
    check("different seed -> different initial weights", h42a != h43)

    # S4 training ----------------------------------------------------------------------
    print("S4 training loop (two epochs each; the constrained arm and the MC-dropout twin of earlier package versions are not part of this version)")
    verlaeufe = {}
    for fam in ("GRU_det", "T1_det"):
        hp = HP_T if fam.startswith("T1") else HP_RNN
        C.seed_alles(42)
        m = C.baue_modell_natural(hp, fam)
        m, verlauf, s_ep = C.trainiere_mit_val(m, daten, hp, fam)
        ok = len(verlauf) == hp["num_epochs"] and all(np.isfinite(v) and v > 0 for v in verlauf)
        check(f"{fam}: {hp['num_epochs']} epoch(s), finite validation BCE", ok,
              f"val BCE {verlauf} ({s_ep:.2f} s/epoch)")
        verlaeufe[fam] = (m, verlauf)
    C.seed_alles(42)
    m2 = C.baue_modell_natural(HP_RNN, "GRU_det")
    _, verlauf2, _ = C.trainiere_mit_val(m2, daten, HP_RNN, "GRU_det")
    check("GRU_det: repeated training with the same seed reproduces the validation history",
          verlauf2 == verlaeufe["GRU_det"][1], f"{verlauf2} vs {verlaeufe['GRU_det'][1]}")

    # S5 Monte Carlo dropout of the trained model -------------------------------------
    print("S5 Monte Carlo dropout (passes of the trained GRU with its dropout switched on)")
    m_det = verlaeufe["GRU_det"][0]
    torch.manual_seed(C.MC_ZIEHUNGS_SEED)
    M = A.mc_passes(m_det, daten["Xte"], T=5, batch_size=64)
    check("5 passes x N logits, finite", M.shape == (5, N_TE) and np.isfinite(M).all(), str(M.shape))
    anteil = float((M.max(0) - M.min(0) > 1e-6).mean())
    check("the passes differ (dropout active, recurrent encoder sampled); model back in eval mode",
          anteil > 0.5 and not m_det.training and not m_det.rnn.training,
          f"share of patients with differing passes {anteil:.2f}")
    sig = A.mc_signale(M)
    check("U_total = U_alea + MI, MI >= 0, logit variance >= 0",
          np.allclose(sig["u_total"], sig["u_alea"] + sig["mi"], atol=1e-9) and (sig["mi"] >= -1e-9).all()
          and (sig["logit_var"] >= 0).all(), f"max MI {sig['mi'].max():.3f}, max logit variance {sig['logit_var'].max():.3f}")
    torch.manual_seed(C.MC_ZIEHUNGS_SEED)
    M2 = A.mc_passes(m_det, daten["Xte"], T=5, batch_size=64)
    check("the same mask seed reproduces the passes bit for bit", np.array_equal(M, M2))

    # S6 representation density --------------------------------------------------------
    print("S6 representation density")
    fkt, quelle = D.signal_funktionen()
    m_det = verlaeufe["GRU_det"][0]
    Ffit = fkt["extrahiere_features"](m_det, daten["Xtr"])
    Fte = fkt["extrahiere_features"](m_det, daten["Xte"])
    Food = fkt["extrahiere_features"](m_det, Xood)
    check("penultimate features (N, hidden)", Ffit.shape == (N_TR, HP_RNN["hidden_size"]), f"{Ffit.shape}, source {quelle}")
    y_fit = ytr.astype(int)
    aurocs = {}
    for tied in (False, True):
        g = fkt["fit_gda"](Ffit, y_fit, tied=tied)
        s_id, s_ood = fkt["neg_log_dichte"](Fte, g), fkt["neg_log_dichte"](Food, g)
        au = fkt["ood_auroc"](s_id, s_ood)
        aurocs[tied] = au
        check(f"class-conditional Gaussian (tied={tied}): finite scores, level-1 AUROC in [0, 1]",
              np.isfinite(s_id).all() and np.isfinite(s_ood).all() and 0.0 <= au <= 1.0, f"AUROC {au:.3f}")
    var = fit_varianten(Ffit, y_fit, fkt)
    for name, fn in var.items():
        s_id, s_ood = fn(Fte), fn(Food)
        au = fkt["ood_auroc"](s_id, s_ood)
        check(f"variant {name}: finite scores, AUROC in [0, 1]", np.isfinite(s_id).all() and np.isfinite(s_ood).all()
              and 0.0 <= au <= 1.0, f"AUROC {au:.3f}")
    check("L2 normalisation gives unit rows", np.allclose(np.linalg.norm(_l2norm(Ffit), axis=1), 1.0))

    # S7 performance and calibration ---------------------------------------------------
    print("S7 performance and calibration")
    lv = _logits(m_det, daten["Xval"])
    T = _temperatur(lv, daten["yval"])
    check("temperature by LBFGS is finite and > 0", np.isfinite(T) and T > 0, f"T = {T:.3g}")
    p = torch.sigmoid(_logits(m_det, daten["Xte"])).numpy().astype(np.float64)
    kanten = np.quantile(p, np.linspace(0, 1, 11))
    e = _ece(p, yte.astype(np.float64), kanten)
    check("decile ECE in [0, 1]", 0.0 <= e <= 1.0, f"ECE {e:.3f}")
    try:
        slope, intercept, citl = kalibrierung(p, yte.astype(np.float64))
        check("IRLS calibration slope / intercept / CITL finite", all(np.isfinite([slope, intercept, citl])),
              f"slope {slope:.3f}, intercept {intercept:.3f}, CITL {citl:.3f}")
    except SystemExit as ex:
        check("IRLS calibration", False, str(ex))
    H = fkt["binaere_entropie"](p)
    check("binary entropy in [0, ln 2]", np.isfinite(H).all() and (H >= 0).all() and (H <= np.log(2) + 1e-9).all())

    # S8 selective prediction ----------------------------------------------------------
    print("S8 selective prediction")
    fehler = ((p > C.SCHWELLEN[0][1]).astype(int) != yte.astype(int))
    aurc, eaurc, r80, kurve = _risk_coverage(H, fehler)
    check("risk-coverage: AURC in [0, 1], E-AURC >= 0, risk at 80 % in [0, 1], 96-point curve",
          0.0 <= aurc <= 1.0 and eaurc >= -1e-12 and 0.0 <= r80 <= 1.0 and kurve.shape == (len(COVERAGE_GITTER),)
          and kurve.shape == (96,), f"AURC {aurc:.3f}, E-AURC {eaurc:.3f}, risk@80 {r80:.3f}")
    check("the sorting is stable: a perfect score gives E-AURC 0", abs(_risk_coverage(fehler.astype(float), fehler)[1]) < 1e-12)

    n_ok, n = sum(RESULTS), len(RESULTS)
    print(f"\n{n_ok} of {n} checks passed -> {'PASS' if n_ok == n else 'FAIL'}")
    sys.stdout.flush()
    return 0 if n_ok == n else 1



# ==========================================================================================
# section from the full package: checks/compare_with_results.py
# ------------------------------------------------------------------------------------------
# compare_with_results.py -- compare the aggregate outputs of a run with the shipped results
#
# Paper:    acceptance test of the package: a new run must reproduce every shipped table and figure
# Origin:   new for the package
# Reads:    <reference>: by default the package's results/ folder (same layout as rerun_natural/);
#           <run>/rerun_natural/{05_signale, 06_report, 07_abbildungen, 08_appendix}
# Writes:   <run>/rerun_natural/logs/compare_with_results.md  (one line per file)
# Checks:   CSV  identical text, or identical parsed values when only the formatting differs; floats
#                compared exactly (bit-identical) unless --tolerance is given
#           MD   identical after dropping lines that carry a date, a time or a script version
#           PNG  identical pixels (RGB); a size difference is reported as such
#           Rows of the sigmaReparam transformer (T2, T2R), which the package does not train, are dropped
#           on both sides before comparing (column familie/arch/family, or MD lines naming it).
#           Reports, patient-level files and figures outside the manuscript are not compared (EXCLUDE).
#           Exit code 1 if any compared file differs or is missing in the run, 0 otherwise; --allow GLOB marks
#           files whose difference is expected (reported, not counted).
# Run:      python checks/compare_with_results.py [--run <UQ_ROOT>] [--reference <folder>] [--tolerance 0] [--allow GLOB]
# ==========================================================================================

PACKAGE_RESULTS = config.PACKAGE_DIR / "results"
SUBFOLDERS = ("05_signale", "06_report", "07_abbildungen", "08_appendix")
# not results of the manuscript, or not comparable by construction
EXCLUDE = ("*.npz", "*.pdf", "*_rohdaten*", "MANIFEST.md", "manifest_*.json", "gamma_log.csv", "spektralnormen.csv",
           "figure_s_signal_distribution_*", "figure_5_mutual_information.*", "figure_deferral_patients.*",
           "figure_s_rq1_distributions*", "folie_*", "figure_s_calibration_flexible*", "*.log",
           "table_s_rq1_distribution_tests*",   # distribution tests: not part of the manuscript, script not in the package
           "table_s8_strict_cohort.md")         # eight-family range table of the old script; the package writes Supplementary Table 2 (seven families)
VOLATILE = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{2}:\d{2}|SKRIPT_STAND|Skriptstand|generated|Generated|Stand |version|Version|origin)", re.I)
T2_ROW = re.compile(r"(?:T2R?_det\b|\bT2R?\b|sigma ?Reparam|σReparam)", re.I)
FAMILY_COLUMNS = ("familie", "arch", "family", "Familie", "Model family", "model_family")


def excluded(rel):
    name = os.path.basename(rel)
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE)


def drop_t2_rows(df):
    """Drop every row that names the sigmaReparam transformer in any cell (its family column is not
    always the first one: some appendix tables stack blocks with different layouts)."""
    cols = [c for c in FAMILY_COLUMNS if c in df.columns] or list(df.columns)
    hit = np.zeros(len(df), dtype=bool)
    for c in cols:
        hit |= df[c].astype(str).str.contains(T2_ROW, regex=True).fillna(False).to_numpy()
    return df[~hit].reset_index(drop=True)


def norm_md(text):
    return "\n".join(l.rstrip() for l in text.splitlines() if not VOLATILE.search(l) and not T2_ROW.search(l)).strip()


def compare_csv(a, b, tol):
    ta, tb = open(a, encoding="utf-8").read(), open(b, encoding="utf-8").read()
    if ta == tb:
        return "identical", ""
    try:
        da, db = pd.read_csv(io.StringIO(ta)), pd.read_csv(io.StringIO(tb))
    except Exception as e:  # ragged tables and the like
        return "differs", f"text differs and could not be parsed: {e}"
    da, db = drop_t2_rows(da), drop_t2_rows(db)
    if list(da.columns) != list(db.columns):
        return "differs", f"columns {list(da.columns)} vs {list(db.columns)}"
    if da.shape != db.shape:
        return "differs", f"shape {da.shape} vs {db.shape} (after dropping T2/T2R rows)"
    worst = 0.0
    for c in da.columns:
        x, y = da[c], db[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xa, ya = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            both_nan = np.isnan(xa) & np.isnan(ya)
            diff = np.abs(xa - ya)
            diff[both_nan] = 0.0
            if np.isnan(diff).any():
                return "differs", f"column {c}: NaN pattern differs"
            worst = max(worst, float(diff.max()) if len(diff) else 0.0)
        elif not x.astype(str).equals(y.astype(str)):
            k = int(np.argmax(~(x.astype(str).to_numpy() == y.astype(str).to_numpy())))
            return "differs", f"column {c}, row {k}: '{x.iloc[k]}' vs '{y.iloc[k]}'"
    if worst <= tol:
        return ("identical" if worst == 0 else "within tolerance"), (f"max |Δ| {worst:.3g}" if worst else "values identical, formatting differs")
    return "differs", f"max |Δ| {worst:.3g}"


def compare_md(a, b):
    na, nb = norm_md(open(a, encoding="utf-8").read()), norm_md(open(b, encoding="utf-8").read())
    if na == nb:
        return "identical", ""
    la, lb = na.splitlines(), nb.splitlines()
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return "differs", f"first difference at normalised line {i + 1}: '{x[:60]}' vs '{y[:60]}'"
    return "differs", f"{len(la)} vs {len(lb)} normalised lines"


def compare_png(a, b):
    from PIL import Image
    ia, ib = Image.open(a), Image.open(b)
    if ia.size != ib.size:
        return "differs", f"size {ia.size} vs {ib.size}"
    xa, xb = np.asarray(ia.convert("RGB"), dtype=np.int16), np.asarray(ib.convert("RGB"), dtype=np.int16)
    d = int(np.abs(xa - xb).max())
    return ("identical", "") if d == 0 else ("differs", f"max pixel difference {d}")


def main_compare(argv=None):
    ap = argparse.ArgumentParser(description="Compare a run with the shipped results (package acceptance).")
    ap.add_argument("--run", default=str(config.ROOT), help="working directory of the run (UQ_ROOT)")
    ap.add_argument("--reference", default=str(PACKAGE_RESULTS), help="reference tree with the same subfolders")
    ap.add_argument("--tolerance", type=float, default=0.0, help="allowed |Δ| per numeric cell (default 0 = bit-identical)")
    ap.add_argument("--only", default=None, help="glob on the relative path, e.g. '06_report/*' or '*/table_s3*'")
    ap.add_argument("--allow", action="append", default=[], metavar="GLOB",
                    help="relative path (glob) whose difference is expected; reported as 'differs (allowed)', not counted")
    a = ap.parse_args(argv)
    run = Path(a.run) / "rerun_natural"
    ref = Path(a.reference)
    if not ref.exists():
        print(f"ABORT: reference folder missing: {ref}")
        return 2
    rows, n_bad = [], 0
    for sub in SUBFOLDERS:
        for p in sorted(glob.glob(str(ref / sub / "**" / "*"), recursive=True)):
            if not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, ref).replace(os.sep, "/")
            if excluded(rel) or (a.only and not fnmatch.fnmatch(rel, a.only)):
                continue
            q = run / rel
            if not q.exists():
                rows.append((rel, "missing in run", ""))
                n_bad += 1
                continue
            try:
                if p.endswith(".csv"):
                    status, detail = compare_csv(str(q), p, a.tolerance)
                elif p.endswith(".md"):
                    status, detail = compare_md(str(q), p)
                elif p.endswith(".png"):
                    status, detail = compare_png(str(q), p)
                else:
                    status, detail = ("identical", "") if open(q, "rb").read() == open(p, "rb").read() else ("differs", "bytes differ")
            except Exception as e:
                status, detail = "differs", f"error: {e}"
            if status == "differs" and any(fnmatch.fnmatch(rel, g) for g in a.allow):
                status = "differs (allowed)"
            elif status == "differs":
                n_bad += 1
            rows.append((rel, status, detail))
    lines = [f"# Comparison of the run {run} with {ref}", "",
             f"{datetime.now():%Y-%m-%d %H:%M}; {len(rows)} files; {n_bad} differ or are missing; tolerance {a.tolerance}; "
             "rows of the sigmaReparam transformer dropped on both sides; excluded patterns: " + ", ".join(EXCLUDE) + "."
             + (f" Allowed differences (not counted): {', '.join(a.allow)}." if a.allow else ""), "",
             "| file | status | detail |", "|---|---|---|"] + [f"| {r} | {s} | {d} |" for r, s, d in rows]
    out = run / "logs" / "compare_with_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r, s, d in rows:
        if s != "identical":
            print(f"  [{s.upper():15s}] {r}  {d}")
    print(f"{len(rows)} files compared, {n_bad} differ or missing -> {out}")
    return 1 if n_bad else 0



def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("smoke", "compare"):
        print("usage: python checks.py smoke [--cpu]   |   python checks.py compare [--run DIR] [--reference DIR] [--tolerance X] [--allow GLOB]")
        return 2
    if argv[0] == "smoke":
        return main_smoke([x for x in argv[1:] if x != "--cpu"] + (["--cpu"] if "--cpu" in argv else []))
    return main_compare(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
