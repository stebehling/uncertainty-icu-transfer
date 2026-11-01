"""
cohort_tables.py -- Table 1: exclusion cascades, split sizes, age and sex per cohort part

Paper:    Table 1 (cohort construction MIMIC-IV / eICU-CRD, baseline characteristics);
          Methods, "Study design and data sources"; Limitation (the single-ICU-stay
          rule is evaluated on the raw data in MIMIC-IV but after the other filters in
          eICU-CRD)
Origin:   src/kohorten_kaskade.py v1.4 / 2026-09-10 (cascades, anchors K/V/E/D),
          src/tabelle1_demographie.py (no SKRIPT_STAND; age/sex logic),
          src/tabelle1_demographie_split.py v1.2 / 2026-08-25 (per-part IDs, anchors A0-A4)
Reads:    MIMIC_DIR/icu/icustays.csv.gz     (subject_id, hadm_id, stay_id, los, intime)
          MIMIC_DIR/hosp/patients.csv.gz    (anchor_age, anchor_year, gender)
          EICU_DIR/patient.csv.gz or .csv   (age, gender, offsets, discharge status)
          config.DATASET_MIMIC              (stay_ids_train/test, y_ihm_train/test)
          config.DATASET_EICU               (X_eicu, y_ihm_eicu, stay_ids_eicu)
          config.SPLIT/split_v2.csv, config.SPLIT/manifest_split_v2.json
          config.ROOT/stay_subject_map.csv  (stay_id -> subject_id; data/split.py --export-map)
Writes:   config.REPORT/kohorten_kaskade.csv        (kohorte, schritt, n_stays, n_patienten, entfernt, kommentar)
          config.APPENDIX/table_1_cohort_flow.md    (Table 1 cascade + eICU baseline block)
          config.APPENDIX/table_1_demographics.csv and .md  (age and sex: training, internal
                                                    test, external test)
Checks:   K1  n_stays falls monotonically along the MIMIC-IV cascade
          K2  final MIMIC-IV count == len(stay_ids_train) + len(stay_ids_test) of the dataset
          A0  stay_subject_map.csv matches the SHA-256 in the split manifest; split_seed 20260817
          A1  training / validation / internal test / eICU have 23,472 / 5,878 / 7,298 / 66,934
              stays and match n_stays of the manifest
          A1b patient counts match n_patienten of the manifest
          A2  training and internal test share no stay; training and validation share no patient
          A3  training and internal test share no patient (stay_subject_map.csv)
          A4  training + internal test <= development cohort (36,648)
          V1  training + validation == training pool of the dataset
          K3  no patient in training pool and test set; K3b test share at patient level ~20 %
          K4  training-pool prevalence equals the classification threshold config.THRESHOLD
          E1  the mirrored eICU chain ends at or above the tensor size; E2 remainder < 5 %
          D1  mortality of the reconstructed eICU set == mean of y_ihm_eicu (same cohort)
          E3  (no threshold) eICU stays whose admission had another ICU stay in the raw data
          Checks print PASS/FAIL; outputs are written in any case; exit code 1 on any FAIL.
          The A-checks abort before any output (they are prerequisites of the split rows).
Run:      python data/cohort_tables.py        (minutes; reads raw PhysioNet tables locally,
                                               writes aggregate counts only)

Age conventions: MIMIC-IV age = anchor_age + (year(intime) - anchor_year); patients
over 89 are de-identified to anchor_age 91, so ages > 89 are expected. eICU ages are
strings; "> 89" is set to 90, empty fields are missing. Age is reported as median
[IQR] (ICU age is skewed) and as mean +- SD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SKRIPT_STAND = "v2.0 / 2026-10-08 (package: kohorten_kaskade v1.4 + tabelle1_demographie + tabelle1_demographie_split v1.2 merged)"

MIN_ALTER = 18
MIN_LOS_STUNDEN = 48
TEST_ANTEIL = 0.2
ZUFALLS_SEED = 42

# expected cohort sizes (structural anchors of the frozen cohort)
N_TRAIN = 23472
N_VAL = 5878
N_TEST_INTERN = 7298
N_TEST_EXTERN = 66934
N_ENTWICKLUNG = 36648
SPLIT_SEED_SOLL = 20260817
EICU_ALTER_90PLUS = 90       # eICU codes "> 89" as a string; replaced numerically

# raw tables
MIMIC_PATIENTS = config.MIMIC_DIR / "hosp" / "patients.csv.gz"
MIMIC_ICUSTAYS = config.MIMIC_DIR / "icu" / "icustays.csv.gz"
EICU_PATIENT = config.EICU_DIR / "patient.csv.gz"
# minimum sizes in bytes; the PhysioNet demo databases are three orders of magnitude smaller
MINDEST = {MIMIC_PATIENTS: 1_000_000,
           MIMIC_ICUSTAYS: 1_000_000,
           EICU_PATIENT: 5_000_000}

MAP_DATEI = config.ROOT / "stay_subject_map.csv"
SPLIT_CSV = config.SPLIT / "split_v2.csv"
SPLIT_MANIFEST = config.SPLIT / "manifest_split_v2.json"

pruefungen = []


def abbruch(t):
    raise SystemExit("ABORT (fail-fast): " + t)


def pruefe(k, text, ok, detail=""):
    pruefungen.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {k}  {text}"
          + (f"   ({detail})" if detail else ""))


def _eicu_patient_pfad():
    p = EICU_PATIENT
    if not p.exists() and p.with_suffix("").exists():   # patient.csv instead of patient.csv.gz
        p = p.with_suffix("")
    return p


def pruefe_rohdaten():
    """Abort if a raw table is missing or suspiciously small (demo database)."""
    for pfad, mindest in MINDEST.items():
        if pfad == EICU_PATIENT:
            pfad = _eicu_patient_pfad()
        if not pfad.exists():
            raise SystemExit(
                "ABORT (fail-fast): raw table missing:\n  " + str(pfad) +
                "\nCheck MIMIC_DIR / EICU_DIR (config.py, environment variables).")
        groesse = pfad.stat().st_size
        if groesse < mindest:
            raise SystemExit(
                f"ABORT (fail-fast): {pfad} has only {groesse:,} bytes, "
                f"expected at least {mindest:,}.\n"
                "This is probably the DEMO database (100 patients), not the full version.")
    return True


def sha256_datei(pfad):
    h = hashlib.sha256()
    with open(pfad, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------- demographics ---------------------------------
def _stats_alter(alter):
    alter = np.asarray(alter, dtype=float)
    fehlt = np.isnan(alter)
    gueltig = alter[~fehlt]
    if gueltig.size == 0:
        raise SystemExit("ABORT: not a single valid age -- wrong join?")
    q1, med, q3 = np.percentile(gueltig, [25, 50, 75])
    return dict(n=gueltig.size, fehlt=int(fehlt.sum()),
                median=med, q1=q1, q3=q3,
                mittel=float(gueltig.mean()), sd=float(gueltig.std(ddof=1)))


def _stats_geschlecht(serie_norm):
    """serie_norm contains 'F'/'M'/'U' (U = unknown/other)."""
    n = len(serie_norm)
    vc = serie_norm.value_counts()
    nf, nm, nu = int(vc.get("F", 0)), int(vc.get("M", 0)), int(vc.get("U", 0))
    return dict(n=n, nf=nf, nm=nm, nu=nu,
                pf=100*nf/n, pm=100*nm/n, pu=100*nu/n)


def _block(name, alter, serie_norm):
    """Print one demographics block and return its numbers as a flat dict."""
    a = _stats_alter(alter)
    g = _stats_geschlecht(serie_norm)
    print(f"--- {name}  (n = {a['n']+a['fehlt']}) ---")
    print(f"  age, median [IQR] : {a['median']:.0f} [{a['q1']:.0f}-{a['q3']:.0f}]")
    zus = f"   (age missing: {a['fehlt']})" if a["fehlt"] else ""
    print(f"  age, mean +- SD   : {a['mittel']:.1f} +- {a['sd']:.1f}{zus}")
    print(f"  female            : {g['nf']} ({g['pf']:.1f} %)")
    print(f"  male              : {g['nm']} ({g['pm']:.1f} %)")
    if g["nu"]:
        print(f"  unknown/other     : {g['nu']} ({g['pu']:.1f} %)")
    print()
    return {"kohorte": name, "n": a["n"] + a["fehlt"],
            "alter_median": a["median"], "alter_q1": a["q1"], "alter_q3": a["q3"],
            "alter_mittel": a["mittel"], "alter_sd": a["sd"], "alter_fehlend": a["fehlt"],
            "weiblich_n": g["nf"], "weiblich_pct": g["pf"],
            "maennlich_n": g["nm"], "maennlich_pct": g["pm"],
            "unbekannt_n": g["nu"], "unbekannt_pct": g["pu"]}


def mimic_demographie(stay_ids):
    """Age and sex of the given MIMIC-IV stays from patients.csv.gz and icustays.csv.gz."""
    ids = set(int(s) for s in stay_ids)
    pat = pd.read_csv(MIMIC_PATIENTS,
                      usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    icu = pd.read_csv(MIMIC_ICUSTAYS,
                      usecols=["subject_id", "stay_id", "intime"],
                      parse_dates=["intime"])
    icu = icu[icu["stay_id"].isin(ids)].copy()
    if len(icu) != len(ids):
        raise SystemExit(
            f"ABORT MIMIC: {len(icu)} of {len(ids)} stay IDs found in icustays. "
            "Wrong MIMIC directory or wrong cohort source?")
    df = icu.merge(pat, on="subject_id", how="left")
    df["alter"] = df["anchor_age"] + (df["intime"].dt.year - df["anchor_year"])
    gender = df["gender"].map({"M": "M", "F": "F"}).fillna("U")
    return df["alter"].to_numpy(dtype=float), gender


def eicu_demographie(unit_ids):
    """Age and sex of the given eICU stays from patient.csv(.gz)."""
    ids = set(int(s) for s in unit_ids)
    pat = pd.read_csv(_eicu_patient_pfad(), usecols=["patientunitstayid", "gender", "age"])
    pat = pat[pat["patientunitstayid"].isin(ids)].copy()
    if len(pat) != len(ids):
        raise SystemExit(
            f"ABORT eICU: {len(pat)} of {len(ids)} stay IDs found in patient.csv. "
            "Wrong eICU directory or wrong cohort source?")

    def parse_alter(a):
        if pd.isna(a):
            return np.nan
        s = str(a).strip()
        if s == "":
            return np.nan
        if s.startswith(">"):
            return float(EICU_ALTER_90PLUS)
        try:
            return float(s)
        except ValueError:
            return np.nan

    alter = pat["age"].map(parse_alter).to_numpy(dtype=float)
    gender = pat["gender"].map({"Male": "M", "Female": "F"}).fillna("U")
    return alter, gender


# ------------------------------ split IDs -----------------------------------
def lade_split_ids(s_tr, s_te):
    """Stay IDs of the three MIMIC-IV parts from the frozen split, with anchors A0-A4.

    Returns (id_train, id_val, id_test, pat_train, pat_val, pat_test, manifest).
    The origin script (tabelle1_demographie_split.py) aborts on any violation
    before printing anything; kept.
    """
    for p, was in ((SPLIT_MANIFEST, "manifest_split_v2.json"), (SPLIT_CSV, "split_v2.csv"),
                   (MAP_DATEI, "stay_subject_map.csv")):
        if not p.exists():
            abbruch(f"{was} not found at {p} (data/split.py writes the split; "
                    "data/split.py --export-map writes the map).")
    print(f"  inner split     : {SPLIT_CSV}")
    print(f"  manifest        : {SPLIT_MANIFEST}")
    print(f"  patient map     : {MAP_DATEI}\n")
    with open(SPLIT_MANIFEST, encoding="utf-8") as f:
        sp = json.load(f)

    # A0: identity of the files read
    ist_m = sha256_datei(MAP_DATEI)
    soll_m = sp.get("eingabe_pruefsummen", {}).get("stay_subject_map_datei_sha256", "")
    if soll_m and ist_m != soll_m:
        abbruch(f"A0: SHA-256 of {MAP_DATEI} is {ist_m[:16]}..., manifest says "
                f"{soll_m[:16]}... -- a different patient map than at split time.")
    if int(sp.get("split_seed", -1)) != SPLIT_SEED_SOLL:
        abbruch(f"A0: split_seed {sp.get('split_seed')} instead of {SPLIT_SEED_SOLL}.")
    print(f"  [OK] A0: stay_subject_map.csv identical to the manifest, split_seed {SPLIT_SEED_SOLL}.")

    zu = pd.read_csv(SPLIT_CSV)
    c_stay = next((c for c in ("stay_id", "stayid", "icustay_id") if c in zu.columns), None)
    c_teil = next((c for c in zu.columns
                   if zu[c].astype(str).str.lower().isin(["train", "val", "validation"]).any()), None)
    if not c_stay or not c_teil:
        abbruch(f"split_v2.csv: columns not recognised. Present: {list(zu.columns)}")
    teil = zu[c_teil].astype(str).str.lower()
    id_train = np.asarray(sorted({int(s) for s in zu.loc[teil == "train", c_stay]}))
    id_val = np.asarray(sorted({int(s) for s in zu.loc[teil.isin(["val", "validation"]), c_stay]}))
    print(f"  assignment read from columns '{c_stay}' and '{c_teil}'.")

    id_test = np.asarray(sorted({int(s) for s in s_te}))

    n_soll = sp.get("n_stays", {})
    if len(id_train) != n_soll.get("train", N_TRAIN):
        abbruch(f"A1: training has {len(id_train)} instead of {n_soll.get('train')} (manifest).")
    if len(id_val) != n_soll.get("val", N_VAL):
        abbruch(f"A1: validation has {len(id_val)} instead of {n_soll.get('val')} (manifest).")
    if len(id_train) != N_TRAIN:
        abbruch(f"A1: training has {len(id_train)} instead of {N_TRAIN} stays.")
    if len(id_test) != N_TEST_INTERN:
        abbruch(f"A1: internal test has {len(id_test)} instead of {N_TEST_INTERN} stays.")
    ueberlappung = np.intersect1d(id_train, id_test)
    if len(ueberlappung):
        abbruch(f"A2: {len(ueberlappung)} stays in training AND test.")
    if len(id_train) + len(id_test) > N_ENTWICKLUNG:
        abbruch(f"A4: training plus test gives {len(id_train) + len(id_test)} > {N_ENTWICKLUNG}.")

    mp = pd.read_csv(MAP_DATEI)
    sp_stay = next((c for c in ("stay_id", "stayid", "icustay_id") if c in mp.columns), None)
    sp_subj = next((c for c in ("subject_id", "subjectid", "patient_id") if c in mp.columns), None)
    if not sp_stay or not sp_subj:
        abbruch(f"stay_subject_map.csv: columns not recognised. Present: {list(mp.columns)}")
    karte = dict(zip(mp[sp_stay].astype(int), mp[sp_subj].astype(int)))
    pat_train = {karte[s] for s in id_train if s in karte}
    pat_val = {karte[s] for s in id_val if s in karte}
    pat_test = {karte[s] for s in id_test if s in karte}
    n_pat = sp.get("n_patienten", {})
    if n_pat and len(pat_train) != n_pat.get("train"):
        abbruch(f"A1b: training has {len(pat_train)} patients instead of {n_pat.get('train')} (manifest).")
    if n_pat and len(pat_val) != n_pat.get("val"):
        abbruch(f"A1b: validation has {len(pat_val)} patients instead of {n_pat.get('val')} (manifest).")
    if pat_train & pat_val:
        abbruch(f"A2: {len(pat_train & pat_val)} patients in training AND validation -- "
                "the split is not patient-level.")
    gemeinsam = pat_train & pat_test
    if gemeinsam:
        abbruch(f"A3: {len(gemeinsam)} patients in training AND internal test (patient-level leakage).")
    print(f"  [OK] A0-A4 passed. training {len(id_train)} stays / {len(pat_train)} patients; "
          f"validation {len(id_val)} / {len(pat_val)}; internal test {len(id_test)} / {len(pat_test)}.")
    praev = sp.get("praevalenz_stays", {})
    if praev:
        print(f"       manifest prevalence train {praev.get('train')}, val {praev.get('val')}.")
    print()
    return id_train, id_val, id_test, pat_train, pat_val, pat_test, sp


# ---------------------------------- main ------------------------------------
def main():
    print("=" * 78)
    print("COHORT CASCADE AND DEMOGRAPHICS  (Table 1)")
    print(f"  script version: {SKRIPT_STAND}")
    print("=" * 78)
    pruefe_rohdaten()
    config.ensure_run_folders()

    zeilen = []

    def schritt(kohorte, name, df, kommentar="", vorher=None):
        n = len(df)
        pat = df["subject_id"].nunique() if "subject_id" in df.columns else \
            (df["patientunitstayid"].nunique()
             if "patientunitstayid" in df.columns else np.nan)
        entfernt = "" if vorher is None else f"{vorher - n}"
        zeilen.append({"kohorte": kohorte, "schritt": name, "n_stays": n,
                       "n_patienten": pat, "entfernt": entfernt,
                       "kommentar": kommentar})
        print(f"    {name:<44}{n:>8}"
              + (f"   (-{vorher - n})" if vorher is not None else ""))
        return n

    # ------------------------------------------------------------ MIMIC ---
    print("\nMIMIC-IV v3.1")
    icu = pd.read_csv(MIMIC_ICUSTAYS,
                      usecols=["subject_id", "hadm_id", "stay_id", "los"])
    n0 = schritt("MIMIC-IV", "ICU stays in MIMIC-IV v3.1", icu,
                 "icustays.csv.gz")

    icu["n_icu_pro_aufnahme"] = icu.groupby("hadm_id")["stay_id"].transform(
        "size")
    df = icu[icu["n_icu_pro_aufnahme"] == 1]
    n1 = schritt("MIMIC-IV", "Admissions with more than one ICU stay removed",
                 df, "exactly one ICU stay per hospital admission (raw table)", n0)

    pat = pd.read_csv(MIMIC_PATIENTS,
                      usecols=["subject_id", "anchor_age"])
    df = df.merge(pat, on="subject_id", how="left")
    df = df[df["anchor_age"] >= MIN_ALTER]
    n2 = schritt("MIMIC-IV", f"Age below {MIN_ALTER} years removed", df,
                 "anchor_age; MIMIC-IV contains adults only, so the filter "
                 "removes no stays", n1)

    df = df[df["los"] >= MIN_LOS_STUNDEN / 24.0]
    n3 = schritt("MIMIC-IV", f"ICU stay shorter than {MIN_LOS_STUNDEN} h "
                 "removed", df, "los", n2)

    p_ds = config.DATASET_MIMIC
    if not p_ds.exists():
        abbruch(f"{p_ds} not found.")
    ds = np.load(p_ds, allow_pickle=True)
    s_tr = np.asarray(ds["stay_ids_train"]).ravel()
    s_te = np.asarray(ds["stay_ids_test"]).ravel()
    im_datensatz = set(s_tr.tolist()) | set(s_te.tolist())
    df = df[df["stay_id"].isin(im_datensatz)]
    n4 = schritt("MIMIC-IV", "Without complete 48-hour feature window or "
                 "label removed", df, "intersection with the dataset", n3)

    pruefe("K1", "n_stays falls monotonically in every step",
           n0 >= n1 >= n2 >= n3 >= n4, f"{n0} -> {n4}")
    pruefe("K2", "final count equals the dataset",
           n4 == len(s_tr) + len(s_te),
           f"{n4} against {len(s_tr)} + {len(s_te)} = {len(s_tr) + len(s_te)}")

    # The outer split is READ FROM THE DATASET (authoritative source); only the
    # patient disjointness is checked. Re-running the permutation is not needed.
    subj = df.drop_duplicates("stay_id").set_index("stay_id")["subject_id"]
    pat_tr = {int(subj.loc[s_]) for s_ in s_tr if s_ in subj.index}
    pat_te = {int(subj.loc[s_]) for s_ in s_te if s_ in subj.index}
    ueberlappung = pat_tr & pat_te
    print(f"    {'Patient-level split (from the dataset)':<44}"
          f"{len(pat_tr) + len(pat_te):>8} patients, {len(pat_te)} in test")

    # inner split (training / validation) from the frozen split files, anchors A0-A4
    print("\n  Inner split (data/split.py)")
    id_train, id_val, id_test, pat_train, pat_val, pat_test, sp = lade_split_ids(s_tr, s_te)

    zeilen.append({"kohorte": "MIMIC-IV", "schritt": "Training set",
                   "n_stays": len(s_tr), "n_patienten": len(pat_tr),
                   "entfernt": "",
                   "kommentar": "patient-level split, 20 % of patients held out"})
    zeilen.append({"kohorte": "MIMIC-IV", "schritt": "  of which: model training",
                   "n_stays": len(id_train), "n_patienten": len(pat_train), "entfernt": "",
                   "kommentar": "frozen split (data/split.py); 20 % validation "
                   "for hyperparameters and temperature"})
    zeilen.append({"kohorte": "MIMIC-IV", "schritt": "  of which: validation",
                   "n_stays": len(id_val), "n_patienten": len(pat_val), "entfernt": "",
                   "kommentar": "frozen split (data/split.py)"})
    if len(id_train) + len(id_val) != len(s_tr):
        abbruch(f"V1: {len(id_train)} + {len(id_val)} != {len(s_tr)}")
    pruefe("V1", "training + validation = training pool of the dataset",
           True, f"{len(id_train):,} + {len(id_val):,} = {len(s_tr)}".replace(",", " "))
    zeilen.append({"kohorte": "MIMIC-IV", "schritt": "Internal test set",
                   "n_stays": len(s_te), "n_patienten": len(pat_te),
                   "entfernt": "", "kommentar": ""})
    pruefe("K3", "no patient in training pool and test set at the same time",
           len(ueberlappung) == 0,
           f"{len(pat_tr)} + {len(pat_te)} patients, overlap "
           f"{len(ueberlappung)}")
    anteil = len(pat_te) / (len(pat_tr) + len(pat_te))
    pruefe("K3b", "test share at patient level about 20 %",
           abs(anteil - 0.20) < 0.005, f"{100 * anteil:.2f} %")

    y_tr = np.asarray(ds["y_ihm_train"], float)
    y_te = np.asarray(ds["y_ihm_test"], float)
    print(f"    prevalence  training pool {y_tr.mean():.4f} | "
          f"test {y_te.mean():.4f}")
    pruefe("K4", f"training-pool prevalence equals the classification threshold {config.THRESHOLD}",
           abs(y_tr.mean() - config.THRESHOLD) < 6e-4,
           f"{y_tr.mean():.4f} / test {y_te.mean():.4f}")

    # ------------------------------------------------------------- eICU ---
    print("\neICU-CRD v2.0")
    p_eicu = config.DATASET_EICU
    demo_eicu = None
    id_eicu = None
    if not p_eicu.exists():
        pruefe("E1", f"{p_eicu.name} found", False, "skipped")
    else:
        de = np.load(p_eicu, allow_pickle=True)
        schluessel = [k for k in de.files if k.startswith("X")
                      and getattr(de[k], "ndim", 0) == 3]
        if not schluessel:
            abbruch(f"eICU: no 3-D X in {de.files}")
        n_eicu = int(de[schluessel[0]].shape[0])
        # order exactly as in data/prepare_eicu.load_cohort:
        # status -> age -> 48 h -> exactly one ICU stay
        pt = pd.read_csv(_eicu_patient_pfad(),
                         usecols=["patientunitstayid",
                                  "patienthealthsystemstayid", "age",
                                  "gender", "unitdischargeoffset",
                                  "hospitaldischargestatus"])
        e0 = schritt("eICU-CRD", "ICU stays in eICU-CRD v2.0", pt,
                     "patient.csv.gz")

        status = pt["hospitaldischargestatus"].astype(str).str.strip(
            ).str.lower()
        pt1 = pt[status.isin(["alive", "expired"])]
        e1 = schritt("eICU-CRD", "Without a recorded discharge status "
                     "removed", pt1, "valid = {alive, expired}", e0)

        def alter(a):
            a = str(a).strip()
            if a in ("", "nan"):
                return np.nan
            if ">" in a:
                return 90.0
            try:
                return float(a)
            except ValueError:
                return np.nan

        pt2 = pt1[pt1["age"].map(alter) >= MIN_ALTER]
        e2 = schritt("eICU-CRD", f"Age below {MIN_ALTER} years removed", pt2,
                     "'> 89' counted as 90; empty entries drop out", e1)

        pt3 = pt2[pt2["unitdischargeoffset"] >= MIN_LOS_STUNDEN * 60]
        e3 = schritt("eICU-CRD", f"ICU stay shorter than {MIN_LOS_STUNDEN} h "
                     "removed", pt3, "unitdischargeoffset", e2)

        eins = pt3.groupby("patienthealthsystemstayid")[
            "patientunitstayid"].transform("count") == 1
        pt4 = pt3[eins]
        e4 = schritt("eICU-CRD", "Admissions with more than one ICU stay "
                     "removed", pt4, "exactly one ICU stay per hospital admission "
                     "(after the status, age and length-of-stay filters)", e3)

        zeilen.append({"kohorte": "eICU-CRD",
                       "schritt": "Without usable measurements removed",
                       "n_stays": n_eicu, "n_patienten": np.nan,
                       "entfernt": f"{e4 - n_eicu}",
                       "kommentar": "external transfer cohort"})
        print(f"    {'Without usable measurements removed':<44}{n_eicu:>8}"
              f"   (-{e4 - n_eicu})")
        pruefe("E1", "mirrored chain ends at or above the tensor size",
               n_eicu <= e4, f"{n_eicu} <= {e4}")
        # D1: mortality of the reconstructed set against the file
        y_e = np.asarray(de["y_ihm_eicu"], float).ravel()
        ihm_rek = (pt4["hospitaldischargestatus"].astype(str).str.strip()
                   .str.lower() == "expired").mean()
        pruefe("D1", "mortality of the reconstructed eICU set == mean "
               "y_ihm_eicu (same cohort)",
               abs(float(ihm_rek) - float(y_e.mean())) < 1e-4,
               f"{ihm_rek:.5f} against {y_e.mean():.5f}")
        alt_e = pt4["age"].map(alter)
        q1, med, q3 = np.nanpercentile(alt_e, [25, 50, 75])
        g = pt4["gender"].astype(str).str.strip().str.lower()
        n_w, n_m = int((g == "female").sum()), int((g == "male").sum())
        print(f"    [DEMOGRAPHICS eICU] age median {med:.0f} [{q1:.0f}-{q3:.0f}], "
              f"mean {np.nanmean(alt_e):.1f} +- {np.nanstd(alt_e, ddof=1):.1f}; "
              f"female {n_w} ({100 * n_w / len(pt4):.1f} %), male {n_m} "
              f"({100 * n_m / len(pt4):.1f} %), other/missing "
              f"{len(pt4) - n_w - n_m}; mortality {100 * ihm_rek:.1f} %")
        demo_eicu = {"alter_median": med, "alter_q1": q1, "alter_q3": q3,
                     "alter_mittel": float(np.nanmean(alt_e)),
                     "alter_sd": float(np.nanstd(alt_e, ddof=1)),
                     "weiblich_n": n_w, "weiblich_pct": 100 * n_w / len(pt4),
                     "n": len(pt4), "ihm_pct": 100 * ihm_rek}
        pruefe("E2", "remainder below 5 % of the filtered cohort "
               "(otherwise a step is missing)", (e4 - n_eicu) / max(e4, 1) < 0.05,
               f"{e4 - n_eicu} of {e4} = "
               f"{100 * (e4 - n_eicu) / max(e4, 1):.1f} %")
        # E3: MIMIC rule (count on raw data) applied to the eICU final cohort
        roh_n = pt.groupby("patienthealthsystemstayid")[
            "patientunitstayid"].transform("size")
        einzel_roh = set(pt.loc[roh_n == 1, "patientunitstayid"])
        in_datensatz = set(pt4["patientunitstayid"])
        betroffen = len(in_datensatz - einzel_roh)
        print(f"  [E3] eICU stays in the dataset whose admission had more than one "
              f"ICU stay in the RAW data: {betroffen} "
              f"of {len(in_datensatz)} ({100 * betroffen / len(in_datensatz):.1f} %) "
              "-- excluded under the MIMIC-IV rule, included here.")
        zeilen.append({"kohorte": "eICU-CRD",
                       "schritt": "Of these: admission had another ICU stay "
                       "in the raw data (would be excluded under the MIMIC-IV "
                       "rule)", "n_stays": betroffen, "n_patienten": np.nan,
                       "entfernt": "", "kommentar": "definition difference, "
                       "see Limitations"})
        # stay IDs of the external cohort for the demographics block: from the
        # tensor if present, else the reconstructed set (D1 above ties them together)
        if "stay_ids_eicu" in de.files:
            id_eicu = np.asarray(de["stay_ids_eicu"]).ravel()
        else:
            id_eicu = pt4["patientunitstayid"].to_numpy()

    # ----------------------------------------------------------- output ---
    df_out = pd.DataFrame(zeilen)
    df_out.to_csv(config.REPORT / "kohorten_kaskade.csv", index=False)
    md_pfad = config.APPENDIX / "table_1_cohort_flow.md"
    with open(md_pfad, "w", encoding="utf-8") as f:
        f.write("# Table 1. Cohort construction\n\n")
        f.write("Steps are listed in the order in which each preprocessing "
                "pipeline applied them; the order differs between the two "
                "databases and the criterion for single ICU stays per admission "
                "is evaluated on the raw data in MIMIC-IV but after the other "
                "filters in eICU-CRD (see the last eICU row).\n\n")
        f.write("| Cohort | Step | ICU stays | Patients | Removed |\n")
        f.write("|---|---|---|---|---|\n")
        for _, r in df_out.iterrows():
            pat_txt = "" if pd.isna(r["n_patienten"]) else \
                f"{int(r['n_patienten']):,}".replace(",", " ")
            f.write(f"| {r['kohorte']} | {r['schritt']} | "
                    f"{int(r['n_stays']):,}".replace(",", " ")
                    + f" | {pat_txt} | {r['entfernt']} |\n")
    if demo_eicu is not None:
        with open(md_pfad, "a", encoding="utf-8") as f:
            f.write("\n\n## (c) Baseline characteristics, eICU-CRD "
                    "(reconstructed cohort, anchor D1)\n\n")
            f.write("| Characteristic | eICU-CRD external cohort |\n|---|---|\n")
            f.write(f"| ICU stays | {demo_eicu['n']:,} |\n".replace(",", " "))
            f.write(f"| Age, median [IQR] | {demo_eicu['alter_median']:.0f} "
                    f"[{demo_eicu['alter_q1']:.0f}-{demo_eicu['alter_q3']:.0f}] |\n")
            f.write(f"| Age, mean (SD) | {demo_eicu['alter_mittel']:.1f} "
                    f"({demo_eicu['alter_sd']:.1f}) |\n")
            f.write(f"| Female, n (%) | {demo_eicu['weiblich_n']:,} "
                    f"({demo_eicu['weiblich_pct']:.1f}) |\n".replace(",", " "))
            f.write(f"| In-hospital mortality, % | {demo_eicu['ihm_pct']:.1f} |\n")
            f.write("\nAges recorded as '> 89' were set to 90. MIMIC-IV rows "
                    "from tabelle1_demographie_split.py.\n")
    print("\n  written: 06_report/kohorten_kaskade.csv and "
          "08_appendix/table_1_cohort_flow.md")

    # ---------------------------------------- demographics per cohort part ---
    print("\n" + "=" * 78)
    print("TABLE 1  age and sex per cohort part")
    print("=" * 78)
    bloecke = []
    a, g = mimic_demographie(id_train)
    bloecke.append(_block("MIMIC-IV v3.1 -- training", a, g))
    a, g = mimic_demographie(id_test)
    bloecke.append(_block("MIMIC-IV v3.1 -- internal test", a, g))
    if id_eicu is not None:
        id_eicu = np.asarray(sorted({int(s) for s in id_eicu}))
        if len(id_eicu) != N_TEST_EXTERN:
            abbruch(f"A1: eICU has {len(id_eicu)} instead of {N_TEST_EXTERN} stays.")
        a, g = eicu_demographie(id_eicu)
        bloecke.append(_block("eICU-CRD v2.0 -- external test", a, g))
    demo = pd.DataFrame(bloecke)
    demo.to_csv(config.APPENDIX / "table_1_demographics.csv", index=False)
    with open(config.APPENDIX / "table_1_demographics.md", "w", encoding="utf-8") as f:
        f.write("# Table 1. Age and sex per cohort part\n\n")
        f.write("| Characteristic | " + " | ".join(demo["kohorte"]) + " |\n")
        f.write("|---|" + "---|" * len(demo) + "\n")
        f.write("| ICU stays | " + " | ".join(f"{int(n):,}".replace(",", " ") for n in demo["n"]) + " |\n")
        f.write("| Age, median [IQR] | " + " | ".join(
            f"{r.alter_median:.0f} [{r.alter_q1:.0f}-{r.alter_q3:.0f}]" for r in demo.itertuples()) + " |\n")
        f.write("| Age, mean (SD) | " + " | ".join(
            f"{r.alter_mittel:.1f} ({r.alter_sd:.1f})" for r in demo.itertuples()) + " |\n")
        f.write("| Female, n (%) | " + " | ".join(
            f"{int(r.weiblich_n):,} ({r.weiblich_pct:.1f})".replace(",", " ") for r in demo.itertuples()) + " |\n")
        f.write("| Male, n (%) | " + " | ".join(
            f"{int(r.maennlich_n):,} ({r.maennlich_pct:.1f})".replace(",", " ") for r in demo.itertuples()) + " |\n")
        if (demo["unbekannt_n"] > 0).any():
            f.write("| Unknown/other, n (%) | " + " | ".join(
                f"{int(r.unbekannt_n):,} ({r.unbekannt_pct:.1f})".replace(",", " ") for r in demo.itertuples()) + " |\n")
        f.write("\nMIMIC-IV caps ages over 89 at 91 for de-identification; eICU-CRD codes them "
                f"as '> 89', set to {EICU_ALTER_90PLUS} here. Age = anchor_age + (year of ICU "
                "admission - anchor_year) in MIMIC-IV.\n")
    print("  written: 08_appendix/table_1_demographics.csv and .md")

    n_fail = sum(1 for o in pruefungen if not o)
    print("\n" + "=" * 78)
    if n_fail:
        print(f"RESULT: {n_fail} of {len(pruefungen)} checks FAIL. "
              "Do not use the cascade.")
        raise SystemExit(1)
    print(f"RESULT: all {len(pruefungen)} checks PASS. Table 1 "
          "reconstructed and covered by the dataset.")
    print("=" * 78)


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Table 1: exclusion cascades (MIMIC-IV, eICU-CRD), split sizes and age/sex per "
                    "cohort part from the raw PhysioNet tables, the datasets and the frozen split. "
                    "Writes kohorten_kaskade.csv, table_1_cohort_flow.md, table_1_demographics.csv/.md. "
                    "No options."
    ).parse_args()
    main()
