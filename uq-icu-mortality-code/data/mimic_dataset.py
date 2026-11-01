"""
mimic_dataset.py -- MIMIC-IV from the raw PhysioNet tables to dataset.npz and the frozen split

Paper:    Methods, "Study design and data sources" (cohort, 48 x 13 features, patient-level split,
          train-only standardisation; frozen train/validation split)
Origin:   the modules data/csv_loader.py, mimic_cohort.py, mimic_height.py, mimic_features.py,
          build_dataset.py and split.py of the full package, merged for the short version; every
          section below keeps its original header as a comment and its code verbatim
Reads:    MIMIC_DIR/icu/icustays.csv.gz, icu/chartevents.csv.gz (chunked), hosp/admissions.csv.gz,
          hosp/patients.csv.gz, hosp/omr.csv.gz
Writes:   config.DATASET_MIMIC (dataset.npz); config.ROOT/stay_subject_map.csv; config.SPLIT/split_v2.csv
          and manifest_split_v2.json
Run:      python data/mimic_dataset.py --build        (about 20 min; dataset.npz)
          python data/mimic_dataset.py --export-map   (stay -> patient map, once)
          python data/mimic_dataset.py --split        (seconds; frozen split)
          or all three through run_pipeline.py --data
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import numpy as np
import os
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data.feature_mapping import GRENZEN  # noqa: E402
from data.feature_mapping import (  # noqa: E402
    ITEMID_MAPPING,
    GCS_COMPONENTS,
    ALLE_ITEMIDS,
    FINALE_VARIABLEN,
    IMPUTE_WERTE,
    GRENZEN,
    fahrenheit_zu_celsius,
)
from data.feature_mapping import FINALE_VARIABLEN  # noqa: E402


# ==========================================================================================
# section from the full package: data/csv_loader.py
# ------------------------------------------------------------------------------------------
# csv_loader.py -- memory-safe chunked loader for the large PhysioNet CSV tables
#
# Paper:    Methods, "Data sources and preprocessing" (MIMIC-IV extraction)
# Origin:   src/daten_laden.py (no SKRIPT_STAND; working copy of 2026-10-08)
# Reads:    whatever table the caller passes (e.g. icu/chartevents.csv.gz)
# Writes:   nothing
# Checks:   none (library module)
# Run:      not a script; imported by data/mimic_height.py and data/mimic_features.py
#
# The full MIMIC-IV chartevents table has hundreds of millions of rows. Loading it
# whole exhausts memory, so the table is read in chunks, every chunk is filtered
# immediately to the rows of interest (e.g. the ~29 item IDs of the 13 variables),
# and only the filtered rows are kept. Peak memory stays small regardless of the
# size of the source table; the result is identical to a filtered full load.
# ==========================================================================================

# rows per chunk when reading large CSV files; 1-3 million is a good RAM/speed trade-off
CHUNKSIZE = 2_000_000


def lade_gefiltert(
    pfad,
    usecols,
    filter_spalte=None,
    erlaubte_werte=None,
    dtype=None,
    datum_spalten=None,
    chunksize=CHUNKSIZE,
    log=True,
):
    """
    Read a (possibly very large) CSV in chunks and keep only the rows whose value
    in `filter_spalte` is contained in `erlaubte_werte`.

    Parameters
    ----------
    pfad           : path to the .csv.gz (or .csv)
    usecols        : columns to read at all (reading fewer columns saves memory)
    filter_spalte  : column to filter on, e.g. "itemid" (MIMIC chartevents) or
                     "patientunitstayid" (eICU tables); None = no filter
    erlaubte_werte : set/list of allowed values in filter_spalte
    dtype          : optional dtype mapping, e.g. {"itemid": "int32", "valuenum": "float32"}
    datum_spalten  : optional list of columns parsed to datetime right after
                     filtering, per chunk, so that only the kept rows are parsed
    chunksize      : rows per chunk (default CHUNKSIZE)
    log            : show progress on the console

    Returns
    -------
    One concatenated DataFrame with the filtered rows only.
    """
    # sets are faster than lists for isin look-ups
    if erlaubte_werte is not None:
        erlaubte_werte = set(erlaubte_werte)

    teile = []
    gelesen = 0
    behalten = 0

    # chunksize=... turns read_csv into an iterator over chunks
    leser = pd.read_csv(
        pfad,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
    )

    for i, chunk in enumerate(leser):
        gelesen += len(chunk)
        # 1. filter (keep relevant rows only); .copy() makes the slice an
        #    independent frame so that the date parsing below does not raise
        #    a SettingWithCopyWarning
        if filter_spalte is not None:
            chunk = chunk[chunk[filter_spalte].isin(erlaubte_werte)].copy()
        # 2. parse dates only now, on the few kept rows
        if datum_spalten and len(chunk) > 0:
            for spalte in datum_spalten:
                chunk[spalte] = pd.to_datetime(chunk[spalte])
        # 3. keep
        if len(chunk) > 0:
            teile.append(chunk)
            behalten += len(chunk)
        if log:
            # \r overwrites the line -> a running progress display
            print(f"  chunk {i + 1:>4}: read {gelesen:>14,}   kept {behalten:>12,}",
                  end="\r", flush=True)

    if log:
        print()  # line break after the progress line

    if teile:
        return pd.concat(teile, ignore_index=True)
    # nothing kept -> empty DataFrame with the right columns
    return pd.DataFrame(columns=usecols)



# ==========================================================================================
# section from the full package: data/mimic_cohort.py
# ------------------------------------------------------------------------------------------
# mimic_cohort.py -- MIMIC-IV cohort and labels per ICU stay (Harutyunyan cascade)
#
# Paper:    Methods, "Study design and data sources" (cohort selection counts);
#           Table 1 (MIMIC-IV cascade, reproduced by data/cohort_tables.py)
# Origin:   src/build_labels.py (no SKRIPT_STAND; working copy of 2026-10-08)
# Reads:    MIMIC_DIR/icu/icustays.csv.gz        (stay_id, hadm_id, subject_id, intime, outtime, los)
#           MIMIC_DIR/hosp/admissions.csv.gz     (hospital_expire_flag)
#           MIMIC_DIR/hosp/patients.csv.gz       (anchor_age)
# Writes:   nothing (returns a DataFrame; data/build_dataset.py and data/split.py call it)
# Checks:   none beyond the printed cascade counts
# Run:      python data/mimic_cohort.py        (prints the cascade and label summary)
#
# Per ICU stay (stay_id) two targets are produced:
#   - los : length of stay in days      (from icustays; stored in the dataset, not
#                                         used by the manuscript)
#   - ihm : in-hospital mortality       (from admissions.hospital_expire_flag)
#
# Cohort selection follows Harutyunyan et al. (2019): three filters, each
# switchable, with a count per step (the patient-selection cascade of Table 1):
#   1. only hospital admissions with EXACTLY ONE ICU stay (admissions with several
#      ICU stays / transfers are excluded); counted on the raw table, before any
#      other filter
#   2. adults only (anchor_age >= 18); MIMIC-IV contains adults only, so this
#      removes nothing, kept for completeness
#   3. only stays with LOS >= 48 hours, required for a 48-hour input window
# The returned columns are those of the original pipeline so that the downstream
# feature extraction is unchanged.
# ==========================================================================================

ICU = config.MIMIC_DIR / "icu"
HOSP = config.MIMIC_DIR / "hosp"

# ---------------------------------------------------------------------------
# filter switches (set to None / False to release a filter)
# ---------------------------------------------------------------------------
NUR_EINE_ICU_PRO_AUFNAHME = True   # exclude admissions with > 1 ICU stay
MIN_ALTER = 18                     # adults; None = off
MIN_LOS_STUNDEN = 48               # 48-hour lower bound; None = off


def load_labels() -> pd.DataFrame:
    # ICU stays: already contains 'los' (days)
    icustays = pd.read_csv(ICU / "icustays.csv.gz")

    # hospital admissions: contains 'hospital_expire_flag' (1 = died in hospital)
    admissions = pd.read_csv(
        HOSP / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "hospital_expire_flag"],
    )

    # merge on hadm_id so that every ICU stay receives its IHM label
    df = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="left")

    # IHM under a clear label name
    df["ihm"] = df["hospital_expire_flag"].astype("Int64")

    # ICU stays per admission counted on the RAW data (before any row filter)
    df["n_icu_pro_aufnahme"] = df.groupby("hadm_id")["stay_id"].transform("size")

    # -------------------- cohort cascade --------------------
    print("=" * 60)
    print("COHORT SELECTION")
    print("=" * 60)
    n = len(df)
    print(f"Start (all ICU stays):                   {n:>7d}")

    # 1) only admissions with exactly one ICU stay
    if NUR_EINE_ICU_PRO_AUFNAHME:
        vor = len(df)
        df = df[df["n_icu_pro_aufnahme"] == 1]
        print(f"  - admissions with > 1 ICU stay (-{vor - len(df):>6d}) -> {len(df):>7d}")

    # 2) adults only
    if MIN_ALTER is not None:
        patients = pd.read_csv(HOSP / "patients.csv.gz", usecols=["subject_id", "anchor_age"])
        df = df.merge(patients, on="subject_id", how="left")
        vor = len(df)
        df = df[df["anchor_age"] >= MIN_ALTER]
        print(f"  - age < {MIN_ALTER}                (-{vor - len(df):>6d}) -> {len(df):>7d}")

    # 3) only LOS >= 48 h
    if MIN_LOS_STUNDEN is not None:
        min_los_tage = MIN_LOS_STUNDEN / 24.0
        vor = len(df)
        df = df[df["los"] >= min_los_tage]
        print(f"  - LOS < {MIN_LOS_STUNDEN} h ({min_los_tage:.2f} days)   (-{vor - len(df):>6d}) -> {len(df):>7d}")

    # 4) drop missing labels (should be 0 if every stay has an admission)
    vor = len(df)
    df = df.dropna(subset=["los", "ihm"])
    if vor - len(df):
        print(f"  - missing LOS/IHM          (-{vor - len(df):>6d}) -> {len(df):>7d}")

    print("-" * 60)
    print(f"Final cohort:                            {len(df):>7d}")
    print("=" * 60)

    # reduce to the label columns (schema unchanged)
    labels = df[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los", "ihm"]].reset_index(drop=True)

    return labels



# ==========================================================================================
# section from the full package: data/mimic_height.py
# ------------------------------------------------------------------------------------------
# mimic_height.py -- height per MIMIC-IV ICU stay (chartevents 226730/226707, else OMR)
#
# Paper:    Methods, "Data sources and preprocessing" (height variable)
# Origin:   src/hoehe.py (no SKRIPT_STAND; working copy of 2026-10-08)
# Reads:    MIMIC_DIR/icu/icustays.csv.gz       (subject_id, stay_id, intime)
#           MIMIC_DIR/icu/chartevents.csv.gz    (items 226730 and 226707 only, chunked)
#           MIMIC_DIR/hosp/omr.csv.gz           (result_name containing "height")
# Writes:   nothing (returns dict {stay_id: height_cm}; called by data/mimic_features.py)
# Checks:   fail-loud if no OMR heights are found
# Run:      python data/mimic_height.py        (prints coverage statistics)
#
# Follows the official MIMIC-IV concept (mimiciv_derived.height / measurement/height.sql):
#   - chartevents item 226730  Height (cm)      -> already cm
#   - chartevents item 226707  Height (Inches)  -> times 2.54
#   - plausibility 120 < height < 230 cm (GRENZEN["height"] in feature_mapping)
#   - one value per stay (median of the plausible values)
# Added for coverage: OMR height (not part of the official concept), the plausible
# measurement closest to the ICU admission (intime).
#
# Ranking per stay:
#   1. chartevents height (226730 / 226707), median of plausible values
#   2. else OMR height closest to intime
#   3. else no value -> the calling pipeline imputes the default (170)
# ==========================================================================================

ZOLL_ZU_CM = 2.54
HOEHE_MIN_CM, HOEHE_MAX_CM = GRENZEN["height"]   # ONE source: (120, 230)
HEIGHT_CM_ITEM = 226730     # chartevents: Height (cm)
HEIGHT_INCH_ITEM = 226707   # chartevents: Height (Inches)


def _plausibel(cm: pd.Series) -> pd.Series:
    return (cm >= HOEHE_MIN_CM) & (cm <= HOEHE_MAX_CM)


def lade_hoehen_pro_stay(verbose: bool = True) -> dict:
    icustays = pd.read_csv(
        ICU / "icustays.csv.gz", usecols=["subject_id", "stay_id", "intime"]
    )
    icustays["intime"] = pd.to_datetime(icustays["intime"])

    # ======================================================================
    # A. chartevents height (226730 cm + 226707 inch) -- official procedure
    # ======================================================================
    ce = lade_gefiltert(
        ICU / "chartevents.csv.gz",
        usecols=["stay_id", "itemid", "valuenum"],
        filter_spalte="itemid",
        erlaubte_werte=[HEIGHT_CM_ITEM, HEIGHT_INCH_ITEM],
        dtype={"itemid": "int32", "valuenum": "float32"},
    )
    ce = ce.dropna(subset=["valuenum"])
    # 226730 stays cm; 226707 (inches) -> cm
    ce["hoehe_cm"] = np.where(
        ce["itemid"] == HEIGHT_INCH_ITEM,
        ce["valuenum"] * ZOLL_ZU_CM,
        ce["valuenum"],
    )
    if verbose:
        n_cm = int((ce["itemid"] == HEIGHT_CM_ITEM).sum())
        n_in = int((ce["itemid"] == HEIGHT_INCH_ITEM).sum())
        print(f"chartevents height rows: 226730 (cm)={n_cm:,}  226707 (inch)={n_in:,}")
    ce = ce[_plausibel(ce["hoehe_cm"])]
    ce_pro_stay = ce.groupby("stay_id")["hoehe_cm"].median()

    # ======================================================================
    # B. OMR height (supplement), closest to the admission
    # ======================================================================
    omr = pd.read_csv(
        HOSP / "omr.csv.gz",
        usecols=["subject_id", "chartdate", "result_name", "result_value"],
    )
    omr["chartdate"] = pd.to_datetime(omr["chartdate"])
    namen = omr["result_name"].astype(str)
    ist_hoehe = namen.str.contains("height", case=False, na=False)
    if not ist_hoehe.any():
        raise ValueError(
            "FAIL-LOUD: no OMR heights found. Check result_name."
        )
    omr_h = omr[ist_hoehe].copy()
    omr_h["wert_roh"] = pd.to_numeric(omr_h["result_value"], errors="coerce")
    omr_h = omr_h.dropna(subset=["wert_roh"])
    # OMR records inches; values > 100 are already cm (robust to mixed units)
    omr_h["hoehe_cm"] = np.where(
        omr_h["wert_roh"] <= 100, omr_h["wert_roh"] * ZOLL_ZU_CM, omr_h["wert_roh"]
    )
    omr_h = omr_h[_plausibel(omr_h["hoehe_cm"])]

    gem = icustays.merge(
        omr_h[["subject_id", "chartdate", "hoehe_cm"]], on="subject_id", how="inner"
    )
    gem["abstand"] = (gem["chartdate"] - gem["intime"]).abs()
    gem = gem.sort_values("abstand").drop_duplicates("stay_id", keep="first")
    omr_pro_stay = gem.set_index("stay_id")["hoehe_cm"]

    # ======================================================================
    # C. ranking: chartevents first, else OMR
    # ======================================================================
    ce_idx = set(ce_pro_stay.index.tolist())
    omr_idx = set(omr_pro_stay.index.tolist())

    lookup, n_ce, n_omr = {}, 0, 0
    for sid in icustays["stay_id"].tolist():
        if sid in ce_idx:
            lookup[sid] = float(ce_pro_stay.loc[sid])
            n_ce += 1
        elif sid in omr_idx:
            lookup[sid] = float(omr_pro_stay.loc[sid])
            n_omr += 1

    if verbose:
        n = len(icustays)
        n_total = len(lookup)
        werte = np.array(list(lookup.values()), dtype=float)
        print("-" * 60)
        print(f"stays total:                   {n:>7d}")
        print(f"  via chartevents (226730/707): {n_ce:>7d}")
        print(f"  additionally via OMR only:    {n_omr:>7d}")
        print(f"  total with height:            {n_total:>7d}  ({n_total / n * 100:.1f}%)")
        print(f"  without (default 170):        {n - n_total:>7d}")
        if n_total:
            print(
                f"height cm: mean={werte.mean():.1f}  std={werte.std():.1f}  "
                f"min={werte.min():.1f}  max={werte.max():.1f}"
            )
        print("-" * 60)

    return lookup



# ==========================================================================================
# section from the full package: data/mimic_features.py
# ------------------------------------------------------------------------------------------
# mimic_features.py -- 48 x 13 hourly feature matrix per MIMIC-IV ICU stay
#
# Paper:    Methods, "Data sources and preprocessing" (MIMIC-IV preprocessing rules
#           R1-R3 that data/prepare_eicu.py mirrors for eICU-CRD)
# Origin:   src/extract_features.py (no SKRIPT_STAND; working copy of 2026-10-08)
# Reads:    MIMIC_DIR/icu/icustays.csv.gz       (stay_id, intime)
#           MIMIC_DIR/icu/chartevents.csv.gz    (the item IDs of feature_mapping, chunked)
#           via data/mimic_height.py: chartevents (226730/226707), hosp/omr.csv.gz
# Writes:   nothing (returns dict {stay_id: 48x13 array}; called by data/build_dataset.py)
# Checks:   none beyond printed counts (stays without a real height)
# Run:      python data/mimic_features.py      (extracts all stays and prints shape checks)
#
# Steps per stay_id:
#   1. keep the measurements of the relevant item IDs
#   2. restrict to the 48-hour window from admission (intime)
#   3. hour since admission (0 to 47)
#   4. item ID -> variable name (pooling happens here automatically, because
#      several item IDs share one name)
#   5. temperature Fahrenheit -> Celsius (values >= 79)
#   5b. FiO2 percent -> fraction (conditional, MIMIC stores both)
#   6. mean per hour and variable, reshape wide (hours x variables)
#   7. full 48-hour grid
#   7b. outliers -> NaN (plausibility limits, BEFORE filling)
#   8. fill gaps: forward (LOCF), then backward
#   9. GCS total from the three components
#   9b. gcs_total against its plausibility limits
#   10. completely missing variables -> fixed default value
#   11. fixed column order -> 48 x 13
# Steps 5b, 7b and 9b give MIMIC-IV and eICU-CRD EXACTLY the same outlier and unit
# treatment (same GRENZEN table from feature_mapping).
# ==========================================================================================

OBS_WINDOW_HOURS = 48  # observation window


def itemid_zu_name(itemid):
    """Variable name for an item ID (resolves the pooling)."""
    if itemid in ITEMID_MAPPING:
        return ITEMID_MAPPING[itemid]
    if itemid in GCS_COMPONENTS:
        return GCS_COMPONENTS[itemid]
    return None


def extrahiere_einen_stay(messungen_des_stays, intime):
    """
    Build the 48 x 13 matrix for ONE ICU stay.

    messungen_des_stays: DataFrame with columns itemid, charttime, valuenum
                         (already restricted to this stay)
    intime:              admission timestamp of this stay
    """

    # --- step 1: relevant item IDs only ---
    relevant = messungen_des_stays[
        messungen_des_stays["itemid"].isin(ALLE_ITEMIDS)
    ].copy()

    # --- step 2: restrict to the 48-hour window ---
    fensterende = intime + pd.Timedelta(hours=OBS_WINDOW_HOURS)
    im_fenster = relevant[
        (relevant["charttime"] >= intime)
        & (relevant["charttime"] < fensterende)
    ].copy()

    # --- step 3: hour since admission (0 to 47) ---
    zeit_seit_aufnahme = im_fenster["charttime"] - intime
    stunden = zeit_seit_aufnahme.dt.total_seconds() / 3600.0
    im_fenster["stunde"] = stunden.astype(int)

    # --- step 4: item ID -> variable name (resolves the pooling) ---
    im_fenster["variable"] = im_fenster["itemid"].apply(itemid_zu_name)

    # --- step 5: temperature Fahrenheit -> Celsius ---
    # safety rule as in Harutyunyan: values >= 79 are certainly Fahrenheit
    ist_temp = im_fenster["variable"] == "temperature"
    ist_fahrenheit = ist_temp & (im_fenster["valuenum"] >= 79)
    im_fenster.loc[ist_fahrenheit, "valuenum"] = fahrenheit_zu_celsius(
        im_fenster.loc[ist_fahrenheit, "valuenum"]
    )

    # --- step 5b: FiO2 percent -> fraction (MIMIC stores both) ---
    # MIMIC records FiO2 partly as percent (e.g. 40), partly as fraction (e.g.
    # 0.40). The rule is conditional:
    #   value > 1.0  -> recorded as percent -> divide by 100
    #   value <= 1.0 -> already a fraction  -> unchanged
    # The threshold 1.0 separates cleanly: a real fraction is never above 1.0, a
    # percent value never meaningfully below. Values that fall below 0.21 (room
    # air) after dividing are caught by the plausibility check.
    ist_fio2 = im_fenster["variable"] == "fio2"
    ist_prozent = ist_fio2 & (im_fenster["valuenum"] > 1.0)
    im_fenster.loc[ist_prozent, "valuenum"] = (
        im_fenster.loc[ist_prozent, "valuenum"] / 100.0
    )

    # --- step 6: mean per hour and variable, reshape wide ---
    # several measurements of a variable within one hour -> mean (this is also the pooling)
    pro_stunde = im_fenster.groupby(["stunde", "variable"])["valuenum"].mean()
    tabelle = pro_stunde.unstack("variable")

    # --- step 7: full 48-hour grid (hour 0 to 47) ---
    volles_raster = range(OBS_WINDOW_HOURS)
    tabelle = tabelle.reindex(volles_raster)

    # --- step 7b: outliers -> NaN (BEFORE filling) ---
    # Values outside the central plausibility limits are measurement errors.
    # They are set to missing so that the next step treats them like any other
    # gap: first the real neighbouring value of the same stay (LOCF), else the
    # fixed default. Same GRENZEN table and same order (filter, then fill) as
    # the eICU chain. gcs_total does not exist yet (only the three components),
    # so it is checked after the summation (step 9b).
    for variable, (low, high) in GRENZEN.items():
        if variable in tabelle.columns:
            ausserhalb = (tabelle[variable] < low) | (tabelle[variable] > high)
            tabelle.loc[ausserhalb, variable] = np.nan

    # --- step 8: fill gaps (forward first, then backward) ---
    tabelle = tabelle.ffill()
    tabelle = tabelle.bfill()

    # --- step 9: GCS total (after filling) ---
    gcs_teile = ["gcs_eye", "gcs_verbal", "gcs_motor"]
    vorhandene_gcs = [s for s in gcs_teile if s in tabelle.columns]
    if len(vorhandene_gcs) == 3:
        tabelle["gcs_total"] = tabelle[gcs_teile].sum(axis=1)
        tabelle = tabelle.drop(columns=gcs_teile)

    # --- step 9b: gcs_total against its plausibility limits ---
    # The sum of valid components always lies between 3 and 15. If a component
    # was broken, the sum may fall outside -> NaN, so that step 10 fills it with
    # the default. Same logic as the eICU chain.
    if "gcs_total" in tabelle.columns:
        low, high = GRENZEN["gcs_total"]
        ausserhalb = (tabelle["gcs_total"] < low) | (tabelle["gcs_total"] > high)
        tabelle.loc[ausserhalb, "gcs_total"] = np.nan

    # --- step 10: completely missing variables -> default value ---
    for variable in FINALE_VARIABLEN:
        if variable not in tabelle.columns:
            # variable never measured for this stay -> default
            tabelle[variable] = IMPUTE_WERTE[variable]
        else:
            # variable present but possibly still NaN (e.g. incomplete GCS)
            tabelle[variable] = tabelle[variable].fillna(IMPUTE_WERTE[variable])

    # --- step 11: fixed column order -> 48 x 13 ---
    # .copy() so that setting the height column in extrahiere_alle_stays writes
    # into its own matrix (no SettingWithCopy on a view)
    matrix = tabelle[FINALE_VARIABLEN].copy()

    return matrix


def extrahiere_alle_stays():
    """
    Iterate over all ICU stays and build one 48 x 13 matrix each.

    Loading strategy (affects only the loading, not the feature logic):
      1. chartevents is read in chunks and filtered immediately to the ~29
         relevant item IDs (lade_gefiltert).
      2. Instead of searching the table per stay, it is grouped ONCE by stay_id;
         groupby(...).indices gives the row positions per stay.
    The feature computation itself (extrahiere_einen_stay) is unaffected.
    """

    # --- icustays: small, load normally ---
    icustays = pd.read_csv(ICU / "icustays.csv.gz", usecols=["stay_id", "intime"])
    icustays["intime"] = pd.to_datetime(icustays["intime"])

    # --- chartevents: large -> chunked, filtered to the item IDs ---
    print("Loading chartevents (chunked, relevant item IDs only) ...")
    chartevents = lade_gefiltert(
        ICU / "chartevents.csv.gz",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        filter_spalte="itemid",
        erlaubte_werte=ALLE_ITEMIDS,
        dtype={"itemid": "int32", "valuenum": "float32"},
        datum_spalten=["charttime"],  # parse while loading (saves RAM)
    )
    print(f"  {len(chartevents):,} relevant measurement rows kept.")

    # --- group ONCE by stay_id ---
    # .indices is a dict {stay_id: numpy array of row positions}
    print("Grouping by stay_id ...")
    gruppen_positionen = chartevents.groupby("stay_id").indices

    # empty slice (correctly shaped columns) for stays without any measurement
    leer = chartevents.iloc[0:0]

    # --- height separately (official MIMIC concept: chartevents 226730/226707 + OMR) ---
    # Height is static and does not belong in the 48-hour time-series pooling;
    # it is therefore NOT taken from chartevents in step 4 but looked up per stay
    # and written into the height column. Without a real value the default (170)
    # from step 10 remains.
    print("Loading heights separately (chartevents 226730/226707 + OMR) ...")
    hoehe_lookup = lade_hoehen_pro_stay(verbose=True)

    # --- one matrix per ICU stay (order as in icustays) ---
    alle_matrizen = {}
    n = len(icustays)
    n_ohne_hoehe = 0   # stays without a real height -> default 170
    for k, (stay_id, intime) in enumerate(
        zip(icustays["stay_id"], icustays["intime"])
    ):
        if stay_id in gruppen_positionen:
            messungen = chartevents.iloc[gruppen_positionen[stay_id]]
        else:
            # this stay has none of the relevant measurements ->
            # extrahiere_einen_stay returns a fully imputed matrix
            messungen = leer

        matrix = extrahiere_einen_stay(messungen, intime)

        # set height (static): real value from mimic_height, else default (170)
        if stay_id in hoehe_lookup:
            matrix["height"] = hoehe_lookup[stay_id]
        else:
            n_ohne_hoehe += 1

        alle_matrizen[stay_id] = matrix.values  # 48 x 13 numpy array

        if (k + 1) % 5000 == 0 or (k + 1) == n:
            print(f"  {k + 1:,} / {n:,} stays processed", end="\r", flush=True)
    print()

    # --- loud count: fail-loud instead of silently wrong ---
    print(f"  [height] {n_ohne_hoehe} stays without a real height -> default 170.")

    return alle_matrizen



# ==========================================================================================
# section from the full package: data/build_dataset.py
# ------------------------------------------------------------------------------------------
# build_dataset.py -- MIMIC-IV dataset: features + labels, patient-level split, standardization
#
# Paper:    Methods, "Study design and data sources" (dataset construction,
#           patient-level 80/20 split, train-only standardization)
# Origin:   src/build_dataset.py (no SKRIPT_STAND; working copy of 2026-10-08)
# Reads:    raw MIMIC-IV via data/mimic_features.py (icu/icustays, icu/chartevents,
#           hosp/omr) and data/mimic_cohort.py (icu/icustays, hosp/admissions, hosp/patients)
# Writes:   config.DATASET_MIMIC  (dataset.npz)
# Checks:   none beyond the printed counts and the post hoc mean/std control
# Run:      python data/build_dataset.py        (about 20 min on the authors' machine; needs the raw
#                                                MIMIC-IV tables)
#
# Order (matters for leakage avoidance):
#   1. join features (48 x 13 per stay) and labels (los, ihm) on stay_id
#      -> labels come from mimic_cohort.load_labels(), i.e. already FILTERED
#         (>= 48 h, single-ICU admissions, adults); only these stays enter
#   2. train/test split AT PATIENT LEVEL (subject_id), not per stay
#      -> no patient appears in train AND test
#   3. standardization: mean/std from train ONLY, applied to train and test
#   4. everything into one .npz file
#
# dataset.npz contains:
#   X_train, y_los_train, y_ihm_train, stay_ids_train
#   X_test,  y_los_test,  y_ihm_test,  stay_ids_test
#   mittelwert, std        (training-pool mean/std per variable)
#   variablen              (the 13 variable names, column order of X)
# ==========================================================================================

TEST_ANTEIL = 0.2     # 20 % of the patients go to the test set
ZUFALLS_SEED = 42     # fixed seed -> reproducible split


def baue_datensatz():

    # ===================================================================
    # step 1: extract features (dict: stay_id -> 48 x 13 array)
    # ===================================================================
    print("1) Extracting features ...")
    features_dict = extrahiere_alle_stays()

    # ===================================================================
    # step 2: labels + patient assignment of the FILTERED cohort
    # ===================================================================
    print("2) Loading filtered labels and patient assignment ...")

    # load_labels() returns the Harutyunyan-filtered cohort (>= 48 h,
    # single-ICU admissions, adults); only these stays belong in the dataset
    labels = load_labels()

    # look-up tables: stay_id -> value (filtered stays only)
    los_pro_stay = dict(zip(labels["stay_id"], labels["los"]))
    ihm_pro_stay = dict(zip(labels["stay_id"], labels["ihm"]))
    subject_pro_stay = dict(zip(labels["stay_id"], labels["subject_id"]))

    # ===================================================================
    # step 3: bring everything into equally ordered arrays
    # ===================================================================
    print("3) Joining features and labels in the same order ...")

    alle_stay_ids = list(features_dict.keys())

    # COHORT MATCH: keep only stays with a valid label in the filtered cohort.
    # Stays removed by the filters are absent from the look-up tables ->
    # .get(sid) returns None -> pd.notna(None) is False -> they drop out here.
    stay_ids = [
        sid for sid in alle_stay_ids
        if pd.notna(ihm_pro_stay.get(sid)) and pd.notna(los_pro_stay.get(sid))
    ]
    ausgeschlossen = len(alle_stay_ids) - len(stay_ids)
    if ausgeschlossen:
        print(f"   {ausgeschlossen} extracted stays are not in the filtered "
              f"cohort (or have no label) -> excluded.")
    print(f"   stays in the dataset: {len(stay_ids)}")

    # float32 instead of float64 halves memory and file size; y_ihm is 0/1 -> int
    X = np.array([features_dict[sid] for sid in stay_ids], dtype=np.float32)
    y_los = np.array([los_pro_stay[sid] for sid in stay_ids], dtype=np.float32)
    y_ihm = np.array([ihm_pro_stay[sid] for sid in stay_ids], dtype=np.int64)
    subjects = np.array([subject_pro_stay[sid] for sid in stay_ids])
    stay_ids = np.array(stay_ids)

    print("   X shape:", X.shape, "| labels:", y_los.shape)

    # ===================================================================
    # step 4: train/test split AT PATIENT LEVEL
    # ===================================================================
    print("4) Splitting at patient level (no patient in both sets) ...")

    einzigartige_patienten = np.unique(subjects)

    rng = np.random.default_rng(ZUFALLS_SEED)
    rng.shuffle(einzigartige_patienten)

    n_test = int(len(einzigartige_patienten) * TEST_ANTEIL)
    test_patienten = set(einzigartige_patienten[:n_test])

    # mask: which stay belongs to a test patient?
    ist_test = np.array([s in test_patienten for s in subjects])
    ist_train = ~ist_test

    print(f"   patients total: {len(einzigartige_patienten)}")
    print(f"   of which test: {len(test_patienten)}")
    print(f"   stays train: {ist_train.sum()} | stays test: {ist_test.sum()}")

    # ===================================================================
    # step 5: standardization -- statistics from train ONLY
    # ===================================================================
    print("5) Standardizing (mean/std from train only) ...")

    X_train = X[ist_train]
    X_test = X[ist_test]

    # mean and std per variable over all training stays and hours;
    # X_train has shape (n_train, 48, 13) -> over axes 0 and 1 -> 13 values.
    # Statistics are computed in float64 even though X is float32: float32
    # accumulates a bias over millions of values, especially for variables with
    # small relative spread (temperature ~37, pH ~7.4). The standardized result
    # is cast back to float32 to keep the memory advantage.
    mittelwert = X_train.mean(axis=(0, 1), dtype=np.float64)
    std = X_train.std(axis=(0, 1), dtype=np.float64)
    std[std == 0] = 1.0  # avoid division by zero (constant variable)

    X_train_std = ((X_train - mittelwert) / std).astype(np.float32)
    X_test_std = ((X_test - mittelwert) / std).astype(np.float32)   # test with TRAIN statistics

    # ===================================================================
    # step 6: save
    # ===================================================================
    print(f"6) Saving {config.DATASET_MIMIC} ...")

    # savez_compressed: vital signs compress well -> much smaller file
    config.DATASET_MIMIC.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.DATASET_MIMIC,
        X_train=X_train_std,
        y_los_train=y_los[ist_train],
        y_ihm_train=y_ihm[ist_train],
        stay_ids_train=stay_ids[ist_train],
        X_test=X_test_std,
        y_los_test=y_los[ist_test],
        y_ihm_test=y_ihm[ist_test],
        stay_ids_test=stay_ids[ist_test],
        mittelwert=mittelwert,
        std=std,
        variablen=np.array(FINALE_VARIABLEN),
    )

    print(f"   Done. {config.DATASET_MIMIC} saved.")
    return X_train_std, X_test_std



# ==========================================================================================
# section from the full package: data/split.py
# ------------------------------------------------------------------------------------------
# split.py -- stage S1: frozen, checksummed train/validation split of the MIMIC-IV training pool
#
# Paper:    Methods, "Model development" (train/validation split, grouped by patient,
#           stratified by outcome; one split for all families and seeds)
# Origin:   src/_archiv/erzeuge_split_v2 (1).py (no SKRIPT_STAND; the archive copy that
#           produced the frozen split), plus src/exportiere_stay_subject_map.py for the
#           stay -> patient map (--export-map)
# Reads:    config.DATASET_MIMIC                 (stay_ids_train, y_ihm_train, stay_ids_test)
#           config.ROOT/stay_subject_map.csv     (stay_id, subject_id, partition)
#           --export-map only: raw MIMIC-IV via data/mimic_cohort.load_labels()
#           (icu/icustays, hosp/admissions, hosp/patients)
# Writes:   config.SPLIT/split_v2.csv            (stay_id, subject_id, teil)
#           config.SPLIT/manifest_split_v2.json  (counts, prevalences, SHA-256 checksums)
#           --export-map only: config.ROOT/stay_subject_map.csv
# Checks:   A  the map covers exactly the stay_ids of the dataset (train 29,350; test 7,298;
#              total 36,648) and train/test patients are disjoint
#           B  every training stay is assigned to exactly one part; train and validation
#              PATIENTS are disjoint; no test patient in the split
#           C  |validation stay share - 0.20| < 0.03
#           D  |prevalence(train) - prevalence(val)| < 0.01 (stay level)
#           Any violation aborts. An existing split_v2.csv is never overwritten: identical
#           assignment -> confirmed (idempotent), different assignment -> abort.
# Run:      python data/split.py --export-map    (once; writes stay_subject_map.csv, minutes)
#           python data/split.py                 (seconds)
#
# The split of the 29,350 training stays is 80/20, GROUPED by subject_id (no patient
# in both parts) and STRATIFIED at patient level (a patient is positive if any of the
# patient's stays is positive). It is fully deterministic (numpy PCG64, SPLIT_SEED,
# sorted patient lists before shuffling) and therefore bit-identical across machines:
# a run on another computer MUST print the same assignment SHA-256.
# ==========================================================================================

SPLIT_SEED = 20260817
VAL_ANTEIL = 0.20
AUSGABE_ORDNER = os.path.join("rerun_natural", "01_split")   # relative name recorded in the manifest
MAP_DATEI = config.ROOT / "stay_subject_map.csv"


def _sicher_ausgabepfad(dateiname):
    """Fail-closed: outputs only below config.RUNS."""
    ziel = os.path.abspath(str(config.SPLIT / dateiname))
    basis = os.path.abspath(str(config.RUNS))
    if os.path.commonpath([ziel, basis]) != basis:
        abbruch(f"output path {ziel} lies outside {basis}.")
    os.makedirs(os.path.dirname(ziel), exist_ok=True)
    return ziel


def abbruch(t):
    raise SystemExit("ABORT (fail-fast): " + t)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def exportiere_stay_subject_map():
    """stay_id -> subject_id for the grouped split (origin: exportiere_stay_subject_map.py).

    dataset.npz stores stay_ids but no subject_ids. The map is taken from the same
    source build_dataset.py used (mimic_cohort.load_labels(), the filtered cohort).
    Checks (any violation aborts):
      1  every stay_id of the dataset (train and test) has exactly one subject_id
      2  no conflict: a stay_id never maps to two patients
      3  train and test patients are disjoint (existing outer split)
      4  row count = 29,350 + 7,298 = 36,648
    """
    # load_labels: defined above in this module (section data/mimic_cohort.py)

    if not config.DATASET_MIMIC.exists():
        abbruch(f"{config.DATASET_MIMIC} not found.")
    d = np.load(config.DATASET_MIMIC, allow_pickle=True)
    labels = load_labels()
    zu = dict(zip(labels["stay_id"], labels["subject_id"]))
    if len(zu) != len(labels):
        raise SystemExit("ABORT (2): duplicate stay_id in load_labels().")

    zeilen, patienten = [], {"train": set(), "test": set()}
    for part in ("train", "test"):
        sids = d[f"stay_ids_{part}"]
        fehlt = [int(s) for s in sids if int(s) not in zu]
        if fehlt:
            raise SystemExit(f"ABORT (1): {len(fehlt)} stay_ids of partition "
                             f"'{part}' without subject_id, e.g. {fehlt[:5]}.")
        for s in sids:
            sub = int(zu[int(s)])
            zeilen.append((int(s), sub, part))
            patienten[part].add(sub)

    schnitt = patienten["train"] & patienten["test"]
    if schnitt:
        raise SystemExit(f"ABORT (3): {len(schnitt)} patients in train AND "
                         f"test -- outer split not patient-level?")
    if len(zeilen) != 36648:
        raise SystemExit(f"ABORT (4): {len(zeilen)} rows instead of 36,648.")

    df = pd.DataFrame(zeilen, columns=["stay_id", "subject_id", "partition"])
    df.to_csv(MAP_DATEI, index=False)
    print(f"  written: {MAP_DATEI} ({len(df)} rows)")
    print(f"  patients: train {len(patienten['train'])}, "
          f"test {len(patienten['test'])}, intersection 0  "
          f"[outer split confirmed patient-level]")


def main_split(argv=None):
    print("=" * 78)
    print(f"S1  train/validation split v2  (grouped + stratified, seed {SPLIT_SEED})")
    print("=" * 78)

    p_npz = config.DATASET_MIMIC
    p_map = MAP_DATEI
    if not p_npz.exists():
        abbruch(f"{p_npz} not found.")
    if not p_map.exists():
        abbruch(f"{p_map} not found; create it with: python data/split.py --export-map")
    d = np.load(p_npz, allow_pickle=True)
    m = pd.read_csv(p_map)

    ids_tr = np.asarray(d["stay_ids_train"], dtype=np.int64)
    y_tr = np.asarray(d["y_ihm_train"], dtype=np.int64)
    ids_te = np.asarray(d["stay_ids_test"], dtype=np.int64)

    # ---- A: map covers the NPZ partitions, outer split patient-wise ---------
    mt = m[m.partition == "train"]
    me = m[m.partition == "test"]
    if len(m) != 36648 or len(mt) != 29350 or len(me) != 7298:
        abbruch(f"A: map rows {len(m)}/{len(mt)}/{len(me)} instead of "
                "36,648/29,350/7,298.")
    if set(mt.stay_id.astype(np.int64)) != set(ids_tr.tolist()):
        abbruch("A: map training set != dataset.npz stay_ids_train.")
    if set(me.stay_id.astype(np.int64)) != set(ids_te.tolist()):
        abbruch("A: map test set != dataset.npz stay_ids_test.")
    pat_tr = set(mt.subject_id.tolist())
    pat_te = set(me.subject_id.tolist())
    if pat_tr & pat_te:
        abbruch(f"A: {len(pat_tr & pat_te)} patients in train AND test.")
    print(f"  [OK] A  map covers the NPZ (29,350 / 7,298); "
          f"{len(pat_tr)} / {len(pat_te)} patients, intersection 0")

    # ---- patient level: outcome = max over the patient's stays --------------
    y_je_stay = pd.Series(y_tr, index=ids_tr)
    subj_je_stay = mt.set_index("stay_id").subject_id
    subj_je_stay = subj_je_stay.reindex(y_je_stay.index)
    if subj_je_stay.isna().any():
        abbruch("internal inconsistency: stay without subject after reindex.")
    pat_y = y_je_stay.groupby(subj_je_stay).max()

    # ---- stratified, grouped draw (deterministic) ---------------------------
    rng = np.random.default_rng(SPLIT_SEED)
    val_pat = []
    for stratum in (0, 1):
        kandidaten = np.array(sorted(pat_y.index[pat_y == stratum].tolist()),
                              dtype=np.int64)
        rng.shuffle(kandidaten)
        n_val = int(round(VAL_ANTEIL * len(kandidaten)))
        val_pat.append(kandidaten[:n_val])
        print(f"       stratum patient outcome={stratum}: "
              f"{len(kandidaten):>6,} patients -> {n_val:>5,} in validation")
    val_pat = set(np.concatenate(val_pat).tolist())

    teil = np.where(subj_je_stay.isin(val_pat).to_numpy(), "val", "train")
    raus = pd.DataFrame({"stay_id": y_je_stay.index.to_numpy(),
                         "subject_id": subj_je_stay.to_numpy().astype(np.int64),
                         "teil": teil}).sort_values("stay_id").reset_index(drop=True)

    # ---- B/C/D: hard criteria ----------------------------------------------
    if len(raus) != 29350 or raus.stay_id.duplicated().any():
        abbruch("B: assignment does not cover exactly the 29,350 training stays.")
    p_val = set(raus[raus.teil == "val"].subject_id)
    p_trn = set(raus[raus.teil == "train"].subject_id)
    if p_val & p_trn:
        abbruch(f"B: {len(p_val & p_trn)} patients in train AND validation.")
    if (p_val | p_trn) & pat_te:
        abbruch("B: test patient in the inner split.")
    anteil = float((raus.teil == "val").mean())
    if abs(anteil - VAL_ANTEIL) >= 0.03:
        abbruch(f"C: validation stay share {anteil:.4f} outside 0.20 +/- 0.03.")
    ymap = dict(zip(ids_tr.tolist(), y_tr.tolist()))
    y_raus = raus.stay_id.map(ymap)
    prv_t = float(y_raus[raus.teil == "train"].mean())
    prv_v = float(y_raus[raus.teil == "val"].mean())
    if abs(prv_t - prv_v) >= 0.01:
        abbruch(f"D: prevalence difference {abs(prv_t - prv_v):.4f} >= 0.01.")
    print(f"  [OK] B  29,350 stays assigned once; train/validation patients disjoint; "
          f"no test patient")
    print(f"  [OK] C  validation share {anteil:.4f} "
          f"({int((raus.teil == 'val').sum()):,} stays, {len(p_val):,} patients)")
    print(f"  [OK] D  prevalence train {prv_t:.4f} / validation {prv_v:.4f}")

    # ---- checksums and output ----------------------------------------------
    kanon = "".join(f"{int(s)},{t}\n" for s, t in
                    zip(raus.stay_id, raus.teil)).encode()
    zuordnung_sha = sha(kanon)
    ordnung = np.argsort(ids_tr)
    eingaben = {
        "dataset_stay_ids_train_sha256": sha(np.sort(ids_tr).tobytes()),
        "dataset_y_ihm_train_sha256": sha(y_tr[ordnung].tobytes()),
        "stay_subject_map_datei_sha256": sha(open(p_map, "rb").read()),
    }

    ziel_csv = _sicher_ausgabepfad("split_v2.csv")
    if os.path.exists(ziel_csv):
        alt = pd.read_csv(ziel_csv)
        alt_kanon = "".join(f"{int(s)},{t_}\n" for s, t_ in
                            zip(alt.stay_id, alt.teil)).encode()
        if sha(alt_kanon) == zuordnung_sha:
            print(f"\n  [OK] {ziel_csv} already exists with the identical "
                  f"assignment (SHA {zuordnung_sha[:12]}...); frozen "
                  "split unchanged, nothing written.")
            print("=" * 78)
            return
        abbruch(f"{ziel_csv} exists with a DIFFERENT assignment "
                f"(old {sha(alt_kanon)[:12]}..., new {zuordnung_sha[:12]}...). "
                "The frozen split is not overwritten; find the cause "
                "(inputs changed?).")
    with io.open(ziel_csv, "w", encoding="utf-8", newline="") as f:
        raus.to_csv(f, index=False, lineterminator="\n")
    manifest = {
        "erzeugt_am": datetime.datetime.now().isoformat(timespec="seconds"),
        "skript": "data/split.py (port of erzeuge_split_v2.py)",
        "deklaration": "E3, protocol of 2026-08-17 (grouped by subject_id, "
                       "stratified at patient level, one split for all "
                       "seeds and architectures)",
        "split_seed": SPLIT_SEED,
        "val_anteil_soll": VAL_ANTEIL,
        "n_stays": {"train": int((raus.teil == "train").sum()),
                    "val": int((raus.teil == "val").sum())},
        "n_patienten": {"train": len(p_trn), "val": len(p_val)},
        "praevalenz_stays": {"train": round(prv_t, 6), "val": round(prv_v, 6)},
        "zuordnung_sha256": zuordnung_sha,
        "eingabe_pruefsummen": eingaben,
    }
    manifest["ausgabe_ordner"] = AUSGABE_ORDNER
    with io.open(_sicher_ausgabepfad("manifest_split_v2.json"), "w",
                 encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(f"\n  written: {ziel_csv} ({len(raus):,} rows) + manifest")
    print(f"  assignment SHA-256: {zuordnung_sha}")
    print("  A run on any other machine with identical inputs MUST print "
          "exactly this SHA.")
    print("=" * 78)



def build_dataset():
    """dataset.npz from the raw tables, with the control prints of the original build script."""
    X_train_std, X_test_std = baue_datensatz()
    print("\n=== Control ===")
    print("X_train shape:", X_train_std.shape)
    print("X_test shape:", X_test_std.shape)
    print("train mean (should be ~0):", round(float(X_train_std.mean()), 3))
    print("train std (should be ~1):", round(float(X_train_std.std()), 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description="MIMIC-IV: raw tables -> dataset.npz, stay/patient map, frozen split.")
    ap.add_argument("--build", action="store_true", help="build dataset.npz (about 20 min)")
    ap.add_argument("--export-map", action="store_true", help="write stay_subject_map.csv (once, before --split)")
    ap.add_argument("--split", action="store_true", help="frozen train/validation split into 01_split/")
    a = ap.parse_args(argv)
    if not (a.build or a.export_map or a.split):
        ap.print_help()
        return 0
    if a.build:
        build_dataset()
    if a.export_map:
        exportiere_stay_subject_map()
    if a.split:
        main_split()
    return 0


if __name__ == "__main__":
    sys.exit(main())
