"""
prepare_eicu.py -- eICU-CRD input tensor built with the MIMIC-IV per-hour rules

Paper:    Methods, "Data sources and preprocessing"; Supplementary Table 1;
          supplementary sensitivity analysis "pre-harmonization vs. harmonized
          eICU input" (harmonization declared 2026-10-07, DECISIONS.md D1)
Origin:   src/npj_code_package/data/prepare_eicu.py v1.0 / 2026-10-07
Reads:    EICU_DIR/patient, vitalPeriodic, vitalAperiodic, nurseCharting, lab
          (.csv.gz or .csv; credentialed PhysioNet access; patient-level, local only)
          config.DATASET_MIMIC      MIMIC-IV training data; only 'variablen' and 'std' are read
          --legacy-tensor PATH      optional pre-harmonization tensor; read only for the
                                    reproduction check C3 and the change attribution
Writes:   config.DATASET_EICU       (eicu_dataset_harmonized.npz; patient-level, local only)
          config.EICU_MANIFEST      (checksums, counts)
          config.APPENDIX/harmonization/harmonization_report.md and .json (aggregates only)
Checks:   C1 variable order equals MIMIC-IV
          C2 cohort identical to the pre-harmonization tensor (labels bit-identical; only
             with --legacy-tensor)
          C3 the pre-harmonization rules, re-implemented here, reproduce the
             pre-harmonization tensor (bit-identical, or within 1e-4 relative; only
             with --legacy-tensor)
          C4 no missing values; masks binary and shaped like the values
          C5 hours before the first measurement carry the first measured value;
             never-measured variables carry the default in all 48 hours
          C6 pipeline guards of the training pipeline (shape, prevalence window)
          Any failure aborts before the output file is written.
Run:      python data/prepare_eicu.py --self-test   (seconds, no data needed)
          python data/prepare_eicu.py               (full build; 4.5 min on the authors' machine,
                                                     several GB RAM)

--------------------------------------------------------------------------------
THE RULES, AND WHERE THEY COME FROM
--------------------------------------------------------------------------------
Per stay and variable, the MIMIC-IV chain (data/mimic_features.py) does:
  step 2   keep measurements in [admission, admission + 48 h)
  step 3   hour since admission = floor(minutes / 60), 0..47
  step 5   temperature values >= 79 are Fahrenheit -> Celsius          (unit rule)
  step 5b  FiO2 values > 1 are percent -> fraction                     (unit rule)
  step 6   MEAN of all raw values of the variable within the hour,
           pooling every source (item ID) of that variable            (rule R3)
  step 7b  plausibility limits applied TO THE HOURLY MEAN; outside -> missing (rule R2)
  step 8   forward fill, then backward fill along the 48 hours        (rule R1)
  step 10  variables never measured in the window -> default value
The pre-harmonization eICU chain differed in
  R2  limits applied to raw values before averaging,
  R3  sources combined by preference (arterial before cuff, vitalPeriodic
      before nurseCharting) instead of by averaging,
  R1  forward fill only; hours before the first measurement got the default.
This script implements both rule sets. "legacy" reproduces the old tensor
(check C3) and serves as the baseline of the change attribution; "harmonized"
is written to disk. Everything else -- cohort, sources, mappings, limits,
defaults, variable order -- is unchanged and imported from the same
feature_mapping.py the MIMIC chain uses.

Measurement masks: 1 = the stay has at least one raw value of the variable in
this hour and the resulting hourly value is within the limits. The model
pipeline never loads masks; they serve the checks and the audit scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data.feature_mapping import (  # noqa: E402
    FINALE_VARIABLEN, GRENZEN, IMPUTE_WERTE, fahrenheit_zu_celsius)

VERSION = "1.0 / 2026-10-07"
RESULTS = config.APPENDIX / "harmonization"
WINDOW_HOURS = 48
MIN_LOS_HOURS = 48
MIN_AGE = 18
FAHRENHEIT_MIN = 79.0        # MIMIC chain, step 5: temperature values >= 79 are Fahrenheit
CHUNK_ROWS = 2_000_000
VARIABLES = list(FINALE_VARIABLEN)
REL_TOL_C3 = 1e-4            # tolerance for C3 if the pandas version differs from the 2026-06 build
PREVALENCE_WINDOW = (0.105, 0.115)   # guard of the training pipeline's eICU loader

# Raw sources per variable: (table, offset column, value column, row filter).
# Order = preference order of the pre-harmonization chain (first source wins
# under preference pooling). Static variables are handled separately.
SOURCES = {
    "heart_rate":        [("vitalPeriodic", "observationoffset", "heartrate", None)],
    "respiratory_rate":  [("vitalPeriodic", "observationoffset", "respiration", None)],
    "bp_systolic":       [("vitalPeriodic", "observationoffset", "systemicsystolic", None),
                          ("vitalAperiodic", "observationoffset", "noninvasivesystolic", None)],
    "bp_diastolic":      [("vitalPeriodic", "observationoffset", "systemicdiastolic", None),
                          ("vitalAperiodic", "observationoffset", "noninvasivediastolic", None)],
    "bp_mean":           [("vitalPeriodic", "observationoffset", "systemicmean", None),
                          ("vitalAperiodic", "observationoffset", "noninvasivemean", None)],
    "oxygen_saturation": [("vitalPeriodic", "observationoffset", "sao2", None)],
    "temperature":       [("vitalPeriodic", "observationoffset", "temperature", None),
                          ("nurseCharting", "nursingchartoffset", "nursingchartvalue",
                           ("nursingchartcelltypevalname", ["Temperature (C)"]))],
    "fio2":              [("lab", "labresultoffset", "labresult", ("labname", ["FiO2"]))],
    "glucose":           [("lab", "labresultoffset", "labresult",
                           ("labname", ["glucose", "bedside glucose"]))],
    "ph":                [("lab", "labresultoffset", "labresult", ("labname", ["pH"]))],
    "gcs_total":         [("nurseCharting", "nursingchartoffset", "nursingchartvalue",
                           ("nursingchartcelltypevalname", ["GCS Total"]))],
}
STATIC = {"weight": "admissionweight", "height": "admissionheight"}   # patient table, one value per stay
TABLE_COLUMNS = {
    "vitalPeriodic": ["patientunitstayid", "observationoffset", "heartrate", "sao2", "respiration",
                      "temperature", "systemicsystolic", "systemicdiastolic", "systemicmean"],
    "vitalAperiodic": ["patientunitstayid", "observationoffset", "noninvasivesystolic",
                       "noninvasivediastolic", "noninvasivemean"],
    "nurseCharting": ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevalname",
                      "nursingchartvalue"],
    "lab": ["patientunitstayid", "labresultoffset", "labname", "labresult"],
}
TABLE_OFFSET = {"vitalPeriodic": "observationoffset", "vitalAperiodic": "observationoffset",
                "nurseCharting": "nursingchartoffset", "lab": "labresultoffset"}


# ------------------------------------------------------------------ utilities
def abort(message, code=1):
    print("ABORT (fail-closed): " + message)
    raise SystemExit(code)


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def table_path(eicu_dir, name):
    for k in (name + ".csv.gz", name + ".csv"):
        p = os.path.join(eicu_dir, k)
        if os.path.exists(p):
            return p
    abort(f"{name}.csv(.gz) not found in {eicu_dir}", 2)


# -------------------------------------------------------------------- cohort
def load_cohort(eicu_dir):
    """Cohort and labels from patient.csv. Same filters, same order as the
    pre-harmonization chain: known hospital discharge status, age >= 18, ICU
    stay >= 48 h, exactly one ICU stay per hospital stay -- counted after the
    other filters, as stated in the Methods and Supplementary Table 2
    (deliberately not changed here)."""
    cols = ["patientunitstayid", "patienthealthsystemstayid", "age", "gender",
            "unitdischargeoffset", "hospitaldischargestatus", "admissionheight", "admissionweight"]
    pat = pd.read_csv(table_path(eicu_dir, "patient"), usecols=cols)

    def parse_age(a):
        a = str(a).strip()
        if a in ("", "nan"):
            return np.nan
        if ">" in a:                     # "> 89" is coded as 90
            return 90
        try:
            return int(float(a))
        except ValueError:
            return np.nan

    pat["age_num"] = pat["age"].map(parse_age)
    pat["los_days"] = pat["unitdischargeoffset"] / 1440.0
    status = pat["hospitaldischargestatus"].astype(str).str.strip().str.lower()
    n_all = len(pat)
    pat = pat[status.isin(["alive", "expired"])].copy()      # unknown outcome is not "survived"
    pat["ihm"] = (pat["hospitaldischargestatus"].astype(str).str.strip().str.lower()
                  == "expired").astype(int)
    n_status = len(pat)
    pat = pat[pat["age_num"] >= MIN_AGE]
    n_age = len(pat)
    pat = pat[pat["unitdischargeoffset"] >= MIN_LOS_HOURS * 60]
    n_los = len(pat)
    single = pat.groupby("patienthealthsystemstayid")["patientunitstayid"].transform("count") == 1
    pat = pat[single]
    log(f"cohort: {n_all:,} stays -> known status {n_status:,} -> age >= {MIN_AGE}: {n_age:,} "
        f"-> ICU stay >= {MIN_LOS_HOURS} h: {n_los:,} -> one ICU stay per hospital stay: {len(pat):,} "
        f"| in-hospital mortality {pat['ihm'].mean():.4f}")
    return pat[["patientunitstayid", "los_days", "ihm", "admissionheight",
                "admissionweight"]].reset_index(drop=True)


# ----------------------------------------------------------------- raw rows
def load_table(eicu_dir, name, stay_ids):
    """Chunked read restricted to cohort stays and to the 48-hour window.
    Restricting the window here changes nothing: the binning discards these
    rows anyway, and row order is preserved (relevant for check C3)."""
    offset = TABLE_OFFSET[name]
    parts = []
    t0 = time.time()
    for chunk in pd.read_csv(table_path(eicu_dir, name), usecols=TABLE_COLUMNS[name],
                             chunksize=CHUNK_ROWS):
        c = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        c = c[(c[offset] >= 0) & (c[offset] < WINDOW_HOURS * 60)]
        parts.append(c)
    df = (pd.concat(parts, ignore_index=True) if parts
          else pd.DataFrame(columns=TABLE_COLUMNS[name]))
    log(f"{name}: {len(df):,} rows in the window for the cohort ({time.time() - t0:.0f} s)")
    return df


def raw_rows(var, source, tables):
    """Raw values of one source as a frame (stay, offset, value), in file
    order. Unit rule for FiO2 (percent -> fraction) applied here, as in both
    chains (MIMIC step 5b)."""
    table, offset_col, value_col, row_filter = source
    df = tables[table]
    if row_filter is not None:
        col, allowed = row_filter
        df = df[df[col].isin(allowed)]
    value = pd.to_numeric(df[value_col], errors="coerce")
    if var == "fio2":
        value = value.where(value <= 1.0, value / 100.0)
    return pd.DataFrame({"stay": df["patientunitstayid"].to_numpy(),
                         "offset": df[offset_col].to_numpy(),
                         "value": value.to_numpy()})


def fahrenheit_rule(rows):
    """MIMIC step 5: temperature values >= 79 are Fahrenheit -> Celsius."""
    v = rows["value"]
    return rows.assign(value=v.where(~(v >= FAHRENHEIT_MIN), fahrenheit_zu_celsius(v)))


# ------------------------------------------------------------------- binning
def within_limits(s, var):
    lo, hi = GRENZEN[var]
    return s.where((s >= lo) & (s <= hi))


def hourly_mean(rows):
    """MIMIC step 6: mean of all raw values per stay and hour."""
    rows = rows.dropna(subset=["value"])
    if rows.empty:
        return pd.Series(dtype="float64")
    hour = (rows["offset"] // 60).astype(int)
    return rows.assign(hour=hour).groupby(["stay", "hour"])["value"].mean()


def bin_variable(var, sources_rows, filter_stage, pooling):
    """Hourly values of one variable as a Series indexed by (stay, hour).

    sources_rows  raw-row frames, one per source, in preference order
    filter_stage  'raw'  limits on raw values before averaging   (pre-harmonization eICU)
                  'mean' limits on the hourly mean               (MIMIC step 7b, rule R2)
    pooling       'preference' first source with a value in the hour wins (pre-harmonization eICU)
                  'mean'       all sources averaged within the hour       (MIMIC step 6, rule R3)
    """
    if pooling == "mean":
        rows = pd.concat(sources_rows, ignore_index=True)
        if filter_stage == "raw":
            rows = rows.assign(value=within_limits(rows["value"], var))
        s = hourly_mean(rows)
        return within_limits(s, var) if filter_stage == "mean" else s
    out = None
    for rows in sources_rows:
        if filter_stage == "raw":
            rows = rows.assign(value=within_limits(rows["value"], var))
        s = hourly_mean(rows)
        if filter_stage == "mean":
            s = within_limits(s, var)
        out = s if out is None else out.combine_first(s)
    return out


def to_matrix(series, index_of_stay, n):
    """Series (stay, hour) -> float32 matrix (n, 48) with NaN where no value."""
    M = np.full((n, WINDOW_HOURS), np.nan, dtype=np.float32)
    if series is None or len(series) == 0:
        return M
    d = series.reset_index()
    d.columns = ["stay", "hour", "value"]
    row = d["stay"].map(index_of_stay)
    keep = (row.notna() & (d["hour"] >= 0) & (d["hour"] < WINDOW_HOURS)
            & d["value"].notna()).to_numpy()
    d = d[keep]
    M[row[keep].astype(int).to_numpy(), d["hour"].astype(int).to_numpy()] = \
        d["value"].to_numpy(dtype=np.float32)
    return M


def static_matrix(cohort, var):
    """Admission height / weight: one value per stay, within limits,
    repeated over the 48 hours (unchanged from the pre-harmonization chain)."""
    vals = cohort[STATIC[var]].to_numpy(dtype=np.float32)
    lo, hi = GRENZEN[var]
    vals = np.where((vals >= lo) & (vals <= hi), vals, np.nan)
    return np.repeat(vals[:, None], WINDOW_HOURS, axis=1).astype(np.float32)


# ------------------------------------------------------------------- filling
def fill_forward(M):
    """Last observation carried forward along the hour axis. Hours before the
    first observation stay missing."""
    n, t = M.shape
    valid = ~np.isnan(M)
    pos = np.where(valid, np.arange(t)[None, :], 0)
    np.maximum.accumulate(pos, axis=1, out=pos)
    return M[np.arange(n)[:, None], pos]


def fill_backward(M):
    return fill_forward(M[:, ::-1])[:, ::-1]


def fill_default(M, var):
    M = M.copy()
    M[np.isnan(M)] = IMPUTE_WERTE[var]
    return M


def fill_legacy(M, var):
    """Pre-harmonization eICU: forward fill, remainder default."""
    return fill_default(fill_forward(M), var)


def fill_harmonized(M, var):
    """MIMIC step 8 + 10: forward fill, backward fill, remainder default."""
    return fill_default(fill_backward(fill_forward(M)), var)


def change_magnitude(a, b, sd):
    """Size of the differences between two filled matrices over the cells that
    differ, in units of the MIMIC training-pool SD of the variable. A cell
    counts as changed at any non-zero difference, so the magnitude is needed to
    read the counts."""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    d = d[d > 0] / sd
    if d.size == 0:
        return {"mean_abs_change_sd": 0.0, "p95_abs_change_sd": 0.0}
    return {"mean_abs_change_sd": float(d.mean()), "p95_abs_change_sd": float(np.percentile(d, 95))}


# --------------------------------------------------------------------- build
def build(cohort, tables, compare_legacy, mimic_sd):
    """Builds the harmonized tensor and, cumulatively, the attribution of
    changed cells to R2, R3 and R1 (counts and magnitudes). Returns tensors,
    masks and the per-variable report. The legacy tensor is built only when
    compare_legacy is set. mimic_sd: per-variable SD of the MIMIC training pool,
    the unit in which change magnitudes are reported."""
    n = len(cohort)
    index_of_stay = {sid: i for i, sid in enumerate(cohort["patientunitstayid"])}
    X_new = np.full((n, WINDOW_HOURS, len(VARIABLES)), np.nan, dtype=np.float32)
    M_new = np.zeros_like(X_new)
    X_old = np.full_like(X_new, np.nan) if compare_legacy else None
    M_old = np.zeros_like(X_new) if compare_legacy else None
    report = {}
    for j, var in enumerate(VARIABLES):
        t0 = time.time()
        if var in STATIC:
            S = static_matrix(cohort, var)
            binned = {"legacy": S, "r2": S, "r23": S}
            raw_stats = {"raw_values": int(np.isfinite(cohort[STATIC[var]].to_numpy(dtype=float)).sum()),
                         "raw_outside_limits": int((np.isfinite(cohort[STATIC[var]].to_numpy(dtype=float))
                                                    & np.isnan(S[:, 0])).sum()),
                         "fahrenheit_converted": 0}
        else:
            srcs = [raw_rows(var, s, tables) for s in SOURCES[var]]
            if var == "temperature":
                n_f = int(sum((r["value"] >= FAHRENHEIT_MIN).sum() for r in srcs))
                srcs_h = [fahrenheit_rule(r) for r in srcs]
            else:
                n_f, srcs_h = 0, srcs
            all_values = pd.concat([r["value"] for r in srcs_h], ignore_index=True).dropna()
            lo, hi = GRENZEN[var]
            raw_stats = {"raw_values": int(len(all_values)),
                         "raw_outside_limits": int(((all_values < lo) | (all_values > hi)).sum()),
                         "fahrenheit_converted": n_f}
            binned = {
                "legacy": to_matrix(bin_variable(var, srcs, "raw", "preference"), index_of_stay, n),
                "r2":     to_matrix(bin_variable(var, srcs_h, "mean", "preference"), index_of_stay, n),
                "r23":    to_matrix(bin_variable(var, srcs_h, "mean", "mean"), index_of_stay, n),
            }
            del srcs, srcs_h, all_values
        mask_old = ~np.isnan(binned["legacy"])
        mask_new = ~np.isnan(binned["r23"])
        f_legacy = fill_legacy(binned["legacy"], var)
        f_r2 = fill_legacy(binned["r2"], var)
        f_r23 = fill_legacy(binned["r23"], var)
        f_new = fill_harmonized(binned["r23"], var)
        X_new[:, :, j], M_new[:, :, j] = f_new, mask_new
        if compare_legacy:
            X_old[:, :, j], M_old[:, :, j] = f_legacy, mask_old
        changed_total = f_new != f_legacy
        sd_var = float(mimic_sd[j]) if mimic_sd[j] > 0 else 1.0
        magnitudes = {f"{step}_{k}": v
                      for step, (a, b) in (("R2", (f_r2, f_legacy)), ("R3", (f_r23, f_r2)),
                                           ("R1", (f_new, f_r23)), ("total", (f_new, f_legacy)))
                      for k, v in change_magnitude(a, b, sd_var).items()}
        report[var] = {
            **raw_stats,
            "cells_changed_R2_limits_on_hourly_mean": int((f_r2 != f_legacy).sum()),
            "cells_changed_R3_mean_pooling": int((f_r23 != f_r2).sum()),
            "cells_changed_R1_backward_fill": int((f_new != f_r23).sum()),
            "cells_changed_total": int(changed_total.sum()),
            "stays_changed_total": int(changed_total.any(axis=1).sum()),
            # R2 also moves hours between "measured" and "missing": an hour whose mean
            # is outside the limits is lost; an hour that had only out-of-range raw
            # values but a plausible mean is gained.
            "hours_lost_R2": int((np.isnan(binned["r2"]) & mask_old).sum()),
            "hours_gained_R2": int((~np.isnan(binned["r2"]) & ~mask_old).sum()),
            **magnitudes,
            "share_never_measured_legacy": float((~mask_old.any(axis=1)).mean()),
            "share_never_measured_harmonized": float((~mask_new.any(axis=1)).mean()),
            "share_stays_with_lead_in": float((mask_new.any(axis=1) & (mask_new.argmax(axis=1) > 0)).mean()),
        }
        log(f"{var:<18} R2 {report[var]['cells_changed_R2_limits_on_hourly_mean']:>9,}  "
            f"R3 {report[var]['cells_changed_R3_mean_pooling']:>9,}  "
            f"R1 {report[var]['cells_changed_R1_backward_fill']:>9,}  "
            f"total {report[var]['cells_changed_total']:>9,} cells, "
            f"mean |change| {report[var]['total_mean_abs_change_sd']:.2f} SD ({time.time() - t0:.0f} s)")
    return X_new, M_new, X_old, M_old, report


# -------------------------------------------------------------------- checks
def check_lead_in(X, M, tol_rel=1e-6):
    """C5: for every (stay, variable) with at least one measurement, all hours
    before the first measured hour equal the value of that hour; for every
    (stay, variable) without measurement, all 48 hours equal the default.
    Returns (violations_lead_in, violations_default)."""
    n, t, v = X.shape
    hours = np.arange(t)[None, :]
    bad_lead, bad_default = 0, 0
    for j, var in enumerate(VARIABLES):
        m = M[:, :, j] > 0.5
        x = X[:, :, j]
        measured = m.any(axis=1)
        t0 = m.argmax(axis=1)
        first = x[np.arange(n), t0]
        lead = (hours < t0[:, None]) & measured[:, None]
        tol = tol_rel * np.maximum(1.0, np.abs(first))[:, None]
        bad_lead += int((lead & (np.abs(x - first[:, None]) > tol)).sum())
        bad_default += int((~measured[:, None] & (x != IMPUTE_WERTE[var])).sum())
    return bad_lead, bad_default


# ----------------------------------------------------------------- self-test
def self_test():
    """Synthetic check of the binning and filling rules on three constructed
    stays (no data needed). Exits non-zero on the first failed expectation."""
    var, default = "heart_rate", IMPUTE_WERTE["heart_rate"]          # limits (20, 250), default 86
    index_of_stay = {1: 0, 2: 1, 3: 2}
    # stay 1: first value at hour 2 (50); hour 3 has raw values 50 and 5000; hour 5 has 60
    # stay 2: two sources in hour 0 (100 from the preferred source, 120 from the other)
    # stay 3: never measured
    source_a = pd.DataFrame({"stay": [1, 1, 1, 1, 2], "offset": [120, 190, 200, 300, 10],
                             "value": [50.0, 50.0, 5000.0, 60.0, 100.0]})
    source_b = pd.DataFrame({"stay": [2], "offset": [20], "value": [120.0]})
    legacy = to_matrix(bin_variable(var, [source_a, source_b], "raw", "preference"), index_of_stay, 3)
    harmon = to_matrix(bin_variable(var, [source_a, source_b], "mean", "mean"), index_of_stay, 3)
    fl, fh = fill_legacy(legacy, var), fill_harmonized(harmon, var)
    expectations = [
        ("legacy: default before the first measurement", fl[0, 0] == default and fl[0, 1] == default),
        ("harmonized (R1): first measured value before the first measurement", fh[0, 0] == 50 and fh[0, 1] == 50),
        ("legacy: raw filter removes 5000, hour 3 mean 50", legacy[0, 3] == 50),
        ("harmonized (R2): hour mean 2525 implausible -> missing -> carried forward (50)",
         np.isnan(harmon[0, 3]) and fh[0, 3] == 50),
        ("both: hour 4 carried forward (50), hour 5 measured (60)",
         fl[0, 4] == 50 and fh[0, 4] == 50 and fl[0, 5] == 60 and fh[0, 5] == 60),
        ("legacy: preference pooling keeps 100", fl[1, 0] == 100 and fl[1, 47] == 100),
        ("harmonized (R3): mean pooling gives 110", fh[1, 0] == 110 and fh[1, 47] == 110),
        ("both: never measured -> default in all hours", (fl[2] == default).all() and (fh[2] == default).all()),
    ]
    # C5 on a full-width synthetic tensor: the test variable in its own channel,
    # every other channel never measured and at its default
    Xs = np.zeros((3, WINDOW_HOURS, len(VARIABLES)), dtype=np.float32)
    Ms = np.zeros_like(Xs)
    for j, v in enumerate(VARIABLES):
        Xs[:, :, j] = fh if v == var else IMPUTE_WERTE[v]
        Ms[:, :, j] = (~np.isnan(harmon)) if v == var else 0.0
    expectations.append(("C5 on the synthetic tensor (lead-in = first value, never measured = default)",
                         check_lead_in(Xs, Ms) == (0, 0)))
    failed = [name for name, ok in expectations if not bool(ok)]
    for name, ok in expectations:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if failed:
        abort(f"self-test failed: {failed}")
    print("self-test passed")


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the eICU-CRD input tensor with the MIMIC-IV per-hour rules (DECISIONS.md D1).")
    ap.add_argument("--eicu-dir", default=str(config.EICU_DIR),
                    help="folder with the eICU-CRD tables (default: config.EICU_DIR / env EICU_DIR)")
    ap.add_argument("--output", default=str(config.DATASET_EICU),
                    help="output tensor (default: config.DATASET_EICU)")
    ap.add_argument("--manifest", default=str(config.EICU_MANIFEST),
                    help="output manifest (default: config.EICU_MANIFEST)")
    ap.add_argument("--legacy-tensor", default=None, metavar="PATH",
                    help="pre-harmonization eICU tensor (not part of the package); enables the "
                         "reproduction check C3, the cohort check C2 and the change attribution baseline")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    ap.add_argument("--self-test", action="store_true",
                    help="run the synthetic self-test of the rules and exit")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    t_start = time.time()
    print("=" * 78)
    print(f"prepare_eicu.py  {VERSION}  --  eICU-CRD with the MIMIC-IV per-hour rules")
    print(f"root:    {config.ROOT}\neICU:    {args.eicu_dir}\noutput:  {args.output}")
    print("=" * 78)
    if os.path.exists(args.output) and not args.force:
        abort(f"{args.output} exists; pass --force to overwrite it")
    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # C1 variable order against the MIMIC training data
    p_mimic = config.DATASET_MIMIC
    if not p_mimic.is_file():
        abort(f"{p_mimic} not found (data/build_dataset.py writes it)", 2)
    mimic = np.load(p_mimic, allow_pickle=True)
    mimic_vars = [str(v) for v in mimic["variablen"]]
    if mimic_vars != VARIABLES:
        abort(f"C1: variable order differs from MIMIC-IV\n  MIMIC: {mimic_vars}\n  here:  {VARIABLES}")
    mimic_sd = np.asarray(mimic["std"], dtype=np.float64)   # scale for reporting change magnitudes
    log("C1 ok: variable order equals MIMIC-IV (13 channels)")

    p_legacy = args.legacy_tensor
    compare_legacy = p_legacy is not None
    if compare_legacy and not os.path.isfile(p_legacy):
        abort(f"--legacy-tensor {p_legacy} not found", 2)
    if not compare_legacy:
        log("no --legacy-tensor given: checks C2/C3 and the change attribution baseline are skipped")

    cohort = load_cohort(args.eicu_dir)
    stay_ids = set(cohort["patientunitstayid"].tolist())
    tables = {name: load_table(args.eicu_dir, name, stay_ids) for name in TABLE_COLUMNS}
    X_new, M_new, X_old, M_old, report = build(cohort, tables, compare_legacy, mimic_sd)
    del tables

    y_los = cohort["los_days"].to_numpy(dtype=np.float32)
    y_ihm = cohort["ihm"].to_numpy(dtype=np.float32)
    stay_arr = cohort["patientunitstayid"].to_numpy(dtype=np.int64)
    checks = {"C1_variable_order": "ok"}

    # C2 / C3 against the pre-harmonization tensor
    if compare_legacy:
        old = np.load(p_legacy, allow_pickle=True)
        if not (np.array_equal(old["y_los_eicu"], y_los) and np.array_equal(old["y_ihm_eicu"], y_ihm)):
            abort("C2: cohort differs from the pre-harmonization tensor (labels not bit-identical)")
        checks["C2_cohort_identical"] = "ok"
        old_X = np.asarray(old["X_eicu"], dtype=np.float32)
        exact = bool(np.array_equal(old_X, X_old))
        diff = np.abs(old_X.astype(np.float64) - X_old.astype(np.float64))
        rel_bad = int((diff > REL_TOL_C3 * np.maximum(1.0, np.abs(old_X))).sum())
        masks_equal = bool(np.array_equal(np.asarray(old["X_mask_eicu"]), M_old))
        if rel_bad:
            abort(f"C3: legacy re-implementation does not reproduce {os.path.basename(p_legacy)}: "
                  f"{rel_bad:,} cells beyond {REL_TOL_C3} relative, max |diff| {diff.max():.3e}")
        checks["C3_legacy_reproduction"] = {
            "bit_identical": exact, "max_abs_diff": float(diff.max()),
            "cells_beyond_tolerance": rel_bad, "masks_identical": masks_equal}
        log(f"C3 ok: legacy rules reproduce {os.path.basename(p_legacy)} "
            f"({'bit-identical' if exact else f'max |diff| {diff.max():.2e}'}; masks identical: {masks_equal})")
        del old, old_X, diff

    # C4
    if np.isnan(X_new).any():
        abort(f"C4: {int(np.isnan(X_new).sum()):,} missing values in the harmonized tensor")
    if M_new.shape != X_new.shape or not set(np.unique(M_new).tolist()) <= {0.0, 1.0}:
        abort("C4: masks not binary or not shaped like the values")
    checks["C4_complete_and_masks_binary"] = "ok"
    # C5
    bad_lead, bad_default = check_lead_in(X_new, M_new)
    if bad_lead or bad_default:
        abort(f"C5: {bad_lead:,} lead-in cells without the first value, "
              f"{bad_default:,} never-measured cells without the default")
    checks["C5_lead_in_and_defaults"] = "ok"
    # C6
    prevalence = float(y_ihm.mean())
    if X_new.shape[1:] != (WINDOW_HOURS, len(VARIABLES)):
        abort(f"C6: shape {X_new.shape}")
    if not (PREVALENCE_WINDOW[0] <= prevalence <= PREVALENCE_WINDOW[1]):
        abort(f"C6: prevalence {prevalence:.4f} outside {PREVALENCE_WINDOW}")
    checks["C6_pipeline_guards"] = {"shape": list(X_new.shape), "prevalence": prevalence}
    log("C4-C6 ok")

    # write tensor (same keys as the pre-harmonization file, plus stay IDs)
    np.savez_compressed(args.output, X_eicu=X_new, y_los_eicu=y_los, y_ihm_eicu=y_ihm,
                        variablen=np.array(VARIABLES), X_mask_eicu=M_new, stay_ids_eicu=stay_arr)
    arrays = {"X_eicu": X_new, "y_los_eicu": y_los, "y_ihm_eicu": y_ihm, "X_mask_eicu": M_new,
              "stay_ids_eicu": stay_arr}
    n_cells = int(X_new.size)
    totals = {
        "stays": int(len(cohort)), "cells": n_cells,
        "cells_changed_R2": sum(r["cells_changed_R2_limits_on_hourly_mean"] for r in report.values()),
        "cells_changed_R3": sum(r["cells_changed_R3_mean_pooling"] for r in report.values()),
        "cells_changed_R1": sum(r["cells_changed_R1_backward_fill"] for r in report.values()),
        "cells_changed_total": sum(r["cells_changed_total"] for r in report.values()),
    }
    if compare_legacy:
        totals["stays_changed_total"] = int((X_new != X_old).any(axis=(1, 2)).sum())
    manifest = {
        "file": os.path.basename(args.output), "script": "data/prepare_eicu.py",
        "version": VERSION, "built": datetime.now().isoformat(timespec="seconds"),
        "decision": "DECISIONS.md D1 (declared 2026-10-07)",
        "rules": {"R1": "forward fill, then backward fill, remainder default",
                  "R2": "plausibility limits applied to the hourly mean",
                  "R3": "all sources of a variable averaged within the hour; temperature >= 79 treated as Fahrenheit",
                  "unchanged": "cohort, sources, mappings, limits, defaults, variable order"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "inputs": {name: {"path": table_path(args.eicu_dir, name),
                          "bytes": os.path.getsize(table_path(args.eicu_dir, name))}
                   for name in list(TABLE_COLUMNS) + ["patient"]},
        "arrays": {k: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256_array(a)}
                   for k, a in arrays.items()},
        "variables": VARIABLES, "prevalence_ihm": prevalence, "checks": checks, "totals": totals,
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    write_report(report, totals, checks, manifest, time.time() - t_start)
    log(f"written: {args.output} and {args.manifest}; report in {RESULTS}  ({(time.time() - t_start) / 60:.1f} min)")
    return 0


def write_report(report, totals, checks, manifest, seconds):
    with open(RESULTS / "harmonization_report.json", "w", encoding="utf-8") as f:
        json.dump({"per_variable": report, "totals": totals, "checks": checks,
                   "built": manifest["built"], "version": VERSION}, f, indent=2)
    n_cells = totals["cells"]
    lines = [
        "# Harmonization of the eICU-CRD input tensor with the MIMIC-IV rules",
        "",
        f"Built {manifest['built']} by `{manifest['script']}` {VERSION}; run time {seconds / 60:.1f} min. "
        "Declared before computation: DECISIONS.md D1. Aggregate counts only.",
        "",
        "Changed cells are counted cumulatively in the order R2 -> R3 -> R1: R2 = limits applied to the hourly "
        "mean instead of to raw values (the MIMIC unit rule for temperature, raw values >= 79 treated as "
        "Fahrenheit and converted, enters at this step; column F->C counts the converted raw values); "
        "R3 = sources averaged within the hour instead of preferred; R1 = backward fill before the first "
        "measurement instead of the default. 'Total' compares the harmonized with the pre-harmonization tensor.",
        "",
        "### Counts",
        "",
        "| Variable | Raw values | Outside limits | F->C | R2 cells | R3 cells | R1 cells | Total cells | % of channel | Stays changed % | Never measured, before -> after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ch = n_cells / len(VARIABLES)
    for var, r in report.items():
        lines.append(
            f"| {var} | {r['raw_values']:,} | {r['raw_outside_limits']:,} | {r['fahrenheit_converted']:,} | "
            f"{r['cells_changed_R2_limits_on_hourly_mean']:,} | {r['cells_changed_R3_mean_pooling']:,} | "
            f"{r['cells_changed_R1_backward_fill']:,} | {r['cells_changed_total']:,} | "
            f"{100 * r['cells_changed_total'] / ch:.2f} | "
            f"{100 * r['stays_changed_total'] / totals['stays']:.1f} | "
            f"{100 * r['share_never_measured_legacy']:.1f} % -> {100 * r['share_never_measured_harmonized']:.1f} % |")
    lines += [
        "",
        "### Magnitudes",
        "",
        "Absolute change per changed cell, in units of the MIMIC-IV training-pool SD of the variable: mean and "
        "95th percentile. 'Hours lost / gained' under R2: hours whose mean falls outside the limits (lost) or "
        "that had only out-of-range raw values but a plausible mean (gained).",
        "",
        "| Variable | R2 mean / p95 (SD) | R2 hours lost / gained | R3 mean / p95 (SD) | R1 mean / p95 (SD) | Total mean / p95 (SD) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for var, r in report.items():
        f = lambda s: f"{r[s + '_mean_abs_change_sd']:.3f} / {r[s + '_p95_abs_change_sd']:.3f}"  # noqa: E731
        lines.append(f"| {var} | {f('R2')} | {r['hours_lost_R2']:,} / {r['hours_gained_R2']:,} | "
                     f"{f('R3')} | {f('R1')} | {f('total')} |")
    lines += [
        "",
        f"**Totals:** {totals['stays']:,} stays, {n_cells:,} cells. Changed: R2 {totals['cells_changed_R2']:,}, "
        f"R3 {totals['cells_changed_R3']:,}, R1 {totals['cells_changed_R1']:,}; total "
        f"{totals['cells_changed_total']:,} ({100 * totals['cells_changed_total'] / n_cells:.2f} %)"
        + (f"; stays with at least one changed cell {totals['stays_changed_total']:,} "
           f"({100 * totals['stays_changed_total'] / totals['stays']:.1f} %)." if "stays_changed_total" in totals else "."),
        "",
        "## Checks",
        "",
    ]
    for k, v in checks.items():
        lines.append(f"- {k}: {v if isinstance(v, str) else json.dumps(v)}")
    lines += ["", "## Not changed (database differences, documented in Supplementary Table 1)", "",
              "Sources and mappings of all variables, the cohort rule, plausibility limits, default values "
              "and the variable order are identical to the pre-harmonization chain. GCS remains "
              "`nurseCharting` 'GCS Total'; FiO2 remains `lab` 'FiO2'; height and weight remain the static "
              "admission values.", ""]
    with open(RESULTS / "harmonization_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
