"""
feature_mapping.py -- the 13 clinical variables: item IDs, units, GCS, defaults, limits

Paper:    Methods, "Data sources and preprocessing" (variable set, imputation,
          plausibility limits); Supplementary Table 1
Origin:   src/feature_mapping.py (no SKRIPT_STAND; working copy of 2026-10-08)
Reads:    nothing
Writes:   nothing
Checks:   none (library module)
Run:      not a script; imported by the MIMIC-IV chain (mimic_height, mimic_features,
          build_dataset) and by the eICU chain (prepare_eicu)

Basis: the Harutyunyan benchmark (itemid_to_variable_map.csv) transferred to
MIMIC-IV. All item IDs are MetaVision IDs (MIMIC-IV uses the MetaVision schema only).

Defined in ONE place:
  1. POOLING        which item IDs are combined into which variable
  2. UNITS          which variables need a unit conversion
  3. GCS            summed from three components
  4. DEFAULTS       values for variables that are completely missing for a stay
  5. LIMITS         plausibility bounds shared by the MIMIC-IV and the eICU chain

Decisions:
  - blood pressure (sys/dia/mean), SpO2, glucose: POOLED (several sources, as in
    Harutyunyan)
  - temperature, weight, height: unit conversion only, no real pooling
  - pH and respiratory rate: deliberately NOT pooled (arterial vs. venous and
    spontaneous vs. total are clinically different quantities)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402,F401  (package convention; this module needs no path)

# ===========================================================================
# 1. POOLING: itemid -> variable name
#    Several item IDs may map to the same variable name; the aggregation then
#    averages them (= pooling).
# ===========================================================================
ITEMID_MAPPING = {
    # --- heart rate (one source) ---
    220045: "heart_rate",

    # --- respiratory rate (NOT pooled: standard measurement only) ---
    220210: "respiratory_rate",

    # --- systolic blood pressure (POOLED: arterial + non-invasive + manual) ---
    220050: "bp_systolic",   # Arterial Blood Pressure systolic
    225309: "bp_systolic",   # ART BP Systolic
    220179: "bp_systolic",   # Non Invasive Blood Pressure systolic
    224167: "bp_systolic",   # Manual Blood Pressure Systolic Left
    227243: "bp_systolic",   # Manual Blood Pressure Systolic Right

    # --- diastolic blood pressure (POOLED) ---
    220051: "bp_diastolic",  # Arterial Blood Pressure diastolic
    225310: "bp_diastolic",  # ART BP Diastolic
    220180: "bp_diastolic",  # Non Invasive Blood Pressure diastolic
    224643: "bp_diastolic",  # Manual Blood Pressure Diastolic Left

    # --- mean blood pressure (POOLED) ---
    220052: "bp_mean",       # Arterial Blood Pressure mean
    225312: "bp_mean",       # ART BP mean
    224322: "bp_mean",       # IABP Mean
    220181: "bp_mean",       # Non Invasive Blood Pressure mean

    # --- oxygen saturation (POOLED: arterial + pulse oximetry) ---
    220227: "oxygen_saturation",  # Arterial O2 Saturation
    220277: "oxygen_saturation",  # O2 saturation pulseoxymetry

    # --- temperature (one source here; conversion see below) ---
    223761: "temperature",   # Temperature Fahrenheit
    223762: "temperature",   # Temperature Celsius (rare, but present)

    # --- FiO2 (one source) ---
    223835: "fio2",          # Inspired O2 Fraction

    # --- glucose (POOLED: serum + finger stick + whole blood) ---
    220621: "glucose",       # Glucose (serum)
    225664: "glucose",       # Glucose finger stick
    226537: "glucose",       # Glucose (whole blood)

    # --- pH (NOT pooled: arterial only) ---
    223830: "ph",            # PH (Arterial)

    # --- weight (one source here; conversion see below) ---
    224639: "weight",        # Daily Weight (kg)

    # --- height: NOT here. Extracted separately in mimic_height.py
    #     (official MIMIC concept: chartevents 226730 + 226707, plus OMR).
    #     Height is static and does not belong in the 48-hour time-series pooling.
}

# ===========================================================================
# 2. GCS components: MIMIC-IV has no ready-made "total" item.
#    Summed from the three components (range 3 to 15).
# ===========================================================================
GCS_COMPONENTS = {
    220739: "gcs_eye",      # GCS - Eye Opening
    223900: "gcs_verbal",   # GCS - Verbal Response
    223901: "gcs_motor",    # GCS - Motor Response
}

# ===========================================================================
# 3. The 13 final variable names (GCS as one combined variable).
#    This order is the column order of the feature matrix.
# ===========================================================================
FINALE_VARIABLEN = [
    "heart_rate",
    "respiratory_rate",
    "bp_systolic",
    "bp_diastolic",
    "bp_mean",
    "oxygen_saturation",
    "temperature",
    "fio2",
    "glucose",
    "ph",
    "weight",
    "height",
    "gcs_total",
]

# ===========================================================================
# 4. DEFAULT VALUES (Harutyunyan, Table 2).
#    Used when a variable is COMPLETELY missing for a stay (not a single
#    measurement -> forward filling cannot apply).
# ===========================================================================
IMPUTE_WERTE = {
    "heart_rate": 86.0,
    "respiratory_rate": 19.0,
    "bp_systolic": 118.0,
    "bp_diastolic": 59.0,
    "bp_mean": 77.0,
    "oxygen_saturation": 98.0,
    "temperature": 36.6,
    "fio2": 0.21,
    "glucose": 128.0,
    "ph": 7.4,
    "weight": 81.0,
    "height": 170.0,
    "gcs_total": 15.0,
}

# ===========================================================================
# 5. PLAUSIBILITY LIMITS (outlier filter) -- THE single central source.
#    Values outside (low, high) are treated as measurement errors and set to
#    missing (NaN), then imputed like any other gap (first the neighbouring
#    value of the same stay by forward/backward fill, else the default above).
#
#    IMPORTANT: this table is used by BOTH chains, the MIMIC-IV extraction AND
#    the eICU preparation. Both databases therefore receive exactly the same
#    outlier treatment with the same limits; a change here applies to both.
#
#    The limits are deliberately WIDE. They catch real measurement errors (a pH
#    of 999999, an SpO2 of 20000) but not rare, medically real extreme values.
#    All limits apply before any analysis, to all stays alike, independent of
#    any result.
# ===========================================================================
GRENZEN = {
    "heart_rate":        (20, 250),    # beats/min
    "respiratory_rate":  (4, 60),      # breaths/min
    "bp_systolic":       (40, 250),    # mmHg
    "bp_diastolic":      (10, 200),    # mmHg
    "bp_mean":           (20, 220),    # mmHg
    "oxygen_saturation": (50, 100),    # percent (above 100 is physically impossible)
    "temperature":       (25, 45),     # degrees Celsius (core)
    "fio2":              (0.21, 1.0),  # fraction (0.21 = room air, 1.0 = pure O2)
    "glucose":           (10, 1000),   # mg/dL
    "ph":                (6.5, 8.0),   # blood pH (physiologically narrow)
    "weight":            (20, 400),    # kg (adults)
    "height":            (120, 230),   # cm (official MIMIC concept measurement/height.sql)
    "gcs_total":         (3, 15),      # GCS scale (minimum 3, maximum 15)
}


def fahrenheit_zu_celsius(temp_f):
    """Convert Fahrenheit to Celsius: (F - 32) * 5/9."""
    return (temp_f - 32.0) * 5.0 / 9.0


# all item IDs needed from chartevents (variables + GCS components)
ALLE_ITEMIDS = list(ITEMID_MAPPING.keys()) + list(GCS_COMPONENTS.keys())
