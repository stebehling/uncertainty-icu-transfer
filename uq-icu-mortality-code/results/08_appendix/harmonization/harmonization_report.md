# Harmonization of the eICU-CRD input tensor with the MIMIC-IV rules

Built 2026-10-09T23:50:35 by `data/prepare_eicu.py` 1.0 / 2026-10-07; run time 4.6 min. Declared before computation: DECISIONS.md D1. Aggregate counts only.

Changed cells are counted cumulatively in the order R2 -> R3 -> R1: R2 = limits applied to the hourly mean instead of to raw values (the MIMIC unit rule for temperature, raw values >= 79 treated as Fahrenheit and converted, enters at this step; column F->C counts the converted raw values); R3 = sources averaged within the hour instead of preferred; R1 = backward fill before the first measurement instead of the default. 'Total' compares the harmonized with the pre-harmonization tensor.

### Counts

| Variable | Raw values | Outside limits | F->C | R2 cells | R3 cells | R1 cells | Total cells | % of channel | Stays changed % | Never measured, before -> after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| heart_rate | 36,437,336 | 4,500 | 0 | 2,439 | 0 | 26,204 | 28,639 | 0.89 | 12.6 | 1.5 % -> 1.5 % |
| respiratory_rate | 32,266,284 | 378,289 | 0 | 101,376 | 0 | 46,594 | 146,931 | 4.57 | 35.2 | 8.4 % -> 8.4 % |
| bp_systolic | 14,030,922 | 16,667 | 0 | 5,718 | 364,044 | 31,927 | 398,311 | 12.40 | 40.0 | 1.5 % -> 1.5 % |
| bp_diastolic | 14,031,667 | 14,846 | 0 | 5,434 | 363,547 | 31,923 | 397,314 | 12.37 | 39.5 | 1.5 % -> 1.5 % |
| bp_mean | 14,129,496 | 46,255 | 0 | 13,609 | 366,960 | 31,791 | 402,633 | 12.53 | 40.0 | 1.5 % -> 1.5 % |
| oxygen_saturation | 33,503,196 | 11,967 | 0 | 5,283 | 0 | 32,810 | 38,002 | 1.18 | 14.3 | 2.2 % -> 2.2 % |
| temperature | 4,810,428 | 3,924 | 84,780 | 18,602 | 185,026 | 83,852 | 282,864 | 8.80 | 48.3 | 2.8 % -> 2.7 % |
| fio2 | 99,610 | 1,758 | 0 | 1,410 | 0 | 169,020 | 170,277 | 5.30 | 28.3 | 51.6 % -> 51.6 % |
| glucose | 700,733 | 151 | 0 | 22 | 0 | 304,408 | 304,427 | 9.48 | 66.9 | 1.7 % -> 1.7 % |
| ph | 108,112 | 8 | 0 | 0 | 0 | 207,938 | 207,938 | 6.47 | 33.5 | 51.1 % -> 51.1 % |
| weight | 65,318 | 25 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.0 | 2.5 % -> 2.5 % |
| height | 66,465 | 210 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.0 | 1.0 % -> 1.0 % |
| gcs_total | 689,232 | 0 | 0 | 0 | 0 | 83,166 | 83,166 | 2.59 | 18.2 | 25.7 % -> 25.7 % |

### Magnitudes

Absolute change per changed cell, in units of the MIMIC-IV training-pool SD of the variable: mean and 95th percentile. 'Hours lost / gained' under R2: hours whose mean falls outside the limits (lost) or that had only out-of-range raw values but a plausible mean (gained).

| Variable | R2 mean / p95 (SD) | R2 hours lost / gained | R3 mean / p95 (SD) | R1 mean / p95 (SD) | Total mean / p95 (SD) |
|---|---:|---:|---:|---:|---:|
| heart_rate | 0.795 / 2.465 | 28 / 0 | 0.000 / 0.000 | 0.808 / 1.935 | 0.806 / 1.963 |
| respiratory_rate | 0.842 / 2.475 | 11,099 / 14 | 0.000 / 0.000 | 0.902 / 2.430 | 0.843 / 2.391 |
| bp_systolic | 1.149 / 3.330 | 47 / 3 | 0.122 / 0.386 | 0.914 / 2.306 | 0.194 / 0.835 |
| bp_diastolic | 1.817 / 5.221 | 39 / 2 | 0.108 / 0.323 | 0.881 / 2.279 | 0.183 / 0.815 |
| bp_mean | 1.901 / 5.640 | 90 / 2 | 0.140 / 0.375 | 0.909 / 2.314 | 0.230 / 1.043 |
| oxygen_saturation | 3.866 / 10.934 | 484 / 0 | 0.000 / 0.000 | 0.940 / 2.520 | 1.328 / 4.914 |
| temperature | 1.186 / 4.853 | 88 / 6,633 | 0.040 / 0.069 | 0.902 / 2.702 | 0.364 / 1.748 |
| fio2 | 0.703 / 2.281 | 68 / 0 | 0.000 / 0.000 | 1.630 / 3.703 | 1.622 / 3.703 |
| glucose | 7.591 / 15.291 | 3 / 0 | 0.000 / 0.000 | 0.654 / 1.932 | 0.655 / 1.932 |
| ph | 0.000 / 0.000 | 0 / 0 | 0.000 / 0.000 | 1.323 / 3.659 | 1.323 / 3.659 |
| weight | 0.000 / 0.000 | 0 / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| height | 0.000 / 0.000 | 0 / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| gcs_total | 0.000 / 0.000 | 0 / 0 | 0.000 / 0.000 | 1.245 / 2.986 | 1.245 / 2.986 |

**Totals:** 66,934 stays, 41,766,816 cells. Changed: R2 153,893, R3 1,279,577, R1 1,049,633; total 2,460,502 (5.89 %).

## Checks

- C1_variable_order: ok
- C4_complete_and_masks_binary: ok
- C5_lead_in_and_defaults: ok
- C6_pipeline_guards: {"shape": [66934, 48, 13], "prevalence": 0.10934652388095856}

## Not changed (database differences, documented in Supplementary Table 1)

Sources and mappings of all variables, the cohort rule, plausibility limits, default values and the variable order are identical to the pre-harmonization chain. GCS remains `nurseCharting` 'GCS Total'; FiO2 remains `lab` 'FiO2'; height and weight remain the static admission values.
