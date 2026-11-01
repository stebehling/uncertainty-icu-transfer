# Table 1. Cohort construction

Steps are listed in the order in which each preprocessing pipeline applied them; the order differs between the two databases and the criterion for single ICU stays per admission is evaluated on the raw data in MIMIC-IV but after the other filters in eICU-CRD (see the last eICU row).

| Cohort | Step | ICU stays | Patients | Removed |
|---|---|---|---|---|
| MIMIC-IV | ICU stays in MIMIC-IV v3.1 | 94 458 | 65 366 |  |
| MIMIC-IV | Admissions with more than one ICU stay removed | 77 547 | 60 757 | 16911 |
| MIMIC-IV | Age below 18 years removed | 77 547 | 60 757 | 0 |
| MIMIC-IV | ICU stay shorter than 48 h removed | 36 648 | 31 459 | 40899 |
| MIMIC-IV | Without complete 48-hour feature window or label removed | 36 648 | 31 459 | 0 |
| MIMIC-IV | Training set | 29 350 | 25 168 |  |
| MIMIC-IV |   of which: model training | 23 472 | 20 134 |  |
| MIMIC-IV |   of which: validation | 5 878 | 5 034 |  |
| MIMIC-IV | Internal test set | 7 298 | 6 291 |  |
| eICU-CRD | ICU stays in eICU-CRD v2.0 | 200 859 | 200 859 |  |
| eICU-CRD | Without a recorded discharge status removed | 199 108 | 199 108 | 1751 |
| eICU-CRD | Age below 18 years removed | 198 490 | 198 490 | 618 |
| eICU-CRD | ICU stay shorter than 48 h removed | 76 953 | 76 953 | 121537 |
| eICU-CRD | Admissions with more than one ICU stay removed | 66 934 | 66 934 | 10019 |
| eICU-CRD | Without usable measurements removed | 66 934 |  | 0 |
| eICU-CRD | Of these: admission had another ICU stay in the raw data (would be excluded under the MIMIC-IV rule) | 10 378 |  |  |


## (c) Baseline characteristics, eICU-CRD (reconstructed cohort, anchor D1)

| Characteristic | eICU-CRD external cohort |
|---|---|
| ICU stays | 66 934 |
| Age, median [IQR] | 66 [54-77] |
| Age, mean (SD) | 64.3 (16.3) |
| Female  n (%) | 30 703 (45.9) |
| In-hospital mortality, % | 10.9 |

Ages recorded as '> 89' were set to 90. MIMIC-IV rows from tabelle1_demographie_split.py.
