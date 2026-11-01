# Table 1. Age and sex per cohort part

| Characteristic | MIMIC-IV v3.1 -- training | MIMIC-IV v3.1 -- internal test | eICU-CRD v2.0 -- external test |
|---|---|---|---|
| ICU stays | 23 472 | 7 298 | 66 934 |
| Age, median [IQR] | 67 [56-78] | 67 [56-78] | 66 [54-77] |
| Age, mean (SD) | 65.6 (16.3) | 65.5 (16.4) | 64.3 (16.3) |
| Female, n (%) | 10 225 (43.6) | 3 256 (44.6) | 30 703 (45.9) |
| Male, n (%) | 13 247 (56.4) | 4 042 (55.4) | 36 219 (54.1) |
| Unknown/other, n (%) | 0 (0.0) | 0 (0.0) | 12 (0.0) |

MIMIC-IV caps ages over 89 at 91 for de-identification; eICU-CRD codes them as '> 89', set to 90 here. Age = anchor_age + (year of ICU admission - anchor_year) in MIMIC-IV.
