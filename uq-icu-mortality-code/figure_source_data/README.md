# Source data of the figures

One CSV per figure of the article (main text: Figure 1 to 4; supplement: Supplementary Figures 1 to 7 and 9),
recorded while the figure scripts of the full package drew the submitted figure files from the results of the
reported run (2026-10-26). The drawing code itself is not part of this package (it belongs to the full package, see the
article's Code availability statement); the values here are what the figures show, so every graph can be
re-plotted from them. Figure 5 and Supplementary Figure 8 are hand-drawn schematics whose numbers are listed in
`Figure_5_SF08_handset_numbers.csv`.

Where a figure summarises patient-level data (distributions, calibration curves, risk-coverage curves), the CSV
holds the summary that is drawn (curve points, bin values, bands), never patient-level values: the PhysioNet data
use agreements forbid their redistribution.

## Columns

| column | meaning |
|---|---|
| `figure` | figure name (file name without `.csv`) |
| `file_stem` | file name under which the figure script saved the figure |
| `panel`, `row`, `col` | index of the panel (axes) in the figure and its position in the panel grid |
| `panel_title`, `xlabel`, `ylabel` | panel title and axis labels as drawn (empty when the figure carries them in the caption) |
| `element` | `line` (x, y of a curve), `marker` (points drawn without a connecting line), `scatter`, `bar` (x = bar centre, y = bar top; `x_low`/`x_high` = left/right edge, `y_low`/`y_high` = bottom/top), `errorbar` (x, y = point; `x_low`/`x_high` or `y_low`/`y_high` = interval), `band` (vertices of a shaded region, in drawing order), `segment` (a line segment from (`x_low`, `y_low`) to (`x_high`, `y_high`)), `rectangle`, `polygon`, `text_data` / `text_axes` (an annotation at x, y in data or axes coordinates; the text in `text`), `xtick` / `ytick` (labels of a categorical axis at the tick position) |
| `series` | legend label of the element (empty for unlabelled helpers such as reference lines) |
| `index` | running index of the point within its element (for bands: `polygon:vertex`) |
| `x`, `y` | the plotted coordinates |
| `x_low`, `x_high`, `y_low`, `y_high`, `width`, `height` | extents of bars, intervals, bands and segments (see `element`) |
| `text` | text of an annotation or tick label |

Values are written in their shortest exact decimal form (Python `repr`). Coordinates are in data units of the
respective panel; categorical axes (Figure 4, Supplementary Figures 2 to 4) are given by the `ytick` / `xtick`
rows.

## Figures and their inputs in the result files of this package

| figure | content | inputs of the figure script (file names of `results/` and of a run) |
|---|---|---|
| Figure_1_calibration | calibration: decile points, logistic recalibration curve and flexible (spline) curve per family and cohort | `kalibrierung_kennzahlen.csv`; patient-level `scores_*.npz` of a run |
| Figure_2_signal_distributions | distributions of the uncertainty signals by cohort | `ddu_seeds_natural.csv`, `mc_seeds_natural.csv`, `mc_mi_seeds_natural.csv`; patient-level `scores_*.npz`, `mc_mi_*.npz` |
| Figure_3_deferral | remaining errors against the share of patients deferred, external cohort | `ebene3_natural.csv`, `ebene3_e14.csv` (anchors); patient-level `scores_*.npz`, `mc_mi_*.npz`, `features_*.npz` |
| Figure_4_levels | nine signals x three levels x three families (Table 4 drawn; logit variance with reversed sign) | `ddu_seeds_natural.csv`, `mc_seeds_natural.csv`, `mc_mi_seeds_natural.csv`, `mi_ensemble_natural.csv`, `ebene3_natural.csv`, `ebene3_e14.csv`, `table_3_cells.csv` |
| SF01_prediction_distribution | raw predicted probabilities of both cohorts | patient-level `scores_*.npz` |
| SF02_cohort_separation, SF03_error_detection, SF04_selective_prediction | level 1, 2 and 3 of every signal and family | the same aggregate CSVs as Figure 4 |
| SF05_calibration_equal_width | calibration in ten equal-width bins | patient-level `scores_*.npz` |
| SF06_risk_coverage | risk-coverage curves, external cohort | as Figure 3 |
| SF07_cross_entropy_skill | cross-entropy and skill against the prevalence baseline | `crossentropy_natural.csv` (full package) |
| SF09_validation_curves | validation loss per epoch of the seven deterministic families | the training checkpoints (`val_verlauf`) |
| Figure_5_SF08_handset_numbers | numbers typed into the two schematics | `entropie_dezile.csv`, `fehlerarten_deferral.csv` (full package) |
