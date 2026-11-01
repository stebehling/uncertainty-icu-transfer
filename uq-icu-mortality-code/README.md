# Code for "Evaluating uncertainty for detecting performance loss and guiding deferral after model transfer between hospital systems"

This is the compact companion code of the article: fourteen Python files that take the raw
PhysioNet tables of MIMIC-IV v3.1 and eICU-CRD v2.0 through preprocessing, model training,
Monte Carlo dropout, deep ensembles and the three evaluation levels, and end with the main-text
tables. It reproduces Tables 1-4 and every RQ1 / RQ2a / RQ2b value of the main text at the
prevalence threshold 0.1288. It does not draw the figures and does not contain the supplementary
analyses (constrained encoder variants, sensitivity analyses, supplementary tables and figures);
the complete package that produces those is not part of this repository. The numerical values
behind every figure of the article and the supplement are included as source data
(`figure_source_data/`, see below).

`results/` holds the selected hyperparameters and the aggregate outputs of the reported run, so
that a new run needs no hyperparameter search and can be compared with the article file by file.
No patient-level data and no trained model is included.

Version 1.1 (2026-10-26). Changes against version 1.0 (2026-10-10): the Monte Carlo dropout
measures are sampled from the trained models themselves (section "Monte Carlo dropout" below); the
three separately trained MC-dropout model families with a heteroscedastic output head of version
1.0 and their hyperparameter files are gone, so three families are trained instead of six; the
Monte Carlo rows of `results/` changed accordingly, every other result file is unchanged; the
figure source data were added.

## Install and run one command

The repository folder `uq-icu-mortality-code/` contains the fourteen Python files (nine at the top
level, four in `data/`, two in `models/`), this README, `requirements.txt`, `results/` and
`figure_source_data/`. They belong together in one folder; do not move single files out of it.

1. Clone the repository into a project folder, for example `uq-study/`:
   `git clone https://github.com/stebehling/uncertainty-icu-transfer uq-study` (or unpack the
   release archive into an empty `uq-study/`). You get `uq-study/uq-icu-mortality-code/` with the
   files above. Everything the code produces (the two tensors, `rerun_natural/` with all outputs)
   is written next to that folder, into `uq-study/`, unless you set `UQ_ROOT` to another working
   directory. Run every command below from inside `uq-icu-mortality-code/`.
2. Get credentialed access to [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) and
   [eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/) on PhysioNet and download the
   compressed CSV tables (MIMIC-IV: `icu/icustays`, `icu/chartevents`, `hosp/admissions`,
   `hosp/patients`, `hosp/omr`; eICU-CRD: `patient`, `vitalPeriodic`, `vitalAperiodic`,
   `nurseCharting`, `lab`).
3. Install Python 3.11 and the packages in the versions that produced `results/` (the plain PyPI
   wheel of `torch` is CPU-only; install the CUDA build matching your driver first, the build used
   here was CUDA 12.8):

   ```
   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
   pip install numpy==2.3.1 pandas==2.3.1 scipy==1.17.1 scikit-learn==1.9.0 optuna==4.9.0
   ```

   The same pins are listed in `requirements.txt`; after the `torch` line above,
   `pip install -r requirements.txt` installs the remaining packages (the installed CUDA build of
   `torch` satisfies its pin).

   Training, the prediction caches and the Monte Carlo passes use the GPU; everything else runs on
   a CPU.

4. Tell the code where the raw tables are (defaults: `uq-study/physionet/mimiciv/3.1` and
   `uq-study/physionet/eicu-crd/2.0`, i.e. below the working directory):

   ```
   set MIMIC_DIR=D:\physionet\mimiciv\3.1        (PowerShell: $env:MIMIC_DIR = "D:\physionet\mimiciv\3.1")
   set EICU_DIR=D:\physionet\eicu-crd\2.0
   ```

5. Check the installation without any data (under a minute, prints `PASS` per check, 28 checks):

   ```
   python checks.py smoke
   ```

6. Run everything in the correct order with one command:

   ```
   python run_pipeline.py --data --all
   ```

   `--data` builds `dataset.npz` and `eicu_dataset_harmonized.npz` from the raw tables, the
   stay/patient map, the frozen split and Tables 1-2. `--all` checks the inputs, trains the ten
   seeds of the three families with the shipped hyperparameters (`results/02_suchen/beste_hp_*.json`
   are picked up automatically; delete them to repeat the Optuna search, hours per family), writes
   the prediction caches and the uncertainty measures, runs the Monte Carlo passes, evaluates
   performance and the three levels, runs the deep ensembles, and writes Tables 3 and 4. Every
   product that already exists is kept, so the command can be interrupted and restarted.

7. Compare your run with the shipped results (CSV cell-wise, Markdown line-wise):

   ```
   python checks.py compare
   ```

Run time on the authors' machine (Windows 11, NVIDIA RTX 3060): data about 25 min (CPU), training
about 40 min for the three families (GPU; version 1.0 measured about 1.3 h for its six), prediction
caches about 10 min (GPU), measures about 5 min (CPU), Monte Carlo passes 27 min (GPU), evaluation,
ensembles and tables about 1 min; about 2 h in total without the hyperparameter search.

## The fourteen files

| File | Role | Output |
|---|---|---|
| `config.py` | paths (`UQ_ROOT`, `MIMIC_DIR`, `EICU_DIR`), stage folders, seeds 42-51, threshold 0.1288, Monte Carlo constants, families, glossary | — |
| `common.py` | data loading (frozen split, eICU with the MIMIC statistics), model factory, training loop, logging | — |
| `run_pipeline.py` | the stages in order; `--data`, `--all` or single stages | everything below `rerun_natural/` |
| `data/feature_mapping.py` | the thirteen variables, item IDs, plausibility limits, clinical defaults | — |
| `data/mimic_dataset.py` | MIMIC-IV: cohort, height, 48 × 13 features, patient-level split, standardisation; stay map; frozen split | `dataset.npz`, `01_split/` |
| `data/prepare_eicu.py` | eICU-CRD with the same per-hour rules as MIMIC-IV, with checksums and a change report | `eicu_dataset_harmonized.npz` |
| `data/cohort_tables.py` | cohort cascades and baseline characteristics of both databases | Tables 1-2 (`08_appendix/table_1_*`) |
| `models/architectures.py` | GRU, LSTM, Transformer; the loss; the Monte Carlo dropout passes of a trained model | — |
| `models/training.py` | Optuna search (30 trials) and the ten seeds per family | `02_suchen/`, `03_modelle/` |
| `uncertainty.py` | representation density, prediction caches, level-1 and level-2 measures of density and entropy | `04_vorhersagen/`, `05_signale/` |
| `mc_dropout.py` | 30 Monte Carlo passes of every trained model: logit variance, U_total / U_alea / MI, levels 1 and 2 | `mc_seeds_natural.csv`, `mc_mi_seeds_natural.csv` |
| `ensembles.py` | deep ensembles of the ten seeds, levels 1 and 2 | `mi_ensemble_natural.csv` |
| `evaluation.py` | AUROC / ECE / temperature, calibration slope, AUPRC, ensemble performance, level 3, Tables 3-4 | `06_report/*.csv`, `08_appendix/table_3*.md` |
| `checks.py` | smoke test on synthetic data; comparison of a run with `results/` | — |

Every file starts with a header naming the manuscript element it serves, the script of the full
package it comes from, what it reads and writes, and how it is run. Identifiers, column names and
output file names are those of the original study (German); the glossary below translates them.

## Monte Carlo dropout

The Monte Carlo dropout measures come from the trained deterministic models themselves
(`mc_dropout.py`): every dropout layer that was active in training is switched on at prediction
time (the input, inter-layer and output dropout of the recurrent models; every dropout of the
Transformer), the model is run 30 times per patient and cohort with masks drawn after
`torch.manual_seed(42)` per cohort, and the measures are formed on the probability scale exactly
as for the deep ensembles: predictive entropy U_total of the mean probability, aleatoric part
U_alea (mean entropy of the passes), mutual information MI = U_total − U_alea, and the variance of
the 30 logits per patient (`mc_epistemisch`). Level 2 and 3 score these measures against the errors
of the same model. No model is retrained for this; the stage runs after the measures and before the
evaluation, because level 3 reads the logit variance it adds to the scores files.

## Where the paper's numbers are

| Manuscript | File of the run |
|---|---|
| Tables 1-2 | `08_appendix/table_1_cohort_flow.md`, `table_1_demographics.md` |
| Table 3 | `08_appendix/table_3.md` (from `06_report/leistung_natural.csv`, `auprc_natural.csv`, `kalibrierung_kennzahlen.csv`, `08_appendix/table_2_ensemble_rows.csv`) |
| Table 4 | `08_appendix/table_3_cells.md` (file name of the original run; nine measures × three levels × three families) |
| RQ1 cohort separation, RQ2a error detection (every measure, seed and threshold) | `05_signale/ddu_seeds_natural.csv`, `mc_seeds_natural.csv`, `06_report/mc_mi_seeds_natural.csv`, `mi_ensemble_natural.csv` |
| RQ2b selective prediction | `06_report/ebene3_natural.csv`, `ebene3_e14.csv` |
| Figures 1-4, Supplementary Figures 1-7 and 9 | `figure_source_data/<figure>.csv` (shipped; the plotted values, see below) |

## Figure source data

`figure_source_data/` holds one CSV per figure of the article and the supplement with the values
that are drawn: curve points, markers, bars, error bars, shaded bands, annotations and the labels
of categorical axes, one row per plotted point (columns described in `figure_source_data/README.md`).
They were recorded while the figure scripts of the full package drew the submitted figure files
from the results of the reported run, so every graph can be re-plotted from them without the
drawing code. Figures that summarise patient-level data (distributions, calibration and
risk-coverage curves) are represented by the summary that is drawn, never by patient-level values.
The two hand-drawn schematics (Figure 5, Supplementary Figure 8) have their numbers in
`Figure_5_SF08_handset_numbers.csv`.

## Glossary

`familie` model family · `arch` architecture · `seed` training seed · `domaene` cohort (`in_domain` =
MIMIC-IV test set, `transfer` = eICU-CRD) · `ebene1_ood` level 1 cohort separation · `ebene2_fehler`
level 2 error detection · `ebene3` level 3 selective prediction · `schwelle` classification threshold ·
`wert` value · `signal` uncertainty measure (column name) · `softmax_entropie` predictive entropy · `dichte`
representation density · `mc_epistemisch` logit variance across the Monte Carlo dropout passes ·
`mi_gesamt` / `mi_aleatorisch` / `mi_epistemisch` total / aleatoric / epistemic (MI) uncertainty
(Monte Carlo passes in `mc_mi_seeds_natural.csv`, deep ensembles in `mi_ensemble_natural.csv`) ·
`variante` = `mittel` the probability-scale decomposition (the only variant in this version) ·
`fit_menge` density fitting set · `kovarianz` covariance (per class or `tied`) · `steigung`
calibration slope · `achsenabschnitt` intercept · `ece` expected calibration error · `temperatur`
temperature · `aurc` area under the risk-coverage curve · `eaurc` excess AURC · `risiko80` risk at
80 % coverage · `praevalenz` prevalence · `leistung` performance · `vorhersagen` predictions ·
`suchen` search · `modelle` models · `beste_hp` best hyperparameters · `teil` split part (`train` /
`val`) · `ihm` in-hospital mortality · `det` deterministic model (the suffix of the family and cache
file names; the only variant) · `T1` Transformer.

## Platform notes

- Every training run seeds Python, NumPy and PyTorch and sets cuDNN to its deterministic mode. On
  the authors' machine, retraining reproduced the original checkpoints bit for bit; other GPUs or
  library versions can differ in the last bits, and `checks.py compare --tolerance 1e-6` then judges
  the aggregates within that tolerance.
- The Monte Carlo passes draw their dropout masks from torch's global random number generator,
  seeded immediately before the passes of each cohort; on the authors' machine a second run
  reproduced the passes bit for bit (the smoke test checks this on the synthetic data).
- Checkpoints are never overwritten; a completed search is frozen; every stage resumes from
  existing files. Checkpoints are loaded with `torch.load(..., weights_only=False)` (plain
  dictionaries with the state_dict, the hyperparameters and the validation history).
- On Windows a process that ran a cuDNN GRU/LSTM can end with exit code `0xC0000409` when the CUDA
  libraries are unloaded at process exit, after all outputs were written; the printed verdict is
  the result.

## Licence

MIT License. Copyright (c) 2026 Stefan Behling.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of the Software. THE
SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This licence covers the code only. MIMIC-IV and eICU-CRD are available on PhysioNet under their
own data use agreements; no patient-level data are part of this repository. Please cite the article.
