# Table 3. Mortality discrimination and calibration in-domain and after external transfer

| Model | AUROC, in-domain | AUROC, transfer | Paired loss [seed interval] | AUPRC lift, in-domain / transfer | Slope, in-domain | Slope, transfer |
|---|---|---|---|---|---|---|
| GRU | 0.873 | 0.773 | 0.100 [0.098; 0.102] | 4.15 / 3.48 | 0.96 | 0.64 |
| LSTM | 0.868 | 0.762 | 0.105 [0.102; 0.109] | 4.10 / 3.38 | 0.92 | 0.57 |
| Transformer | 0.858 | 0.766 | 0.092 [0.090; 0.094] | 3.73 / 3.14 | 0.84 | 0.59 |
| Deep ensemble of the GRU | 0.877 | 0.777 | 0.100 | 4.22 / 3.54 | 1.01 | 0.67 |
| Deep ensemble of the LSTM | 0.872 | 0.767 | 0.105 | 4.17 / 3.44 | 0.96 | 0.60 |
| Deep ensemble of the transformer | 0.867 | 0.772 | 0.095 | 3.90 / 3.27 | 0.94 | 0.64 |

Means over ten training seeds; paired loss = AUROC in-domain minus AUROC transfer per seed, with the 95 % t-interval across the seeds. Deep ensembles: mean-probability ensemble of the ten seed models (performance/ensemble_performance.py), no seed interval.
