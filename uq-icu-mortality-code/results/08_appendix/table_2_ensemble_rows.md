| Model | AUROC, in-domain | AUROC, transfer | Paired loss | AUPRC lift, in-domain / transfer | Slope, in-domain | Slope, transfer |
|---|---|---|---|---|---|---|
| Deep ensemble of the GRU | 0.877 | 0.777 | 0.100 | 4.22 / 3.54 | 1.01 | 0.67 |
| Deep ensemble of the LSTM | 0.872 | 0.767 | 0.105 | 4.17 / 3.44 | 0.96 | 0.60 |
| Deep ensemble of the Transformer | 0.867 | 0.772 | 0.095 | 3.90 / 3.27 | 0.94 | 0.64 |

Ensemble = mean of the ten seed models' probabilities (6); one ensemble per family, no seed interval. Jackknife ranges (nine-member ensembles) in the CSV.
