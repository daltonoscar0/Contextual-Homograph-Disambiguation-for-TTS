# Summary

## Evaluation-split accuracy

| system | micro | macro |
|---|---:|---:|
| MLE baseline (ours) | 0.840 | 0.841 |
| POS-rule baseline (ours) | 0.954 | 0.955 |
| Frozen BERT probe (ours, last4) | 0.986 | 0.986 |
| Embedded: rules (Gorman et al. 2018) | 0.870 | 0.867 |
| Server: rules (Gorman et al. 2018) | 0.890 | 0.886 |
| Embedded: ML (Gorman et al. 2018) | 0.926 | 0.924 |
| Server: ML (Gorman et al. 2018) | 0.954 | 0.951 |
| Server: rules + ML (Gorman et al. 2018) | 0.990 | 0.990 |

## Ten homographs where the probe most underperforms the POS baseline

| homograph | type | n_eval | POS baseline | probe | delta |
|---|---|---:|---:|---:|---:|
| graduate | Lexical/Morphosyntactic | 10 | 1.000 | 0.800 | -0.200 |
| compress | Lexical | 10 | 1.000 | 0.900 | -0.100 |
| discharge | Morphosyntactic | 10 | 1.000 | 0.900 | -0.100 |
| escort | Morphosyntactic | 10 | 1.000 | 0.900 | -0.100 |
| incline | Lexical | 10 | 1.000 | 0.900 | -0.100 |
| invite | Morphosyntactic | 10 | 1.000 | 0.900 | -0.100 |
| minute | Lexical | 10 | 1.000 | 0.900 | -0.100 |
| perfume | Morphosyntactic | 10 | 1.000 | 0.900 | -0.100 |
| abstract | Morphosyntactic | 10 | 1.000 | 1.000 | +0.000 |
| abuse | Morphosyntactic | 10 | 1.000 | 1.000 | +0.000 |
