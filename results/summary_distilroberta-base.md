# Summary

## Evaluation-split accuracy

95% intervals: Wilson score for micro, percentile bootstrap over the
162 homographs for macro. The paper reports point estimates only, so
its rows have no interval; they are single numbers on n=1615 and carry
comparable uncertainty.

| system | micro | 95% CI | macro | 95% CI |
|---|---:|---:|---:|---:|
| MLE baseline (ours) | 0.840 | [0.822, 0.857] | 0.841 | [0.810, 0.870] |
| POS-rule baseline (ours) | 0.954 | [0.943, 0.963] | 0.955 | [0.939, 0.969] |
| Frozen probe (ours, distilroberta-base, last4) | 0.989 | [0.983, 0.993] | 0.990 | [0.985, 0.994] |
| Embedded: rules (Gorman et al. 2018) | 0.870 | n/a | 0.867 | n/a |
| Server: rules (Gorman et al. 2018) | 0.890 | n/a | 0.886 | n/a |
| Embedded: ML (Gorman et al. 2018) | 0.926 | n/a | 0.924 | n/a |
| Server: ML (Gorman et al. 2018) | 0.954 | n/a | 0.951 | n/a |
| Server: rules + ML (Gorman et al. 2018) | 0.990 | n/a | 0.990 | n/a |

## Ten homographs where the probe most underperforms the POS baseline

| homograph | type | n_eval | POS baseline | probe | delta |
|---|---|---:|---:|---:|---:|
| associate | Lexical/Morphosyntactic | 10 | 1.000 | 0.900 | -0.100 |
| discharge | Morphosyntactic | 10 | 1.000 | 0.900 | -0.100 |
| abstract | Morphosyntactic | 10 | 1.000 | 1.000 | +0.000 |
| abuse | Morphosyntactic | 10 | 1.000 | 1.000 | +0.000 |
| abuses | Morphosyntactic | 10 | 1.000 | 1.000 | +0.000 |
| addict | Morphosyntactic | 10 | 1.000 | 1.000 | +0.000 |
| advocate | Morphosyntactic | 10 | 1.000 | 1.000 | +0.000 |
| affect | Lexical | 10 | 1.000 | 1.000 | +0.000 |
| affiliate | Morphosyntactic | 10 | 1.000 | 1.000 | +0.000 |
| aggregate | Morphosyntactic | 10 | 1.000 | 1.000 | +0.000 |
