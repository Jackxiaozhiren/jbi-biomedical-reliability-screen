# Master results report

## K-resolution (WN18RR TransE spike)
| K | m | floor fraction | pi0 | BH rejected rate | BH threshold | scores/s |
|---|---|---|---|---|---|---|
| 100 | 2924 | 0.510 | 0.228 | 0.558 | 0.0198 | 65928 |
| 500 | 2924 | 0.494 | 0.308 | 0.586 | 0.0279 | 157275 |
| 1000 | 2924 | 0.396 | 0.223 | 0.595 | 0.0290 | 554254 |


## Hetionet audit (J9_K500)

### Claimed vs realized FDR
| method | claimed | realized | precision | recall | coverage |
|---|---|---|---|---|---|
| nominal BH | 0.05 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| calibrated | 0.05 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| calibrated | 0.1 | 0.0859 | 0.9141 | 0.4102 | 4.5% |
| calibrated | 0.2 | 0.2160 | 0.7840 | 0.6323 | 8.0% |
| cost-aware 1:1 | None | 0.2322 | 0.7678 | 0.6474 | 8.4% |
| cost-aware 5:1 | None | 0.0859 | 0.9141 | 0.4102 | 4.5% |
| cost-aware 1:5 | None | 0.4394 | 0.5606 | 0.7998 | 14.2% |

## Realized-FDR (realized_fdr_FB15k237_J9_K500)
| method | claimed | realized | precision | recall | coverage |
|---|---|---|---|---|---|
| nominal BH | 0.05 | 0.0314 | 0.9686 | 0.5485 | 5.7% |
| calibrated | 0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| calibrated | 0.05 | 0.0327 | 0.9673 | 0.5463 | 5.7% |
| calibrated | 0.1 | 0.0893 | 0.9107 | 0.7403 | 8.2% |

## Realized-FDR (realized_fdr_WN18RR_J4_K200)
| method | claimed | realized | precision | recall | coverage |
|---|---|---|---|---|---|
| nominal BH | 0.05 | 0.0083 | 0.9917 | 0.5950 | 12.0% |
| calibrated | 0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| calibrated | 0.05 | 0.0133 | 0.9867 | 0.6916 | 15.0% |
| calibrated | 0.1 | 0.0370 | 0.9630 | 0.7290 | 16.2% |

## Realized-FDR (realized_fdr_WN18RR_J9_K500)
| method | claimed | realized | precision | recall | coverage |
|---|---|---|---|---|---|
| nominal BH | 0.05 | 0.0327 | 0.9673 | 0.4956 | 5.1% |
| calibrated | 0.01 | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| calibrated | 0.05 | 0.0364 | 0.9636 | 0.4934 | 5.1% |
| calibrated | 0.1 | 0.0975 | 0.9025 | 0.5300 | 5.8% |