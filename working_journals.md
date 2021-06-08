2021-June-07

## Problems: 
if we make each vector to be columns of features of one year for a particular company, then we lose the comparison of that year's record against its previous years to detect the abnormal year over year change

![Abnomaly in Year over Year Change](https://github.com/casualcomputer/accounting_fraud/blob/master/time_series_anomaly.jpg)

## Solutions:
solution 1: instead of just just the features of that year in the vector, we should use year-over-year ratio of the features to compose the vector 
solution 2: treat dataset as panel data/logitudinal
