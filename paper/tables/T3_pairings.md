| Pairing | Total cost (USD) | Fill rate | Period CSL | Avg on-hand | Cost, % revenue |
|---|---|---|---|---|---|
| lgbm_q + P3 (s,S) daily @ τ=0.99 | 35,209 | 0.9981 | 0.9942 | 25.7 | 1.42% |
| lgbm_q + P3 (s,S) daily @ τ=0.95 | 43,982 | 0.9881 | 0.9415 | 15.6 | 1.77% |
| lgbm_q + P3 (s,S) weekly @ τ=0.99 | 46,602 | 0.9786 | 0.9460 | 24.0 | 1.88% |
| lgbm_q + P3 (s,S) daily @ τ=0.9 | 56,193 | 0.9773 | 0.8445 | 12.1 | 2.27% |
| lgbm_q + P2 quantile base stock @ τ=0.99 | 60,176 | 0.9931 | 0.9887 | 30.2 | 2.43% |
| lgbm_mean + P1 normal base stock *(accuracy-driven default)* | 84,646 | 0.9562 | 0.7960 | 13.5 | 3.41% |
| hw + P1 normal base stock *(best P1 forecaster on TEST)* | 84,306 | 0.9581 | 0.7775 | 14.1 | 3.40% |
| naive + P1 normal base stock *(worst)* | 121,427 | 0.9259 | 0.6455 | 18.8 | 4.89% |
