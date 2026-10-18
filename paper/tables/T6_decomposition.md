| Step | VAL pairing | VAL cost | VAL vs A | TEST pairing | TEST cost | TEST vs A | Fill (TEST) | On-hand (TEST) |
|---|---|---|---|---|---|---|---|---|
| A  Accuracy-driven default | lgbm_mean + P1 normal base stock | 209,719 | — | lgbm_mean + P1 normal base stock | 84,646 | — | 0.956 | 13.5 |
| B  Best point forecaster, same rule | tsb + P1 normal base stock | 202,620 | -3.4% [-4.8, -2.0] | hw + P1 normal base stock | 84,306 | -0.4% [-2.5, +1.5] | 0.958 | 14.1 |
| C  Calibrated quantile rule | cal_comb + P2 quantile base stock @ τ=0.99 | 254,213 | +21.2% [+6.6, +39.4] | cal_comb + P2 quantile base stock @ τ=0.99 | 103,127 | +21.8% [+4.3, +44.6] | 0.957 | 63.8 |
| D  Trained quantile rule | lgbm_q + P2 quantile base stock @ τ=0.99 | 147,690 | -29.6% [-31.7, -27.5] | lgbm_q + P2 quantile base stock @ τ=0.99 | 60,176 | -28.9% [-32.0, -25.8] | 0.993 | 30.2 |
| E  Cost-optimal pairing | lgbm_q + P3 (s,S) daily @ τ=0.99 | 88,253 | -57.9% [-59.3, -56.6] | lgbm_q + P3 (s,S) daily @ τ=0.99 | 35,209 | -58.4% [-60.5, -56.3] | 0.998 | 25.7 |
