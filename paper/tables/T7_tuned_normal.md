| Configuration | Selected on VAL | TEST cost | Fill | vs default | Excess over cost-optimal |
|---|---|---|---|---|---|
| Accuracy-driven default (untuned) | lgbm_mean + P1 normal base stock | 84,646 | 0.956 | — | +140.4% [+128.8, +153.4] |
| Normal theory, weekly review, sigma fixed, z tuned | tsb, z=5 | 73,921 | 0.978 | -12.7% [-15.7, -9.7] | +109.9% [+99.1, +123.9] |
| Normal theory, weekly review, sigma refreshed, z tuned | lgbm_mean, z=5 | 65,734 | 0.983 | -22.3% [-24.6, -20.2] | +86.7% [+79.6, +95.2] |
| Normal theory, (s,S) daily review, sigma fixed, z tuned | lgbm_mean, z=6 | 46,959 | 0.989 | -44.5% [-46.5, -42.5] | +33.4% [+27.5, +40.3] |
| Normal theory, (s,S) daily review, sigma refreshed, z tuned | lgbm_mean, z=6 | 44,288 | 0.990 | -47.7% [-49.6, -45.8] | +25.8% [+20.2, +32.5] |
| Normal theory, daily review, sigma fixed, z tuned | hw, z=6 | 159,966 | 0.976 | +89.0% [+81.4, +96.4] | +354.3% [+340.9, +369.6] |
| Normal theory, daily review, sigma refreshed, z tuned | lgbm_mean, z=8 | 147,992 | 0.986 | +74.8% [+66.9, +82.5] | +320.3% [+310.8, +330.6] |
| Trained quantile base stock (lgbm_q, P2, tau selected on VAL) | lgbm_q + P2 quantile base stock @ τ=0.99 | 60,176 | 0.993 | -28.9% [-32.0, -25.8] | +70.9% [+68.2, +74.1] |
| Cost-optimal pairing | lgbm_q + P3 (s,S) daily @ τ=0.99 | 35,209 | 0.998 | -58.4% [-60.5, -56.3] | +0.0% [+0.0, +0.0] |
