| Configuration | Best pairing | Best cost | Default cost | Default excess | Within-rule gain | Across-rule gain | Accuracy-best = cost-best in P1? |
|---|---|---|---|---|---|---|---|
| Baseline (L=2, K=$0.50, R=7, hold 25%, cost 60%) | lgbm_q + P3 (s,S) daily @ τ=0.99 | 35,209 | 84,646 | +140.4% | 0.4% | 58.4% | No |
| Lead time 7 d | lgbm_q + P3 (s,S) daily @ τ=0.99 | 45,117 | 99,634 | +120.8% | 2.5% | 54.7% | No |
| Lead time 14 d | lgbm_q + P3 (s,S) daily @ τ=0.99 | 62,752 | 110,037 | +75.4% | 2.2% | 43.0% | No |
| Order cost $5.00 per line | lgbm_q + P3 (s,S) weekly @ τ=0.99 | 139,428 | 367,165 | +163.3% | 2.7% | 62.0% | No |
| Review interval 14 d | lgbm_q + P3 (s,S) daily @ τ=0.95 | 39,246 | 78,248 | +99.4% | 0.7% | 49.8% | No |
| Holding rate 40% | lgbm_q + P3 (s,S) daily @ τ=0.99 | 47,691 | 93,396 | +95.8% | 0.2% | 48.9% | No |
| Purchase cost 75% of price | lgbm_q + P3 (s,S) daily @ τ=0.99 | 39,903 | 73,446 | +84.1% | 0.5% | 45.7% | No |
| Stockout penalty = 3× margin | lgbm_q + P3 (s,S) daily @ τ=0.99 | 37,913 | 154,231 | +306.8% | 0.0% | 75.4% | Yes |
| (s,S) reorder point 0.30×S | lgbm_q + P3 (s,S) daily @ τ=0.99 | 32,471 | 84,646 | +160.7% | 0.4% | 61.6% | No |
| DDMRP buffer 0.80 | lgbm_q + P3 (s,S) daily @ τ=0.99 | 35,209 | 84,646 | +140.4% | 0.4% | 58.4% | No |
