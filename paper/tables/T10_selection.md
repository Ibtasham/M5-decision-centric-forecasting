| Strategy | Selection rule | VAL-B cost | TEST cost | vs S1 (VAL-B) | vs S1 (TEST) | Distinct choices |
|---|---|---|---|---|---|---|
| S1 Fixed, pooled best | one pairing, chosen on all of VAL | 48,571 | 35,209 | +0.0% | +0.0% | 1 |
| S2 Fixed, split-half | one pairing, chosen on VAL-A | 48,571 | 35,209 | +0.0% | +0.0% | 1 |
| S3 Per-SKU, all of VAL | argmin per SKU on VAL | 41,772 | 47,481 | -14.0% | +34.9% | 41 |
| S4 Per-SKU, split-half | argmin per SKU on VAL-A | 73,730 | 64,995 | +51.8% | +84.6% | 51 |
| S5 Menu: policy families | per SKU, forecaster fixed | 43,866 | 42,146 | -9.7% | +19.7% | 14 |
| S6 Menu: forecasters | per SKU, policy fixed | 47,240 | 36,983 | -2.7% | +5.0% | 2 |
| S7 Shrinkage λ=0.25 | blend SKU and pooled cost | 47,465 | 36,703 | -2.3% | +4.2% | 5 |
| S7 Shrinkage λ=0.5 |  | 45,972 | 38,660 | -5.4% | +9.8% | 10 |
| S7 Shrinkage λ=0.9 |  | 42,863 | 42,340 | -11.8% | +20.3% | 21 |
| S8 Accuracy-driven default | best RMSSE + P1 | 115,698 | 84,646 | +138.2% | +140.4% | 1 |
| S9 Oracle on VAL-B | *(unattainable)* | 39,495 | 57,407 | -18.7% | +63.0% | 48 |
| S9 Oracle on TEST | *(unattainable)* | 74,590 | 28,275 | +53.6% | -19.7% | 49 |
