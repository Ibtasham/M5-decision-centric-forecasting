# Review report: what was wrong and what was fixed

Author: Md Ibtashamur Rahman (sole author), Dept. of IPE, SUST.

## A. Errors in the analysis (changed the numbers)

| # | Problem | Fix | Effect |
|---|---|---|---|
| 1 | **Information leak.** Calibration pools, per-series level ratios and the selector's meta-features used statistics computed over the whole panel, including VAL and TEST days. | Pools (`groups_at`) and features (`lf_*`) are computed from data before each origin. Everything was rerun. | The cold-start exception (a pooled empirical rule winning by 17%) disappeared. Meta-classifier training accuracy is 0.990, not 1.000. |
| 2 | **Sensitivity runs under-sized the protection interval.** Changing lead time or review interval did not resize P = L + R for the quantile rules. | `rebuild_pi_q` recomputes all protection-interval quantities. | The "lead 14 d: DDMRP wins" result and the "+496%" review-14 gap were artefacts. Corrected range is 75–307%. |
| 3 | **Bootstrap orientation mislabelled.** The 17.9% / 33.8% figures were changes against the default, but were described as excess/saving in the opposite sense. | `paired_gap` returns both orientations. The paper states step C = +21.8% (cost rises) and step D = −28.9%. | Wording and figure 2 corrected. |
| 4 | **Block lengths differed** (368/145 days vs 364/140 elsewhere). | Blocks aligned: VAL [1465,1829), TEST [1829,1969). | All numbers regenerated. |
| 5 | **Untuned straw man.** P1 used a fixed z and a once-measured σ. | New tuned normal-theory analysis (13 z values, σ refresh, (s,S) structure; selected on VAL). | Headline is now reported as two numbers: +140% untuned, +26% fully tuned. |
| 6 | Smoke test and several scripts had API drift. | Patched; smoke test passes. | None on results. |

## B. Errors in the text

- Placeholders ("Author One", "[to be completed]", CITE-CHECK banners, the team status note) removed. Solo-author title page.
- "Within-rule gain 0.0%" and the "leverage ratio above 100:1" were unstable (near-zero denominator on TEST). Now reported as differences with CIs (0.4% [−1.5, +2.5] on TEST, 3.4% on VAL), and no ratio is quoted.
- "Cannot comply with a 95% fill obligation": false. The untuned family reaches 0.957 fill. Replaced by a like-for-like cost comparison at 95% fill (+130% VAL, +139% TEST).
- "Step E worth 29–32%": actual D→E reduction is 41.5%.
- "17× cost range": actual 12.9×. "58–101 units on hand": actual 63.8 vs 30.2 vs 13.5.
- "The oracle bounds end-to-end methods": it only bounds selection within the 59-pairing menu. Reworded.
- "Error in the literature" claim about the critical ratio softened to a statement of the convention used; the derivation is flagged as first order.
- Mechanism (iii) corrected: daily review shortens the effective exposure at the same S.
- Table cross-references, "nine vs eight strategies", "two distributional forecasters" (it is one trained quantile model plus three calibrated variants), abstract listing 8 of 9 point forecasters, "3,049-series submissions", "forged margin" → "forgone", the 42,840 note (now a clarification), total units 68,156,315, pool count and HW tuning description.
- Disclosed: τ = 0.995 duplicates τ = 0.99 (56 distinct decisions; optimum on the grid edge), quantile monotonisation rates, deterministic lead time, lost sales only, one dataset, LightGBM only, review frequency bundled into "rule", no terminal-inventory credit, short TEST block.

## C. References

Corrected: Kolassa (IJF 32(3), 788–803, and what it actually does), Hyndman 2011 dropped, DeepAR (IJF 36(3)), "Stock & Orlicky" replaced by Ptak & Smith (2016), M5 accuracy paper (IJF 38(4), 1346–1364). Added and verified: Prak et al. 2017, Boylan & Syntetos 2006, Gneiting 2011, Ban & Rudin 2019, Fildes et al. 2022, Kourentzes 2014, Willemain et al. 2004, Syntetos et al. 2015, Miclo et al. 2019, Beutel & Minner 2012, Oroojlooyjadid et al. 2020, Wallström & Segerstedt 2010. Please re-check page ranges against the publisher pages before submission.

## D. Still needs the author

- Corresponding-author e-mail, funding statement, competing-interest statement and any AI-use disclosure required by the target journal.
- Journal name and addressee in the cover letter.
