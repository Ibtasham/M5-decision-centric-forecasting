---
title: "Editor's diagnosis and defence guide"
subtitle: "For the author only. Not part of the submission."
---

# Part 1. Diagnosis before rewriting

## 1.1 What the paper actually asks and claims

**Research question.** If a retailer chooses its forecasting model by out-of-sample accuracy and plugs it into a textbook replenishment rule, how much inventory cost does that cost, and which part of the pipeline (forecast, treatment of uncertainty, policy structure, selection method) causes it?

**What the evidence supports** (M5 panel, 3,999 series, one 140-day test block, simulated lost-sales system):

1. Accuracy-driven default (best-RMSSE forecaster + normal-theory base stock) costs 140.4% more than the best of 56 distinct pairings, CI [128.8, 153.4], with lower fill.
2. Changing only the forecaster inside that rule changes cost by little (0.4% on test, 3.4% on validation). Changing the rule changes it by 58%.
3. A tuned normal-theory rule narrows the gap to 25.8% [20.2, 32.5].
4. Pooled empirical calibration made things worse (+21.8%); the trained quantile model made them better (−28.9%).
5. Per-SKU selection and a meta-classifier lose to one fixed pairing on test (+34.9%, +32.6%).
6. The same pairing is best in every regime cell, and in 8 of 10 environments.

**What the literature supports.** That accuracy is an imperfect proxy for decision value is not new (Gneiting, 2011; Kolassa, 2016; Fildes et al., 2022). The paper's contribution is quantitative and empirical: a decomposition on a public retail panel, a tuned classical baseline, and a test of selection methods. Do not present it as discovering the principle.

## 1.2 Factual or methodological problems (not writing problems)

| # | Issue | Status | What you should do |
|---|---|---|---|
| M1 | Bootstrap resamples series, not time. Intervals reflect variation across series in one 140-day window and say nothing about other periods. The test block is five forecast origins. | Was not stated clearly. Now added to the limitations. | Be ready to say "the intervals are conditional on this window". Consider a rolling-origin replication as future work. |
| M2 | Step C compares two pipelines that differ in estimation method *and* in how daily information is aggregated over the protection interval. The explanation (pooling imports the group's tail) is plausible, supported by the on-hand figure (63.8 units vs 30.2), but not tested. | Reworded as a likely explanation. | Do not defend it as proven. A direct test (train on P-day sums) is the clean follow-up. |
| M3 | "Calibration after the fact raises cost" is true for *our* pooled empirical implementation. Other post-hoc methods (conformal, per-series residual quantiles) were not tried. | Scope narrowed in abstract, highlights and implications. | Never generalise beyond this implementation. |
| M4 | The "oracle headroom" bounds gains from per-SKU selection within the 59-pairing menu only. | Already scoped. | Do not say it bounds end-to-end methods. |
| M5 | The 25.8% residual after tuning depends on the tuning grid and on rules chosen from 13 z values × 9 forecasters × 6 variants on one validation year. | Disclosed. | Know that the grid maximum (z=10) was never selected; chosen z values were 5–8. |
| M6 | The high critical ratio (0.9929) comes from the margin economics (small holding cost versus forgone margin). If a retailer values service less, the cost-optimal rule and the size of the gap change. | Sensitivity runs cover holding rate, cost ratio, penalty multiplier. The sign of the gap is stable; the size ranges 75–307%. | Quote the range, not only 140%. |
| M7 | τ=0.995 gives decisions identical to τ=0.99, so the quantile grid has five effective levels and the optimum is on the upper edge. | Disclosed. | Be able to say that CR=0.9929 lies between the two, so the optimum is probably close. The cause of the duplicate was not investigated. |
| M8 | Lead time is deterministic; demand is independent across series; no stockout substitution; no terminal inventory credit; P3 daily uses S sized for the weekly interval. | Disclosed. | These are modelling choices, not errors. |
| M9 | Sampling: 3,999 series, not 4,000, because per-stratum rounding of the proportional allocation gives 3,635 + 304 + 60. No zero-sales series exist in M5, so none were dropped. | Explained in the text. | |
| M10 | "Within-rule gain" for the forecaster is the *ex post* best on each block, an optimistic figure for forecaster choice. That makes the conclusion conservative. | Stated. | |

## 1.3 Unsupported or risky statements found and what was done

- "Largest forecasting exercise on retail data" (M5): unsupported as stated. Removed the superlative.
- "To our knowledge this ratio has not been measured before": removed. No novelty claim of that kind can be checked here.
- The "leverage ratio above 100:1": unstable (near-zero denominator). Removed in favour of differences with intervals.
- "Cannot comply with a 95% fill obligation": false. Removed.
- Cold-start exception: came from leaked pools. Removed.
- Descriptions of cited papers (Section 2): I wrote them from the titles and my knowledge of the papers. **You must read the abstract of each cited work and confirm the sentence attributed to it.** Highest-risk ones: Kolassa (2016), Fildes et al. (2022), Boylan and Syntetos (2006), Wallström and Segerstedt (2010), Prak et al. (2017), Miclo et al. (2019). I softened the wording but cannot verify it.

## 1.4 Contradictory or confusing numbers checked

Checked against the result files: 140.4% / 58.4% / 0.4% / 21.8% / 28.9% / 25.8% / 34.9% / 32.6% / 19.7% and the sensitivity range 75–307%. They agree across abstract, tables, text and conclusion. Two things that look contradictory but are not: (a) the gap is "+140%" measured as default cost over best cost, while the best saves "58%" measured as a fraction of the default's cost (1 − 1/2.404); (b) 59 pairings but 56 distinct decisions.

## 1.5 Writing problems (the actual revision)

Redundancy between abstract, introduction, results, discussion and conclusion; repeated caveats; bold run-in labels and parallel lists; stock phrases; numbered lists where prose reads better. The revised text keeps every verified number and limitation, and states each caveat once where it matters.

# Part 2. Defence guide: be ready for these questions

Format: *choice, why, what a reviewer will say, short answer.*

**1. Lost sales, not backorders.** Retail shoppers usually leave. Reviewer: "backorders are standard in the literature." Answer: lost sales is the realistic case for shelf retail; it raises the cost of under-stocking, which favours high-service rules. Listed as a limitation.

**2. Margin-based stockout penalty.** Uses only observed price, a purchase-cost ratio and a holding rate. Reviewer: "your penalty ignores goodwill." Answer: true, it is conservative; the multiplier-3 run shows the gap grows (+307%).

**3. Critical ratio π/(π + hR).** Per-day holding means an unneeded unit is carried about one review cycle. First-order result; the simulator does not depend on it exactly because τ is tuned on cost. Do not claim others have the formula wrong.

**4. Protection-interval blend (λ = 0.5).** Summing daily quantiles overstates the tail; independence understates it. The blend is an approximation. Reviewer: "arbitrary." Answer: it is a known source of confound (Section 6.2); the calibrated methods avoid it, and the tuned-normal results do not depend on it.

**5. Time split and leak-free pools.** TRAIN 1,465 days, VAL 364, TEST 140. Everything selected on VAL, reported on TEST. Pools, features and regime labels use only data before each origin. Know the story: an earlier version leaked, you found it, fixed it, and reran everything.

**6. Choice of forecasters.** Standard intermittent-demand methods (Croston, SBA, TSB), seasonal baselines, a combination, damped Holt–Winters, and one global LightGBM, the model family behind the M5 winners. Reviewer: "no deep learning." Answer: the main effects are on the decision side; listed as a limitation.

**7. One LightGBM, trained once per block.** Mirrors periodic retraining. Training used four trailing origins on a 2,321-series subsample because of memory limits.

**8. Why RMSSE defines the "accuracy-driven" default.** It is the scaled squared-error metric used in M5. WAPE/MAE pick `hw` instead; the paper says so. The default's cost is the same order whichever you use (the within-P1 spread is small).

**9. P1 default.** z from the critical ratio, σ from the trailing 56 days at the start of the block. Reviewer: "straw man." Answer: Section 5.3 tunes z, σ refresh and structure on VAL and still finds +25.8%.

**10. Why (s,S) with s = 0.5S.** A simple standard form; sensitivity at 0.3 changes little. Not optimised, so the best rule is, if anything, understated.

**11. Decomposition A→E.** Each step changes one thing: forecaster, uncertainty method, trained quantiles, rule. Steps are not additive and the order is a choice. Step E changes review frequency and trigger together.

**12. Paired bootstrap over series.** Same series, same realised demand. It does not capture variation over time (see M1).

**13. Selection protocols.** S3 shows overfitting (looks −14% on VAL, +34.9% on TEST). S4 uses disjoint halves of VAL. Reviewer: "per-SKU selection could work with better regularisation." Answer: shrinkage was tried; the claim is limited to one validation year and our menu.

**14. Regimes.** Labels from trailing 56 days versus the prior 364. Null result: the same pairing wins everywhere. Weak evidence for the `new` and `vol_up` cells (65 and 145 TEST cells).

**15. Sensitivity suite.** Nine alternatives to the baseline; lead time, review interval, order cost, holding rate, cost ratio, penalty, s fraction, DDMRP buffer. One earlier run produced artefacts (P not resized) and was corrected.

**16. Generalisation.** One dataset, one sector, one ML family. State this plainly; do not claim the numbers transfer.

# Part 3. What only you can check

1. Read the abstract of each cited paper and confirm Section 2 describes it fairly.
2. Add DOIs where the publisher page shows one (I confirmed only four).
3. Confirm the declarations (competing interests, funding, AI use).
4. Open the repository on GitHub and check that the README renders and the release contains what you expect.
