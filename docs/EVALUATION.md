# Evaluation protocol

What was measured, on what, and in what order. The order is the part that matters: an evaluation
protocol decided after seeing the results is not an evaluation.

## The rule

**The test window is scored once.** Every choice — which model, which hyperparameters, which
features, where the threshold sits — is settled inside the development window using expanding-window
cross-validation. Nothing measured on the test window feeds back into a decision made before it.

This is stated plainly because it is easy to violate by accident. Trying a second model after seeing
a disappointing test score, or nudging a threshold because the test recall looked low, converts the
test set into a validation set and the reported number into an optimistic one.

## Splitting

```
|<──────────── development, 80% of the window ────────────>|<── test, 20% ──>|
|  fold 1 train |         validate                          |                |
|      fold 2 train      |      validate                    |                |
|          fold 3 train        |  validate                  |                |
|             fold 4 train           |     validate         |                |
                                                             ^
                                                    scored once, at the end
```

Four expanding-window folds, each training on everything before a boundary and validating on the
block that follows it — the shape of the deployed problem: fit on the past, score the future. A
10-minute purge gap sits after each training edge. Nothing in this dataset carries state across
rows, but the gap makes the guarantee structural rather than dependent on that remaining true.

| | Rows | Frauds | Window |
|---|---|---|---|
| Fold 1 | 16,188 → 56,724 | 73 → 103 | ≤7.7h → 15.4h |
| Fold 2 | 73,534 → 63,640 | 179 → 76 | ≤15.4h → 23.0h |
| Fold 3 | 138,438 → 17,714 | 259 → 78 | ≤23.0h → 30.7h |
| Fold 4 | 157,286 → 52,490 | 337 → 38 | ≤30.7h → 38.4h |
| **Test** | **73,434** | **97** | **38.4h → 48.0h** |

**Why not a single validation slice.** A 20% validation window holds roughly 45 frauds. A threshold
estimated from 45 positives has a confidence interval wide enough to be meaningless. Pooling
out-of-fold predictions across the four folds gives 295 positives to place it with.

**Why deduplication runs before splitting.** 1,081 rows are byte-identical to an earlier row. A
duplicate pair straddling the boundary is the same transaction in both training and test — the model
graded on a row it had already memorised. Removing them costs 0.023 of average precision, which is
the size of the leak.

## Metrics, and why these

**Average precision** (area under the precision-recall curve) is primary. At 0.17% prevalence the
false-positive rate barely moves across the useful range of thresholds, so ROC-AUC compresses large
practical differences into the third decimal place. It is still reported, as a secondary.

**Recall at a fixed alert budget** is the operational metric. A fraud lead asks "we can review a
thousand transactions a day — how many frauds does that buy?" A model with a better PR curve that
only reaches it at a 5% alert rate loses to a worse one that fits inside the budget.

**Brier score** because the threshold is chosen by expected cost, which requires probabilities that
mean what they say rather than scores that merely rank correctly.

**Bootstrap confidence intervals**, stratified by class over 1,000 replicates. With 97 positives in
the test window a point estimate carries a lot of sampling noise; the interval is the difference
between "0.798" and "0.798, and a rerun could plausibly give 0.72". Stratification keeps the
positive count fixed per replicate — unstratified resampling can draw zero positives, and average
precision on such a replicate is undefined.

**Accuracy is not reported.** Predicting "legitimate" for everything scores 99.87% here.

## Threshold selection

Expected cost, over the pooled out-of-fold development predictions:

```
cost = (alerts × review_cost)
     + (missed fraud, at full amount)
     + (flagged legitimate × amount × decline friction)
```

Assumptions: €3 per review, 5% friction on a false decline, 100% recovery on a caught fraud. Varied
across review costs of €1–€30 and frictions of 0–20% — twelve combinations, all of which pick the
**same threshold**, because the alert-rate constraint binds before the cost trade-off does. That is
a result, not an oversight: the operating point is determined by review capacity, and the cost
assumptions are not load-bearing.

**Candidate thresholds are enumerated in the tail.** With a 0.1% budget the interesting thresholds
are the top ~70 scores out of 73,000. An evenly spaced quantile grid of 300 points cannot resolve
that — its finest step moves the alert count by hundreds, so the optimiser silently returns whichever
single point happens to be feasible. This was a real bug; the fix enumerates every distinct score
down to five times the budget and covers the rest coarsely.
`tests/test_cost.py::test_candidate_thresholds_resolve_the_tail` prevents its return.

## What was tried and rejected

Each of these is in the repository, with the script that produced the number.

| Idea | Result | Verdict |
|---|---|---|
| `scale_pos_weight` on XGBoost | AP 0.779 vs 0.794 | rejected |
| `is_unbalance` on LightGBM | AP 0.625 vs 0.790, Brier 4× worse | rejected |
| Isotonic calibration | Brier 0.00074 vs 0.00054, AP −0.06 | rejected |
| Platt calibration | Brier 0.00064 vs 0.00054 | rejected |
| Amount-weighted training | value recall +0.021, AP −0.021, calibration broken | rejected, revisit with real data |
| `amount % 10 == 0` as a feature | lift 1.00×, *p* = 1.00 | rejected |
| Random forest | AP within noise, 61 ms/row, 240 s to fit | rejected on latency |

SMOTE and random undersampling were not tried at all, deliberately: both change the base rate the
model sees, which shifts every predicted probability away from the real world. The system needs
calibrated probabilities.

## Reproducing

```bash
python scripts/download_data.py          # verifies row count, fraud count and checksum
make all
```

Every figure and table in the README is written by these scripts into `artifacts/` and
`docs/figures/`. Randomness is seeded (`RANDOM_SEED = 20260907` in `src/backstop/config.py`); the
remaining variation across machines comes from thread scheduling inside XGBoost and is small
relative to the confidence intervals.
