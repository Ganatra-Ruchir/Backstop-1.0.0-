# Model card — Backstop fraud scorer

Following the structure of Mitchell et al., *Model Cards for Model Reporting* (2019).

## Model details

| | |
|---|---|
| Version | `20260906-212200` |
| Type | XGBoost gradient-boosted trees, 400 estimators, depth 5, learning rate 0.06 |
| Inputs | 35 features: 28 anonymised principal components plus 7 derived from amount and time |
| Output | An estimated probability of fraud, and a *review* / *approve* decision at threshold 0.852681 |
| Trained on | 210,292 transactions (376 fraud) spanning the first 38.4 hours of the observation window |
| Training data checksum | SHA-256 `4cc331ec41b28f3e…` |
| Libraries | Python 3.12.3, XGBoost 3.4.1, scikit-learn 1.9.0, NumPy 2.5.3, pandas 3.0.5 |
| Licence | MIT |

The threshold is part of the model, not a deployment setting. It was chosen by minimum expected cost
over 190,568 pooled out-of-fold development predictions (295 frauds), constrained to an alert rate
at or below 0.1% of traffic, and it is recorded in the artifact manifest alongside that description.

## Intended use

**Intended.** Ranking card-authorisation events for human review, inside a team with a fixed review
capacity, on traffic drawn from the same population as the training window.

**Out of scope.** Automatically declining a transaction without review. Any decision affecting an
individual that is not reviewed by a person. Traffic from a different issuer, geography, product or
era. Anything requiring the model to explain itself in business terms — see *Explainability* below.

**Explicitly not suitable** as a compliance artifact in a jurisdiction requiring specific adverse-
action reasons. The model can say *which* feature drove a decision; for 28 of its 35 features it
cannot say what that feature means.

## Data

The [ULB / Worldline credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
284,807 real transactions made by European cardholders over two days in September 2013, of which 492
(0.172%) are fraud. Features V1–V28 are principal components published in place of the original
fields — a privacy measure that is also the project's central constraint. `Time` and `Amount` are
unmodified.

**Preparation.** 1,081 byte-identical duplicate rows are removed before splitting (19 of them
fraud), leaving 283,726 rows and 473 frauds. The 1,825 zero-amount transactions are kept: they are
real traffic the model will be asked to score, and they turn out to carry 7× the base fraud rate.

**Split.** Temporal, not random. Development is the first 80% of the observation window (210,292
rows, 376 frauds); test is the last 20% (73,434 rows, 97 frauds). Model choice, hyperparameters and
the threshold are all settled inside development using four expanding-window folds with a 10-minute
purge gap. The test window was scored once.

## Metrics

Measured on the held-out test window, at the shipped threshold.

| | |
|---|---|
| Average precision | 0.798 (95% CI [0.721, 0.870], stratified bootstrap, 1,000 replicates) |
| ROC-AUC | 0.985 |
| Brier score | 0.00040 |
| Recall at a 0.1% alert budget | 0.701, precision 0.932 |
| Recall at 0.1% false-positive rate | 0.794 |
| Decisions | 66 alerts · 65 true positives · 1 false positive · 32 false negatives |
| Value-weighted recall | 0.627 |
| Fraud loss avoided | €7,071 of €11,596 (62.5% of the achievable maximum) |
| Latency | p50 2.1 ms, p99 2.7 ms for a single transaction |

Accuracy is deliberately not reported. Predicting "legitimate" for everything is 99.87% accurate on
this window and worth nothing.

## Performance across groups

The dataset carries no demographic attributes, so a fairness audit in the usual sense is impossible.
What can be examined is performance across transaction characteristics, and it is not uniform:

| Amount band | Frauds | Recall |
|---|---|---|
| €0–1 | 26 | 0.73 |
| €1–10 | 31 | 0.68 |
| €10–50 | 10 | **0.30** |
| €50–200 | 13 | 0.92 |
| €200–1k | 15 | 0.60 |
| €1k+ | 2 | 0.50 |

The €10–50 band is a real gap. Value-weighted recall (0.627) sits below count-weighted recall
(0.670), meaning the model misses proportionally more money than transactions. Amount-weighted
training was tested as a remedy, improved value recall to 0.638, cost 2 points of average precision
and broke probability calibration, and was not shipped — see the README.

## Explainability

Every alert carries the four features with the largest SHAP contributions, with direction and
magnitude. Contributions sum back to the model's own output, and a test asserts it: an explanation
that does not reconstruct the prediction is decoration.

For the seven engineered features the explanation is in plain language — *"zero-amount authorisation
(card testing pattern)"*, *"occurred overnight"*, *"amount is high for this portfolio"*. For the 28
principal components it says exactly what is known and no more: *"V14 is low (−3.83); anonymised
component"*. That is a genuine signal an analyst can compare across alerts. It is not a business
reason, and this project does not present it as one.

## Ethical considerations

**Being wrong has a cost that is not symmetric.** A false positive is a customer whose card is
declined — potentially in a shop, potentially for a transaction they need. The system is designed
around review rather than automatic decline for this reason, and its precision at the operating
point (0.932) is high specifically to keep that cost low.

**A model trained on caught fraud learns the fraud that was caught.** The label in this dataset is
confirmed fraud, which is a biased sample of actual fraud: patterns that consistently evade
detection are absent from training and therefore absent from what the model can learn.

**Deployment changes the data.** A blocked transaction never produces a chargeback, so a deployed
model degrades the labels available for its own successor. Handling that properly needs a held-out
control group.

**Anonymised features hide bias rather than removing it.** V1–V28 are linear combinations of the
original fields. If those included anything correlated with a protected characteristic, the
correlation survives the transformation — it is simply no longer visible or auditable. Absence of
evidence here is not evidence of absence.

## Caveats and recommendations

- Two days of data. No weekly seasonality, no holidays, no campaign effects. The confidence
  intervals are wide and honest.
- The 48-hour horizon means the drift monitor's reference window is thin. In production it should be
  a rolling reference, not a fixed one.
- Retraining cadence is unaddressed here for want of a longer window; the drift monitor is what
  would trigger it.
- Before deployment: measure the real cost of a review and a false decline, re-run the sensitivity
  analysis, and expect the alert-rate constraint to keep binding first.
