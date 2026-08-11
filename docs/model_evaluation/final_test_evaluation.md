# Final Test Evaluation — Logistic Regression with Isotonic Calibration

## 1. Purpose

This document records the final out-of-sample evaluation of the project's main logistic-regression credit-risk model after probability calibration.

The purpose of this stage was no longer to select or tune a model.

Calibration-method selection had already been completed using held-out validation data, where isotonic regression was selected over:

* the uncalibrated logistic-regression probabilities;
* intercept-only calibration;
* Platt calibration.

The purpose of the final test evaluation was therefore to answer:

> Does the frozen baseline logistic-regression model combined with an isotonic calibrator fitted on the complete validation dataset generalize successfully to unseen future data?

The test dataset was treated as the final untouched evaluation period.

No modeling, feature-selection, calibration-method-selection, or threshold-tuning decisions were made using the test results.

---

# 2. Final Modeling System

The final scoring system consists of three separate fitted components:

```text
training-fitted preprocessing pipeline
        ↓
training-fitted baseline logistic regression
        ↓
validation-fitted isotonic calibrator
        ↓
final calibrated bad-loan probability
```

These components have different roles.

## 2.1 Preprocessing pipeline

The preprocessing pipeline was fitted using the training dataset only.

It learned transformations such as:

* numeric missing-value imputation;
* numeric scaling parameters;
* categorical missing-value handling;
* one-hot encoding categories.

The same frozen training-fitted preprocessing rules were later applied to validation and test data.

Validation and test therefore did not influence the preprocessing parameters.

---

## 2.2 Baseline logistic-regression model

The feature-level logistic-regression model was fitted using the training dataset.

The model was not retrained using validation or test data.

For each loan, the logistic-regression model produces a raw score in log-odds form:

```text
z = baseline logistic-regression log-odds
```

The original uncalibrated probability is:

```text
p = sigmoid(z)
```

The logistic-regression coefficients therefore remained frozen throughout the calibration and final-evaluation stages.

---

## 2.3 Final isotonic calibrator

During calibration-method selection, isotonic regression produced the strongest probability calibration while causing only negligible changes in discrimination.

After that method had been selected, a fresh isotonic calibrator was fitted using the entire validation dataset.

Its inputs were:

```text
input:
baseline logistic-regression log-odds for every validation loan

target:
actual validation bad-loan outcomes
```

The isotonic calibrator therefore does not use the original borrower features directly.

It learns only a monotonic mapping:

```text
baseline logistic log-odds
        ↓
calibrated bad-loan probability
```

The fitted calibrator was saved as a reusable model artifact so it can later be applied without refitting.

The intended artifact is:

```text
models/logistic_regression_isotonic_calibrator.joblib
```

---

# 3. Final Evaluation Design

The final workflow was:

```text
complete validation feature matrix
        ↓
frozen baseline logistic regression
        ↓
validation log-odds
        ↓
fit final isotonic calibrator using all validation outcomes
        ↓
freeze and save isotonic calibrator

untouched test feature matrix
        ↓
same frozen baseline logistic regression
        ↓
test baseline probabilities + test log-odds
        ↓
frozen validation-fitted isotonic calibrator
        ↓
calibrated test probabilities
        ↓
final evaluation
```

Two probability sets were evaluated on the same test loans:

```text
uncalibrated
isotonic
```

This allowed the final evaluation to measure whether isotonic calibration improved probability quality on unseen data and whether it materially damaged risk ranking.

---

# 4. Test-Set Portfolio Risk

The observed bad-loan rate in the final test dataset was:

```text
26.44%
```

More precisely:

```text
0.2643548355692284
```

The original logistic-regression model produced a mean predicted bad-loan probability of:

```text
20.77%
```

More precisely:

```text
0.2077062012946916
```

The uncalibrated model therefore substantially underestimated the overall level of risk in the final test period.

The difference between observed and mean predicted portfolio risk was approximately:

```text
26.44% - 20.77%
≈ 5.66 percentage points
```

After isotonic calibration, the mean predicted probability became:

```text
25.98%
```

More precisely:

```text
0.2597985515356377
```

Compared with the observed rate of:

```text
26.44%
```

the remaining difference was only approximately:

```text
0.46 percentage points
```

This provides strong evidence that the broad probability-level underestimation identified during validation also existed in the unseen test period and that isotonic calibration substantially corrected it.

---

# 5. Overall Calibration Results

The final calibration summary was:

| Method       | Observed bad-loan rate | Mean predicted probability | Weighted mean absolute calibration gap | Log loss | Brier score |
| ------------ | ---------------------: | -------------------------: | -------------------------------------: | -------: | ----------: |
| Uncalibrated |                 26.44% |                     20.77% |                                6.27 pp |  0.56271 |     0.18550 |
| Isotonic     |                 26.44% |                     25.98% |                                1.00 pp |  0.54245 |     0.18083 |

All major calibration-oriented metrics improved after isotonic calibration.

---

# 6. Weighted Calibration Error

The weighted mean absolute calibration gap decreased from:

```text
Uncalibrated:
0.06268780271944416
≈ 6.27 percentage points
```

to:

```text
Isotonic:
0.009953160531097583
≈ 1.00 percentage point
```

This represents an approximate reduction of:

```text
84%
```

in this calibration-error measure.

The weighting uses the number of loans inside each probability band, meaning heavily populated regions of the probability distribution contribute more strongly to the overall result than sparsely populated tail regions.

This is important because the goal is not merely to achieve small errors in tiny extreme-risk groups but to produce useful probabilities across the parts of the portfolio where most loans actually occur.

---

# 7. Log-Loss Results

Log loss decreased from:

```text
Uncalibrated:
0.562711017047382
```

to:

```text
Isotonic:
0.5424530134773192
```

Lower log loss is better.

This result indicates that the calibrated probabilities were more consistent with the actual observed outcomes than the original logistic-regression probabilities.

The improvement also shows that isotonic calibration did more than simply correct the portfolio-wide average probability.

It improved the probability predictions across individual loans as measured by a proper probabilistic scoring rule.

---

# 8. Brier-Score Results

The Brier score decreased from:

```text
Uncalibrated:
0.18549598410440324
```

to:

```text
Isotonic:
0.18082726682792918
```

Lower Brier score is better.

The Brier score measures squared differences between predicted probabilities and actual binary outcomes.

The improvement therefore provides additional evidence that the calibrated probabilities are more reliable than the original probabilities.

---

# 9. Probability-Band Calibration

The probability-band report shows where the calibration improvement occurred.

## 9.1 0–10% probability band

### Uncalibrated

```text
Row count:
44,641

Mean predicted probability:
6.60%

Observed bad-loan rate:
11.24%

Calibration gap:
+4.64 percentage points
```

### Isotonic

```text
Row count:
20,179

Mean predicted probability:
7.00%

Observed bad-loan rate:
8.39%

Calibration gap:
+1.39 percentage points
```

The positive calibration gap indicates underprediction.

Isotonic substantially reduced the underprediction in this range.

---

## 9.2 10–20% probability band

### Uncalibrated

```text
Mean predicted probability:
14.74%

Observed bad-loan rate:
22.44%

Calibration gap:
+7.70 percentage points
```

### Isotonic

```text
Mean predicted probability:
14.89%

Observed bad-loan rate:
16.29%

Calibration gap:
+1.39 percentage points
```

This was a major improvement.

The baseline model severely underestimated bad-loan risk in this populated region of the score distribution.

---

## 9.3 20–30% probability band

### Uncalibrated

```text
Mean predicted probability:
24.40%

Observed bad-loan rate:
32.20%

Calibration gap:
+7.80 percentage points
```

### Isotonic

```text
Mean predicted probability:
24.79%

Observed bad-loan rate:
25.62%

Calibration gap:
+0.83 percentage points
```

Again, isotonic corrected most of the baseline model's underprediction.

---

## 9.4 30–40% probability band

### Uncalibrated

```text
Mean predicted probability:
34.51%

Observed bad-loan rate:
38.71%

Calibration gap:
+4.20 percentage points
```

### Isotonic

```text
Mean predicted probability:
34.31%

Observed bad-loan rate:
34.24%

Calibration gap:
-0.07 percentage points
```

This isotonic band was almost perfectly calibrated.

---

## 9.5 40–50% probability band

### Uncalibrated

```text
Mean predicted probability:
44.44%

Observed bad-loan rate:
45.56%

Calibration gap:
+1.12 percentage points
```

### Isotonic

```text
Mean predicted probability:
43.70%

Observed bad-loan rate:
43.35%

Calibration gap:
-0.35 percentage points
```

Both were relatively close in this range, but isotonic remained slightly better aligned.

---

# 10. Higher-Risk Probability Bands

Calibration was less precise in the higher predicted-probability ranges.

For example:

## 50–60% isotonic band

```text
Mean predicted probability:
53.22%

Observed bad-loan rate:
50.55%

Calibration gap:
-2.67 percentage points
```

## 60–70% isotonic band

```text
Mean predicted probability:
60.52%

Observed bad-loan rate:
54.34%

Calibration gap:
-6.18 percentage points
```

The negative calibration gaps mean the calibrated model overpredicted risk in these groups.

However, isotonic still improved the 60–70% band relative to the uncalibrated model, which had a calibration gap of approximately:

```text
-8.58 percentage points
```

The calibrator is therefore not perfectly calibrated at every probability level, but its overall behavior is materially better.

---

# 11. Extreme Tail Bands

The extreme probability ranges should not drive conclusions because they contain very few isotonic observations.

For example:

```text
70–80% isotonic band:
13 loans

80–90% isotonic band:
0 loans

90–100% isotonic band:
9 loans
```

Observed bad-loan rates calculated from such small samples are highly unstable.

Large calibration gaps in these bands are therefore not strong evidence of general calibration failure.

The absence of observations in some probability ranges is also not necessarily an implementation problem.

Isotonic regression learns a stepwise monotonic mapping, so it may produce no probabilities inside certain intervals.

---

# 12. Score-Decile Risk Ordering

The final test score-decile results show that the model continues to separate borrowers by risk.

For isotonic-calibrated probabilities, observed bad-loan rates were approximately:

```text
Decile 1: 48.77%
Decile 2: 39.08%
Decile 3: 34.25%
Decile 4: 30.76%
Decile 5: 27.99%
Decile 6: 23.61%
Decile 7: 20.54%
Decile 8: 17.45%
Decile 9: 13.46%
Decile 10: 8.46%
```

Decile 1 represents the highest predicted-risk group.

Decile 10 represents the lowest predicted-risk group.

The observed bad-loan rate decreases consistently as the model moves from higher-risk to lower-risk deciles.

This provides clear evidence that the logistic model retains useful discrimination.

---

# 13. Bad-Loan Concentration

The isotonic-calibrated ranking captured approximately:

```text
Top 10% of borrowers:
18.45% of all bad loans

Top 20%:
33.23%

Top 30%:
46.19%

Top 50%:
68.41%

Top 70%:
85.11%
```

This means bad loans are meaningfully concentrated toward the high-risk side of the model's ranking.

For example, the highest-risk 10% of loans contain considerably more than 10% of all observed bad loans.

The model therefore provides useful ranking information even though its discrimination is not perfect.

---

# 14. Decile-Level Calibration Improvement

The original model substantially underpredicted risk through most of the score distribution.

Approximate observed-minus-predicted gaps were:

| Decile | Uncalibrated gap | Isotonic gap |
| ------ | ---------------: | -----------: |
| 1      |         -2.41 pp |     -2.71 pp |
| 2      |         +4.07 pp |     -0.08 pp |
| 3      |         +6.92 pp |     -0.17 pp |
| 4      |         +8.22 pp |     +0.59 pp |
| 5      |         +9.11 pp |     +1.00 pp |
| 6      |         +7.61 pp |     +0.80 pp |
| 7      |         +7.43 pp |     +0.92 pp |
| 8      |         +6.61 pp |     +1.59 pp |
| 9      |         +5.15 pp |     +1.22 pp |
| 10     |         +3.94 pp |     +1.39 pp |

The largest improvements occurred in the middle and lower-risk portions of the distribution.

For example, Decile 5 changed from:

```text
Uncalibrated predicted:
18.89%

Observed:
28.01%
```

to approximately:

```text
Isotonic predicted:
26.99%

Observed:
27.99%
```

This is a substantial improvement in probability reliability.

---

# 15. Highest-Risk Decile

The highest-risk decile behaved somewhat differently.

The uncalibrated model was already reasonably close:

```text
Uncalibrated predicted:
51.17%

Observed:
48.76%
```

After isotonic:

```text
Isotonic predicted:
51.48%

Observed:
48.77%
```

The calibrated model therefore slightly worsened calibration in the highest-risk decile.

This is consistent with findings during validation.

The baseline model's calibration problem was not simply that every probability needed to be shifted upward uniformly.

Instead, the magnitude and direction of calibration error varied across the score distribution.

This was one reason isotonic regression was preferred over a simpler intercept-only correction.

---

# 16. Final Discrimination Results

The final test discrimination summary was:

| Method       |  ROC AUC |     Gini | Average Precision | Spearman correlation to baseline |
| ------------ | -------: | -------: | ----------------: | -------------------------------: |
| Uncalibrated | 0.672351 | 0.344701 |          0.405126 |                         1.000000 |
| Isotonic     | 0.672310 | 0.344620 |          0.403387 |                         0.999674 |

---

# 17. ROC AUC

ROC AUC changed from:

```text
0.6723507463
```

to:

```text
0.6723098981
```

The difference was approximately:

```text
0.000041
```

This is negligible.

Isotonic calibration therefore did not materially damage the logistic model's ability to rank bad loans above good loans.

---

# 18. Gini

Gini changed from:

```text
0.3447014926
```

to:

```text
0.3446197962
```

This change is also negligible.

Because:

```text
Gini = 2 × ROC AUC - 1
```

the result is consistent with the ROC AUC findings.

---

# 19. Average Precision

Average precision changed from:

```text
Uncalibrated:
0.4051264334

Isotonic:
0.4033867845
```

The change is small.

Average precision summarizes model precision across different levels of recall.

It is primarily a discrimination/ranking measure rather than a calibration measure.

The test-set bad-loan prevalence was approximately:

```text
0.2644
```

so a no-skill ranking would have average precision near the approximately 26.44% positive-class rate.

The logistic model's average precision of approximately 40.5% therefore provides useful ranking power relative to the base rate, although the model's discrimination should still be described as moderate rather than exceptional.

Calibration was not expected to materially improve average precision because isotonic regression changes probability reliability rather than fundamentally creating new information for distinguishing good loans from bad loans.

---

# 20. Spearman Rank Correlation

The Spearman rank correlation between isotonic-calibrated probabilities and the original logistic-regression probabilities was:

```text
0.9996740611
```

This is extremely close to 1.

It indicates that borrower ordering remained almost identical after calibration.

The very small deviation from perfect rank preservation is expected because isotonic regression can assign identical probabilities to groups of nearby scores, creating ties.

---

# 21. Interpretation of Model Quality

The final logistic model should not be interpreted as perfectly predicting which individual loans will become bad.

Its discrimination is useful but moderate.

For example:

```text
Overall test bad-loan rate:
26.44%

Highest-risk decile observed bad-loan rate:
approximately 48.8%

Lowest-risk decile observed bad-loan rate:
approximately 8.5%
```

The model therefore separates borrowers into meaningfully different risk groups.

However, substantial overlap remains between borrowers who eventually become good and bad loans.

The appropriate interpretation is therefore:

> The model provides useful estimates of relative and absolute credit risk, but it does not determine individual loan outcomes with certainty.

This is acceptable for the intended MVP because the calibrated probabilities can serve as inputs to:

* risk scores;
* risk bands;
* decision policies;
* portfolio monitoring;
* applicant-level risk estimates.

---

# 22. Calibration Versus Discrimination

The final evaluation reinforces the distinction between two different model properties.

## Discrimination asks:

```text
Does the model rank riskier borrowers above safer borrowers?
```

Relevant metrics include:

* ROC AUC;
* Gini;
* average precision;
* score-decile ordering;
* bad-loan concentration.

The logistic model demonstrates useful but moderate discrimination.

## Calibration asks:

```text
When the model predicts a particular probability of bad-loan risk,
does approximately that proportion of borrowers actually become bad?
```

Relevant metrics include:

* calibration gaps;
* mean predicted probability versus observed bad-loan rate;
* log loss;
* Brier score.

The original logistic-regression probabilities were materially miscalibrated.

Isotonic regression substantially improved this aspect of the model.

---

# 23. Final Modeling Conclusion

The final out-of-sample results support the selected calibration strategy.

The baseline logistic-regression model originally:

* materially underestimated overall portfolio risk;
* strongly underpredicted risk through several populated probability ranges;
* nevertheless demonstrated useful borrower ranking.

The validation-fitted isotonic calibrator:

* reduced the overall portfolio probability bias;
* reduced weighted mean absolute calibration error from approximately 6.27 percentage points to approximately 1.00 percentage point;
* improved log loss;
* improved Brier score;
* strongly improved probability alignment across the major populated probability bands;
* strongly improved decile-level probability alignment;
* preserved risk ranking almost entirely;
* caused only negligible changes in ROC AUC and Gini.

The final conclusion is:

> The training-fitted baseline logistic regression combined with an isotonic calibrator fitted on the full validation dataset generalized successfully to the unseen test period. Isotonic calibration materially improved the reliability of predicted bad-loan probabilities while causing essentially no meaningful loss of discrimination.

---

# 24. Final Frozen System

The main model system should now be treated as frozen:

```text
training-fitted preprocessing pipeline
+
training-fitted baseline logistic regression
+
validation-fitted isotonic calibrator
```

The output of this system is the final calibrated estimate of bad-loan probability.

The final system is suitable for use in the next project stages:

```text
calibrated bad-loan probability
        ↓
risk score
        ↓
risk band
        ↓
decision policy
        ↓
monitoring and interface
```

---

# 25. Test-Set Governance

The test set has now served its intended purpose as a final unbiased evaluation dataset.

From this point forward, the project should not:

* switch calibration methods because of test results;
* retune isotonic calibration using the test data;
* modify the baseline logistic model because of test performance;
* change feature selection based on the test results;
* repeatedly evaluate alternative main models against this same test set and select the winner.

Doing so would turn the final test set into another validation set.

The final test results should instead be recorded as the out-of-sample estimate of the frozen model system's performance.

---

# 26. Discrimination Improvement as Future Work

The final model's discrimination is useful but not exceptionally strong.

The baseline test metrics were approximately:

```text
ROC AUC:
0.672

Average precision:
0.405
```

The current MVP should therefore not claim that the model achieves highly accurate individual default classification.

However, improving discrimination is not required before continuing the main project.

The current calibrated model provides sufficient risk separation and probability reliability to support the risk-scoring, decision-policy, monitoring, and deployment portions of the MVP.

Future work may investigate a separate challenger model, such as a nonlinear boosting model, to answer:

> Can a more flexible model materially improve discrimination beyond the interpretable logistic-regression baseline?

Such work should be treated as a post-MVP extension rather than a prerequisite for completing the current system.

---

# 27. Next Project Stage

With model training, calibration selection, final isotonic fitting, and final test evaluation complete, the next stage is:

# Risk Scoring and Decision Policy

The next workflow will transform:

```text
final calibrated bad-loan probability
```

into:

```text
risk score
→ risk band
→ decision policy
```

The score transformation and risk-band thresholds should be chosen systematically rather than arbitrarily.

The calibrated logistic-regression system established in this document will serve as the underlying probability model for that work.
