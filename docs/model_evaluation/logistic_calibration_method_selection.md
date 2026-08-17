# Logistic Regression Calibration Method Selection

## Purpose

This document records why **isotonic regression calibration** was selected as the preferred calibration method for the baseline logistic regression model.

The purpose is to make the decision understandable and reproducible even if this analysis is revisited much later.

The calibration-selection workflow was:

1. Train the original logistic regression model using the training dataset.
2. Keep the trained logistic regression model fixed.
3. Split the validation dataset chronologically into:

   * a **training-validation subset** used to fit candidate calibration methods;
   * a **testing-validation subset** used only to compare those calibration methods.
4. Compare:

   * uncalibrated probabilities;
   * intercept-only calibration;
   * Platt calibration;
   * isotonic calibration.
5. Judge the methods using both:

   * calibration quality;
   * preservation of discrimination and borrower ranking.
6. Select the preferred calibration method.
7. Refit the selected calibrator using the **entire validation dataset**.
8. Evaluate the finalized model and calibrator once on the untouched test dataset.

The untouched test dataset was therefore not involved in selecting the calibration method.

---

# Why Calibration Was Needed

The baseline logistic regression model showed a clear mismatch between predicted risk and observed risk.

On the testing-validation subset:

* **Observed bad-loan rate:** 25.62%
* **Mean uncalibrated predicted bad-loan probability:** 20.53%

The original logistic regression model was therefore materially **underestimating bad-loan risk on average**.

This was also visible across probability bands.

| Probability Band | Mean Predicted Probability | Observed Bad-Loan Rate | Calibration Gap |
| ---------------- | -------------------------: | ---------------------: | --------------: |
| 0–10%            |                      6.81% |                 10.48% |        +3.66 pp |
| 10–20%           |                     14.79% |                 21.32% |        +6.53 pp |
| 20–30%           |                     24.36% |                 31.43% |        +7.07 pp |
| 30–40%           |                     34.46% |                 37.83% |        +3.37 pp |

In these reports:

**Calibration gap = observed bad-loan rate − mean predicted probability**

Therefore:

* positive gap = risk is being underpredicted;
* negative gap = risk is being overpredicted;
* gap near zero = predicted and observed risk are closely aligned.

The baseline model was particularly underpredicting risk throughout the heavily populated low-to-middle probability ranges.

Importantly, the model's borrower ranking was still useful. The main problem was not that the logistic regression could not distinguish relatively risky borrowers from relatively safe borrowers.

The problem was that the **numerical probabilities themselves did not correspond closely enough to realized bad-loan frequencies**.

---

# Candidate Calibration Methods

## 1. Uncalibrated Baseline

The original logistic regression probability is:

`p = sigmoid(z)`

where `z` represents the original logistic regression log-odds.

This serves as the reference against which all calibration methods are compared.

---

## 2. Intercept-Only Calibration

Intercept calibration applies:

`p_calibrated = sigmoid(z + c)`

where `c` is a single learned adjustment.

This method can correct a broad difference between predicted and observed risk.

However, because it only shifts the log-odds, it cannot substantially change the shape of the probability relationship.

---

## 3. Platt Calibration

Platt calibration applies:

`p_calibrated = sigmoid(a × z + b)`

It learns:

* a scale parameter `a`;
* an intercept parameter `b`.

This allows the calibration curve to change both its overall level and its scale.

It remains a monotonic sigmoid transformation of the original logistic regression score.

---

## 4. Isotonic Calibration

Isotonic regression learns a flexible monotonic relationship:

`original log-odds → calibrated probability`

Unlike intercept or Platt calibration, isotonic regression does not assume that the calibration correction must follow a predefined sigmoid shape.

It can therefore adapt more flexibly to the calibration pattern observed in the validation data.

However, it still maintains the important monotonic constraint that higher baseline risk scores cannot systematically map to lower calibrated probabilities.

Because isotonic regression learns a stepwise mapping, multiple original scores can sometimes receive the same calibrated probability.

---

# Validation Design

The calibration methods were deliberately **not evaluated on the same observations used to fit them**.

The validation dataset was split chronologically into two halves:

## Training-validation subset

Used to fit:

* intercept adjustment;
* Platt calibrator;
* isotonic calibrator.

## Testing-validation subset

Used to compare:

* uncalibrated baseline;
* intercept calibration;
* Platt calibration;
* isotonic calibration.

This distinction is important.

Evaluating a calibrator on the same observations used to fit it could make a flexible method such as isotonic regression appear artificially strong.

Using the later half of validation as a separate comparison dataset provides a more realistic assessment of how the calibrators generalize.

The true test dataset remained untouched throughout this process.

---

# Calibration Results

## Overall Calibration Summary

| Method       | Observed Bad-Loan Rate | Mean Predicted Probability | Weighted Mean Absolute Calibration Gap |    Log Loss | Brier Score |
| ------------ | ---------------------: | -------------------------: | -------------------------------------: | ----------: | ----------: |
| Uncalibrated |                 25.62% |                     20.53% |                                5.35 pp |     0.54516 |     0.18039 |
| Intercept    |                 25.62% |                     26.23% |                                2.40 pp |     0.53690 |     0.17833 |
| Platt        |                 25.62% |                     26.33% |                                1.29 pp |     0.53445 |     0.17740 |
| Isotonic     |                 25.62% |                     26.31% |                            **0.75 pp** | **0.53306** | **0.17717** |

All three calibration methods substantially improved the original probability estimates.

---

# Weighted Mean Absolute Calibration Gap

The weighted mean absolute calibration gap summarizes how far predicted probability was from observed bad-loan frequency across probability bands.

The band errors were weighted by the number of loans in each band.

Results:

* Uncalibrated: **5.35 percentage points**
* Intercept: **2.40 percentage points**
* Platt: **1.29 percentage points**
* Isotonic: **0.75 percentage points**

Relative to the uncalibrated model, the approximate reduction in band-level calibration error was:

* Intercept: **55%**
* Platt: **76%**
* Isotonic: **86%**

Isotonic regression therefore produced the smallest overall probability-band calibration error.

---

# Probability-Band Evidence

The most informative probability bands are the lower and middle ranges because these contain most of the loans.

## 0–10% Probability Band

| Method       | Mean Predicted | Observed |          Gap |
| ------------ | -------------: | -------: | -----------: |
| Uncalibrated |          6.81% |   10.48% |     +3.66 pp |
| Intercept    |          6.99% |    7.98% |     +0.98 pp |
| Platt        |          7.50% |    6.65% |     -0.85 pp |
| Isotonic     |          6.92% |    7.29% | **+0.37 pp** |

---

## 10–20% Probability Band

| Method       | Mean Predicted | Observed |                    Gap |
| ------------ | -------------: | -------: | ---------------------: |
| Uncalibrated |         14.79% |   21.32% |               +6.53 pp |
| Intercept    |         15.29% |   16.53% |               +1.24 pp |
| Platt        |         15.61% |   14.88% |               -0.73 pp |
| Isotonic     |         15.02% |   15.02% | **approximately 0 pp** |

---

## 20–30% Probability Band

| Method       | Mean Predicted | Observed |          Gap |
| ------------ | -------------: | -------: | -----------: |
| Uncalibrated |         24.36% |   31.43% |     +7.07 pp |
| Intercept    |         24.63% |   26.16% |     +1.53 pp |
| Platt        |         24.65% |   25.53% |     +0.89 pp |
| Isotonic     |         24.96% |   24.70% | **-0.26 pp** |

---

## 30–40% Probability Band

| Method       | Mean Predicted | Observed |          Gap |
| ------------ | -------------: | -------: | -----------: |
| Uncalibrated |         34.46% |   37.83% |     +3.37 pp |
| Intercept    |         34.49% |   33.27% |     -1.22 pp |
| Platt        |         34.41% |   33.60% | **-0.81 pp** |
| Isotonic     |         34.56% |   33.65% |     -0.92 pp |

These results show that isotonic was not selected because of one isolated aggregate metric.

Its improvement was visible directly across the important parts of the probability distribution.

---

# Why the Extreme Probability Bands Were Not Given Much Weight

The highest predicted-probability bands contained very few loans.

For isotonic calibration:

* 80–90% band: **2 loans**
* 90–100% band: **1 loan**

Observed bad-loan rates of 100% or 0% in groups of this size are not reliable estimates of true calibration.

Even the other methods contained relatively few observations in these extreme bands.

For that reason, the calibration decision was driven primarily by:

* the densely populated probability ranges;
* weighted calibration error;
* Brier score;
* log loss;
* preservation of discrimination.

This prevents very small tail groups from dominating the calibration decision.

---

# Discrimination Results

Calibration should improve probability accuracy without meaningfully damaging the baseline model's ability to rank borrowers according to risk.

The discrimination comparison was:

| Method       |  ROC AUC |     Gini | Average Precision | Spearman Rank Correlation vs Uncalibrated |
| ------------ | -------: | -------: | ----------------: | ----------------------------------------: |
| Uncalibrated | 0.673170 | 0.346341 |          0.400425 |                                  1.000000 |
| Intercept    | 0.673170 | 0.346341 |          0.400425 |                                  1.000000 |
| Platt        | 0.673170 | 0.346341 |          0.400425 |                                  1.000000 |
| Isotonic     | 0.673003 | 0.346007 |          0.396752 |                                  0.999439 |

---

# Intercept and Platt Discrimination

Intercept and Platt calibration preserved the baseline ranking exactly.

Their:

* ROC AUC;
* Gini;
* average precision;
* Spearman rank correlation

were unchanged.

This is consistent with both methods applying monotonic transformations of the original logistic regression score.

---

# Isotonic Discrimination

Isotonic calibration produced a very small decline in discrimination:

* ROC AUC: **0.673170 → 0.673003**
* Gini: **0.346341 → 0.346007**
* Average precision: **0.400425 → 0.396752**
* Spearman rank correlation: **0.999439**

The ROC AUC declined by only approximately:

**0.00017**

The Spearman rank correlation remained extremely close to 1.

This means that isotonic calibration preserved essentially all of the original borrower ordering.

The small change is expected because isotonic regression's stepwise mapping can assign the same calibrated probability to several borrowers whose original scores differed slightly.

This creates ties without substantially reversing the underlying risk ordering.

---

# Score-Decile Evidence

The decile report confirms that the baseline logistic regression already provides meaningful risk ranking.

The observed bad-loan rate declines consistently from the highest-risk decile to the lowest-risk decile.

| Risk Decile      | Observed Bad-Loan Rate |
| ---------------- | ---------------------: |
| 1 — highest risk |                 48.46% |
| 2                |                 37.19% |
| 3                |                 33.12% |
| 4                |                 29.93% |
| 5                |                 26.67% |
| 6                |                 23.25% |
| 7                |                 20.29% |
| 8                |                 16.28% |
| 9                |                 13.01% |
| 10 — lowest risk |                  7.95% |

Therefore, the model's underlying ranking ability is useful.

The calibration exercise was intended to improve the **meaning of the numeric probability**, rather than replace this borrower ranking.

---

# Example: Middle-Risk Decile

For Decile 5:

| Method       | Mean Predicted Probability | Observed Bad-Loan Rate |
| ------------ | -------------------------: | ---------------------: |
| Uncalibrated |                     18.85% |                 26.67% |
| Intercept    |                     24.93% |                 26.67% |
| Platt        |                     25.54% |                 26.67% |
| Isotonic     |                     27.42% |                 26.58% |

The original model substantially underestimated risk for this group.

All three calibration methods corrected much of this error.

---

# Example: Lowest-Risk Decile

For Decile 10:

| Method       | Mean Predicted Probability | Observed Bad-Loan Rate |
| ------------ | -------------------------: | ---------------------: |
| Uncalibrated |                      5.01% |                  7.95% |
| Intercept    |                      7.00% |                  7.95% |
| Platt        |                      8.90% |                  7.95% |
| Isotonic     |                      7.60% |                  8.00% |

Again, isotonic placed the predicted probability very close to the realized bad-loan frequency.

---

# Limitation of Intercept-Only Calibration

Intercept calibration substantially improved the broad underprediction problem.

However, it also demonstrated why a single global probability-level adjustment was insufficient.

Consider the highest-risk decile:

| Method       | Mean Predicted Probability | Observed Bad-Loan Rate |
| ------------ | -------------------------: | ---------------------: |
| Uncalibrated |                     49.01% |                 48.46% |
| Intercept    |                     57.62% |                 48.46% |
| Platt        |                     53.04% |                 48.46% |
| Isotonic     |                     51.81% |                 48.32% |

The baseline model was already reasonably calibrated for these very high-risk borrowers.

The intercept adjustment pushed these probabilities upward along with the rest of the distribution and therefore substantially overpredicted their risk.

This shows that the original calibration problem was not simply:

> Every predicted probability is too low by the same amount.

Instead, the calibration error varied across the risk distribution.

That favors calibration methods that can adjust the shape of the relationship rather than applying only one global shift.

---

# Why Isotonic Was Selected

The decision was not based on one metric.

The evidence across several independent evaluations consistently favored isotonic calibration.

## 1. Lowest Weighted Probability-Band Calibration Error

Weighted mean absolute calibration gap:

* Uncalibrated: **5.35 pp**
* Intercept: **2.40 pp**
* Platt: **1.29 pp**
* Isotonic: **0.75 pp**

Isotonic produced the smallest band-level calibration error.

---

## 2. Lowest Log Loss

Results:

* Uncalibrated: **0.54516**
* Intercept: **0.53690**
* Platt: **0.53445**
* Isotonic: **0.53306**

Lower log loss is better.

Isotonic achieved the best result.

---

## 3. Lowest Brier Score

Results:

* Uncalibrated: **0.18039**
* Intercept: **0.17833**
* Platt: **0.17740**
* Isotonic: **0.17717**

Lower Brier score is better.

Again, isotonic achieved the best result.

---

## 4. Strongest Calibration Across Important Probability Ranges

Isotonic substantially corrected the baseline model's underprediction throughout the densely populated low-to-middle risk ranges.

The improvement appeared directly in predicted-versus-observed bad-loan rates rather than only in aggregate metrics.

---

## 5. Discrimination Loss Was Negligible

Isotonic's primary tradeoff was a very small deterioration in discrimination.

However:

* ROC AUC changed by only about **0.00017**;
* Gini changed only minimally;
* Spearman rank correlation remained **0.99944**;
* the overall decile structure remained almost unchanged.

This indicates that almost all of the model's original ranking information was preserved.

The small discrimination cost was therefore judged acceptable relative to the improvement in probability reliability.

---

## 6. Reliable Probabilities Matter for This Project

The project's downstream goal is not merely to rank loans from safest to riskiest.

The predicted probabilities are intended to support:

* borrower risk scores;
* risk bands;
* credit decision policy;
* portfolio interpretation;
* model monitoring;
* an interactive application that presents estimated applicant risk.

For those uses, the numerical meaning of the probability matters directly.

A borrower receiving an estimated probability near 20%, for example, should ideally belong to a group where approximately 20% of similar borrowers actually experience the bad-loan outcome.

Improved calibration is therefore an important property of the final model.

---

# Why Platt Was Not Selected

Platt calibration performed very well and is the strongest alternative to isotonic.

It:

* substantially improved calibration;
* preserved the baseline ranking exactly;
* retained identical ROC AUC;
* retained identical Gini;
* retained identical average precision;
* is simpler and parametric.

However, isotonic outperformed Platt on every major calibration-focused summary measure:

| Metric                   |   Platt |    Isotonic | Better   |
| ------------------------ | ------: | ----------: | -------- |
| Weighted calibration gap | 1.29 pp | **0.75 pp** | Isotonic |
| Log loss                 | 0.53445 | **0.53306** | Isotonic |
| Brier score              | 0.17740 | **0.17717** | Isotonic |

The discrimination deterioration introduced by isotonic was sufficiently small that its additional calibration accuracy was considered worth the tradeoff.

Platt remains an important benchmark and fallback option if later evidence suggests that isotonic calibration does not generalize adequately.

---

# Why the Current Isotonic Calibrator Is Not the Final Artifact

The isotonic calibrator used during the comparison was trained only on the **training-validation half** of the validation dataset.

That was intentional.

It allowed the testing-validation half to remain independent for method comparison.

Now that isotonic has been selected, retaining a calibrator trained on only half the available validation observations would unnecessarily discard useful calibration information.

Therefore, the comparison calibrator itself should not be treated as the final production artifact.

The next step is to:

> Fit a fresh isotonic regression calibrator using the entire validation dataset.

The underlying baseline logistic regression model remains unchanged.

The full validation dataset is used only to learn the final mapping between the baseline logistic regression log-odds and observed bad-loan probabilities.

---

# Final Evaluation Plan

The finalized workflow is:

`Training data`

→ fit preprocessing rules
→ fit baseline logistic regression

`Full validation data`

→ apply frozen preprocessing
→ generate baseline logistic regression log-odds
→ fit final isotonic calibrator

`Untouched test data`

→ apply frozen preprocessing
→ apply frozen baseline logistic regression
→ generate test log-odds
→ apply frozen isotonic calibrator
→ evaluate final calibrated probabilities

The test dataset must remain untouched until this final evaluation.

The calibration method should **not** be changed based on the test-set results.

Doing so would effectively turn the test dataset into another validation dataset and would remove its usefulness as an unbiased final estimate of out-of-sample performance.

---

# What Should Be Checked on the Untouched Test Set

The final test evaluation should compare the uncalibrated baseline probabilities with the finalized isotonic probabilities using the same major categories of evaluation.

## Calibration

Evaluate:

* observed bad-loan rate;
* mean predicted bad-loan probability;
* weighted mean absolute calibration gap;
* calibration across probability bands;
* log loss;
* Brier score.

## Discrimination

Evaluate:

* ROC AUC;
* Gini;
* average precision;
* score-decile ordering;
* bad-loan concentration by decile;
* rank preservation relative to the uncalibrated baseline.

The expectation is not that every individual metric must improve.

The main question is:

> Does isotonic calibration continue to provide materially more reliable probabilities on completely unseen data without meaningfully damaging discrimination?

---

# Final Decision

**Selected calibration method: Isotonic Regression**

Isotonic regression was selected because it produced the strongest held-out validation calibration across:

* probability-band calibration;
* weighted mean absolute calibration error;
* log loss;
* Brier score;
* score-decile probability alignment.

Its main cost was a very small decline in discrimination metrics, while the underlying borrower ranking remained almost entirely intact.

Because this project requires meaningful and interpretable borrower risk probabilities rather than ranking alone, this tradeoff was accepted.

The next implementation step is therefore to:

1. generate baseline log-odds for the entire validation dataset;
2. fit a fresh isotonic regression calibrator using the entire validation dataset;
3. freeze that calibrator;
4. apply it to the untouched test dataset;
5. evaluate final calibration and discrimination;
6. save the finalized calibrator as the calibration artifact.

---

# Short Decision Record

**Problem:**
The baseline logistic regression materially underestimated bad-loan probabilities across much of the portfolio.

**Comparison design:**
Intercept, Platt, and isotonic calibration were fitted using the first chronological half of validation and evaluated using the second chronological half.

**Calibration result:**
Isotonic achieved the lowest weighted calibration gap, lowest log loss, lowest Brier score, and strongest probability-band alignment.

**Discrimination tradeoff:**
Isotonic caused only a negligible decline in ranking performance. ROC AUC declined by approximately 0.00017, while Spearman rank correlation remained approximately 0.99944.

**Decision:**
Select isotonic regression calibration.

**Next step:**
Refit isotonic regression using the entire validation dataset and conduct the final evaluation on the untouched test dataset.
