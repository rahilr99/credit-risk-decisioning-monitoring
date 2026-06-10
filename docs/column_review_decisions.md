# Column Review Decisions

## 1. Purpose

This document records the manual review decisions for the LendingClub columns that were not fully resolved by the preliminary column-review inventory script.

The target-defined interim dataset contains 152 columns. The preliminary inventory script automatically classified columns with obvious roles, including the target, the target-source field, identifiers, monitoring-only fields, post-application leakage fields, and constant or empty fields.

After the automatic classification rules were applied, 109 columns remained marked as:

```text
requires_review
```

These unresolved columns require manual review before preprocessing and modeling begin.

The goal of this review is to determine whether each column:

* is eligible as an input for the core applicant-style credit-risk model;
* should be retained for monitoring, benchmarking, reporting, or another secondary use;
* contains post-application information that would introduce leakage; or
* should be excluded from model features for another documented reason.

The core model should use only information that would realistically be available when an applicant is evaluated. A column should not become a model input merely because it may improve predictive performance.

Each reviewed column will receive:

```text
column_description
available_at_application
primary_purpose
additional_use_cases
decision_reason
```

The detailed decisions recorded in this document will later be converted into a lightweight CSV inventory containing:

```text
column_name
primary_purpose
additional_use_cases
```

That CSV will serve as the machine-readable lookup table used by later project modules.

## 2. Batch 1 — Application and loan setup fields

This batch contains general information about the application, the requested loan structure, and borrower-provided background information.

| column_name                 | column_description                                                                                                      | available_at_application | primary_purpose           | additional_use_cases                 | decision_reason                                                                                                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `application_type`          | Indicates whether the loan is an individual application or a joint application with two co-borrowers.                   | `yes`                    | `modeling`                | `portfolio_segmentation`             | The application structure is known when the loan request is evaluated and may provide relevant context for the risk estimate.                                    |
| `term`                      | Number of scheduled payments on the loan. The dataset uses either 36 or 60 months.                                      | `yes`                    | `modeling`                | `portfolio_segmentation`             | The requested repayment period is known before origination and directly affects the structure of the loan.                                                       |
| `verification_status`       | Indicates whether the borrower's income was verified, not verified, or whether the income source was verified.          | `conditional`            | `conditionally_available` | `portfolio_segmentation`             | The field may be available during underwriting, but it may not exist at the initial application-submission stage. Its use depends on the intended scoring point. |
| `verification_status_joint` | Indicates whether the co-borrowers' joint income was verified, not verified, or whether the income source was verified. | `conditional`            | `conditionally_available` | `portfolio_segmentation`             | The field is relevant only for joint applications and may depend on whether verification has occurred before scoring.                                            |
| `home_ownership`            | Borrower's reported home-ownership status, such as rent, own, or mortgage.                                              | `yes`                    | `modeling`                | `portfolio_segmentation`             | This borrower-provided characteristic is available during the application process and may be considered as a candidate model input.                              |
| `purpose`                   | Borrower-provided category describing the reason for the loan request.                                                  | `yes`                    | `modeling`                | `portfolio_segmentation`,`reporting` | The requested use of funds is known during the application process and may be relevant for both risk modeling and portfolio analysis.                            |
| `emp_length`                | Borrower's employment length in years. Values range from less than one year to ten or more years.                       | `yes`                    | `modeling`                |                                      | Employment tenure is borrower-provided application information and may be considered as a candidate model input.                                                 |

### Batch 1 checkpoint

```text
Columns reviewed: 7
Modeling candidates: 5
Conditionally available fields: 2
Post-application leakage fields: 0
```

## 3. Batch 2 — Operational and timing-sensitive fields

This batch contains fields related to LendingClub's operational workflow and fields that may be generated after repayment difficulties arise.

| column_name                                  | column_description                                                                                                          | available_at_application | primary_purpose               | additional_use_cases                 | decision_reason                                                                                                                                                                                                                          |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ----------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `disbursement_method`                        | Method by which the borrower receives the loan. Possible values include cash disbursement and direct payment to creditors.  | `conditional`            | `conditionally_available`     | `portfolio_segmentation`,`reporting` | The disbursement method may be known during underwriting or loan setup, but its availability at the initial scoring point is not guaranteed. It is also an operational product characteristic rather than a raw borrower-risk attribute. |
| `initial_list_status`                        | Initial listing status assigned to the loan by LendingClub. Possible values include `W` and `F`.                            | `no`                     | `exclude_from_model_features` | `portfolio_segmentation`,`reporting` | This field reflects LendingClub's listing workflow rather than applicant information. It should not enter the core applicant-style model.                                                                                                |
| `pymnt_plan`                                 | Indicates whether a payment plan has been placed on the loan.                                                               | `no`                     | `post_application_leakage`    | `portfolio_monitoring`,`reporting`   | A payment plan is associated with the loan after origination and may reflect repayment difficulty. Using it as a model input would introduce future information.                                                                         |
| `orig_projected_additional_accrued_interest` | Original projected additional interest expected to accrue under a hardship payment plan as of the hardship-plan start date. | `no`                     | `post_application_leakage`    | `portfolio_monitoring`,`reporting`   | This value exists only because a hardship plan was created after origination. It contains post-application repayment information and must not enter the model.                                                                           |

### Batch 2 checkpoint

```text
Columns reviewed: 4
Conditionally available fields: 1
Excluded operational fields: 1
Post-application leakage fields: 2
```

## 4. Batch 3 — Geography and high-cardinality text fields

This batch contains borrower-provided geographic information and free-text fields. These values may be available during the application process, but they require conservative handling.

Geographic fields are retained for portfolio analysis rather than used as core underwriting inputs. Free-text fields are excluded from the structured-feature MVP because they would require separate text-processing decisions and may be inconsistent or overly granular.

| column_name  | column_description                                                                   | available_at_application | primary_purpose            | additional_use_cases               | decision_reason                                                                                                                                                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------ | ------------------------ | -------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `addr_state` | State provided by the borrower in the loan application.                              | `yes`                    | `monitoring`               | `portfolio_segmentation,reporting` | State-level geography may be useful for portfolio analysis and reporting. It is excluded from the core applicant-risk model to avoid relying on geographic differences as underwriting signals without a separate fairness and stability review. |
| `zip_code`   | First three digits of the ZIP code provided by the borrower in the loan application. | `yes`                    | `monitoring`               | `portfolio_segmentation`           | The field is more geographically granular than state and may be useful for portfolio analysis. It is excluded from the core model because it may introduce unstable or difficult-to-justify geographic effects.                                  |
| `title`      | Loan title provided by the borrower.                                                 | `yes`                    | `high_cardinality_or_text` |                                    | This borrower-entered text field may overlap with the structured `purpose` category and may contain inconsistent wording. It is excluded from the structured-feature MVP.                                                                        |
| `desc`       | Loan description provided by the borrower.                                           | `yes`                    | `high_cardinality_or_text` |                                    | This is an unstructured borrower-entered text field. Using it would require a separate text-processing workflow outside the scope of the structured-feature MVP.                                                                                 |
| `emp_title`  | Job title supplied by the borrower when applying for the loan.                       | `yes`                    | `high_cardinality_or_text` |                                    | Employer titles can contain inconsistent wording and many distinct values. The field is excluded from the structured-feature MVP unless a separate occupation-grouping strategy is designed later.                                               |

### Batch 3 checkpoint

```text
Columns reviewed: 5
Monitoring fields: 2
High-cardinality or text fields: 3
Modeling candidates: 0
Post-application leakage fields: 0
```

## 5. Batch 4 — Lender-derived, pricing, and funding fields

This batch contains fields related to LendingClub's internal risk classification, pricing process, loan-amount determination, and funding workflow.

These fields require careful handling because some may exist before origination but still embed LendingClub's prior underwriting or operational decisions. The core model should estimate credit risk from applicant-style information rather than reproduce LendingClub's existing risk grade or pricing logic.

| column_name       | column_description                                                                                                        | available_at_application | primary_purpose           | additional_use_cases                            | decision_reason                                                                                                                                                                                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `grade`           | LendingClub-assigned loan grade representing the platform's internal risk classification.                                 | `conditional`            | `lender_derived_feature`  | `benchmarking,portfolio_segmentation,reporting` | The grade may exist during underwriting, but it is generated by LendingClub's own risk-assessment process. Including it in the core model would cause the project to rely on an existing lender-derived risk signal. Retain it for comparison and portfolio analysis. |
| `sub_grade`       | More granular LendingClub-assigned risk classification within the broader loan grade.                                     | `conditional`            | `lender_derived_feature`  | `benchmarking,portfolio_segmentation,reporting` | The sub-grade contains an even more detailed version of LendingClub's internal risk assessment. It should be excluded from the core model by default but retained for benchmarking.                                                                                   |
| `int_rate`        | Interest rate assigned to the loan.                                                                                       | `conditional`            | `lender_derived_feature`  | `benchmarking,portfolio_segmentation,reporting` | The interest rate reflects LendingClub's pricing decision and may embed information from its underwriting process. It should not enter the independent core model but may be useful for comparison and portfolio reporting.                                           |
| `installment`     | Monthly payment owed by the borrower if the loan originates.                                                              | `conditional`            | `lender_derived_feature`  | `portfolio_segmentation,reporting`              | The installment amount is derived from the loan amount, repayment term, and assigned interest rate. Because it partly reflects lender pricing, exclude it from the independent core model by default.                                                                 |
| `loan_amnt`       | Listed loan amount applied for by the borrower. The recorded value may reflect a reduction made by the credit department. | `conditional`            | `conditionally_available` | `portfolio_segmentation,reporting`              | The requested loan amount is conceptually relevant to credit risk. However, the dataset field may reflect a lender adjustment rather than the applicant's original request. Its use depends on the scoring point and should be revisited before modeling.             |
| `funded_amnt`     | Total amount committed to the loan at that point in time.                                                                 | `no`                     | `monitoring`              | `portfolio_segmentation,reporting`              | This value reflects the funding stage rather than the initial application. It should not become an applicant-risk model input, but it is useful for measuring portfolio exposure.                                                                                     |
| `funded_amnt_inv` | Total amount committed by investors to the loan at that point in time.                                                    | `no`                     | `monitoring`              | `portfolio_segmentation,reporting`              | Investor funding is an operational outcome that is not available at the initial scoring point. Retain the field for portfolio analysis rather than applicant-risk modeling.                                                                                           |

### Batch 4 checkpoint

```text
Columns reviewed: 7
Lender-derived features: 4
Conditionally available fields requiring later review: 1
Monitoring fields: 2
Post-application leakage fields: 0
```


## 6. Batch 5 — Delinquency, collections, and public-record fields

This batch contains credit-bureau attributes describing the borrower's prior or current credit difficulties outside the LendingClub loan being evaluated.

These fields must be distinguished from post-application leakage. A bureau attribute describing an applicant's existing delinquency history may be a valid application-time risk signal. By contrast, a field describing repayment performance on the newly issued LendingClub loan would reveal future information and must be excluded.

Most fields in this batch are retained as modeling candidates. Two fields remain conditionally available because their definitions state that the values were updated within the past two months. Their provenance should be verified before they enter the core model.

| column_name                  | column_description                                                                                                       | available_at_application | primary_purpose           | additional_use_cases                        | decision_reason                                                                                                                                                                                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------ | ------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_tl_30dpd`               | Number of accounts currently 30 days past due, updated within the past two months.                                       | `conditional`            | `conditionally_available` | `data_quality_check,portfolio_segmentation` | The field may represent a recent credit-bureau snapshot that is available during underwriting. However, the timing language creates uncertainty about whether the stored value consistently reflects the application-time snapshot. Verify provenance before modeling. |
| `num_tl_120dpd_2m`           | Number of accounts currently 120 days past due, updated within the past two months.                                      | `conditional`            | `conditionally_available` | `data_quality_check,portfolio_segmentation` | The field may be a legitimate bureau attribute, but its update timing must be verified before it enters the model. Retain it for review rather than making a premature eligibility decision.                                                                           |
| `acc_now_delinq`             | Number of accounts on which the borrower is currently delinquent.                                                        | `yes`                    | `modeling`                | `portfolio_segmentation`                    | Current delinquency on existing credit accounts is an applicant-level bureau signal that may be available when the application is evaluated.                                                                                                                           |
| `chargeoff_within_12_mths`   | Number of charge-offs within the previous 12 months.                                                                     | `yes`                    | `modeling`                | `portfolio_segmentation`                    | Recent charge-offs describe the applicant's prior credit history and may be used as an application-time risk indicator.                                                                                                                                                |
| `pub_rec_bankruptcies`       | Number of public-record bankruptcies.                                                                                    | `yes`                    | `modeling`                | `portfolio_segmentation`                    | Bankruptcy history is an applicant-level public-record attribute that may be available during credit assessment.                                                                                                                                                       |
| `collections_12_mths_ex_med` | Number of collections within the previous 12 months, excluding medical collections.                                      | `yes`                    | `modeling`                | `portfolio_segmentation`                    | Recent non-medical collections are existing credit-history information and may be considered as a core model input.                                                                                                                                                    |
| `num_tl_90g_dpd_24m`         | Number of accounts that were 90 or more days past due within the previous 24 months.                                     | `yes`                    | `modeling`                | `portfolio_segmentation`                    | Serious prior delinquencies are bureau-level risk indicators that may be available when the applicant is assessed.                                                                                                                                                     |
| `delinq_2yrs`                | Number of delinquency incidents of 30 or more days past due in the borrower's credit file during the previous two years. | `yes`                    | `modeling`                | `portfolio_segmentation`                    | This field summarizes prior credit behavior and does not depend on the future performance of the new LendingClub loan.                                                                                                                                                 |
| `tax_liens`                  | Number of tax liens.                                                                                                     | `yes`                    | `modeling`                | `portfolio_segmentation`                    | Tax liens are public-record attributes that may be available during the applicant's credit assessment.                                                                                                                                                                 |
| `num_accts_ever_120_pd`      | Number of accounts that have ever been 120 or more days past due.                                                        | `yes`                    | `modeling`                | `portfolio_segmentation`                    | Severe historical delinquency is a relevant bureau-level attribute that may be known before loan origination.                                                                                                                                                          |
| `pub_rec`                    | Number of derogatory public records.                                                                                     | `yes`                    | `modeling`                | `portfolio_segmentation`                    | Derogatory public records describe the borrower's pre-existing credit history and may be retained as a candidate model input.                                                                                                                                          |
| `delinq_amnt`                | Past-due amount owed on accounts for which the borrower is currently delinquent.                                         | `yes`                    | `modeling`                | `portfolio_segmentation`                    | The amount relates to existing delinquent accounts rather than future repayments on the LendingClub loan. It may be used as an application-time bureau feature.                                                                                                        |
| `tot_coll_amt`               | Total collection amounts ever owed.                                                                                      | `yes`                    | `modeling`                | `portfolio_segmentation`                    | Historical collection amounts are bureau-level attributes that may be available during underwriting and may provide useful context about prior credit difficulties.                                                                                                    |

### Batch 5 checkpoint

```text
Columns reviewed: 13
Modeling candidates: 11
Conditionally available fields requiring provenance verification: 2
Post-application leakage fields: 0
```

## 7. Batch 6 — Credit inquiries and recently opened accounts

This batch contains credit-bureau attributes related to recent borrowing activity, newly opened accounts, and the time elapsed since recent credit events.

These fields may be useful indicators of credit-seeking behavior. For example, numerous recent inquiries or several newly opened accounts may provide relevant context when estimating applicant risk.

All fields in this batch are retained as modeling candidates because they describe the applicant's existing credit profile rather than repayment performance on the newly issued LendingClub loan.

Some variables capture similar concepts across different account types or time windows. They remain eligible at this stage. Redundancy, missingness, and feature-selection decisions will be addressed during later preprocessing and modeling work.

| column_name             | column_description                                                                    | available_at_application | primary_purpose | additional_use_cases     | decision_reason                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------- | ------------------------ | --------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `open_acc_6m`           | Number of credit accounts opened within the previous six months.                      | `yes`                    | `modeling`      | `portfolio_segmentation` | Recent account-opening activity is a credit-bureau attribute that may be available when the applicant is evaluated.              |
| `open_il_12m`           | Number of installment accounts opened within the previous 12 months.                  | `yes`                    | `modeling`      | `portfolio_segmentation` | Recent installment-account activity may provide useful context about the applicant's borrowing behavior.                         |
| `mths_since_recent_inq` | Number of months since the applicant's most recent credit inquiry.                    | `yes`                    | `modeling`      | `portfolio_segmentation` | The recency of the latest inquiry is an application-time bureau signal and may help characterize recent credit-seeking behavior. |
| `open_rv_12m`           | Number of revolving-credit accounts opened within the previous 12 months.             | `yes`                    | `modeling`      | `portfolio_segmentation` | Newly opened revolving accounts are part of the applicant's existing credit profile and may be considered during risk modeling.  |
| `inq_last_6mths`        | Number of credit inquiries during the previous six months.                            | `yes`                    | `modeling`      | `portfolio_segmentation` | Recent inquiries may indicate credit-seeking activity and are available from the applicant's credit file during assessment.      |
| `inq_fi`                | Number of personal-finance inquiries.                                                 | `yes`                    | `modeling`      | `portfolio_segmentation` | Personal-finance inquiries are bureau-level information that may provide additional context about recent borrowing activity.     |
| `open_il_24m`           | Number of installment accounts opened within the previous 24 months.                  | `yes`                    | `modeling`      | `portfolio_segmentation` | This field captures installment-account activity across a longer recent window and may be used as a candidate feature.           |
| `num_tl_op_past_12m`    | Number of credit accounts opened within the previous 12 months.                       | `yes`                    | `modeling`      | `portfolio_segmentation` | The field summarizes recent account-opening activity across trade types and may be relevant to applicant risk.                   |
| `inq_last_12m`          | Number of credit inquiries during the previous 12 months.                             | `yes`                    | `modeling`      | `portfolio_segmentation` | This field provides a broader view of recent inquiry activity and may complement shorter-window inquiry measures.                |
| `open_rv_24m`           | Number of revolving-credit accounts opened within the previous 24 months.             | `yes`                    | `modeling`      | `portfolio_segmentation` | The field describes revolving-account activity during a longer recent period and is available from the credit profile.           |
| `acc_open_past_24mths`  | Number of credit accounts opened within the previous 24 months.                       | `yes`                    | `modeling`      | `portfolio_segmentation` | This bureau-level measure captures recent account growth across trade types and may be considered as a model input.              |
| `mo_sin_rcnt_tl`        | Number of months since the applicant's most recently opened credit account.           | `yes`                    | `modeling`      | `portfolio_segmentation` | Account-opening recency is part of the existing credit profile and may provide context about recent borrowing activity.          |
| `mo_sin_rcnt_rev_tl_op` | Number of months since the applicant's most recently opened revolving-credit account. | `yes`                    | `modeling`      | `portfolio_segmentation` | This field provides a revolving-credit-specific measure of account-opening recency.                                              |
| `mths_since_rcnt_il`    | Number of months since the applicant's most recently opened installment account.      | `yes`                    | `modeling`      | `portfolio_segmentation` | This field provides an installment-credit-specific measure of account-opening recency.                                           |
| `mths_since_recent_bc`  | Number of months since the applicant's most recently opened bankcard account.         | `yes`                    | `modeling`      | `portfolio_segmentation` | This bankcard-specific recency measure describes the applicant's existing credit profile at evaluation time.                     |

### Batch 6 checkpoint

```text
Columns reviewed: 15
Modeling candidates: 15
Conditionally available fields: 0
Post-application leakage fields: 0
```

## 8. Batch 7 — Account counts and credit mix

This batch contains credit-bureau attributes describing the number and type of accounts in the applicant's existing credit profile.

The fields capture different aspects of credit mix, including revolving accounts, installment accounts, bankcard accounts, mortgage accounts, and satisfactory accounts.

All fields in this batch are retained as modeling candidates because they describe information that may be available when the applicant is evaluated. Some fields overlap conceptually, but eligibility review is separate from later redundancy analysis and feature selection.

| column_name           | column_description                                                     | available_at_application | primary_purpose | additional_use_cases     | decision_reason                                                                                                                              |
| --------------------- | ---------------------------------------------------------------------- | ------------------------ | --------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_actv_bc_tl`      | Number of currently active bankcard accounts.                          | `yes`                    | `modeling`      | `portfolio_segmentation` | Active bankcard accounts are part of the applicant's existing bureau profile and may provide relevant context about current credit activity. |
| `mort_acc`            | Number of mortgage accounts in the applicant's credit file.            | `yes`                    | `modeling`      | `portfolio_segmentation` | Mortgage-account history contributes to the applicant's credit mix and may be considered during risk modeling.                               |
| `num_rev_tl_bal_gt_0` | Number of revolving-credit accounts with a balance greater than zero.  | `yes`                    | `modeling`      | `portfolio_segmentation` | The field measures active revolving-credit usage and may help characterize the applicant's existing debt profile.                            |
| `open_act_il`         | Number of currently active installment accounts.                       | `yes`                    | `modeling`      | `portfolio_segmentation` | Active installment accounts are part of the applicant's bureau profile and may provide useful context about ongoing obligations.             |
| `num_actv_rev_tl`     | Number of currently active revolving-credit accounts.                  | `yes`                    | `modeling`      | `portfolio_segmentation` | The field captures the applicant's active revolving-credit exposure and may be relevant for risk estimation.                                 |
| `num_bc_sats`         | Number of satisfactory bankcard accounts.                              | `yes`                    | `modeling`      | `portfolio_segmentation` | Satisfactory bankcard accounts describe the applicant's current credit profile and may provide additional credit-quality context.            |
| `total_cu_tl`         | Number of credit-union accounts in the applicant's credit file.        | `yes`                    | `modeling`      | `portfolio_segmentation` | Credit-union account counts contribute to the broader credit-mix profile and may be considered as a candidate feature.                       |
| `num_bc_tl`           | Number of bankcard accounts in the applicant's credit file.            | `yes`                    | `modeling`      | `portfolio_segmentation` | The number of bankcard accounts is an application-time bureau attribute and may help characterize credit mix.                                |
| `num_op_rev_tl`       | Number of open revolving-credit accounts.                              | `yes`                    | `modeling`      | `portfolio_segmentation` | Open revolving accounts are part of the applicant's existing credit profile and may provide relevant risk context.                           |
| `num_sats`            | Number of satisfactory accounts.                                       | `yes`                    | `modeling`      | `portfolio_segmentation` | The field summarizes satisfactory accounts in the bureau profile and may contribute to a broader view of credit quality.                     |
| `open_acc`            | Number of open credit lines in the applicant's credit file.            | `yes`                    | `modeling`      | `portfolio_segmentation` | Open credit-line counts are available from the credit file during assessment and may be relevant to applicant risk.                          |
| `num_rev_accts`       | Number of revolving-credit accounts in the applicant's credit file.    | `yes`                    | `modeling`      | `portfolio_segmentation` | The field captures the applicant's revolving-credit history and may be retained as a candidate model input.                                  |
| `num_il_tl`           | Number of installment accounts in the applicant's credit file.         | `yes`                    | `modeling`      | `portfolio_segmentation` | Installment-account counts contribute to the applicant's credit-mix profile and may be considered during modeling.                           |
| `total_acc`           | Total number of credit lines currently in the applicant's credit file. | `yes`                    | `modeling`      | `portfolio_segmentation` | The total account count provides a broad summary of the applicant's credit history and may be used as an application-time feature.           |

### Batch 7 checkpoint

```text
Columns reviewed: 14
Modeling candidates: 14
Conditionally available fields: 0
Post-application leakage fields: 0
```


## 9. Batch 8 — Credit-history age, derogatory-event recency, and FICO range

This batch contains credit-bureau attributes describing the length of the applicant's credit history, the recency of prior adverse credit events, and the applicant's FICO score range.

These fields describe the applicant's existing credit profile rather than future performance on the LendingClub loan. They are retained as modeling candidates.

The two FICO-range fields require special attention. FICO is an external bureau-derived credit-risk signal rather than a raw borrower characteristic. The fields remain eligible because a credit score may realistically be available during applicant evaluation. However, later modeling work should compare results with and without FICO inputs to measure how much the model relies on the external score.

Some fields may be missing when the applicant has never experienced the relevant event. Missing-value interpretation and preprocessing will be handled in a later module.

| column_name                      | column_description                                                                                                  | available_at_application | primary_purpose | additional_use_cases                        | decision_reason                                                                                                                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------ | --------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fico_range_high`                | Upper boundary of the range containing the borrower's FICO score.                                                   | `yes`                    | `modeling`      | `benchmarking,portfolio_segmentation`       | The score range is available from the applicant's credit profile during evaluation. Because it is an external bureau-derived risk signal, later modeling should compare results with and without this field. |
| `fico_range_low`                 | Lower boundary of the range containing the borrower's FICO score.                                                   | `yes`                    | `modeling`      | `benchmarking,portfolio_segmentation`       | The score range is available during applicant evaluation. It is eligible for modeling but should be analyzed alongside `fico_range_high` for redundancy and dependence on external scoring logic.            |
| `mths_since_last_record`         | Number of months since the applicant's most recent public record.                                                   | `yes`                    | `modeling`      | `portfolio_segmentation`                    | The recency of a public record describes the applicant's existing credit history and may be used as an application-time risk indicator.                                                                      |
| `mths_since_last_delinq`         | Number of months since the applicant's most recent delinquency.                                                     | `yes`                    | `modeling`      | `portfolio_segmentation`                    | Prior-delinquency recency is part of the applicant's bureau history and may provide relevant context during risk assessment.                                                                                 |
| `mths_since_recent_bc_dlq`       | Number of months since the applicant's most recent bankcard delinquency.                                            | `yes`                    | `modeling`      | `portfolio_segmentation`                    | Bankcard-delinquency recency is a bureau-level risk attribute that may be available when the application is evaluated.                                                                                       |
| `mths_since_recent_revol_delinq` | Number of months since the applicant's most recent revolving-credit delinquency.                                    | `yes`                    | `modeling`      | `portfolio_segmentation`                    | The field describes a prior adverse event within the applicant's existing revolving-credit history and does not rely on future LendingClub-loan performance.                                                 |
| `mths_since_last_major_derog`    | Number of months since the applicant's most recent major derogatory credit event, such as a 90-day-or-worse rating. | `yes`                    | `modeling`      | `portfolio_segmentation`                    | The recency of a major adverse credit event may be available from the applicant's bureau profile and is relevant for applicant-risk estimation.                                                              |
| `mo_sin_old_il_acct`             | Number of months since the applicant's oldest installment account was opened.                                       | `yes`                    | `modeling`      | `portfolio_segmentation`                    | The age of the oldest installment account contributes to the applicant's credit-history profile and may be considered as a model input.                                                                      |
| `earliest_cr_line`               | Date on which the applicant's earliest reported credit line was opened.                                             | `yes`                    | `modeling`      | `data_quality_check,portfolio_segmentation` | The field provides information about the length of the applicant's credit history. The raw date will likely need to be converted into a duration during preprocessing.                                       |
| `mo_sin_old_rev_tl_op`           | Number of months since the applicant's oldest revolving-credit account was opened.                                  | `yes`                    | `modeling`      | `portfolio_segmentation`                    | The field measures the age of the applicant's revolving-credit history and may be retained as an application-time feature.                                                                                   |
| `pct_tl_nvr_dlq`                 | Percentage of the applicant's credit accounts that have never been delinquent.                                      | `yes`                    | `modeling`      | `portfolio_segmentation`                    | This bureau-level summary describes the applicant's historical repayment behavior across existing accounts and may be considered as a model input.                                                           |

### Batch 8 checkpoint

```text
Columns reviewed: 11
Modeling candidates: 11
Fields flagged for benchmarking: 2
Conditionally available fields: 0
Post-application leakage fields: 0
```

## 10. Batch 9 — Utilization and revolving-credit measures

This batch contains credit-bureau attributes describing how much of the applicant's available credit is currently being used.

These measures help distinguish between merely having access to credit and actively relying on that available credit. For example, two applicants may have similar credit limits but substantially different outstanding balances and utilization rates.

All fields in this batch are retained as modeling candidates because they describe the applicant's existing credit profile at evaluation time rather than repayment performance on the newly issued LendingClub loan.

Several fields overlap conceptually. This is acceptable during eligibility review. Redundancy analysis, transformations, and final feature selection will be handled during later preprocessing and modeling work.

| column_name        | column_description                                                                                                    | available_at_application | primary_purpose | additional_use_cases     | decision_reason                                                                                                                                    |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- | ------------------------ | --------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `all_util`         | Balance-to-credit-limit ratio across all credit accounts.                                                             | `yes`                    | `modeling`      | `portfolio_segmentation` | Overall utilization summarizes how heavily the applicant is using available credit across account types and may provide relevant risk context.     |
| `il_util`          | Ratio of the total current balance to the high-credit or credit-limit amount across installment accounts.             | `yes`                    | `modeling`      | `portfolio_segmentation` | Installment-credit utilization describes the applicant's existing installment debt burden relative to available credit.                            |
| `percent_bc_gt_75` | Percentage of bankcard accounts using more than 75% of their available credit limits.                                 | `yes`                    | `modeling`      | `portfolio_segmentation` | The field identifies whether the applicant has a high concentration of heavily utilized bankcard accounts and may provide useful risk information. |
| `revol_util`       | Revolving-credit utilization rate, representing the amount of revolving credit used relative to the amount available. | `yes`                    | `modeling`      | `portfolio_segmentation` | Revolving-credit utilization is an application-time bureau measure that may help characterize reliance on available revolving credit.              |
| `bc_util`          | Ratio of the total current balance to the high-credit or credit-limit amount across bankcard accounts.                | `yes`                    | `modeling`      | `portfolio_segmentation` | Bankcard utilization provides a more specific view of revolving-credit usage and may be considered as a candidate model input.                     |
| `bc_open_to_buy`   | Total unused credit available across revolving bankcard accounts.                                                     | `yes`                    | `modeling`      | `portfolio_segmentation` | Available bankcard credit complements utilization measures by capturing the remaining credit capacity available to the applicant.                  |
| `revol_bal`        | Total revolving-credit balance.                                                                                       | `yes`                    | `modeling`      | `portfolio_segmentation` | The outstanding revolving balance is part of the applicant's existing credit profile and may provide context about current debt obligations.       |
| `total_bc_limit`   | Total high-credit or credit-limit amount across bankcard accounts.                                                    | `yes`                    | `modeling`      | `portfolio_segmentation` | The total bankcard limit provides context for balance and utilization fields and may be retained as a candidate model input.                       |
| `total_rev_hi_lim` | Total high-credit or credit-limit amount across revolving-credit accounts.                                            | `yes`                    | `modeling`      | `portfolio_segmentation` | The total revolving-credit limit helps contextualize outstanding revolving balances and utilization levels.                                        |
| `max_bal_bc`       | Maximum current balance owed across revolving accounts.                                                               | `yes`                    | `modeling`      | `portfolio_segmentation` | The maximum balance provides information about the applicant's largest revolving-credit exposure and may be useful during risk estimation.         |

### Batch 9 checkpoint

```text
Columns reviewed: 10
Modeling candidates: 10
Conditionally available fields: 0
Post-application leakage fields: 0
```

## 11. Batch 10 — Income, debt burden, and aggregate balances

This batch contains borrower-income information and credit-bureau measures describing the applicant's existing debt burden, account balances, and available credit limits.

These fields provide broader financial context than the account-specific measures reviewed in earlier batches. For example, utilization ratios show how heavily particular credit lines are used, while aggregate balance fields summarize the overall scale of the applicant's obligations.

All fields in this batch are retained as modeling candidates because they describe information available during applicant evaluation rather than repayment performance on the newly issued LendingClub loan.

Some fields may overlap with more granular bureau attributes. This is acceptable during eligibility review. Redundancy analysis, transformations, and final feature selection will be handled during later preprocessing and modeling work.

| column_name                  | column_description                                                                                                                 | available_at_application | primary_purpose | additional_use_cases                        | decision_reason                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | --------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `dti`                        | Borrower's debt-to-income ratio, calculated from monthly debt payments excluding mortgage divided by self-reported monthly income. | `yes`                    | `modeling`      | `portfolio_segmentation`                    | Debt-to-income ratio summarizes the applicant's existing debt burden relative to income and is available during evaluation.               |
| `annual_inc`                 | Annual income provided by the borrower during registration.                                                                        | `yes`                    | `modeling`      | `data_quality_check,portfolio_segmentation` | Borrower-reported annual income is available during the application process and provides important context for repayment capacity.        |
| `avg_cur_bal`                | Average current balance across the applicant's credit accounts.                                                                    | `yes`                    | `modeling`      | `portfolio_segmentation`                    | The field summarizes the average scale of outstanding balances in the applicant's existing bureau profile.                                |
| `total_bal_il`               | Total current balance across installment accounts.                                                                                 | `yes`                    | `modeling`      | `portfolio_segmentation`                    | Existing installment-account balances contribute to the applicant's overall debt profile and may be relevant for risk estimation.         |
| `total_il_high_credit_limit` | Total high-credit or credit-limit amount across installment accounts.                                                              | `yes`                    | `modeling`      | `portfolio_segmentation`                    | The field provides context for installment balances and utilization levels within the applicant's existing credit profile.                |
| `total_bal_ex_mort`          | Total credit balance excluding mortgage debt.                                                                                      | `yes`                    | `modeling`      | `portfolio_segmentation`                    | Non-mortgage debt balances provide useful context about the applicant's current obligations without being dominated by mortgage exposure. |
| `tot_cur_bal`                | Total current balance across all accounts.                                                                                         | `yes`                    | `modeling`      | `portfolio_segmentation`                    | This field summarizes the applicant's overall outstanding credit balances at evaluation time.                                             |
| `tot_hi_cred_lim`            | Total high-credit or credit-limit amount across all accounts.                                                                      | `yes`                    | `modeling`      | `portfolio_segmentation`                    | The total credit-limit amount provides broader context for the applicant's balance and utilization measures.                              |

### Batch 10 checkpoint

```text
Columns reviewed: 8
Modeling candidates: 8
Conditionally available fields: 0
Post-application leakage fields: 0
```

## 12. Batch 11 — Secondary-applicant bureau fields

This batch contains credit-profile attributes for the secondary applicant on a joint loan application.

These fields do not describe post-origination repayment performance. They may contain legitimate application-time information, but they apply only when the application includes a co-borrower. Individual applications will generally have missing values.

The fields are therefore classified as `conditionally_available` rather than ordinary modeling candidates. Before preprocessing begins, the project should decide how joint applications will be handled. Possible approaches include retaining these fields with explicit missing-value logic, engineering joint-applicant summaries, or building a separate treatment path for joint applications.

The secondary-applicant FICO range is also flagged for benchmarking because it contains an external bureau-derived risk signal.

| column_name                           | column_description                                                                                                            | available_at_application | primary_purpose           | additional_use_cases                        | decision_reason                                                                                                                                                                           |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sec_app_inq_last_6mths`              | Number of credit inquiries during the previous six months for the secondary applicant.                                        | `conditional`            | `conditionally_available` | `portfolio_segmentation`                    | The field may be available when evaluating a joint application, but it does not apply to individual applications.                                                                         |
| `sec_app_collections_12_mths_ex_med`  | Number of collections during the previous 12 months for the secondary applicant, excluding medical collections.               | `conditional`            | `conditionally_available` | `portfolio_segmentation`                    | The field may provide relevant credit-history context for a co-borrower but is only populated for applicable joint applications.                                                          |
| `sec_app_chargeoff_within_12_mths`    | Number of charge-offs during the previous 12 months for the secondary applicant.                                              | `conditional`            | `conditionally_available` | `portfolio_segmentation`                    | Recent charge-offs may be relevant to a joint application, but the field does not apply when there is no secondary applicant.                                                             |
| `sec_app_mort_acc`                    | Number of mortgage accounts for the secondary applicant.                                                                      | `conditional`            | `conditionally_available` | `portfolio_segmentation`                    | Mortgage-account history may contribute to the co-borrower's credit profile but is only relevant for joint applications.                                                                  |
| `sec_app_open_act_il`                 | Number of currently active installment accounts for the secondary applicant.                                                  | `conditional`            | `conditionally_available` | `portfolio_segmentation`                    | The field may describe the co-borrower's existing obligations but requires joint-application handling.                                                                                    |
| `sec_app_open_acc`                    | Number of open credit accounts for the secondary applicant.                                                                   | `conditional`            | `conditionally_available` | `portfolio_segmentation`                    | Open-account counts may be available for a co-borrower during evaluation but will not apply to individual applications.                                                                   |
| `sec_app_fico_range_high`             | Upper boundary of the secondary applicant's FICO-score range.                                                                 | `conditional`            | `conditionally_available` | `benchmarking,portfolio_segmentation`       | The field may be useful when evaluating a joint application. Because it contains an external bureau-derived risk signal, later modeling should measure its incremental effect separately. |
| `sec_app_fico_range_low`              | Lower boundary of the secondary applicant's FICO-score range.                                                                 | `conditional`            | `conditionally_available` | `benchmarking,portfolio_segmentation`       | The field may be useful for joint-applicant risk assessment but should be reviewed alongside the upper boundary for redundancy and external-score dependence.                             |
| `sec_app_num_rev_accts`               | Number of revolving-credit accounts for the secondary applicant.                                                              | `conditional`            | `conditionally_available` | `portfolio_segmentation`                    | The field may provide information about the co-borrower's revolving-credit profile but applies only to joint applications.                                                                |
| `sec_app_mths_since_last_major_derog` | Number of months since the secondary applicant's most recent major derogatory credit event, such as a 90-day-or-worse rating. | `conditional`            | `conditionally_available` | `portfolio_segmentation`                    | The recency of a severe adverse event may be relevant when a co-borrower exists, but it does not apply to individual applications.                                                        |
| `sec_app_earliest_cr_line`            | Date on which the secondary applicant's earliest reported credit line was opened.                                             | `conditional`            | `conditionally_available` | `data_quality_check,portfolio_segmentation` | The field may provide information about the co-borrower's credit-history length. The raw date will likely need to be converted into a duration during preprocessing.                      |
| `sec_app_revol_util`                  | Revolving-credit utilization ratio for the secondary applicant.                                                               | `conditional`            | `conditionally_available` | `portfolio_segmentation`                    | The field may help characterize the co-borrower's reliance on available revolving credit but requires a deliberate strategy for joint applications.                                       |

### Batch 11 checkpoint

```text
Columns reviewed: 12
Conditionally available secondary-applicant fields: 12
Fields flagged for benchmarking: 2
Post-application leakage fields: 0
```

## 13. Batch 12 — Joint-financial fields

This batch contains financial attributes that describe the combined income, debt burden, and revolving-credit balances of co-borrowers on joint applications.

These fields do not reveal future repayment performance. They may contain legitimate application-time information, but they apply only when the application includes a co-borrower. Individual applications will generally have missing values.

The fields are therefore classified as `conditionally_available` rather than ordinary modeling candidates. Their eventual use depends on the preprocessing strategy selected for joint applications.

| column_name        | column_description                                                                | available_at_application | primary_purpose           | additional_use_cases                        | decision_reason                                                                                                                                                                                             |
| ------------------ | --------------------------------------------------------------------------------- | ------------------------ | ------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dti_joint`        | Combined debt-to-income ratio for the co-borrowers on a joint application.        | `conditional`            | `conditionally_available` | `portfolio_segmentation`                    | The value may provide relevant repayment-capacity context for a joint application, but it does not apply to individual applications. Its use requires an explicit joint-application preprocessing strategy. |
| `annual_inc_joint` | Combined annual income reported for the co-borrowers on a joint application.      | `conditional`            | `conditionally_available` | `data_quality_check,portfolio_segmentation` | Joint income may provide useful repayment-capacity context when a co-borrower exists. The field does not apply to individual applications and must be handled deliberately during preprocessing.            |
| `revol_bal_joint`  | Combined revolving-credit balance of the co-borrowers, net of duplicate balances. | `conditional`            | `conditionally_available` | `portfolio_segmentation`                    | The field may help describe the combined revolving-credit burden of co-borrowers. It is relevant only for joint applications and requires an explicit missing-value and preprocessing strategy.             |

### Batch 12 checkpoint

```text
Columns reviewed: 3
Conditionally available joint-financial fields: 3
Post-application leakage fields: 0
```

---

## 14. Closing statement

The manual review of the unresolved LendingClub columns is complete.

The preliminary column-review inventory identified 109 columns that could not be classified safely using simple automatic rules. These columns were reviewed across 12 semantic batches covering:

```text
application and loan setup
operational and timing-sensitive fields
geography and high-cardinality text
lender-derived pricing and funding
delinquency and public records
credit inquiries and recently opened accounts
account counts and credit mix
credit-history age and FICO range
utilization and revolving credit
income and aggregate balances
secondary-applicant bureau attributes
joint-financial attributes
```

The review distinguishes between:

```text
legitimate modeling candidates
monitoring fields
lender-derived features
conditionally available fields
high-cardinality or text fields
post-application leakage
fields excluded from the core model
```

This review establishes feature eligibility rather than the final trained-model feature set.

A column classified as `modeling` is eligible for later preprocessing and model evaluation. It may still be excluded during later modules because of missingness, redundancy, instability, interpretability concerns, or model-selection results.

The next step is to convert the documented decisions into a lightweight machine-readable inventory:

```text
config/column_usage_inventory.csv
```

The CSV will contain:

```text
column_name
primary_purpose
additional_use_cases
```

Before the CSV is used by later project modules, it should be validated to confirm that:

```text
- every dataset column appears exactly once
- all 109 manually reviewed columns are included
- automatically classified columns are also included
- every primary_purpose uses an approved value
- every additional_use_cases entry uses approved discrete values
- no unknown or duplicated column names exist
```
