# Column Review Methodology

## 1. Purpose

This document defines the process for reviewing LendingClub columns before preprocessing and modeling.

The goal is to determine which columns are legitimate inputs for an applicant-style credit risk model and which columns must be excluded, retained only for monitoring, or handled separately.

The model is intended to estimate the probability of a bad-loan outcome using information that would realistically be available when a booked-loan applicant is assessed. It must not learn from repayment outcomes, post-origination events, or fields that reveal future information.

---

## 2. Current review population

The target-defined interim dataset contains:

```text
1,369,566 rows
152 columns
```

The preliminary inventory script automatically classified obvious cases such as:

```text
target
target_source
identifier
monitoring_only
post_application_leakage
constant_or_empty
```

After the automatic rules were applied, the remaining unresolved population was:

```text
109 columns marked as requires_review
```

These columns require manual interpretation before feature eligibility can be finalized.

---

## 3. Core review question

For each unresolved column, the primary question is:

```text
Would this value realistically be known when the applicant is being evaluated?
```

A column should not become a core model feature merely because it improves predictive performance.

The column must also satisfy the project framing:

```text
Use only information that could reasonably exist at application or underwriting time.
```

---

## 4. Review principles

### 4.1 Prevent post-application leakage

Any field generated after loan issuance, during repayment, after delinquency, or during collections must be excluded from model features.

Examples include:

```text
recoveries
total_pymnt
last_pymnt_d
settlement_status
hardship_amount
```

### 4.2 Separate applicant information from lender-derived information

Some fields may exist before origination but were created by LendingClub’s internal underwriting or pricing process.

Examples may include:

```text
grade
sub_grade
int_rate
```

These should not automatically enter the core model because the project should not simply reproduce LendingClub’s existing decision logic.

They may still be useful for benchmarking, comparison, or later sensitivity analysis.

### 4.3 Retain useful monitoring fields separately

Some columns may be valuable for cohort analysis or portfolio monitoring but should not become predictive inputs.

Example:

```text
issue_d
```

### 4.4 Preserve ambiguity instead of forcing premature decisions

When availability or business meaning is uncertain, classify the field conservatively and document the uncertainty.

A field may be marked:

```text
conditionally_available
```

until its operational timing is clarified.

### 4.5 Review related columns together

Columns should be reviewed in semantic batches rather than alphabetical order.

Reviewing related variables together helps identify:

```text
duplicate concepts
derived fields
related families
timing differences
inconsistent classifications
```

---

## 5. Final role categories

Each manually reviewed column should receive one final role.

| Final role                    | Meaning                                                                                                                             |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `candidate_feature`           | Legitimate application-time or bureau-time information that may enter the core model                                                |
| `lender_derived_feature`      | Created by LendingClub’s underwriting, pricing, or risk process; retain for benchmarking but exclude from the core model by default |
| `monitoring_only`             | Useful for portfolio monitoring, cohort analysis, or reporting but not as a predictive feature                                      |
| `post_application_leakage`    | Generated after origination or influenced by repayment performance; exclude                                                         |
| `conditionally_available`     | Potentially valid, but availability depends on the decisioning stage or applicant type                                              |
| `high_cardinality_or_text`    | Textual, geographic, or highly granular field requiring separate treatment and explicit justification                               |
| `exclude_from_model_features` | Not suitable for the core model because of ambiguity, redundancy, operational concerns, or limited value                            |

---

## 6. Recommended review fields

The finalized inventory should preserve both the automatic classification and the manual decision.

Recommended fields:

| Field                      | Purpose                                                     |
| -------------------------- | ----------------------------------------------------------- |
| `column_name`              | Original dataset column                                     |
| `dtype`                    | Data type from the column-profile report                    |
| `missing_count`            | Number of missing values                                    |
| `missing_percentage`       | Percentage of missing values                                |
| `unique_non_missing_count` | Number of unique non-missing values                         |
| `is_constant_non_missing`  | Whether the non-missing values are constant                 |
| `preliminary_role`         | Automatic classification from the inventory script          |
| `final_role`               | Manual classification after review                          |
| `available_at_application` | `yes`, `no`, `conditional`, or `not_applicable`             |
| `recommended_action`       | Whether to retain, benchmark, monitor, or exclude           |
| `reason`                   | Short explanation for the decision                          |
| `review_status`            | `pending_review`, `manually_reviewed`, or `auto_classified` |
| `review_batch`             | Semantic batch used during manual review                    |

---

## 7. Review workflow

For each batch:

1. Review the LendingClub definition of each field.
2. Determine whether the value is available at application time, during underwriting, after origination, or only after repayment activity.
3. Check whether the field is raw applicant information, bureau information, lender-derived information, or post-application information.
4. Compare related columns for overlap, redundancy, or derived relationships.
5. Assign a final role.
6. Record the recommended action and a short reason.
7. Mark the review status as `manually_reviewed`.
8. Confirm that all columns in the batch were addressed before moving to the next batch.

---

## 8. Proposed batching plan

The 109 unresolved columns will be reviewed in 12 semantic batches.

The grouping is organizational only. Placement in a batch does not determine the final role.

### Batch 1 — Application and loan setup

```text
application_type
term
verification_status
verification_status_joint
home_ownership
purpose
emp_length
```

Count:

```text
7
```

### Batch 2 — Operational and timing-sensitive fields

```text
disbursement_method
initial_list_status
pymnt_plan
orig_projected_additional_accrued_interest
```

Count:

```text
4
```

### Batch 3 — Geography and high-cardinality text fields

```text
addr_state
zip_code
title
desc
emp_title
```

Count:

```text
5
```

### Batch 4 — Lender-derived, pricing, and funding fields

```text
grade
sub_grade
int_rate
installment
loan_amnt
funded_amnt
funded_amnt_inv
```

Count:

```text
7
```

### Batch 5 — Delinquency, collections, and public records

```text
num_tl_30dpd
num_tl_120dpd_2m
acc_now_delinq
chargeoff_within_12_mths
pub_rec_bankruptcies
collections_12_mths_ex_med
num_tl_90g_dpd_24m
delinq_2yrs
tax_liens
num_accts_ever_120_pd
pub_rec
delinq_amnt
tot_coll_amt
```

Count:

```text
13
```

### Batch 6 — Credit inquiries and recently opened accounts

```text
open_acc_6m
open_il_12m
mths_since_recent_inq
open_rv_12m
inq_last_6mths
inq_fi
open_il_24m
num_tl_op_past_12m
inq_last_12m
open_rv_24m
acc_open_past_24mths
mo_sin_rcnt_tl
mo_sin_rcnt_rev_tl_op
mths_since_rcnt_il
mths_since_recent_bc
```

Count:

```text
15
```

### Batch 7 — Account counts and credit mix

```text
num_actv_bc_tl
mort_acc
num_rev_tl_bal_gt_0
open_act_il
num_actv_rev_tl
num_bc_sats
total_cu_tl
num_bc_tl
num_op_rev_tl
num_sats
open_acc
num_rev_accts
num_il_tl
total_acc
```

Count:

```text
14
```

### Batch 8 — Credit-history age, derogatory-event recency, and FICO range

```text
fico_range_high
fico_range_low
mths_since_last_record
mths_since_last_delinq
mths_since_recent_bc_dlq
mths_since_recent_revol_delinq
mths_since_last_major_derog
mo_sin_old_il_acct
earliest_cr_line
mo_sin_old_rev_tl_op
pct_tl_nvr_dlq
```

Count:

```text
11
```

### Batch 9 — Utilization and revolving-credit measures

```text
all_util
il_util
percent_bc_gt_75
revol_util
bc_util
bc_open_to_buy
revol_bal
total_bc_limit
total_rev_hi_lim
max_bal_bc
```

Count:

```text
10
```

### Batch 10 — Income, debt burden, and aggregate balances

```text
dti
annual_inc
avg_cur_bal
total_bal_il
total_il_high_credit_limit
total_bal_ex_mort
tot_cur_bal
tot_hi_cred_lim
```

Count:

```text
8
```

### Batch 11 — Secondary-applicant bureau fields

```text
sec_app_inq_last_6mths
sec_app_collections_12_mths_ex_med
sec_app_chargeoff_within_12_mths
sec_app_mort_acc
sec_app_open_act_il
sec_app_open_acc
sec_app_fico_range_high
sec_app_fico_range_low
sec_app_num_rev_accts
sec_app_mths_since_last_major_derog
sec_app_earliest_cr_line
sec_app_revol_util
```

Count:

```text
12
```

### Batch 12 — Joint-financial fields

```text
dti_joint
annual_inc_joint
revol_bal_joint
```

Count:

```text
3
```

---

## 9. Review accounting check

The total number of manually reviewed columns must equal:

```text
7+4+5+7+13+15+14+11+10+8+12+3 = 109
```

Before the inventory is finalized, confirm that:

```text
all 109 requires_review columns were assigned to exactly one batch
no columns were omitted
no columns were duplicated across batches
all manual-review rows have final_role populated
all manual-review rows have review_status = manually_reviewed
```

---

## 10. Module 2 completion condition

Module 2 is complete when:

```text
1. The preliminary column-review inventory has been generated.
2. All automatically classified columns have been verified.
3. All 109 requires_review columns have been manually reviewed.
4. Each column has a documented final role and recommended action.
5. The finalized inventory has been validated.
6. Reusable scripts and documentation have been committed to GitHub.
```

Preprocessing, imputation, encoding, feature engineering, and modeling begin only after this review is complete.
