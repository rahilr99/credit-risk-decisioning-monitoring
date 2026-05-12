# Data Dictionary

## 1. Purpose of the Data Dictionary
The purpose of the data dictionary is to document how each important LendingClub column will be understood and used in this project. This file connects the dataset selection, target definition, and leakage review into a practical column-level reference. 

The data dictionary will help identify which fields are safe candidate features, which fields define the target, which fields are useful only for monitoring, and which fields should be excluded from model training. This is especially important because the LendingClub dataset contains information from different stages of the loan lifecycle, including application/origination fields, lender-generated decision fields, and post-origination performance fields. 

For the first version of this file, the goal is not to fully document every column in the raw dataset. The initial goal is to create a clear review structure and classify the most important known fields. After the raw dataset is loaded and inspected, this file will be updated with the full list of selected columns and their final project roles. 

The main rule is that every column should have a clearly documented purpose before it is used in the project. No field should enter the model training pipeline unless it is understood, classified, and checked against the leakage review framework. 


## 2. Column Classification Framework
Each column will be reviewed using the same classification framework. The goal is to decide whether the field can be used for model training, target creation, monitoring, or should be excluded. 

| Classification | Meaning |
| --- | --- |
|Modeling candidate|A field that may be used as an input feature because it is available at application or origination time.|
|Target/Outcome field|A field used to define or evaluate the repayment outcome, such as `loan_stauts` or the derived target `bad_loan`|
|Monitoring-only field|A field that should not be used for model training but may be useful for portfolio monitoring or outcome analysis.|
|Suspicious field|A field that may be available near origination but could reflect LendingClub's own underwriting, grading, or pricing decision.|
|Excluded field|A field that should not be used because it creates leakage, has no useful modeling value, is redundant, or is outside the project scope|

Each column will also be reviewed based on timing: 

|Timing Category|Meaning|
|---|---|
|Application/origination|Information available before or around the time the loan is issued.|
|Lender-generated|Information created by LendingClub's own decisioning, grading, or pricing process.|
|Post-origination|Information created after the loan starts performing.|
|Outcome|Information that directly describes the final or current repayment status of the loan.|
|Administrative|IDs, URLs, metadata, or fields that do not directly support modeling or monitoring.|

The most important distinction is between fields that can be used before a loan decision is made and fields that are only known after the loan has started performing. Application/origination fields may be considered for modeling. Post-origination and outcome fields should be reserved for target creation, evaluation, monitoring, or exclusion.


## 3. Initial Column Review Table

| Column | Meaning | Data type | Timing | Modeling use | Monitoring use | Leakage status | Notes |
|---|---|---|---|---|---|---|---|
| `loan_amnt` | Requested loan amount | Numeric | Application/origination | Candidate feature | Yes | Safe | Represents requested credit exposure. |
| `term` | Loan repayment term | Categorical/numeric | Application/origination | Candidate feature | Yes | Likely safe | Should verify whether it reflects requested or final approved term. |
| `purpose` | Borrower-stated loan purpose | Categorical | Application/origination | Candidate feature | Yes | Safe | Useful for segmenting loan types. |
| `annual_inc` | Borrower annual income | Numeric | Application/origination | Candidate feature | Yes | Safe | May need outlier and missing-value checks. |
| `dti` | Debt-to-income ratio | Numeric | Application/origination | Candidate feature | Yes | Safe | Important affordability variable. |
| `emp_length` | Borrower employment length | Categorical/ordinal | Application/origination | Candidate feature | Yes | Safe | May require cleaning because values are often stored as text. |
| `home_ownership` | Borrower housing status | Categorical | Application/origination | Candidate feature | Yes | Safe | Can provide borrower stability context. |
| `application_type` | Individual or joint application | Categorical | Application/origination | Candidate feature | Yes | Safe | Useful for distinguishing application structure. |
| `loan_status` | Current or final loan status | Categorical | Outcome/post-origination | No | Yes | Target/outcome | Used to create `bad_loan`; never used as model input. |
| `bad_loan` | Derived binary target | Binary | Derived outcome | No | Yes | Target | Created from `loan_status`. |
| `int_rate` | Interest rate assigned to the loan | Numeric | Lender-generated | Exclude from v1 | Yes | Suspicious | May encode LendingClub’s own risk/pricing decision. |
| `grade` | LendingClub assigned loan grade | Categorical | Lender-generated | Exclude from v1 | Yes | Suspicious | May encode LendingClub’s internal risk assessment. |
| `sub_grade` | LendingClub assigned subgrade | Categorical | Lender-generated | Exclude from v1 | Yes | Suspicious | More granular version of `grade`; exclude from v1 model. |
| `installment` | Monthly payment amount | Numeric | Lender-generated/final loan terms | Review carefully | Yes | Suspicious | May depend on interest rate, term, and approved loan structure. |
| `issue_d` | Loan issue date or month | Date/time | Origination | No for v1 model | Yes | Monitoring field | Useful for historical cohort monitoring. |
| `total_pymnt` | Total amount paid by borrower | Numeric | Post-origination | No | Yes | Leakage for modeling | Describes repayment after loan issue. |
| `total_rec_prncp` | Principal received | Numeric | Post-origination | No | Yes | Leakage for modeling | Payment-performance field. |
| `total_rec_int` | Interest received | Numeric | Post-origination | No | Yes | Leakage for modeling | Payment-performance field. |
| `last_pymnt_d` | Last payment date | Date/time | Post-origination | No | Yes | Leakage for modeling | Reveals post-issue repayment behavior. |
| `last_pymnt_amnt` | Last payment amount | Numeric | Post-origination | No | Yes | Leakage for modeling | Reveals repayment behavior. |
| `out_prncp` | Remaining outstanding principal | Numeric | Post-origination | No | Yes | Leakage for modeling | Balance after loan has started performing. |
| `recoveries` | Amount recovered after default | Numeric | Post-origination | No | Yes | Leakage for modeling | Useful only for monitoring or loss analysis. |
| `collection_recovery_fee` | Collection recovery fee | Numeric | Post-origination | No | Yes | Leakage for modeling | Collection-related field. |


## 4. Modeling Feature Candidates
Modeling feature candidates are fields that may be used as inputs for the applicant scoring model because they describe the borrower, loan request, or credit profile before or around origination. 

Initial candidate categories include loan request information, borrower financial profile, borrower stability, credit history variables, and application structure. Examples include `loan_amnt`, `term`, `purpose`, `annual_inc`, `dti`, `emp_length`, `home_ownership`, and  `application_type`. 

These fields are not automatically guaranteed to be used in the final model. During data ingestion and EDA, each candidate feature still needs to be checked for missing values, extreme outliers, inconsistent formatting, redundancy, and possible timing issues. The final model feature set will be selected only after these checks are complete. 


## 5. Target and Outcome Fields
Target and Outcome fields are used to define or evaluate the repayment outcome of a loan. The main outcome field is `loan_status`, which will be used to create the derived binary target `bad_loan`. 

The target variable `bad_loan` will be used for model training, validation, evaluation, and monitoring. It should never be included as input feature. The model should learn to predict `bad_loan` from application/origination-time information, not from fields that directly reveal the outcome. 


## 6. Monitoring-Only Fields
Monitoring-only fields are useful after the model has already produced scores. These fields help track portfolio behavior, compare predicted risk against realized outcomes, and monitor historical cohorts. 

Examples include `issue_d`, `loan_status`, `bad_loan`, payment fields, recovery fields, remaining balance fields, and derived project fields such as model score, risk band, and simulated policy decision. 

A field can be valid for monitoring while still being invalid for model training. For example, `loan_status` is necessary for outcome monitoring, but it would be direct leakage if used as a model input. 


## 7. Suspicious or Excluded Fields
Suspicious fields are not necessarily future performance fields, but they may still be inappropriate for the first model version because they reflect LendingClub's own decisioning or pricing process. Examples include `int_rate`, `grade`, `sub_grade` and possibly `installment`. 

For this MVP, `int_rate`, `grade`, and `sub_grade` will be excluded from the first model feature set. This helps the project model learn from borrower and loan characteristics directly instead of depending on LendingClub's prior risk assessment. 

Clear leakage fields will also be excluded from model training. This includes payment history, recoveries, collection activity, remaining principal, hardship fields, settlement fields, and final outcome fields. 

## 8. Notes for Later Updates
This file is an intial skeleton and will be updated after the raw LendingClub dataset is loaded and inspected. The final data dictionary should include all selected columns used in the project, their cleaned names if applicable, their role in the system, and any transformation decisions. 

During later modules, each important column should be assigned one of the following roles: modeling features, target/outcome field, monitroing-only fields, suspicious field, or excluded fields. 























