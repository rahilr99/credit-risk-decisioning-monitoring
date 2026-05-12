# Leakage Review
## 1. Purpose of Leakage Review
The purpose of leakage review is to make sure the model only uses information that would have been available before or at the time a loan was originated. Since this project is trying to estimate the probability of a bad loan outcome, the model should not be trained using fields that reveal what happened after the loan was issued. 

This step is especially important for the LendingClub dataset because it contains a mix of different field types. Some fields describe the borrower and loan at origination, such as loan amount, income, debt-to-income ratio, employment length, home ownership, and loan purpose. Other fields describe what happened after the loan started performing, such as payments recieved, recoveries, outstanding principal, collection activity, and final loan status. 

If post-origination fields are used as model inputs, the model may appear to perform well during training and evaluation, but the performance would be misleading. In a real decisioning system, those fields would not be known when the lender is deciding whether to approve, review, or reject a loan application. This would create data leakage. 

The leakage review will create a framework for separating fields into modeling features, target/outcome fields, monitoring-only fields, and excluded fields. This will help keep the applicant scoring layer realistic and will make sure the monitoring layer uses post-outcome information only where it is appropriate.

The core rule for this project is: the model can only use information that would plausibly be available at application time or loan origination time. Fields that reveal future repayments behavior can be used to define the target or support monitoring, but they cannot be used as model inputs. 

## 2. What Leakage Means in this Project
In this project, leakage means that a feature gives the model information about the loan outcome before the model should realistically know that outcome. Since the model is meant to score a loan at application or origination time, any feature created after origination must be treated carefully. 

There are three main leakage risks in this project. 

The first is direct target leakage. This happens when a column directly reveals the outcome the model is trying to predict. For example, `loan_status` cannot be used as a model input because it is used to create the target variable `bad_loan`. 

The second is post-origination leakage. This happens when a column is only known after the loan has started performing. Fields related to payments, recoveries, outstanding principal, collections, settlement, hardship activity, or last payment information fall into this category. 

The third is lender-generated decision leakage. This happens when a column may reflect LendingClub's own prior underwriting or pricing decision. Fields such as `grade`, `sub-grade`, and `int_rate` may summarize risk judgements that LendingClub had already made. If included as model inputs, the model may learn to use these fields as shortcuts instead of learning risk from raw borrower and loan characteristics such as income, debt-to-income ratio, credit history, loan amount, and loan purpose. For this reason, these fields will be excluded from the first version of the model feature set. 

For this MVP, leakage review will separate columns based on timing and business meaning. The main question for each field is: would this information be available to the project lender before making the decision, or does it reveal something that happened after the loan is booked?


## 3. Why Leakage Matters for LendingClub

Leakage matters in the LendingClub dataset because the data contains information from different stages of the loan lifecycle. Some fields describe the borrower before or at origination, some fields are created by LendingClub's own underwriting and pricing process, and some fields describe what happened after the loan was issued. 

This creates a timing problem. The applicant scoring model is supposed to behave as if it is making a prediction before the loan outcome is known. If the model uses fields from the wrong stage of the loan lifecycle, it may no longer be making a realistic prediction. 

For this project, LendingClub fields can be thought of in three broad timing groups: 

| Timing group | Meaning | Example Fields |
|---|---|---|
| Application/origination-time fields | Information that would plausibly be known before or when the loan is issued | `loan_amnt`, `term`, `annual_inc`, `dti`, `emp_length`, `home_ownership`, `purpose`|
| Lender-generated fields | Information created by LendingClub's own underwriting or pricing process | `int_rate`, `grade`, `sub_grade`, `installment` |
| Post-origination fields | Information only known after the loan starts performing | `loan_status`, `total_pymt`, `recoveries`, `last_pymt_d`, `out_prncp` |

The main risk is that these groups can look similar in a raw dataset, even though they should be treated very differently. A column may appear to be useful for prediction, but if it was created after the loan was issued or created as part of LendingClub's own decision process, it may not be valid for the first version of the model. 

The leakage review is therefore needed to separate fields by timing and purpose before the modeling begins. This prevents the model from using future outcome information or LendingClub's existing risk decisions as shortcuts. 


# 4. Feature Timing Categories
To apply the leakage review consistently, each LendingClub column will be classified based on when the information becomes available and how it should be used in the project. 

| Category | Meaning | Modeling use | Monitoring use |
| --- | --- | --- | --- |
| Application/origination-time fields | Information available before or when the loan is issued | Candidate model features | Can also be monitored |
| Lender-generated decision fields | Fields created by LendingClub's own grading or pricing process | Exclude form v1 model | Can be used for analysis or monitoring |
| Target/Outcome fields | Fields createed by LendingClub's own grading or pricing process | Do not use as features | Can be used for evaluation and monitoring |
| Post-origination performance fields | Information created after the loan starts performing | Do not use as features | Can be used for portfolio monitoring |
| Excluded/irrelevant fields | IDs, URLs, text-fields or fields with no useful modeling value | Exclude | Usually Exclude |

Examples of application/origination-time fields include `loan_amnt`, `term`, `annual_inc`, `dti`, `emp_length`, `home_ownership`, `purpose`. 

Examples of lender-generated decision fields include `int_rate`, `grade`, and `sub_grade`. 

Examples of post-origination or outcome fields include `loan_status`, `total_pymnt`, `recoveries`, `last_pymnt_d`, `out_prncp`, hardship fields, settlement fields, and collection-related fields. 

This timing framework will be used later during data ingestion and data dictionary creation to decide which columns are allowed into the model feature set and which columns are reserved for target creation, monitoring, or exclusion. 


## 5. Safe Feature Categories
Safe categories are fields that describe the borrower, loan request, or credit profile before or around origination. These fields are reasonable candidates for model training because they do not reveal the future repayment outcome. 

Examples of safe feature categories include borrower affordability, requested loan characteristics, borrower stability, credit history, and loan purpose. These categories help the model estimate risk from information that could plausibly be available when a lender is evaluating a loan. 

| Feature category | Example fields | Why it may be safe|
|---|---|---|
| Loan request information | `loan_amnt`, `term`, `purpose` | Describes the loan the borrower is requesting |
| Borrower financial profile | `annual_inc`, `dti` | Describes income and debt burden |
| Borrower stability | `emp_length`, `home_ownership` | Gives context about employment and housing situation |
| Basic application structure | `application_type` | Describes whether the application is individual or joint |

These fields are not automatically approved for the final model just because they appear safe. They still need to be checked during data ingestion for missingness, outliers, inconsistent formattting, and exact timing. However, they are the starting pool of candidate modeling features. 

The main rule is that safe features should describe the borrower or loan before the outcome happens. They should not summarize LendingClub's final decision, pricing judgement, payment history, or loan performance after origination. 


## 6. Suspicious Feature Categories
Suspicious feature categories are fields that may be available around origination, but still need careful review because they may reflect LendingClub's own decision process rather than raw borrower information. 

The main examples are lender-generated pricing and risk fields such as `int_rate`, `grade` and `sub_grade`. These fields are not future repayment outcomes, but they may already summarize LendingClub's assessment of borrower risk. If they are used as model inputs, the model may learn shortcuts from LendingClub's existing risk judgement instead of learning risk from borrower and loan characteristics directly. 

For the first version of this MVP, these fields will be excluded from the model feature set. This keeps the model focused on learning from more direct applicant and loan characterisitics such as income, debt-to-income ratio, loan amount, credit history, employment length, home ownership, and loan purpose. 

| Suspicious field type | Example fields | Concern | v1 decision |
| --- | --- | --- | --- |
| Pricing fields | `int_rate`| May reflect LendingClub's risk-based pricing decision | Exclude from model |
| LendingClub risk grades | `grade`, `sub_grade` | May encode LendingClub's internal credit risk assessment | Exclude from model |
| Payment-structure fields | `installment`| May depend on approved loan terms and interest rate | Review carefully/likely exclude |
| Final approved structure fields | fields tied to final approval terms | May reflect the lender's final offer rather than the original request | Review carefully |

These fields may still be useful for analysis or monitoring. For example, the project can compare the model's risk bands against LendingClub's original grades or interest rates to see whether the model's risk estimates move in a similar direction. However, for the first model version, they should not be used as training features. 


## 7. Clear Leakage Fields 
Clear leakage fields are columns that directly reveal the loan outcome or describe events that happened after the loan was issued. These fields should not be used as model inputs because they would give the model access to information that would not be available at application or origination time. 

The most obvious leakage field is `loan_status`, because it is used to create the target variable `bad_loan`. Including `loan_status` as a feature would allow the model to directly see the answer it is supposed to predict. 

Other clear leakage fields include payment, recovery, collection, balance, settlement, and hardship-related fields. These fields describe how the loan performed after origination, so they are invalid for application-time risk scoring.

| Leakage field type | Example fields | Why it is leakage |
| --- | --- | ---- |
| Target/outcome status | `loan_status` | Directly defines the target variable |
| Payment History | `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `last_pymnt_d`, `last_pymnt_amnt` | Describes payments made after the loan was issued |
| Remaining balance | `out_prncp`, `out_prncp_inv`| Reveals how much principal remains after loan performance begins |
| Recovery/collections | `recoveries`, `collection_recovery_fee`, collection related fields | Reveals post-origination workout or settlement activity |
| Settlement fields | settlement-related columns | Reveals post-origination workout or settlement activity |
| Hardship fields | hardship-related columns | Reveals borrower hardship activity after origination |

These fields may still be useful outside of model training. For example, `loan_status` is needed to define the target, and payment or recovery fields may be useful for portfolio monitoring or future loss analysis. However, they must be kept out of the model feature set. 

The rule is simple: if a field tells us what happened after the loan was issued, it cannot be used to predict risk before the loan is issued. 


## 8. Monitoring-Only Fields 
Monitoring-only fields are columns that should not be used to train the model, but may still be useful after the model has already produced risk scores. These fields help evaluate portfolio behavior, track realized outcomes, and compare predicted risk against actual loan performance. 

The key difference is timing. The applicant scoring model should only use information available before or at origination. The monitoring layer, however, is allowed to use later outcome performance information because monitoring happens after lians have been issued and observed over time. 

Examples of monitoring-only fields include final loan status, payment amounts, recoveriesm, remaining principal, collection activity, and historical issue-period fields. These fields can help answer questions such as: Did high-risk bands actually produce high bad-loan rates? Did certain origination cohorts perform worse than other? Is the portfolio becoming riskier over time? 

| Monitoring use case    | Example fields                                                    | Purpose                                                 |
| ---------------------- | ----------------------------------------------------------------- | ------------------------------------------------------- |
| Outcome tracking       | `loan_status`, `bad_loan`                                         | Compare predicted risk against actual repayment outcome |
| Cohort monitoring      | issue date or issue period fields                                 | Group loans by origination period                       |
| Portfolio exposure     | `loan_amnt`, funded amount fields, remaining balance fields       | Track dollar exposure across risk bands                 |
| Payment performance    | `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `last_pymnt_d` | Analyze repayment behavior after origination            |
| Loss/recovery analysis | `recoveries`, `collection_recovery_fee`                           | Track post-default recovery behavior                    |
| Policy monitoring      | model score, risk band, simulated decision                        | Track approve, review, and reject rates over time       |


These fields must be clearly separated from the model training feature set. A field can be valid for monitoring while still being invalid for model training. For example, `loan_status` is necessary for outcome monitoring, but it would be direct leakage if used as a model input. 

For this project, monitoring-only fields will be used after scoring to evaluate risk patterns, cohort behavior, portfolio composition, and realized bad-loan outcomes. They will not be included in the features used to train the applicant scoring model. 

## 9. Leakage Rules for later modules

The leakage review should be applied during data ingestion, preprocessing, feature engineering, model training, and monitoring. Every column should be assigned a clear purpose before it is used in the project. 

The first rule is that `loan_status` should only be used to create the target variable `bad_loan`. It should not be included as a model input because it directly reveals the outcome the model is trying to predict. 

The second rule is that post-origination performance fields should be excluded form the model feature set. This includes fields related to payments, recoveries, remaining principal, collection activity, settlement activity, hardship activity, and last payment information. These fields describe what happened after the loan was issued, so they are not valid inputs for application-time risk scoring. 

The third rule is that lender-generated decision fields should be excluded from the first version of the model. Fields such as `int_rate`, `grade`, `sub_grade` may reflect LendingClub's own risk assessment or pricing decision. Excluding them helps the model learn from borrower and loan characteristics directly. 

The fourth rule is that lender-generated decision fields should be stored separately from modeling features. These fields may be useful after scoring, especially for tracking outcomes, cohort behavior, score distributions, risk band mix, and portfolio performance. However, they should not enter the model training pipeline. 

The fifth rule is that every important column decision should be documented in docs/data_dictionary.md. Each field should be labeled as a modeling candidate, target/outcome field, monitoring-only field, suspicious field, or excluded field. This will make the project easier to audit and explain later. 

A practical rule for later modules is this:
If a field would not be available before or at loan origination, it should not be used as a model input. If a field reflects LendingClub's own decision or pricing process, it should be excluded from the first version unless there is a clear reason to include it later. 


## 10. Summary
The leakage review creates the rules for  deciding which LendingClub fields can be used for mmodel training and which fields must be reserved for target creation, monitoring, or exclusion. 

The main principle is that the applicant scoring model should only use information that would plausibly be available before or at loan origination. Fields that reveal future loan performance, such as payment history, recoveries, remaining principal, collection activity, or final loan status, should not be used as model inputs. 

The project will also exclude lender-generated decision fields such as `int_rate`, `grade`, and `sub_grade` from the first model version. These fields may reflect LendingClub's own risk assessment, so including them could allow the model to rely on LendingClub's prior decisioning process instead of learning from borrower and loan characteristics directly. 

Post-origination fields are not useless. They are just not valid for application-time model training. Some of these fields will still be useful for monitoring portfolio behavior, evaluating realized outcomes, and comparing predicted risk against actual loan performance. 

This leakage framework will guide the next module files, especially docs/data_dictionary.md, where each important column will be assigned a clear role: modeling feature, target/outcome field, monitoring-only field, suspicious field, or excluded field. 
































