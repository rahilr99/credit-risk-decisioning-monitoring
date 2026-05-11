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






























