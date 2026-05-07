# Target Definition

## 1. Purpose of Target Definition
The purpose of target definition is to clearly define what the model is trying to predict. In a credit risk project, this step is important because the model's output, decision policy, evaluation metrics, and monitoring views all depend on how the target variable is defined. 

For this project, the target should represent loan repayment performance after a loan has been issued. The odel is not being trained to predict whether LendingClub historically approved or rejected an applicant. Instead, the model is being trained to estimate the probability that an issued loan becomes a bad loan outcome. 

This distinction is important because the selected LendingClub dataset represents loan that were issued or booked, not the full population of submitted applications. As a result, the model learns risk patterns among loans that were originated. The project's approve, manual review, and reject decisions are simulated business policy decisions based on predicted risk, not a reconstruction of LendingClub's original underwriting decisions. 

The target definition must also support the monitoring layer. Once the model assigns risk scores and risk bands, the monitoring view can compare predicted risk against observed loan outcomes across historical cohorts. For this reason, the target must be defined in a way that is clear, consistent, and defensible. 


## 2. Modeling Objective
The modeling objective is to estimate the probability that a loan will result in a bad repayment outcome after it has been issued. In this project, the model will use borrower and loan characteristics available around application or origination time to produce a risk score for each loan record.

The model's output should be interpreted as predicted bad-loan probability. For example, if the model assigns a score of `0.18`, this means the model estimates a 18% probability that the loan will become a bad loan outcome under the project's target definition. 

This risk score will be used by the decisioning layer, but the model itself will not make the final business decision. The model only estimates risk. A separate policy layer will convert the risk score into a risk band and simulated business action, such as approve, manual review, or reject. 

This separation is important because the credit risk modeling and credit decisioning are not the same thing. The model answers: "How risky does this loan look?". The policy layer answers: "What should the business do with that level of risk?". This allows the project to keep the model logic separate from business rules, which is one of the core design principles of the MVP. 


## 3. Target Variable
The target variable for this project will be called `bad_loan`. 

The target represents whether an issued loan eventually results in a bad repayment outcome. A value of `bad_loan = 1` means the loan experienced serious repayment failure or distress. A value of `bad_loan = 0` means the loan reached a good final repayment outcome. 

The model will be trained as a binary classification model. For each loan, the model will estimate the probability that `bad_loan = 1`. This predicted probability becomes the risk score used by the rest of the decisioning system. 

The target is created from LendingClub loan status field. The exact mapping from loan status to `bad_loan` is defined in the next section. The key idea is that severe negative outcomes such as charge-off, default, or serious delinquency, are treated as bad outcomes, while fully paid loans are treated as good outcomes.

The target variable should only be used for model training, validation, evaluation, and monitoring. It should not be included as an input feature during model training because it directly represents the outcome the model is trying to predict. 


## 4. Loan Status Mapping 
The target variable `bad_loan` will be created from the LendingClub loan status field. The goal is to separate loans with good repayment outcomes from loans with serious repayment failure or distress. 

For the first version of this project, the target will use the following mapping: 
| Loan status | Target Value  | Reason |
|---|---:|---|
| `Fully Paid` | `bad_loan = 0` | The loan reached a good final repayment outcome. |
|`Charged Off` | `bad_loan = 1` | The loan resulted in a serious loss outcome where repayment was not successfully completed. |
|`Default` | `bad_loan = 1` | The loan has reached a severe non-payment state. |
|`Late (31-120 days)` | `bad_loan = 1` | The loan is in serious delinquency and represents meaningful repayment distress. |
|`Current` | Excluded | The loan has not reached a final outcome yet. |
|`In Grace Period` | Excluded | The loan is not yet in a severe or final bad state. |
|`Late (16-30 days)` | Excluded | This is an early delinquency status and may be too noisy to treat as a final bad outcome. |

This project uses the broader term "bad loan outcome" rather than only "default" because the positive class may include serious delinquency in addition to default or charge-off. This makes the target more useful for risk management because lenders often care about early signs of serious repayment distress before a loan reaches final charge-off. 

Loans with unresolved or temporary statuses will be excluded from the modeling dataset. This avoids training the model on records where the final repayment outcome is unclear. For example, a `current` loan may eventually become full paid or may later become delinquent, so it should not be labeled as either good or bad at the time of target creation. 

The exact set of loan status will be verified during the data ingestion stage. If the raw dataset contains additional status categories, each one will be reviewed and assigned to one of the three groups: positive class, negative class, or excluded from modeling. 


## 5. Why Approval/Rejection is Not the Target
The target for this project is based on loan repayment perfomrnace, not LendingClub's historical approval or rejection decision. The model is trying to predict whether an issued loan eventually becomes a bad loan outcome. 

This distinction matters because the selected dataset contains loans that were issued or booked. It does not fully represent every person who applied for a loan, including applicants who were rejected or applicants who received an offer but did not accept it. As a result, the model is trained on the population of loans with observable repayment outcomes. 

If approval or rejecstion were used as targets, the model would be trying to imitate LendingClub's historical underwriting process. That is not the purpose of this MVP. The purpose is to estimate the credit risk using historical loan performance, then apply a separate project-defined business policy to decide whether a similar future loan profile would be approved, sent to manual review, or rejected. 

This means the project's decisioning layer should be interpreted as a simulated risk policy, not as a reconstruction of LendingClub's original approval system. The model provides the risk estimate, while the policy later decides what action to take based on that estimate.


## 6. How the Target connects to Decisioning 
The target variable `bad_loan` is used to train the model to estimate the credit risk. Once the model is trained, it will produce a predicted probability for each loan record. This predicted probability represents the estimated chance that the loan will become a bad loan outcome under the project's target definition. 

For example, if the model produces a score of `0.22`, this means the model estimates a 22% probability that the loan will become a bad loan outcome. This score is not the final decision. It is only the model's risk estimate. 

The decisioning layer will take this predicted risk score and apply business policy rules. These policy rules will convert the score into a risk band and a simulated decision. 

Example: 

| Predicted bad-loan probability | Risk Level | Policy Decision |
|---:|----|---|
| Low risk score | Low Risk | Approve |
| Medium risk score| Medium Risk | Manual Review |
| High risk score | High Risk | Reject |

This separation keeps the model logic and the business decision logic independent. The model estimates the likelihood of a bad outcome, while the policy layer decides what action to take based on the lender's risk appetite. 

This also makes the system more flexible. If the business wants to become more conservative, the approval threshold can be tightened without retraining the model. Conversely, if the business wants to approve more loans, the threshold can be relaxed while keeping the same underlying risk model. 


## 7. Target Limitations
The target definition has some important limitations that should be documented clearly. 

The first limitation is selection bias. The LendingClub dataset contains loans that were issued or booked, not the full population of people who applied for credit. This means the model learns bad-loan risk among loans similar to those that were historically originated. It does not learn the risk profile of applicants who were rejected before a loan was issued. 

The second limitation is that some loan statuses are unresolved or temporary. Statuses such as `Current`,  `In Grace Period`, and `Late (16-30 days)` do not provide a clean final repayment outcome. These rows will be excluded from the supervised training target because labeling them as either good or bad could introduce noise into the model. They may still be used later for scoring or monitoring demonstrations, but they should not be treated as clean training labels. 

The third limitation is that the target is based on observed loan performance, not the full cost of credit risk. A bad-loan outcome captures serious repayment distress, but it does not fully capture loss severity, recovery amount, profitability, or customer lifetime value. For this MVP, the target focuses on whether the loan becomes bad, not how much money is lost. 

The fourth limitation is that the target depends on the exact mapping chosen from `loan_status` to `bad_loan`. Including `Late (31-120 days)` in the positive class makes the model sensitive to serious delinquency, not only final charge-off or default. This is useful for risk management, but it also means the target should be described as `bad_loan` or bad loan outcome rather than strictly default. 

These limitations do not make the target invalid. They simply define what the model can and cannot claim. The model can estimate the probability of a bad repayment outcome for loans similar to the historical LendingClub population. It should bot be interpreted as a complete underwriting model for all possible credit applications. 


## 8. Implications for Later Modules

This target definition will guide the next stages of the project. During data ingestion and preprocessing, the raw `loan_status` field will be transformed into the binary target variable `bad_loan`. Only rows with a clear target label will be used for supervised model training and evaluation. 

The leakage review must make sure that `loan_status` and any post-origination performance fields are not used as model input features. These fields may define the target or support monitoring, but they should not be included in the feature set used to predict `bad_loan`. 

The model evaluation stage should use metrics that are appropriate for credit risk classification. Since bad loans are usually less common than good loans, accuracy alone will not be enough. Later modules should consider metircs such as recall for bad loans, precision, ROC-AUC, PR-AUC, confusion matrices, and threshold-based policy impact. 

The decisioning layer will use the model's predicted probability of `bad_loan = 1` as the risk score. This score will be converted into risk bands and simulated policy decisions such as approve, manual review, or reject. The exact thresholds will be defined later in the decision policy module. 

The monitoring layer will use the target to compare predicted risk against observed outcomes across historical cohorts. This will allow the project to monitor bad-loan rates by cohort, bad-loan rates by risk band, score distributions, risk band mix, and simulated decision rates over time. 

Overall, this target definiton establishes the foundation for modeling, decisioning, and monitoring logic. Later modules should remain consistent with this definition: the project predicts bad-loan risk among issued LendingClub loans and applies simulated business policy rules based on that predicted risk. 

























