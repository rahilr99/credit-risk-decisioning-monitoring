# Project Scope: Credit Risk Decisioning and Monitoring MVP

## 1. Project Overview

This project is an end-to-end credit risk decisioning and monitoring MVP using historical LendingClub loan data. The goal is to build a small fintech-style risk tool that can ingest loan/application-style data, validate it, preprocess features, train and evaluate a credit risk model, score hypothetical loan profiles, assign decision recommendations through a separate business policy, and monitor model and portfolio behavior over time.

This is not intended to be only a machine learning notebook that trains a classifier. The project is designed as a credit risk decisioning and monitoring system. The model estimates the probability of a bad loan outcome, but the final business action comes from a separate decision policy.

Because the selected dataset contains issued LendingClub loans rather than the full population of submitted and rejected applications, the decisioning layer should be interpreted as a simulated risk policy for loans similar to the booked-loan population. It is not a reconstruction of LendingClub's original underwriting or approval process. 


## 2. Business Problem 

A lender needs a repeatable way to estimate credit risk before deciding whether a loan fits within its risk appetite. The lender wants to grow its loan portfolio by approving creditworthy borrowers, but it also needs to control default and charge-off risk so that the booked portfolio does not become too risky over time.

The main business question is:
> Given borrower and loan information available around origination, can we estimate the probability that a loan will become a bad loan, use that estimate to support an approve/manual review/decline policy, and monitor whether the portfolio remains within acceptable risk levels over time?

This project looks at the problem from the lender's perspective. Issued loans are assets for the lender because they are expected to generate future cash flows through repayment and interest. However, those assets carry default risk. The project is meant to help the lender manage that risk at both the individual loan/application-profile level and the portfolio level. 

A key limitation is that LendingClub data represents loans that were already issued. Therefore, the model estimates bad-loan risk conditional on a loan being originated. The project does not claim to model the full application funnel or predict LendingClub's actual historical approval decisions. 


## 3. User Perspective

The intended user is a fintech, banking, or credit risk team that wants a lightweight internal tool for credit decisioning and monitoring. 

The system should be able to answer questions such as:
- How risky is this applicant?
- What is the predicted probability of a bad loan outcome? 
- Under our project-defined policy, should this profile be approved, rejected, or sent to manual review?
- What is the overall risk profile of the booked or simulated approved portfolio?
- Are newer loan cohorts becoming riskier over time?
- Are model scores or risk bands shifting over time?
- Is the model still behaving in a stable and useful way?


## 4. Model Purpose

The model's role is to estimate credit risk at the individual loan/applicant-profile level. More specifically, the model should output a predicted probability that a loan becomes a bad loan based on the target definition created from LendingClub loan outcomes.

For this MVP, a bad loan will likely be defined using resolved loan statuses such as charged-off or defaulted loans, while good loans will likely be defined using fully paid loans. The exact target definition will be documented separately in **docs/target_definitions.md**. 

The model itself does not directly approve or decline a loan. It only estimates the risk. 

For example: 
```text
Applicant A -> predicted default risk = 4%
Applicant B -> predicted default risk = 16%
Applicant C -> predicted default risk = 31%
```
These predicted probabilities are then passed into a separate decision policy. 

Because the model is trained on issued loans, its predictions should be interpreted as estimated bad-outcome risk for profiles similar to loans that were historically originated. The score should not be interpreted as a prediction of whether LendingClub would have approved the applicant. 

## 5. Decision Policy Purpose

The decision policy converts the model's predicted risk into a business action. This separation is important because the model is a prediction tool, while a policy is a business rule. 

A simple MVP decision policy may look like this: 
```text
Low predicted bad-loan risk -> Approve
Medium predicted risk -> Manual review
High predicted bad-loan risk -> Decline
```
The exact thresholds will be chosen later after model evaluation. The thresholds should not be chosen randomly. They should be based on the tradeoff between approving more applicants and controlling bad-loan risk.

In this project, the approve/manual review/ decline recommendation is a simulated internal policy decision. It does not represent LendingClub's actual historical decision. This allows the project to demonstrate how a lender could use model scores to create a risk-based decisioning process while still being honest about the limits of the public dataset.

This project will keep the model and decision policy separate so that the business rules can be changed without altering or retraining the model. 


## 6. Monitoring Purpose

Monitoring is included because a credit risk model is not useful only at training time. The lender also needs to know whether the applicant population being scored,the model scores, the simulated decision outcomes, and the booked portfolio remain stable over time. 

Because LendingClub data includes historical issued loans, the monitoring layer can be built as an offlline historical monitoring prototype. The project will not claim to perform real-time production monitoring. Instead, it will use historical loan cohorts, such as issue-month cohorts, to show how a lender could monitor risk over time. 

The monitoring should help answer questions such as: 
- Is the average predicted bad-loan risk increasing over time?
- Is the simulated approval rate changing?
- Are more applicants falling into the high-risk band?
- Is the distribution of model scores changing?
- Are important borrower or loan features drifting over time?
- Is the observed bad-loan rate changing across historical cohorts?
- Are higher-risk bands actually showing higher observed bad-loan rates?
- Is the portfolio becoming more concentrated in risky segments?

For the MVP, monitoring will focus on portfolio-level and model-level behavior. It will not attempt to build a full real-time borrower behavior monitoring system or a full production MLOps monitoring system. 


## 7. MVP Scope

The MVP will include the following components: 
  1.  Project documentation and business framing
  2.  Dataset selection and data dictionary
  3.  Target definition and leakage review
  4.  Data ingestion and validation
  5.  Exploratory data analysis
  6.  Preprocessing and feature engineering pipeline
  7.  Baseline credit risk model
  8.  Model evaluation using credit-relevant metrics
  9.  Threshold and risk-band analysis
  10. Simulated decision policy for approve/manual review/decline recommendations
  11. Applicant scoring pipeline
  12. Portfolio monitoring metrics
  13. Streamlit dashboard with two interfaces:
      * Applicant scoring view
      * Portfolio monitoring view
  14. Final README


## 8. Out of scope for the first MVP:

- Real-time production deployment
- Live bank or customer data
- Authentication and user accounts
- Full React frontend
- Full FastAPI backend
- Database-backed production system
- Automated retraining pipeline
- Complex MLOps setup
- Docker deployment
- MLflow experiment tracking
- Real-time credit bureau integration
- Final legal or regulatory compliance system


## 9. Design Principles

This project will follow these design principles:

- Deep understanding over speed
- Reusable project code over notebook-only work
- Clear separation between model logic and app/dashboard logic
- Clear separation between model prediction and business decision policy
- Simple, explainable modeling before complex modeling
- Strong leakage prevention
- Evaluation based on business risk tradeoffs, not accuracy alone
- Clean repo organization
- MVP scope discipline


## 10. Core Project Flow

The planned project flow is:

```text
Historical LendingClub issued-loan data
↓
Target definition and leakage review
↓
Data validation
↓
Preprocessing and feature engineering
↓
Model training and evaluation
↓
Predicted bad-loan risk scores
↓
Risk bands
↓
Simulated decision policy
↓
Approve/manual review/decline recommendations
↓
Portfolio and model monitoring metrics
↓
Streamlit dashboard

```
Each stage should produce outputs that can be reused by the next stage.


## 11. Final Product Vision

The final product should feel like a small internal fintech risk tool. It should have two main interfaces. 
The first interface is the applicant scoring view. This view allows a user to inspect, upload, or manually enter a loan/applicant-style profile and receive a predicted bad-loan risk score, risk band, and policy recommendation. The recommendation may be approved, manual review, or declined based on project-defined business thresholds.  

The second interface is the portfolio monitoring view. This view allows a user to monitor the overall risk profile of historical loan cohorts and simulated policy outcomes. It should show metrics such as average predicted risk, simulated approval rate, risk-band distribution, score distribution, observed bad-loan rate, and portfolio exposure across risk bands over time. 

The dashboard should display the outputs of the core risk engine. It should not contain the core validation, preprocessing, modeling, scoring, decision policy, or monitoring logic itself. 

The final MVP should be honest about the data limitation: LendingClub data contains issued loans, not the full application funnel. Therefore, the system should be presented as a bad-loan risk scoring, policy simulation, and monitoring prototype rather than a reconstruction of LendingClub's original underwriting process. 

