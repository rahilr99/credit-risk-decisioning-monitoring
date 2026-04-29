# Project Scope: Credit Risk Decisioning and Monitoring MVP

## 1. Project Overview

This project is an end-to-end credit risk decisioning and monitoring MVP. The goal is to build a small fintech-style risk tool that can ingest applicant-style data, validate it, preprocess features, train and evaluate a credit risk model, score hypothetical applicants, assign decision recommendations, and monitor portfolio/model behavior over time.

This is not intended to just be a machine learning notebook that trains a classifier. The project is designed as a credit risk decisioning system. The model estimates risk, but the final business decision comes from a separate decision policy. 


## 2. Business Problem 

A lender receives credit or loan applications and needs a repeatable way to estimate the risk of each applicant before making a decision. The lender wants to approve enough applicants to grow the credit portfolio, but also needs to control default risk so that the approved book does not become too risky.

The main business question is: 
> Given applicant information available at the time of application, can we estimate the probability that an applicant will become high risk, use that estimate to support an approval decision, and monitor whether the portfolio remains within acceptable risk levels over time?

This project looks at the problem from the lender's perspective. Approved loans or credit accounts are assets for the lender because they are expected to generate future cash flows. However, those assets carry default risk. The project is meant to help the lender manage that risk at both the applicant level and the portfolio level. 


## 3. User Perspective

The intended user is a fintech, banking, or credit risk team that wants a lightweight internal tool for credit decisioning and monitoring. 

The system should be able to answer questions such as:
- How risky is this applicant?
- Should this applicant be approved, rejected, or sent to manual review?
- What is the overall risk profile of the approved portfolio?
- Are newer applicants becoming riskier over time?
- Are model scores or risk bands shifting over time?
- Is the model still behaving in a stable and useful way?


## 4. Model Purpose

The model's role is to estimate the applicant-level credit risk. More specifically, the model should output a predicted probability that an applicant becomes high risk or defaults, depending on the target definition of the selected dataset.

Important to note here that the model itself does not directly approve or reject the applicant. It only estimates risk. 

For example: 
```text
Applicant A -> predicted default risk = 4%
Applicant B -> predicted default risk = 16%
Applicant C -> predicted default risk = 31%
```
These predicted probabilities are then passed into a separate decision policy. 


## 5. Decision Policy Purpose

The decision policy converts the model's predicted risk into a business action. This separation is important because the model is a prediction tool, while a policy is a business rule. 

A simple MVP decision policy may look like this: 
```text
Low predicted risk -> Approve
Medium predicted risk -> Manual review
High predicted risk -> Decline
```
The exact thresholds will be chosen later after model evaluation. The thresholds should not be chosen randomly. They should be based on the tradeoff between approving more applicants and controlling default risk.

This project will keep the model and decision policy separate so that the business rules can be changed without altering or retraining the model. 


## 6. Monitoring Purpose

Monitoring is included because a credit risk model is not useful only at training time. The lender also needs to know whether the applicant population, model scores, and approved portfolio remain stable over time. 

The monitoring should help answer questions such as: 
- Is the average predicted risk increasing?
- Is the approval rate changing?
- Are more applicants falling into the high-risk band?
- Is the distribution of model scores changing?
- Are important applicant features drifting over time?
- If actual outcomes are available, is observed default risk changing?

For the MVP, monitoring will focus on portfolio-level and model-level behavior. It will not attempt to build a full real-time borrower behavior monitoring system unless the dataset supports it. 


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
  10. Decision policy for approve/manual review/decline recommendations
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
Raw applicant-style data
↓
Data validation
↓
Preprocessing and feature engineering
↓
Model training and evaluation
↓
Predicted risk scores
↓
Decision policy
↓
Applicant-level recommendations
↓
Portfolio monitoring metrics
↓
Streamlit dashboard
```
Each stage should produce outputs that can be reused by the next stage.


## 11. Final Product Vision

The final product should feel like a small internal fintech risk tool. It should have two main interfaces. 
The first interface is the applicant scoring view. This view allows a user to inspect or upload applicant-style data and receive predicted risk, risk band, and decision recommendation. 

The second interface is the portfolio monitoring view. This view allows a user to monitor the overall risk profile of the applicant or approved borrower population through metrics such as average predicted risk, approval rate, risk-band distribution, and score distribution over time. 

The dashboard should display the outputs of the core risk engine. It should not contain the core modeling logic itself.
