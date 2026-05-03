# Project Architecture

## Purpose of the Architecture
This project is designed as a credit risk decisioning and monitoring MVP, not as a single machine learning notebook. The purpose of the architecture is to separate the core risk logic from the user interface so that the project is easier to understand, test, maintain, and explain. The model, validation rules, preprocessing steps, decision policy, SQL queries, and monitoring calculations should eventually live in reusable project code instead of being mixed into one notebook or one dashboard file.

This separation matters because a real credit risk system is made of multiple connected parts. The dashboard is only the interface that displays the system's outputs. The actual value  of the project comes from the underlying workflow: validating applicant data, preparing model-ready features, estimating risk, applying decision rules, storing structured outputs, and monitoring portfolio behavior over time. 


## System Overview
The system follows a layered structure. Applicant or loan-level data enters the project from the selected public dataset. The data is first validated to check whether required fields are present, whether values are plausible, and whether the record is usable for analysis or scoring. After validation, preprocessing, and feature engineering, prepare the data for modeling. 

The risk model estimates the likelihood of a bad credit outcome. This model output is not the final business decision. A separate decision policy maps the risk score into a business action, such as approve, manual review, or decline. After scoring and decisioning, the results can be stored in structured tables and displays applicant-level decisions, model outputs, and portfolio monitoring views. 

The high-level flow is: 
```text
Applicant or loan data
→ Data validation
→ Preprocessing and feature engineering
→ Risk model
→ Decision policy
→ SQLite / decision results storage
→ Monitoring calculations
→ Streamlit dashboard
```


### Layer 1: Data Input and Validation
The data input layer is responsible for bringing applicant or loan-level records into the system. In the MVP, this data will come from a public credit risk dataset. Later in the project, the dashboard may also allow hypothetical applicant inputs so a user can test how the system would score a new applicant. 

The validation layer checks whether the input data is usable before it moves further through the system. It should check whether required columns are present, whether data types are reasonable, whether missing values are expected, and whether values fall within plausible ranges. For example, income should not be negative, age should fall within a realistic range, and required applicant fields should not be missing without explanation. 

The main purpose of the validation is to catch obvious data quality problems early. A lender should not blindly score data that is incomplete, malformed, or unrealistic. In a real lending environment, poor input data can lead to poor credit decisions, so validation is part ofthe risk control process. 


### Layer 2: Preprocessing and Feature Engineering
The preprocessing and feature engineering layer prepares raw data for modeling. This layer may handle missing values, encode categorical variables, scale numerical variables when needed, and create useful derived features from existing columns. The exact preprocessing steps will depend on the selected dataset and the model type.

This layer should be designed carefully because preprocessing decisions affect model quality, interpretability, and consistency. The same transformations used during model training should also be used when scoring new or hypothetical applciants. If training and scoring use different preprocessing logic, the model output may become unreliable. 

Feature engineering must also respect the applicant-time rule. The model should only use information that would be available at or before the credit decision point. Any feature created from future repayment behavior , final loan status, collection recoveries, or other post-approval outcomes would create leakage and should not be used for applicant-level scoring. 


### Layer 3: Risk Model
The risk model estimates the likelihood that an applicant or borrower will experience a bad credit outcome. Depending on the selected dataset, this bad outcome may mean default, serious delinquency, repayment difficulty, charge-off, or another clearly defined negative outcome.

The model output should be treated as a risk estimate, not as the final decision. For example, the model may estimate that an applicant has a 7%, 18%, or 35% probability of becoming high risk. Those scores are useful, but they still need to be interpreted through a business policy before a lending action is taken. 

The model should also be explainable enough for a portfolio project. This does not mean the project must only use a simple model, but it shold include methods for understanding why risk is high or low. In credit risk, interpretability matters because lenders need to understand the drivers behind risk estimates, not just the finbal prediction. 


### Layer 4: Decision Policy
The decision policy layer converts the model's risk estimate into a business action. This is where the system maps risk scores into outcomes such as approve, manual review, or decline. For example, low-risk applicants may be approved, medium-risk applicants may be sent to manual review, and high-risk applicants may be declined. 

This layer is separate from the model because the model only estimates risk. The business decides what level of risk is acceptable. A lender may choose stricter thresholds if it wants to reduce losses, or more flexible thresholds if it wants to grow the portfolio. This means the same model could support different policies depending on lender's goals.

Separating the decision policy from the model makes the project more realistic. In real credit systems, risk scores often feed into policy rules, manual review workflows, pricing decisions, or credit limit decisions. The scores matters, but the policy determines what action is taken. 


### Layer 5: SQL and Data Storage Layer
The SQL and data storage layer gives the project a simple structured data layer. Instead of relying only on raw CSV files and notebooks, the project can store cleaned records, model scores, risk bands, decision outputs, and cohort labels in local SQLite database. This helps the MVP feel closer to a small data product rather than a one-time analysis. 

A SQLite database can support tables such as cleaned applicants or loans, model scores, decision results, and monitoring cohorts. For example, one table may store applicant-level records after cleaning, while another may store each applicant's predicted risk score and assigned decision. These structured tables make it easier to query and summarize the project outputs. 

SQL is especially useful for monitoring. The dashboard can  query the database to group the applicants or loans by month, cohort, risk band, or simulated batch. These queries can calculate metrics such as average predicted risks, approval rate, number of applicants, risk-band mix, and observed bad outcome rate. This gives the monitoring layer a realistic data workflow instead of relying only on ad-hoc notebook calculations.


### Layer 6: Monitoring Layer 
The monitoring layer tracks how the model, decision policy, and portfolio behave across groups of applicants or loans. While decisioning focuses on one applicant at a time, monitoring focuses on patterns across many reocrds. It helps answer whether the portfolio is becoming riskier, whether approval rates are changing, whether more applicants are falling into high-risk bands, and whether observed bad outcome rates are moving over time. 

Because this project uses public historical data rather than a live production system, the monitoring layer should be framed as an offline prototype. The project should not claim to perform real-time production monitoring. Instead, it should use historical cohorts or simulated production batches to demonstrate the type of monitoring logic a lender would use in a real environment. 

The monitoring layer may track score distributions, approval rates, risk-band mix, average predicted risk, observed bad outcomes rates, and model performance across cohorts. These views help show that the project is not only about training a model, but also about understanding how the model and portfolio behave after decisions are made. 

### Layer 7: Streamlit Interface
The Streamlit interface is the user-facing layer of the project. It should display model outputs, decision policy results, monitoring views, and model performance summaries in a clear way. The dashboard may eventually include separate pages for applicant scoring, portfolio monitoring, data quality checks, and model evaluation.

Streamlit should not contain core business logic. It should call reusable functions from the project code and display the results. For example, Streamlit can collect applicant inputs, but validation should happen through a validation function. Streamlit can display monitoring charts, but the monitoring calculations should come from reusable functions or SQL queries. 

This makes the project cleaner and more professional. If the interface changes later, the underlying risk engine should still work. The dashboard is the presentation layer, not the system itself. 


### Separation of Concerns
The architecture follows the principle of separation of concerns. Each part of the project should have a clear responsibility. Data validation should validate data. Preprocessing should prepare features. If all the logic lives inside one notebook or one Streamlit file, then the project becomes fragile. A small change to the dashboard could accidentally affect preprocessing, scoring, decision rules, or monitoring calculations. 

A cleaner structure makes the project easier to test, debug, and present in an interview. It also shows that the project was designed like a small product, not just a one-time machine learning experiment. 


### Offline Monitoring Framing
The monitoring component must be described honestly. Since this project uses a static public dataset, it does not have a live stream of new applicants. Therefore, the project should not claim to perform live production monitoring. 

Instead, the monitoring dashboard will be framed as an offline monitoring prototype. It will use historical cohorts, issue dates, application periods, or simulated batches, depending on the selected datasets. The goal is to demonstrate the monitoring logic that a lender would use in production, while being honest about the limits of the available data. 

This framing is important because it avoids overstating the project. The dashboard can still be valuable because it shows how risk distributions, approval rates, risk-band mix, and bad outcomes rates could be tracked over time. But the project should clearly state that the monitoring is a prototype, not a live deployment. 

### Future Architecture Vision
The final version of the project should feel like a small fintech-style risk product. A user should be able to understand the business problem, inspect the data, view model performance, score hypothetical applicants, see the decision policy output, and monitor portfolio behavior across cohorts. 

The long-term structure should support reusable code. Notebooks should be used for exploration and documentation, while the main project logic should live in reusable Python modules. The SQLite database should provide a simple structured data layer, and the Streamlit app should sit on top of the reusable logic and make the results easier to interact with.

The final product should show that the project is not only about building a model. It is about building a responsible decisioning workflow around the model. That workflow includes validation, preprocessing, risk scoring, decision policy, structured storage, monitoring, and clear communication of limitations.



