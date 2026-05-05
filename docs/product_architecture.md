# Project Architecture

## Purpose of the Architecture
This project is designed as a credit risk decisioning and monitoring MVP, not as a single machine learning notebook. The purpose of the architecture is to separate the core risk logic from the user interface so that the project is easier to understand, test, maintain, and explain. The model, validation rules, preprocessing steps, simulated decision policy, SQL queries, and monitoring calculations should eventually live in reusable project code instead of being mixed into one notebook or one dashboard file.

This separation matters because a real credit risk system is made of multiple connected parts. The dashboard is only the interface that displays the system's outputs. The actual value  of the project comes from the underlying workflow: validating loan/applicant-style data, preparing model-ready features, estimating bad-loan risk, applying project-defined decision rules, storing structured outputs, and monitoring portfolio behavior over time. 

Because the project uses LendingClub issued-loan data, the architecture should be understood as a bad-loan risk scoring, simulated policy decisioning, and monitoring system. It is not meant to reconstruct LendingClub's original underwriting or approval process. 


## System Overview
The system follows a layered structure. Historical LendingClub loan-level data enters the project from the selected public dataset. The data is first validated to check whether required fields are present, whether values are plausible, and whether the record is usable for analysis or scoring. After validation and preprocessing, feature engineering prepares the data for modeling. 

The risk model estimates the likelihood that a loan becomes a bad loan based on the project's target definition. This model output is not the final business decision. A separate simulated decision policy maps the risk score into a business action, such as approve, manual review, or decline. After scoring and decisioning, the results can be stored in structured tables that support loan-level decisions, model outputs, and portfolio monitoring views. 

The high-level flow is: 
```text
→ Historical LendingClub issued-loan data
→ Data validation
→ Preprocessing and feature engineering
→ Risk model
→ Risk band
→ Simulated decision policy
→ SQLite / decision results storage
→ Monitoring calculations
→ Streamlit dashboard
```
Again, the approve/manual review/decline output should be interpreted as a simulated internal policy decision. It does not represent LendingClub's actual historical approval decision. The goal is to show how a lender could use model scores and policy thresholds to manage risk. 


### Layer 1: Data Input and Validation
The data input layer is responsible for bringing loan/applicant-style records into the system. In the MVP, this data will come from historical LendingClub issued-loan data. Later in the project, the dashboard may also allow hypothetical loan profile inputs so a user can test how the system would score a new profile. 

The validation layer checks whether the input data is usable before it moves further through the system. It should check whether required columns are present, whether the data types are reasonable, whether missing values are expected, and whether values fall within plausible ranges. For example, income should not be negative, debt-to-income ratio should fall within a reasonable range, loan amount should be positive, and required borrower or loan fields should not be missing without explanation.

The main purpose of validation is to catch obvious data quality problems early. A lender should not blindly score data that is incomplete, malformed, or unrealistic. In a real lending environment, poor input data can lead to poor credit decisions, so validation is part of the risk control process. Because the model is trained on historical issued loans, validation should also check whether a hypothetical profile falls reasonably within the range of the training data. If a profile is far outside the historical LendingClub population, the system should flag that the score may be less reliable. 


### Layer 2: Preprocessing and Feature Engineering
The preprocessing and feature engineering layer prepares raw data for modeling. This layer may handle missing values, encode categorical variables, scale numerical variables when needed, and create useful derived features from existing columns. The exact preprocessing steps will depend on the selected LendingClub columns and the model type.

This layer should be designed carefully because preprocessing decisions affect model quality, interpretability, and consistency. The same transformations used during model training should also be used when scoring new or hypothetical loan profiles. If training and scoring use different preprocessing logic, the model output may become unreliable. 

Feature engineering must also respect the origination-time rule. The model should only use information that would be available at or before the loan decision or origination point. Any feature created from future repayment behavior, final loan status, collection recoveries, hardship status, settlement activity, or other post-origination outcomes would create leakage and should not be used for risk scoring.

Some LendingClub fields may be useful for monitoring but not for model training. For example, final loan status can help define the target, and post-origination performance fields may help with portfolio monitoring, but they should not be used as predictors for application-time risk scoring. 

### Layer 3: Risk Model
The risk model estimates the likelihood that a loan/application-style profile will experience a bad credit outcome. In this project, the bad outcome will be defined using LendingClub loan performance statuses, likely by comparing resolved bad loans such as charged-off or defaulted loans against good loans such as fully paid loans. 

The model should be treated as a risk estimate, not as the final decision. For example, the model may estimate that a loan/application-style profile has a 7%, 18%, or 35% probability of becoming a bad loan. Those scores are useful, but they still need to be interpreted through a business policy before a lending action is taken.

Because the model is trained on issued LendingClub loans, its score should be interpreted carefully. It estimates bad-loan risk for profiles similar to loans that were historically originated. It does not estimate whether LendingClub would have approved or rejected the applicant in the first place. 

The model should also be explainable enough for a portfolio project. This does not mean the project must only use a simple model, but it should include methods for understanding why risk is high or low. In credit risk, interpretability matters because lenders need to understand the drivers behind risk estimates, not just the final prediction. 


### Layer 4: Decision Policy
The decision policy layer converts the model's risk estimate into a business action. This is where the system maps predicted bad-loan risk into outcomes such as approve, manual review, or decline. For example, low-risk profiles may be approved, medium-risk profiles may be sent to manual review, and high-risk profiles may be declined. 

This layer is separate from the model because the model only estimates risk. The business decides what level of risk is acceptable. A lender may choose stricter thresholds if it wants to reduce losses, or more flexible thresholds if it wants to grow the portfolio. This means the same model could support different policies depending on the lender's goals.

In this project, the decision policy is simulated. It represents the project lender's risk appetite, not LendingClub's actual historical underwriting rules. This distinction is important because the LendingClub dataset contains issued loans, not the full population of submitted and rejected applicants. 

Separating the decision policy from the model makes the project more realistic. In real credit systems, risk scores often feed into policy rules, manual review workflows, pricing decisions, or credit limit decisions. The score matters, but the policy determines what action is taken. 


### Layer 5: SQL and Data Storage Layer
The SQL and data storage layer gives the project a simple structured data layer. Instead of relying only on raw CSV files and notebooks, the project can store cleaned loan records, model scores, risk bands, decision outputs, and cohort labels in a local SQLite database. This helps the MVP feel closer to a small data product rather than a one-time analysis. 

A SQLite database can support tables such as cleaned loans, model scores, decision results, and monitoring cohorts. For example, one table may store loan-level records after cleaning, while another may store each record's predicted bad-loan risk score, assigned risk band, and simulated decision. These structured tables make it easier to query and summarize the project outputs. 

SQL is especially useful for monitoring. The dashboard can query the database to group loans by issue month, cohort, risk band, or policy decision. These queries can calculate metrics such as average predicted risk, simulated approval rate, number of loans, loan amount exposure, risk-band mix, and observed  bad-loan rate. This gives the monitoring layer a realistic data workflow instead of relying only on ad-hoc notebook calculations. 


### Layer 6: Monitoring Layer 
The monitoring layer tracks how the model, decision policy, and portfolio behave across groups of applicants or loans. While decisioning focuses on one applicant at a time, monitoring focuses on patterns across many records. It helps answer whether the portfolio is becoming riskier, whether simulated approval rates are changing, whether more loans are failing into high-risk bands, and whether observed bad-loan rates are moving over time.  

Because this project uses public historical data rather than a live production system, the monitoring layer should be framed as an offline historical monitoring prototype. The project should not claim to perform real-time production monitoring. Instead, it should use historical LendingClub cohorts, such as issue-month cohorts, to demonstrate the type of monitoring logic a lender would use in a real environment. 

The monitoring layer may track score distributions, simulated approval rates, risk-band mix, average predicted risk, observed bad-loan rates, feature drift, and model performance across cohorts. These views help show the project is not only about training a model, but also about understanding how the model and portfolio behave over time after risk scores and policy decisions are produced.

### Layer 7: Streamlit Interface
The Streamlit interface is the user-facing layer of the project. It should display model outputs, simulated decision policy results, monitoring views, and model performance summaries in a clear way. The dashboard may eventually include separate pages for loan/application-profile scoring, portfolio monitoring, data quality checks, and model evaluation.

Streamlit should not contain core business logic. It should call reusable functions from the project code and display the results. For example, Streamlit can collect loan profile inputs, but validation should happen through a validation function. Streamlit can display monitoring charts, but the monitoring calculations should come from reusable functions or SQL queries. 

This makes the project cleaner and more professional. If the interface changes later, the underlying risk engine should still work. The dashboard is the presentation layer, not the system itself. 


### Separation of Concerns
The architecture follows the principle of separation of concerns. Each part of the project should have a clear responsibility. Data validation should validate data. Preprocessing should prepare features. The model should estimate risk. The decision policy should convert risk scores into business actions. The monitoring layer should summarize model and portfolio behavior over time. The Streamlit dashboard should display outputs and collect inputs. If all the logic lives inside one notebook or one Streamlit file, then the project becomes fragile. A small change to the dashboard could accidentally affect preprocessing, scoring, decision rules, or monitoring calculations. 

A cleaner structure makes the project easier to test, debug, and present in an interview. It also shows that the project was designed like a small risk product, not just a one-time machine learning experiment. 


### Offline Monitoring Framing
The monitoring component must be described honestly. Since this project uses a static public dataset, it does not have a live stream of new production records. Therefore, the project should not claim to perform live production monitoring. 

Instead, the monitoring dashboard will be framed as an offline historical monitoring prototype. It will use historical LendingClub cohorts, such as issue dates or issue months, to demonstrate the monitoring logic that a lender would use in production, while being honest about the limits of the available data. 

This framing is important because it avoids overstating the project. The dashboard can still be valuable because it shows how risk distributions, simulated approval rates, risk-band mix, portfolio exposure, and bad-loan rates could be tracked over time. But the project should clearly state that the monitoring is a prototype, not a live deployment. 

### Future Architecture Vision
The final version of the project should feel like a small fintech-style risk product. A user should be able to understand the business problem, inspect the data, view model performance, score hypothetical loan/applicant-style profiles, see the simulated decision policy output, and monitor portfolio behavior across historical cohorts. 

The long-term structure should support reusable code. Notebooks should be used for exploration and documentation, while the main project logic should live in reusable Python modules. The SQLite database should provide a simple structured data layer, and the Streamlit app should sit on top of the reusable logic and make the results easier to interact with.

The final product should show that the project is not only about building a model. It is about building a responsible decisioning workflow around the model. That workflow includes validation, preprocessing, leakage prevention, risk scoring, simulated decision policy, structured storage, monitoring, and clear communication of limitations.



