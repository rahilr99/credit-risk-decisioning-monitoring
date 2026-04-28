## Credit Risk Decisioning and Monitoring MVP

This project is an end-to-end credit risk decisioning and monitoring MVP. The goal is to build a small fintech-style risk tool that can ingest applicant-style data, validate it, preprocess features, train and evaluate a credit risk model, score hypothetical applicants, assign decision recommendations, and monitor portfolio/model behavior over time. 


## Project Goal

The project estimates applicant-level credit risk and uses that risk estimate inside a decisioning policy. The model does not directly approve or reject applicants. Instead, it outputs a predicted probability of default or high-risk behavior, and a separate policy maps that probability into decisions such as approve, manual review, or decline. 


## Planned MVP components

- Data ingestion and validation
- Target definition and leakage review
- Exploratory data analysis
- Preprocessing and feature engineering pipeline
- Baseline credit risk model
- Model evaluation and threshold analysis
- Applicant scoring pipeline
- Portfolio monitoring metrics
- Streamlit dashboard with applicant scoring and portfolio monitoring views


## Project Structure

```text
credit-risk-decisioning-monitoring/
├── app/
├── data/
├── docs/
├── models/
├── notebooks/
├── reports/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
