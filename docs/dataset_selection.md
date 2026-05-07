# Dataset Selection

## 1. Purpose of the Dataset Selection
The purpose of dataset selection is to choose a public credit-related dataset that can support the core goals of this MVP: credit risk scoring, policy-based decisioning, and portfolio monitoring. This project is not intended to be a simple notebook classifier. The dataset needs to support a realistic end-to-end workflow where historical borrower and loan information is used to train a model, generate risk scores, apply business decision rules, and monitor portfolio behavior over time. 

The selected dataset must be suitable for both the applicant scoring view and the portfolio monitoring view. For the applicant scoring view, the dataset should contain borrower and loan characteristics that can be treated as application-time or origination-time information. For the monitoring view, the dataset should support tracking risk patterns across historical groups or cohorts, such as changes in score distributions, risk band mix, and bad-loan outcomes over time.

Because this project uses public historical data rather than internal production data from a financial institution, the dataset will not perfectly represent a real lending environment. In a real institution, application records, underwriting decisions, model scores, loan performance, and portfolio monitoring data would be collected within the same internal system. For this MVP, the goal is to choose the public dataset that gives the strongest and most honest foundation for demonstrating those workflows. 

The final dataset choice should therefore balance realism, target clarity, leakage control, monitoring support, and project feasibility. The chosen dataset should allow the project to remain coherent as one credit risk decisioning and monitoring systems rather than separate pieces that are forcefully stitched together.


## 2. Dataset Selection Criteria
The datasets were compared based on how well they support the goals  of the credit rsik decisioning and monitoring MVP. The goal was not simply to choose the dataset with the most rows or the highest modeling potential. The goal was to choose a dataset that could support a coherent project workflow from risk scoring to business decisioning to portfolio monitoring. 

The first criterion was business fit. The datset needed to represent a realistic condimer credit problem where borrower or loan-level information could be used to estimate the credit risk. Since the final project includes an applicant scoring view, the dataset needed to contain variables that resemble information a lender may know at application or origination time, such as income, loan amount, debt-to-income ratio, employment information, credit history, and loan purpose. 

The second criterion was target clarity. The project needs a defensible target variable that represents a bad credit outcome. A strong dataset should make it possible to clearly define which loans are considered bad outcomes and which loans are considered good outcomes. The target should represent repayment performance, not whether the applicant was historically approved or rejected.


## 3. Candidate Datasets Considered
Several public credit-related datasets were considered for this project. Each dataset was evaluated based on its ability to support the full MVP: risk scoring, policy-based decisioning, leakage review, and portfolio monitoring. The main candidates were LendingClub loan data, Home Credit Default Risk, UCI Default of Credit Card Clients, and the German Credit dataset. 

### 3.1 LendingClub Loan Data
LendingClub loan data was considered because it provides historical consumer loan records with borrower characteristics, loan characteristics, issue timing, and final loan outcomes. This makes it suitable for building a model that estimates the probability of a bad loan outcome, such as a default or charge-off, based on borrower and loan information available around origination. 

The main strength of LendingClub is that it supports both sides of this project. For the decisioning layer, the data can be used to trian a model that assigns a bad-loan risk score to applicant-style loan records. For the monitoring layer, the data contains time-related fields, such as loan issue period, that can support historical cohort analysis. This allows the project to monitor score distributions, risk band mix, simulated decision rates, and bad-loan outcomes over time. 

The main limitation is that LendingClub data generally represents issued or booked loans rather than the full population of submitted applications. This means the model does not learn LendingClub's original approval or rejection process. Instead, it learns default or charge-off risk among loans that were already originated. Becuase of this, the project's approve, manual review, and reject decisions must be framed as simulated business policy decisions, not historical LendingClub underwriting decisions.

Despite this limitation, LendingClub provides the strongest overall fit because it allows the project to remain coherent as one decisioning and monitoring system using a single dataset. 


### 3.2 Home Credit Default Risk
Home Credit Default Risk was considered because it is strongly aligned with applicant-level credit risk scoring. The dataset is organized around loan applications, and the target is based on whether the applicant had repayment difficulty. This makes it attractive for the applicant scoring portion of the project. 

The main strength of Home Credit is that it more naturally supports the idea of scoring applicants at the time of application. It also contains multiple related tables that could support richer feature engineering using prior credit history, bureau information, and previous application behavior. 

However, Home Credit is less suitable for monitoring side of this MVP. The project would likely need to create synthetic production batches or artificial cohort fields to demonstrate monitoring over time. While that is possible, it would make the monitoring layer feel more simulated and less naturally connected to the original dataset. 

Home Credit was not selected because the goal of this project is to build one coherent decisioning and monitoring MVP. Choosing Home Credit would strengthen the applicant scoirng story but weaken the historical monitoring story. 


### 3.3 UCI Default of Credit Card Clients
The UCI Default of Credit Card Clients dataset was considered because it has a clear binary default target and a managable structure. It is useful for learning credit default preditction because it contains customer credit, billing, and repayment information over a defined historical period. 

The main strength of this dataset is its simplicity. It is easier to understand, easier to clean, and easier to model compared with larger credit datasets. The target is also clear because the goal is to predict default payment. 

However, the dataset is less suitable for this MVP because it is closer to existing-customer credit card behavior modeling than loan application decisioning. It does not provide the same kind of loan origination structure or historical cohort monitoring support as LendingClub. As a result, it would make the final project feel more like a classroom default-prediction exercise rather than a fintech-style decisioning and monitoring system. 

UCI Default of Credit Card Clients was not selected because it is too narrow for the full scope of this project. 


### 3.4 German Credit Dataset
The German Credit dataset was considered as a simple credit risk benchmark. It is commonly used for teaching credit classification, and it contains examples of good and bad credit risk. 

The main strength of this dataset is that it is simple and easy to work with. It can be useful for explaining basic credit classification concepts and the business cost of approving bad borrowers versus rejecting good borrowers. 

However, it is too small and old-fashioned for this MVP. It does not provide enough realism for a modern fintech-style risk decisioning and monitoring project. It also does not provide strong support for historical cohort monitoring or more advanced portfolio behavior analysis. 

German Credit was not selected because it is better suited for learning exercises than for a serious portfolio project. 


## 4. Single-Dataset Versus Two-Dataset Design

A two-dataset design was considered because the strongest public credit datasets support different parts of the MVP. Home Credit Default Risk is more naturally aligned with applicant-level credit risk scoring, while LendingClub is more naturally aligned with historical laon performance and cohort-based monitoring. 

Using two datasets would allow the project to use Home Credit for model training and applicant scoring, while using LendingClub for portfolio monitoring. On the surface, this seems attractive because each dataset would be used for the part of the project it suports best. 

However, this design would weaken the overall system story. The monitoring layer would not truly be monitoring the same model, feature schema, score distribution, decision policy, or borrower population used in the applicant scoring layer. Instead, the project would become two related but separate credit risk demonstrations. 

A real financial institution would usually collect application data, model scores, underwriting decisions, booked-loan records, repayment outcomes, and monitoring data within the same internal data ecosystem. Since this project uses public data, the available dataset will not perfectly match real-world setup. However, the final MVP should still feel like one coherent system rather than separate workflows stitched together. 

For that reason, this project will use one primary dataset instead of combining separate datasets for decisioning and monitoring. LendingClub was selected because it provides the best overall compromise between risk scoring, target definition, leakage review, and historical monitoring support. 

This choice means the decisioning layer must be framed carefully. The model will not learn LendingClub's historical approval or rejection process. Instead, it will learn bad-loan risk among issued loans. THe approve, manual review, and reject outcomes in this project will be simulated business policy decisions based on the model's predicted risk score. 


## 5. Final Dataset Choice
The selected dataset for this project is LendingClub loan data. 

LendingClub was chosen because it provides the best overall fit for building one coherent credit risk decisioning and monitoiring MVP. The dataset contains borrower-level and loan-level information, final loan performance outcomes, and time-related fields that can support historical cohort monitoring. This makes it possible to use the same dataset for both the applicant scoring view and the portfolio monitoring view. 

The decisioning layer will use LendingClub records to train a model that estimates the probability of a bad loan outcome. In this project, a bad loan outcome will represent serious repayment failure or distress, such as charge-off, default or serious delinquency, depending on the final target definition. The model score will then be passed into a business policy layer that assigns a risk band and produces a simulated decision: approve, manual review, or reject. 

The monitoring layer will use the same dataset to track portfolio behavior over historical cohorts. Because LendingClub contains loan issue timing and final loan outcomes, the project can monitor score distributions, risk band mix, simulated decision rates, and bad-loan rates across different origination periods. This makes the monitoring layer more naturally connected to the model and decision policy. 

LendingClub also supports a meaningful leakage review. The dataset contains some fields that are available around application or origination time, but it also contains fields that are only known after the loan has started performing. Application-time and origination-time fields may be considered for modeling, while post-origination fields such as payment activity, recoveries, collections, remaining principal, hardship activity, and final loan status should not be used as model inputs. Some of these post-origination fields may still be useful for monitoring, but they must be separated from model training feature set. 

This choice does come with an important limitation. LendingClub data represents loans that were issued or booked, not the full population of submitted applications. Therefore, the model should not be interpreted as a reconstruction of LendingClub's original approval process. The model estimates bad-loan risk among loans similar to those that were historically originated, and the approve, manual review, and reject decisions are simulated policy decisions created for this MVP.

Despite this limitation, LendingClub was selected because it allows the project to remain internally consistent. The same dataset can support model training, risk scoring, decision-policy simulation, and portfolio monitoring, which is more appropriate for this project than using datasets for separate parts of the system. 






















































