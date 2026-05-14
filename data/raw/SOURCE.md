# Raw Data Source

Dataset family: LendingClub loan data  
Population used: Accepted/booked loans  
Raw file used: accepted_2007_to_2018Q4.csv.gz  
Source: https://www.kaggle.com/datasets/wordsforthewise/lending-club/data  
Download date: 2026-05-14  

This project uses the accepted/booked LendingClub loan file because the target variable, `bad_loan`, is constructed from repayment outcome information in `loan_status`.

The rejected-loan file is excluded from the main MVP because rejected applications do not have observed repayment outcomes. Including rejected applications would change the project into an acceptance/rejection classification problem, which is outside the scope of this MVP.

Raw data rule:
The file in `data/raw/` should not be manually edited. All transformations should happen through code and should produce outputs in `data/interim/` or later `data/processed/`.
