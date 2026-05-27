import pandas as pd

from data_ingestion import load_raw_lendingclub_data

GOOD_LOAN_STATUSES = {
    "Fully Paid", 
    "Does not meet the credit policy. Status:Fully Paid"
}

BAD_LOAN_STATUSES = {
    "Charged Off", 
    "Default", 
    "Does not meet the credit policy. Status:Charged Off", 
    "Late (31-120 days)",
}

EXCLUDED_LOAN_STATUSES = {
    "Current", 
    "In Grace Period", 
    "Late (16-30 days)",
}

TARGET_MAPPING = {
    **{status: 0 for status in GOOD_LOAN_STATUSES},
    **{status: 1 for status in BAD_LOAN_STATUSES},
}

def add_bad_loan_target(
        df: pd.DataFrame, 
        status_column: str = "loan_status",
        target_column: str = "bad_loan",
) -> pd.DataFrame:
    """
    Add a bad_loan target column based on LnedingClub loan_status.

    bad_loan = 1 means the loan had a bad repayment outcome.
    bad_loan = 0 means the loan had a good repayment outcome. 
    Missing, unresolved, or excluded statuses are left as missing. 
    """

    if status_column not in df.columns: 
        raise ValueError(f"Missing required column: {status_column}")
    
    df = df.copy()

    cleaned_status = df[status_column].astype("string").str.strip()
    df[target_column] = cleaned_status.map(TARGET_MAPPING)
    return df



def filter_target_defined_rows(
        df: pd.DataFrame, 
        target_column: str = "bad_loan",
        ) -> pd.DataFrame:
    """
    Keep only rows where bad_loan is defined. 
    
    This removes Current, In Grace Period, Late (16-30 days), missing loan_status, and any other unresolved/unmapped statuses. 
    """

    if target_column not in df.columns:
        raise ValueError(f"Missing required column: {target_column}")
    
    filtered_df = df[df[target_column].notna()].copy()

    filtered_df[target_column] = filtered_df[target_column].astype(int)
    return filtered_df


def get_target_summary(
        df: pd.DataFrame, 
        target_column: str = "bad_loan"
) -> pd.DataFrame:
    """
    Return a simple count and percentage summary for the target column
    """

    if target_column not in df.columns:
        raise ValueError(f"Missing required columns: {target_column}")
    
    summary_df = (
        df[target_column].value_counts(dropna=False).reset_index()
    )

    summary_df.columns = [target_column, "count"]
    summary_df["percentage"] = summary_df["count"] / len(df)

    return summary_df

def print_target_summary(
        df: pd.DataFrame, 
        target_column: str = "bad_loan", 
) -> None:
    """
    Print the target distribution in a readable format.
    """

    summary_df = get_target_summary(df, target_column=target_column)

    print("\nTarget Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    print("Loading sample raw LendingClub data...")

    sample_df = load_raw_lendingclub_data(nrows=500_000)

    print("Adding bad_loan target...")
    sample_with_target = add_bad_loan_target(sample_df)

    print("\nSummary before filtering unresolved statuses:")
    print_target_summary(sample_with_target)

    print("\nFiltering to rows with defined target...")
    supervised_sample = filter_target_defined_rows(sample_with_target)

    print("\nSummary after filtering unresolved statuses:")
    print_target_summary(supervised_sample)

    print(f"\nOriginal sample rows: {len(sample_df):,}")
    print(f"Rows with defined target: {len(supervised_sample):,}")
    print(f"Rows excluded from supervised sample: {len(sample_df) - len(supervised_sample):,}")