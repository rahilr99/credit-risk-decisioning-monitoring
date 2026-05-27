import pandas as pd
from pathlib import Path
from data_ingestion import load_raw_lendingclub_data, iter_raw_lendingclub_data_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
BAD_LOAN_TARGET_SUMMARY_PATH = REPORTS_TABLES_DIR / "bad_loan_target_summary.csv"


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

def summarize_bad_loan_target_in_chunks(
    chunksize: int = 100_000, 
    output_path: Path = BAD_LOAN_TARGET_SUMMARY_PATH, 
) -> pd.DataFrame:
    """
    Apply the bad_loan target definition to the full raw dataset in chunks and save a target summary report.

    This avoids loading the full LendingClub dataset into memory.
    """
    total_rows = 0
    chunk_count = 0
    target_counts = {}

    print("Starting chunked bad_loan target summary...")
    print(f"Chunk size: {chunksize:,} rows")

    for chunk in iter_raw_lendingclub_data_chunks(chunksize=chunksize):
        chunk_count += 1
        total_rows += len(chunk)

        chunk_with_target = add_bad_loan_target(chunk)

        chunk_target_counts = chunk_with_target["bad_loan"].value_counts(dropna=False)

        for target_value, count in chunk_target_counts.items():
            if pd.isna(target_value):
                clean_target_value = "MISSING"
            else: 
                clean_target_value = int(target_value)
            
            target_counts[clean_target_value] = (
                target_counts.get(clean_target_value, 0) + int(count)
            )
        print(f"Processed chunk {chunk_count:,} | Total rows so far: {total_rows:,}")

    records = []

    target_labels = {
        0: "good_loan", 
        1: "bad_loan", 
        "MISSING": "excluded_or_unmapped", 
    }

    for target_value in [0, 1, "MISSING"]:
        count = target_counts.get(target_value, 0)

        records.append(
            {
                "bad_loan": target_value, 
                "target_label": target_labels[target_value],
                "count": count,
                "percentage_of_total_rows": count / total_rows,
            }
        )

    summary_df = pd.DataFrame(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print("\nChunked bad_loan target summary complete.")
    print(f"Total chunks processed: {chunk_count:,}")
    print(f"Total rows processed: {total_rows:,}")

    print("\nTarget summary:")
    print(summary_df.to_string(index=False))

    print(f"\nSaved bad_loan target summary report to: {output_path}")

       



if __name__ == "__main__":
    # print("Loading sample raw LendingClub data...")

    # sample_df = load_raw_lendingclub_data(nrows=500_000)

    # print("Adding bad_loan target...")
    # sample_with_target = add_bad_loan_target(sample_df)

    # print("\nSummary before filtering unresolved statuses:")
    # print_target_summary(sample_with_target)

    # print("\nFiltering to rows with defined target...")
    # supervised_sample = filter_target_defined_rows(sample_with_target)

    # print("\nSummary after filtering unresolved statuses:")
    # print_target_summary(supervised_sample)

    # print(f"\nOriginal sample rows: {len(sample_df):,}")
    # print(f"Rows with defined target: {len(supervised_sample):,}")
    # print(f"Rows excluded from supervised sample: {len(sample_df) - len(supervised_sample):,}")

    summarize_bad_loan_target_in_chunks(chunksize=100_000)