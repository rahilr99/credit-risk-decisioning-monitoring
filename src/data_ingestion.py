from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"

REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
RAW_LOAN_STATUS_COUNTS_PATHS = REPORTS_TABLES_DIR / "raw_loan_status_counts.csv"

def load_raw_lendingclub_data(nrows: int | None = None) -> pd.DataFrame:
    """
    Load the raw LendingClub accepted-loan dataset.

    Parameters
    ----------
    nrows: 
        Optional number of rows to load. Use this for safe testing before loading the full dataset.

    Returns
    ----------
    pd.DataFrame 
        Raw LendingClub data loaded from data/raw/.
    """

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_PATH}")
    
    df = pd.read_csv (RAW_DATA_PATH, compression='infer', low_memory=False, nrows=nrows,
                      )
    
    return df

def basic_dataframe_report(df: pd.DataFrame) -> None:
    """
    Print a basic inspection report for a DataFrame.
    """

    print("Basic Dataframe report")
    print("=" * 30)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Memory usage: {memory_mb:,.2f} MB")

    print("\nColumn Names: ")
    print(df.columns.to_list())

    print("\nData Types: ")
    print(df.dtypes)

    print("\nFirst 5 rows: ")
    print(df.head())

def save_loan_status_counts_report(loan_status_counts: dict, total_rows: int, output_path: Path = RAW_LOAN_STATUS_COUNTS_PATHS, ) -> pd.DataFrame:
    """
    Save the full loan_status distribution as a CSV report artifact.

    This report is useful for documenting the raw target-related values before target construction. 
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    records = []

    for status, count in sorted(loan_status_counts.items(), key=lambda x: str(x[0])):
        if pd.isna(status):
            clean_status = "MISSING"
        else: 
            clean_status = str(status)
        
        records.append(
            {
                "loan_status": clean_status, 
                "count": int(count), 
                "percentage_of_total_rows": count / total_rows, 
            }
        )

    report_df = pd.DataFrame(records)

    report_df.to_csv(output_path, index=False)
    print(f"\nSaved loan_status count report to: {output_path}")
    return report_df

def inspect_raw_lendingclub_file_in_chunks(chunksize: int = 100_000) -> None:
    """
    Inspect the raw LendingClub file in chunks without loading the full dataset
    into memory at once. 

    This is useful for large-file sanity checks during Module 2.
    """

    total_rows = 0
    chunk_count = 0
    expected_columns = None
    loan_status_counts = {}

    print("Starting chunked raw file inspection...")
    print(f"Chunk size: {chunksize:,} rows")

    for chunk in pd.read_csv(RAW_DATA_PATH, compression='infer', low_memory=False, chunksize=chunksize,):
        chunk_count +=1
        total_rows +=len(chunk)

        if expected_columns is None:
            expected_columns = list(chunk.columns)
            print(f"Detected columns: {len(expected_columns):,}")

        if list(chunk.columns) != expected_columns:
            print(f"WARNING: Column mismatch detected in chunk {chunk_count}")

        if "loan_status" in chunk.columns: 
            chunk_status_counts = chunk["loan_status"].value_counts(dropna = False)

            for status, count in chunk_status_counts.items():
                loan_status_counts[status] = loan_status_counts.get(status, 0) + count 
        print(f"Processed chunk {chunk_count:,} | Total rows so far: {total_rows:,}")
    print("\nChunked inspection complete.")
    print(f"Total chunks processed: {chunk_count:,}")
    print(f"Total rows detected: {total_rows:,}")

    if loan_status_counts:
        print("\n Loan status counts: ")
        for status, count in sorted(loan_status_counts.items(), key=lambda x: str(x[0])):
            print(f"{status}: {count:,}")

    save_loan_status_counts_report(loan_status_counts=loan_status_counts, total_rows=total_rows)




if __name__ == "__main__":
    import time

    start_time = time.time()

    inspect_raw_lendingclub_file_in_chunks(chunksize=100_000)

    end_time = time.time()
    elapsed_seconds = end_time - start_time

    print(f"\nElapsed time :{elapsed_seconds: .2f} seconds")