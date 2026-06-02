from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_DEFINED_DATA_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lendingclub_target_defined.csv.gz"
)

REPORTS_TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

INTERIM_MISSINGNESS_REPORT_PATH = (
    REPORTS_TABLE_DIR / "interim_missingness_report.csv"
)

def validate_target_defined_dataset(
        input_path: Path = TARGET_DEFINED_DATA_PATH,
) -> pd.DataFrame:
    """
    Validate the locally generated target-defined interim dataset

    This confirms that: 
    1. The file exists. 
    2. The dataset can be loaded successfully.
    3. The row count matches the expected target-defined population.
    4. The bad_loan column exists.
    5. bad_loan contains no missing values. 
    """

    expected_row_count = 1_369_566
    target_column = "bad_loan"

    print("Validating target-defined interim dataset...")
    print(f"Input path: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError("Dataset not found: {input_path}")
    
    df = pd.read_csv(
        input_path, 
        compression = "infer", 
        low_memory = False, 
    )

    row_count, column_count = df.shape

    print("\nDataset loaded successfully.")
    print(f"Rows Detected: {row_count:,}")
    print(f"Columns detected: {column_count:,}")

    if row_count != expected_row_count:
        raise ValueError(f"Unexpected row count. "
                        f"Expected {expected_row_count:,} but found {row_count:,}." 
        )

    if target_column not in df.columns: 
        raise ValueError(f"Missing required target column: {target_column}")
    
    target_counts = df[target_column].value_counts(dropna=False).sort_index()

    print("\nTarget counts:")
    print(target_counts)

    missing_target_count = df[target_column].isna().sum()

    if missing_target_count != 0:
        raise ValueError(f"Target column contains {missing_target_count:,} missing values.")
    

    observed_target_values = set(df[target_column].unique())
    expected_target_values = {0,1}

    if observed_target_values != expected_target_values:
        raise ValueError(
            f"Unexpected target values."
            f"Expected {expected_target_values},"
            f"but found {observed_target_values}."
        )

    print("\nValidation passed.")
    print("The interim dataset is ready for the next Module 2 step.")
    return df

def create_missingness_report(
    df: pd.DataFrame, 
    output_path: Path = INTERIM_MISSINGNESS_REPORT_PATH,
) -> pd.DataFrame:
    """
    Create and save a missingness report for the interim dataset. 

    The report contains:
    1. Column name
    2. Data type
    3. Missing-value count
    4. Missing-value percentage
    5. Non-missing-value count
    """

    row_count = df.shape[0]

    report_df = pd.DataFrame(
        {
            "column_name": df.columns, 
            "dtype": df.dtypes.astype(str).to_numpy(),
            "missing_count": df.isna().sum().to_numpy(),
        } 
    )

    report_df["missing_percentage"] = (
        report_df["missing_count"] / row_count
    )

    report_df = report_df.sort_values(
        by=["missing_percentage", "column_name"], ascending=[False, True], 
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(output_path, index=False)

    print("\nMissingness report created.")
    print(f"Saved to: {output_path}")

    print("\nTop 20 columns by missing percentage: ")
    print(report_df.head(20).to_string(index=False))
    return report_df


if __name__ == "__main__":
    validated_df = validate_target_defined_dataset()

    create_missingness_report(validated_df)
    