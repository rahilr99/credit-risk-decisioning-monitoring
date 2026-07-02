# src/create_modeling_splits.py

from pathlib import Path

import pandas as pd

from column_usage import (
    DEFAULT_COLUMN_USAGE_INVENTORY_PATH,
    get_modeling_columns,
    load_column_usage_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELING_BASE_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "lendingclub_modeling_base.csv.gz"
)

DEFAULT_TRAIN_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "modeling_train.csv.gz"
)

DEFAULT_VALIDATION_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "modeling_validation.csv.gz"
)

DEFAULT_TEST_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "modeling_test.csv.gz"
)

DEFAULT_SPLIT_SUMMARY_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "modeling_split_summary.csv"
)

DEFAULT_TIME_SPLIT_COLUMN = "issue_d"
DEFAULT_TARGET_COLUMN = "bad_loan"
TEMP_PARSED_DATE_COLUMN = "_issue_date_parsed"
SPLIT_COLUMN = "modeling_split"

SPLIT_ORDER = ["train", "validation", "test"]

def validate_split_fractions(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> None:
    """
    Validate that split fractions are positive and add up to 1.
    """
    fractions = {
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
    }

    non_positive_fractions = [
        name for name, value in fractions.items() if value <= 0
    ]

    if non_positive_fractions:
        raise ValueError(
            "All split fractions must be positive. Invalid fractions: "
            f"{non_positive_fractions}"
        )

    total_fraction = train_fraction + validation_fraction + test_fraction

    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError(
            "Train, validation, and test fractions must add up to 1. "
            f"Current total: {total_fraction}"
        )
    

    
def load_modeling_base_dataset(path: Path | str) -> pd.DataFrame:
    """
    Load the modeling base dataset. 
    """
    path = Path(path)
    
    return pd.read_csv(path, low_memory=False)



def validate_required_columns(
        available_columns: list[str], 
        required_columns: list[str],
) -> None:
    """
    Validate that required columns exist in the dataset. 
    """
    available_column_set = set(available_columns)

    missing_columns = [
        column for column in required_columns if column not in available_column_set
    ]

    if missing_columns:
        raise ValueError(
            "The modeling base dataset is missing required columns: "
            f"{missing_columns}"
        )

def add_parsed_time_split_column(
    df: pd.DataFrame,
    time_split_column: str = DEFAULT_TIME_SPLIT_COLUMN,
    parsed_date_column: str = TEMP_PARSED_DATE_COLUMN,
) -> pd.DataFrame:
    """
    Parse the time split column into a datetime column used for chronological sorting.
    """
    df = df.copy()

    cleaned_dates = df[time_split_column].astype(str).str.strip()

    parsed_dates = pd.to_datetime(
        cleaned_dates,
        format="%b-%Y",
        errors="coerce",
    )

    fallback_parse_mask = parsed_dates.isna() & cleaned_dates.notna()

    if fallback_parse_mask.any():
        fallback_parsed_dates = pd.to_datetime(
            cleaned_dates.loc[fallback_parse_mask],
            errors="coerce",
        )

        parsed_dates.loc[fallback_parse_mask] = fallback_parsed_dates

    unparseable_mask = parsed_dates.isna()

    if unparseable_mask.any():
        unparseable_examples = (
            df.loc[unparseable_mask, time_split_column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()[:10]
        )

        raise ValueError(
            f"Some values in {time_split_column} could not be parsed as dates. "
            f"Examples: {unparseable_examples}"
        )

    df[parsed_date_column] = parsed_dates

    return df

def assign_time_based_splits(
    df: pd.DataFrame,
    parsed_date_column: str = TEMP_PARSED_DATE_COLUMN,
    split_column: str = SPLIT_COLUMN,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.DataFrame:
    """
    Sort rows chronologically and assign train, validation, and test splits.
    """
    validate_split_fractions(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    df = df.sort_values(
        by=[parsed_date_column],
        ascending=True,
    ).reset_index(drop=True)

    total_rows = len(df)

    train_end_index = int(total_rows * train_fraction)
    validation_end_index = train_end_index + int(total_rows * validation_fraction)

    df[split_column] = "test"
    df.loc[: train_end_index - 1, split_column] = "train"
    df.loc[train_end_index : validation_end_index - 1, split_column] = "validation"

    return df

def create_split_summary(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    parsed_date_column: str = TEMP_PARSED_DATE_COLUMN,
    split_column: str = SPLIT_COLUMN,
) -> pd.DataFrame:
    """
    Create a summary report for the train, validation, and test splits.
    """
    total_rows = len(df)
    summary_rows = []

    for split_name in SPLIT_ORDER:
        split_df = df[df[split_column] == split_name]

        row_count = len(split_df)
        bad_loan_count = int((split_df[target_column] == 1).sum())
        good_loan_count = int((split_df[target_column] == 0).sum())

        if row_count == 0:
            bad_loan_rate = 0
            date_min = None
            date_max = None
        else:
            bad_loan_rate = bad_loan_count / row_count
            date_min = split_df[parsed_date_column].min()
            date_max = split_df[parsed_date_column].max()

        summary_rows.append(
            {
                "split": split_name,
                "row_count": row_count,
                "row_percentage": row_count / total_rows,
                "date_min": date_min,
                "date_max": date_max,
                "good_loan_count": good_loan_count,
                "bad_loan_count": bad_loan_count,
                "bad_loan_rate": bad_loan_rate,
            }
        )

    return pd.DataFrame(summary_rows)

def save_split_datasets(
    df: pd.DataFrame,
    train_dataset_path: Path | str = DEFAULT_TRAIN_DATASET_PATH,
    validation_dataset_path: Path | str = DEFAULT_VALIDATION_DATASET_PATH,
    test_dataset_path: Path | str = DEFAULT_TEST_DATASET_PATH,
    split_column: str = SPLIT_COLUMN,
    temporary_columns: list[str] | None = None,
) -> None:
    """
    Save train, validation, and test datasets.
    """
    if temporary_columns is None:
        temporary_columns = [TEMP_PARSED_DATE_COLUMN, SPLIT_COLUMN]

    output_paths = {
        "train": Path(train_dataset_path),
        "validation": Path(validation_dataset_path),
        "test": Path(test_dataset_path),
    }

    for split_name, output_path in output_paths.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)

        split_df = df[df[split_column] == split_name].copy()

        columns_to_drop = [
            column for column in temporary_columns if column in split_df.columns
        ]

        split_df = split_df.drop(columns=columns_to_drop)

        split_df.to_csv(
            output_path,
            index=False,
            compression="gzip",
        )


def save_dataframe_to_csv(df: pd.DataFrame, path: Path | str) -> None:
    """
    Save a DataFrame to CSV, creating the parent directory if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

def create_modeling_splits(
    modeling_base_dataset_path: Path | str = DEFAULT_MODELING_BASE_DATASET_PATH,
    column_usage_inventory_path: Path | str = DEFAULT_COLUMN_USAGE_INVENTORY_PATH,
    train_dataset_path: Path | str = DEFAULT_TRAIN_DATASET_PATH,
    validation_dataset_path: Path | str = DEFAULT_VALIDATION_DATASET_PATH,
    test_dataset_path: Path | str = DEFAULT_TEST_DATASET_PATH,
    split_summary_report_path: Path | str = DEFAULT_SPLIT_SUMMARY_REPORT_PATH,
    time_split_column: str = DEFAULT_TIME_SPLIT_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.DataFrame:
    """
    Main workflow for creating time-based modeling splits.
    """
    print("\nLoading column usage inventory...")
    column_usage_df = load_column_usage_inventory(column_usage_inventory_path)

    modeling_columns = get_modeling_columns(column_usage_df)
    print(f"Approved modeling feature columns: {len(modeling_columns):,}")

    print("\nLoading modeling base dataset...")
    df = load_modeling_base_dataset(modeling_base_dataset_path)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {df.shape[1]:,}")

    print("\nValidating required columns...")
    required_columns = modeling_columns + [time_split_column, target_column]

    validate_required_columns(
        available_columns=df.columns.tolist(),
        required_columns=required_columns,
    )

    print("Required column validation passed.")

    print("\nParsing time split column...")
    df = add_parsed_time_split_column(
        df=df,
        time_split_column=time_split_column,
    )

    print("\nAssigning time-based splits...")
    df = assign_time_based_splits(
        df=df,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    print("\nCreating split summary...")
    split_summary_df = create_split_summary(
        df=df,
        target_column=target_column,
    )

    print("\nSplit summary:")
    print(split_summary_df.to_string(index=False))

    print("\nSaving split datasets...")
    save_split_datasets(
        df=df,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        test_dataset_path=test_dataset_path,
    )

    print(f"Saved train dataset to: {train_dataset_path}")
    print(f"Saved validation dataset to: {validation_dataset_path}")
    print(f"Saved test dataset to: {test_dataset_path}")

    print("\nSaving split summary report...")
    save_dataframe_to_csv(
        df=split_summary_df,
        path=split_summary_report_path,
    )

    print(f"Saved split summary report to: {split_summary_report_path}")
    print("\nModeling split creation complete.")

    return split_summary_df


if __name__ == "__main__":
    create_modeling_splits()
