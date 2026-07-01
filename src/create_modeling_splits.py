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
    available_column_set = set(available_column_set)

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