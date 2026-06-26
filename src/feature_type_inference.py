# src/feature_type_inference.py

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from column_usage import (
    DEFAULT_COLUMN_USAGE_INVENTORY_PATH,
    get_modeling_columns,
    load_column_usage_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELING_BASE_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "lendingclub_modeling_base.csv.gz"
)

DEFAULT_FEATURE_TYPE_PROFILE_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "modeling_feature_type_profile.csv"
)

DEFAULT_MISSINGNESS_PROFILE_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "modeling_missingness_profile.csv"
)

def get_csv_columns(path: Path | str) -> list[str]:
    """
    Read only the header row of a CSV file and return its column names.
    """
    path = Path(path)

    return pd.read_csv(path, nrows=0).columns.tolist()

def validate_modeling_columns_available(
    available_columns: list[str],
    modeling_columns: list[str],
) -> None:
    """
    Validate that all approved modeling columns exist in the modeling base dataset.
    """
    if not modeling_columns:
        raise ValueError("No modeling columns were found in the column usage inventory.")

    duplicate_modeling_columns = pd.Series(modeling_columns).duplicated()

    if duplicate_modeling_columns.any():
        duplicated_values = (
            pd.Series(modeling_columns)
            .loc[duplicate_modeling_columns]
            .tolist()
        )

        raise ValueError(
            "Duplicate modeling columns were found in the column usage inventory: "
            f"{duplicated_values}"
        )

    available_column_set = set(available_columns)

    missing_modeling_columns = [
        column for column in modeling_columns if column not in available_column_set
    ]

    if missing_modeling_columns:
        raise ValueError(
            "Some approved modeling columns are missing from the modeling base dataset: "
            f"{missing_modeling_columns}"
        )
    
def load_modeling_feature_dataset(
    path: Path | str,
    modeling_columns: list[str],
) -> pd.DataFrame:
    """
    Load only the approved modeling feature columns from the modeling base dataset.
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        usecols=modeling_columns,
        low_memory=False,
    )

    df = df.loc[:, modeling_columns]

    return df

def calculate_missing_percentage(series: pd.Series) -> float:
    """
    Calculate the percentage of missing values in a column
    """
    return series.isna().mean()

def calculate_unique_non_missing_count(series: pd.Series) -> int:
    """
    Count unique non-missing values in a column.
    """
    return series.dropna().nunique()

def is_integer_like_numeric(series: pd.Series) -> bool:
    """
    Check whether a numeric column contains integer-like values.

    Example:
    1.0, 2.0, 3.0 are integer-like.
    1.25, 2.50, 3.75 are not integer-like.
    """
    non_missing = series.dropna()

    if non_missing.empty:
        return False

    if not is_numeric_dtype(non_missing):
        return False

    return (non_missing % 1 == 0).all()