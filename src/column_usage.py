# src/column_usage.py

"""
Utilities for loading, validating, and querying the Module 2 column usage inventory.

This module is the gatekeeper for Module 3.

Rule:
    No column should enter preprocessing or modeling unless it is approved in
    config/column_usage_inventory.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COLUMN_USAGE_INVENTORY_PATH = (
    PROJECT_ROOT / "config" / "column_usage_inventory.csv"
)


REQUIRED_COLUMNS = {
    "column_name",
    "primary_purpose",
    "additional_use_cases",
}


APPROVED_PRIMARY_PURPOSES = {
    "modeling",
    "monitoring",
    "target",
    "target_source",
    "identifier",
    "lender_derived_feature",
    "conditionally_available",
    "high_cardinality_or_text",
    "post_application_leakage",
    "constant_or_empty",
    "exclude_from_model_features",
}


APPROVED_ADDITIONAL_USE_CASES = {
    "benchmarking",
    "data_quality_check",
    "portfolio_monitoring",
    "portfolio_segmentation",
    "reporting",
    "time_split",
}


def parse_additional_use_cases(value: object) -> list[str]:
    """
    Convert a comma-separated additional_use_cases value into a clean list.

    Examples
    --------
    "" -> []
    NaN -> []
    "reporting" -> ["reporting"]
    "portfolio_monitoring, reporting" -> ["portfolio_monitoring", "reporting"]
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if text == "":
        return []

    return [item.strip() for item in text.split(",") if item.strip()]


def load_column_usage_inventory(
    path: Path | str = DEFAULT_COLUMN_USAGE_INVENTORY_PATH,
) -> pd.DataFrame:
    """
    Load the Module 2 column usage inventory from CSV.

    Parameters
    ----------
    path:
        Path to config/column_usage_inventory.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned column usage inventory.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Column usage inventory not found: {path}")

    df = pd.read_csv(path)

    # Only clean fields if they exist. Required-column validation happens later.
    if "column_name" in df.columns:
        df["column_name"] = df["column_name"].astype(str).str.strip()

    if "primary_purpose" in df.columns:
        df["primary_purpose"] = df["primary_purpose"].astype(str).str.strip()

    if "additional_use_cases" in df.columns:
        df["additional_use_cases"] = (
            df["additional_use_cases"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    validate_column_usage_inventory(df)

    return df


def validate_column_usage_inventory(df: pd.DataFrame) -> None:
    """
    Validate that the column usage inventory follows the expected schema and rules.

    Raises
    ------
    ValueError
        If the inventory is missing required columns, has duplicate column names,
        has blank values where they are not allowed, or contains unapproved values.
    """
    missing_required_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_required_columns:
        raise ValueError(
            "Column usage inventory is missing required columns: "
            f"{sorted(missing_required_columns)}"
        )

    if df.empty:
        raise ValueError("Column usage inventory is empty.")

    blank_column_names = df["column_name"].isna() | (df["column_name"].astype(str).str.strip() == "")
    if blank_column_names.any():
        bad_rows = df.index[blank_column_names].tolist()
        raise ValueError(f"Blank column_name values found at rows: {bad_rows}")

    duplicate_column_names = df[df["column_name"].duplicated(keep=False)]["column_name"].tolist()
    if duplicate_column_names:
        raise ValueError(
            "Duplicate column_name values found in column usage inventory: "
            f"{sorted(set(duplicate_column_names))}"
        )

    blank_primary_purpose = (
        df["primary_purpose"].isna()
        | (df["primary_purpose"].astype(str).str.strip() == "")
    )
    if blank_primary_purpose.any():
        bad_rows = df.index[blank_primary_purpose].tolist()
        raise ValueError(f"Blank primary_purpose values found at rows: {bad_rows}")

    invalid_primary_purposes = sorted(
        set(df["primary_purpose"]) - APPROVED_PRIMARY_PURPOSES
    )
    if invalid_primary_purposes:
        raise ValueError(
            "Invalid primary_purpose values found: "
            f"{invalid_primary_purposes}. "
            f"Approved values are: {sorted(APPROVED_PRIMARY_PURPOSES)}"
        )

    invalid_additional_use_cases: set[str] = set()

    for value in df["additional_use_cases"]:
        parsed_values = parse_additional_use_cases(value)

        for use_case in parsed_values:
            if use_case not in APPROVED_ADDITIONAL_USE_CASES:
                invalid_additional_use_cases.add(use_case)

    if invalid_additional_use_cases:
        raise ValueError(
            "Invalid additional_use_cases values found: "
            f"{sorted(invalid_additional_use_cases)}. "
            f"Approved values are: {sorted(APPROVED_ADDITIONAL_USE_CASES)}"
        )


def get_columns_by_primary_purpose(
    df: pd.DataFrame,
    primary_purpose: str | Iterable[str],
) -> list[str]:
    """
    Return column names matching one or more primary_purpose values.

    Examples
    --------
    get_columns_by_primary_purpose(df, "modeling")
    get_columns_by_primary_purpose(df, ["modeling", "monitoring"])
    """
    if isinstance(primary_purpose, str):
        purposes = {primary_purpose}
    else:
        purposes = set(primary_purpose)

    invalid_purposes = purposes - APPROVED_PRIMARY_PURPOSES
    if invalid_purposes:
        raise ValueError(
            f"Invalid primary_purpose requested: {sorted(invalid_purposes)}"
        )

    return (
        df.loc[df["primary_purpose"].isin(purposes), "column_name"]
        .tolist()
    )


def get_columns_by_additional_use_case(
    df: pd.DataFrame,
    additional_use_case: str | Iterable[str],
) -> list[str]:
    """
    Return column names that contain one or more additional_use_cases.

    Examples
    --------
    get_columns_by_additional_use_case(df, "time_split")
    get_columns_by_additional_use_case(df, ["reporting", "portfolio_monitoring"])
    """
    if isinstance(additional_use_case, str):
        requested_use_cases = {additional_use_case}
    else:
        requested_use_cases = set(additional_use_case)

    invalid_use_cases = requested_use_cases - APPROVED_ADDITIONAL_USE_CASES
    if invalid_use_cases:
        raise ValueError(
            f"Invalid additional_use_case requested: {sorted(invalid_use_cases)}"
        )

    mask = df["additional_use_cases"].apply(
        lambda value: bool(requested_use_cases.intersection(parse_additional_use_cases(value)))
    )

    return df.loc[mask, "column_name"].tolist()


def get_modeling_columns(df: pd.DataFrame) -> list[str]:
    """
    Return columns approved as modeling features.
    """
    return get_columns_by_primary_purpose(df, "modeling")


def get_target_columns(df: pd.DataFrame) -> list[str]:
    """
    Return target columns.

    Expected for this project:
        ["bad_loan"]
    """
    return get_columns_by_primary_purpose(df, "target")


def get_target_source_columns(df: pd.DataFrame) -> list[str]:
    """
    Return columns used to define the target but excluded from model features.

    Expected example:
        ["loan_status"]
    """
    return get_columns_by_primary_purpose(df, "target_source")


def get_identifier_columns(df: pd.DataFrame) -> list[str]:
    """
    Return identifier columns.

    These may be useful for tracking rows but must not enter model features.
    """
    return get_columns_by_primary_purpose(df, "identifier")


def get_monitoring_columns(df: pd.DataFrame) -> list[str]:
    """
    Return columns whose primary purpose is monitoring.
    """
    return get_columns_by_primary_purpose(df, "monitoring")


def get_time_split_columns(df: pd.DataFrame) -> list[str]:
    """
    Return columns marked for time-based splitting.

    Expected example:
        ["issue_d"]
    """
    return get_columns_by_additional_use_case(df, "time_split")


def get_excluded_model_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return columns that should not enter the first modeling feature set.

    This includes leakage, identifiers, target source columns, monitoring-only
    fields, and other excluded/non-feature columns.
    """
    excluded_purposes = {
        "monitoring",
        "target",
        "target_source",
        "identifier",
        "lender_derived_feature",
        "conditionally_available",
        "high_cardinality_or_text",
        "post_application_leakage",
        "constant_or_empty",
        "exclude_from_model_features",
    }

    return get_columns_by_primary_purpose(df, excluded_purposes)


def summarize_column_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table showing how many columns belong to each primary purpose.
    """
    summary = (
        df["primary_purpose"]
        .value_counts()
        .rename_axis("primary_purpose")
        .reset_index(name="column_count")
        .sort_values("primary_purpose")
        .reset_index(drop=True)
    )

    return summary


def print_column_usage_summary(df: pd.DataFrame) -> None:
    """
    Print a readable summary of the column usage inventory.
    """
    summary = summarize_column_usage(df)

    print("\nColumn usage inventory loaded successfully.")
    print(f"Total columns: {len(df):,}")

    print("\nColumns by primary purpose:")
    print(summary.to_string(index=False))

    print("\nKey groups:")
    print(f"Modeling columns: {len(get_modeling_columns(df)):,}")
    print(f"Target columns: {len(get_target_columns(df)):,}")
    print(f"Target source columns: {len(get_target_source_columns(df)):,}")
    print(f"Identifier columns: {len(get_identifier_columns(df)):,}")
    print(f"Monitoring columns: {len(get_monitoring_columns(df)):,}")
    print(f"Time split columns: {len(get_time_split_columns(df)):,}")
    print(f"Excluded model-feature columns: {len(get_excluded_model_feature_columns(df)):,}")


if __name__ == "__main__":
    column_usage_df = load_column_usage_inventory()
    print_column_usage_summary(column_usage_df)