"""
Create the modeling base dataset for Module 3.

This script applies the Module 2 column usage inventory to the target-defined LendingClub dataset. 

It does not preprocess features yet. 

Main purpose: 
    - Load config/column_usage_inventory.csv
    - Identify approved modeling columns. 
    - Keep target, identifier, and time-split columns separately. 
    - Select those columns from data/interim/lendingclub_target_defined_csv.gz
    - Save a clean modeling base dataset under data/processed/
    - Save a column selection audit report under reports/tables/

"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from column_usage import(
    DEFAULT_COLUMN_USAGE_INVENTORY_PATH, 
    get_excluded_model_feature_columns, 
    get_identifier_columns, 
    get_modeling_columns, 
    get_target_columns, 
    get_target_source_columns, 
    get_time_split_columns, 
    load_column_usage_inventory, 
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TARGET_DEFINED_DATASET_PATH = (
    PROJECT_ROOT / "data" / "interim" / "lendingclub_target_defined.csv.gz"
)

DEFAULT_MODELING_BASE_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "lendingclub_modeling_base.csv.gz"
)

DEFAULT_COLUMN_SELECTION_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "modeling_column_selection_report.csv"
)


def deduplicate_preserve_order(values: Iterable[str]) ->list[str]:

    """
    Remove duplicates from a sequence while preserving the original order 

    Example
    -------

    ["id", "issue_id", "issue_d", "loan_amnt"]
    becomes 
    ["id", "issue_d", "loan_amnt"]
    """

    seen: set[str] = set()
    deduplicated_values: list[str] = []

    for value in values: 
        if value not in seen:
            seen.add(value)
            deduplicated_values.append(value)
    
    return deduplicated_values

def get_csv_columns(path: Path | str) -> list[str]:
    """
    Read only the header row of a csv file and return the column names. 

    This avoids loading the full dataset just to check what columns exist,
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    header_df = pd.read_csv(path, nrows=0)

    return header_df.columns.tolist()

def load_target_defined_dataset(
        path: Path | str, 
        usecols: list[str], 
) -> pd.DataFrame:
    """
    Load selected columns from the target-defined LendingClub dataset.

    Parameters
    ----------
    path:
        Path to data/interim/lendingclub_target.defined.csv.gz. 

    usecols: 
        Columns to load from the dataset.

    Returns
    -------
    pd.DataFrame
        Dataset containing only the selected modeling-base columns. 

    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Target-defined dataset not found: {path}")
    
    df = pd.read_csv(
        path, 
        usecols=usecols, 
        low_memory=False, 
    )
    
    # read_csv does not always preserve the exact order passed to usecols. 
    # Reorder columns explicitly so the output is predictable. 
    df = df.loc[:, usecols]

    return df


def get_modeling_base_column_groups(
        column_usage_df : pd.DataFrame, 
) -> dict[str, list[str]]:
    """
    Use the column usage inventory to define column groups for the modeling base dataset. 

    The modeling base dataset keeps: 
        - identifier columns for rows tracking
        - time split columns for later train/validation/test splitting
        - target columns as y

        It does not keep target-source columns, leakage columns, or other excluded columns. 
    """

    modeling_columns = get_modeling_columns(column_usage_df)
    target_columns = get_target_columns(column_usage_df)
    target_source_columns = get_target_source_columns(column_usage_df)
    identifier_columns = get_identifier_columns(column_usage_df)
    time_split_columns = get_time_split_columns(column_usage_df)
    excluded_model_feature_columns = get_excluded_model_feature_columns(column_usage_df)

    selected_columns = deduplicate_preserve_order(
        identifier_columns + time_split_columns + modeling_columns + target_columns
    )
     

    return {
        "modeling_columns": modeling_columns, 
        "target_columns": target_columns, 
        "target_source_columns": target_source_columns,
        "identifier_columns": identifier_columns, 
        "time_split_columns": time_split_columns, 
        "excluded_model_feature_columns": excluded_model_feature_columns,
        "selected_columns": selected_columns,
    }

def validate_modeling_base_column_groups(
        available_dataset_columns: list[str],
        column_usage_df: pd.DataFrame, 
        column_groups: dict[str, list[str]], 
) -> None:
    """
    Validate that the selected modeling-base columns are safe and available. 

    This checks column names and feature-selection logic before loading the full-dataset.
    """
    available_dataset_columns_set = set(available_dataset_columns)
    inventory_columns_set = set(column_usage_df["column_name"])

    missing_inventory_columns = sorted(
        inventory_columns_set - available_dataset_columns_set
    )

    if missing_inventory_columns: 
        raise ValueError(
            "Some columns from config/column_usage_inventory.csv are missing "
            "from the target-defined dataset: "
            f"{missing_inventory_columns}"
        )
    
    selected_columns = column_groups["selected_columns"]
    modeling_columns = column_groups["modeling_columns"]
    target_columns = column_groups["target_columns"]
    target_source_columns = column_groups["target_source_columns"]
    identifier_columns = column_groups["identifier_columns"]
    time_split_columns = column_groups["time_split_columns"]
    excluded_model_feature_columns = column_groups["excluded_model_feature_columns"]

    missing_selected_columns = sorted(
        set(selected_columns) - available_dataset_columns_set
    )

    if missing_selected_columns: 
        raise ValueError(
            "Selected modeling-base columns are missing from the dataset: "
            f"{missing_selected_columns}"
        )
    
    if len(target_columns) != 1: 
        raise ValueError(
            "Expected exactly one target column, but found "
            f"{len(target_columns)}: {target_columns}"
        )
    
    if len(time_split_columns) !=1 : 
        raise ValueError(
            "Expected exactly one time-split column, but found"
            f"{len(time_split_columns)}: {time_split_columns}"
        )
    if len(identifier_columns) == 0: 
        raise ValueError(
            "Expected at least one identifier column for row tracking. "
        )
    
    if len(modeling_columns) == 0: 
        raise ValueError(
            "No modeling columns are selected."
        )
    
    duplicate_selected_columns = [
        column 
        for column in selected_columns
        if selected_columns.count(column) > 1
    ]

    if duplicate_selected_columns: 
        raise ValueError(
            "Duplicate columns found in selected modeling-base columns: "
            f"{sorted(set(duplicate_selected_columns))}"
        )
    modeling_column_set = set(modeling_columns)

    unsafe_modeling_columns = sorted(
        modeling_column_set.intersection(
            set(target_columns)
            | set(target_source_columns)
            | set(identifier_columns)
            | set(time_split_columns)
            | set(excluded_model_feature_columns)

        )
    )

    if unsafe_modeling_columns: 
        raise ValueError(
            "Unsafe columns were found inside the modeling feature set: "
            f"{unsafe_modeling_columns}"
        )
    
    selected_target_source_columns = sorted(
        set(selected_columns).intersection(target_source_columns)
    )

    if selected_target_source_columns: 
        raise ValueError(
            "Target-source columns should not be included in the modeling base dataset: "
            f"{selected_target_source_columns}"
        )



def validate_modeling_base_dataset(
        modeling_base_df: pd.DataFrame, 
        column_groups: dict[str, list[str]],
) -> None:
    """
    Validate the loaded modeling dataset after column selection.
    """
    if modeling_base_df.empty:
        raise ValueError("Modeling base dataset is empty.")
    
    duplicated_output_columns = modeling_base_df.columns[modeling_base_df.columns.duplicated()].tolist()

    if duplicated_output_columns:
        raise ValueError(
            "Duplicate columns found in modeling base dataset: "
            f"{duplicated_output_columns}"
        )
    
    target_column = column_groups["target_columns"][0]

    missing_target_count = int(modeling_base_df[target_column].isna().sum())

    if missing_target_count > 0:
        raise ValueError(
            f"Target Column {target_column!r} has {missing_target_count:,} missing_values. "
        )
    
    observed_target_values = set(modeling_base_df[target_column].dropna().unique())

    invalid_target_values = sorted(
        value 
        for value in observed_target_values\
        if value not in {0,1}
    )

    if invalid_target_values: 
        raise ValueError(
            f"Target column {target_column!r} contains invalid values: "
            f"{invalid_target_values}. Expected only 0 and 1."
        )

def create_column_selection_report(
    column_usage_df: pd.DataFrame,
    column_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Create an audit report showing which columns were selected for the modeling base
    dataset and what role each selected column plays.

    The report includes all columns from the column usage inventory, not just the
    selected columns. This makes it easy to see both included and excluded columns.
    """
    report_df = column_usage_df.copy()

    role_by_column: dict[str, list[str]] = {}

    def add_role(columns: list[str], role: str) -> None:
        """
        Attach a modeling-base role to each column in a list.
        """
        for column in columns:
            role_by_column.setdefault(column, [])

            if role not in role_by_column[column]:
                role_by_column[column].append(role)

    add_role(column_groups["identifier_columns"], "identifier_tracking")
    add_role(column_groups["time_split_columns"], "time_split")
    add_role(column_groups["modeling_columns"], "model_feature")
    add_role(column_groups["target_columns"], "target")

    selected_columns = column_groups["selected_columns"]
    selected_column_set = set(selected_columns)

    selected_column_order = {
        column: position
        for position, column in enumerate(selected_columns, start=1)
    }

    report_df["selected_for_modeling_base"] = report_df["column_name"].isin(
        selected_column_set
    )

    report_df["role_in_modeling_base"] = report_df["column_name"].map(
        lambda column: ", ".join(role_by_column.get(column, []))
    )

    report_df["modeling_base_column_order"] = report_df["column_name"].map(
        selected_column_order
    )

    report_df = (
        report_df
        .sort_values(
            by=[
                "selected_for_modeling_base",
                "modeling_base_column_order",
                "column_name",
            ],
            ascending=[False, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return report_df

def save_dataframe_to_csv(
    df: pd.DataFrame,
    path: Path | str,
    compression: str | None = None,
) -> None:
    """
    Save a DataFrame to CSV, creating the parent directory if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False, compression=compression)


def create_modeling_base_dataset(
    target_defined_dataset_path: Path | str = DEFAULT_TARGET_DEFINED_DATASET_PATH,
    column_usage_inventory_path: Path | str = DEFAULT_COLUMN_USAGE_INVENTORY_PATH,
    output_dataset_path: Path | str = DEFAULT_MODELING_BASE_DATASET_PATH,
    column_selection_report_path: Path | str = DEFAULT_COLUMN_SELECTION_REPORT_PATH,
) -> pd.DataFrame:
    """
    Main workflow for creating the modeling base dataset.
    """
    print("\nLoading column usage inventory...")
    column_usage_df = load_column_usage_inventory(column_usage_inventory_path)
    print(f"Inventory rows: {len(column_usage_df):,}")

    print("\nCreating modeling-base column groups...")
    column_groups = get_modeling_base_column_groups(column_usage_df)

    print(f"Modeling feature columns: {len(column_groups['modeling_columns']):,}")
    print(f"Target columns: {len(column_groups['target_columns']):,}")
    print(f"Identifier tracking columns: {len(column_groups['identifier_columns']):,}")
    print(f"Time split columns: {len(column_groups['time_split_columns']):,}")
    print(f"Total selected columns: {len(column_groups['selected_columns']):,}")

    print("\nReading target-defined dataset header...")
    available_dataset_columns = get_csv_columns(target_defined_dataset_path)
    print(f"Available dataset columns: {len(available_dataset_columns):,}")

    print("\nValidating selected columns...")
    validate_modeling_base_column_groups(
        available_dataset_columns=available_dataset_columns,
        column_usage_df=column_usage_df,
        column_groups=column_groups,
    )
    print("Column validation passed.")

    print("\nLoading selected columns from target-defined dataset...")
    modeling_base_df = load_target_defined_dataset(
        path=target_defined_dataset_path,
        usecols=column_groups["selected_columns"],
    )

    print(f"Rows loaded: {len(modeling_base_df):,}")
    print(f"Columns loaded: {modeling_base_df.shape[1]:,}")

    print("\nValidating modeling base dataset...")
    validate_modeling_base_dataset(
        modeling_base_df=modeling_base_df,
        column_groups=column_groups,
    )
    print("Dataset validation passed.")

    target_column = column_groups["target_columns"][0]
    target_rate = modeling_base_df[target_column].mean()

    print("\nTarget summary:")
    print(modeling_base_df[target_column].value_counts().sort_index().to_string())
    print(f"Bad-loan rate: {target_rate:.2%}")

    print("\nSaving modeling base dataset...")
    save_dataframe_to_csv(
        df=modeling_base_df,
        path=output_dataset_path,
        compression="gzip",
    )
    print(f"Saved dataset to: {output_dataset_path}")

    print("\nCreating column selection report...")
    column_selection_report_df = create_column_selection_report(
        column_usage_df=column_usage_df,
        column_groups=column_groups,
    )

    save_dataframe_to_csv(
        df=column_selection_report_df,
        path=column_selection_report_path,
    )
    print(f"Saved report to: {column_selection_report_path}")

    print("\nModeling base dataset creation complete.")

    return modeling_base_df

if __name__ == "__main__":
    create_modeling_base_dataset()