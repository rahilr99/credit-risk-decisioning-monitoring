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


def calculate_date_parse_success_rate(series: pd.Series) -> float | None:
    """
    Estimate whether a non-numeric column looks date-like by checking how many non-missing values can be parsed as dates. 
    """
    non_missing = series.dropna().astype(str).str.strip()
    non_missing = non_missing[non_missing != ""]

    if non_missing.empty: 
        return None
    
    parsed_values = pd.to_datetime(non_missing, errors="coerce")

    return parsed_values.notna().mean()


def get_sample_values(series: pd.Series, max_values: int = 5) -> str:
    """
    Return a small set of sample non-missing values for human review.
    """
    non_missing = series.dropna().astype(str).str.strip()
    non_missing = non_missing[non_missing != ""]

    if non_missing.empty:
        return ""

    unique_values = pd.Series(non_missing.unique()).head(max_values).tolist()

    return ", ".join(unique_values)

def infer_feature_type(
    series: pd.Series,
    low_cardinality_threshold: int = 20,
    discrete_unique_threshold: int = 50,
    discrete_unique_ratio_threshold: float = 0.02,
    date_parse_success_threshold: float = 0.90,
) -> dict[str, object]:
    """
    Infer the feature type and recommended preprocessing group for one column.
    """
    total_count = len(series)
    missing_count = series.isna().sum()
    non_missing_count = total_count - missing_count
    missing_percentage = calculate_missing_percentage(series)

    unique_non_missing_count = calculate_unique_non_missing_count(series)

    if non_missing_count == 0:
        unique_non_missing_ratio = 0
    else:
        unique_non_missing_ratio = unique_non_missing_count / non_missing_count

    dtype = str(series.dtype)
    is_numeric = is_numeric_dtype(series)
    is_integer_like = is_integer_like_numeric(series) if is_numeric else False

    date_parse_success_rate = None

    issue_flags: list[str] = []

    if missing_percentage >= 0.40:
        issue_flags.append("high_missingness")

    if non_missing_count == 0:
        inferred_feature_type = "all_missing"
        recommended_preprocessing_group = "problem_column"
        issue_flags.append("all_missing")

    elif unique_non_missing_count == 1:
        inferred_feature_type = "constant"
        recommended_preprocessing_group = "problem_column"
        issue_flags.append("constant_non_missing_value")

    elif unique_non_missing_count == 2:
        inferred_feature_type = "binary_or_two_value"
        recommended_preprocessing_group = "binary"

    elif is_numeric:
        if (
            is_integer_like
            and (
                unique_non_missing_count <= discrete_unique_threshold
                or unique_non_missing_ratio <= discrete_unique_ratio_threshold
            )
        ):
            inferred_feature_type = "numeric_discrete_or_count"
            recommended_preprocessing_group = "numeric_discrete_or_count"
        else:
            inferred_feature_type = "numeric_continuous"
            recommended_preprocessing_group = "numeric_continuous"

    else:
        date_parse_success_rate = calculate_date_parse_success_rate(series)

        if (
            date_parse_success_rate is not None
            and date_parse_success_rate >= date_parse_success_threshold
        ):
            inferred_feature_type = "date_like"
            recommended_preprocessing_group = "date_like"

        elif unique_non_missing_count <= low_cardinality_threshold:
            inferred_feature_type = "categorical_low_cardinality"
            recommended_preprocessing_group = "categorical_low_cardinality"

        else:
            inferred_feature_type = "categorical_high_cardinality_or_text"
            recommended_preprocessing_group = "categorical_high_cardinality_or_text"
            issue_flags.append("high_cardinality_or_text")

    needs_manual_review = recommended_preprocessing_group == "problem_column" or bool(
        issue_flags
    )

    return {
        "dtype": dtype,
        "total_count": total_count,
        "non_missing_count": non_missing_count,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
        "unique_non_missing_count": unique_non_missing_count,
        "unique_non_missing_ratio": unique_non_missing_ratio,
        "is_numeric": is_numeric,
        "is_integer_like": is_integer_like,
        "date_parse_success_rate": date_parse_success_rate,
        "sample_values": get_sample_values(series),
        "inferred_feature_type": inferred_feature_type,
        "recommended_preprocessing_group": recommended_preprocessing_group,
        "needs_manual_review": needs_manual_review,
        "issue_flags": ", ".join(issue_flags),
    }

def create_feature_type_profile(
    modeling_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a feature type profile for approved modeling columns.
    """
    profile_rows = []

    for column_name in modeling_feature_df.columns:
        feature_profile = infer_feature_type(modeling_feature_df[column_name])
        feature_profile["column_name"] = column_name
        profile_rows.append(feature_profile)

    profile_df = pd.DataFrame(profile_rows)

    ordered_columns = [
        "column_name",
        "inferred_feature_type",
        "recommended_preprocessing_group",
        "needs_manual_review",
        "issue_flags",
        "dtype",
        "total_count",
        "non_missing_count",
        "missing_count",
        "missing_percentage",
        "unique_non_missing_count",
        "unique_non_missing_ratio",
        "is_numeric",
        "is_integer_like",
        "date_parse_success_rate",
        "sample_values",
    ]

    profile_df = profile_df.loc[:, ordered_columns]

    profile_df = profile_df.sort_values(
        by=[
            "needs_manual_review",
            "recommended_preprocessing_group",
            "column_name",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return profile_df

def create_missingness_profile(
    feature_type_profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a missingness-focused report from the feature type profile.
    """
    missingness_columns = [
        "column_name",
        "missing_count",
        "missing_percentage",
        "non_missing_count",
        "inferred_feature_type",
        "recommended_preprocessing_group",
        "needs_manual_review",
        "issue_flags",
    ]

    missingness_df = feature_type_profile_df.loc[:, missingness_columns].copy()

    missingness_df = missingness_df.sort_values(
        by=["missing_percentage", "column_name"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return missingness_df

def save_dataframe_to_csv(df: pd.DataFrame, path: Path | str) -> None:
    """
    Save a DataFrame to CSV, creating the parent directory if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

def create_feature_type_inference_reports(
    modeling_base_dataset_path: Path | str = DEFAULT_MODELING_BASE_DATASET_PATH,
    column_usage_inventory_path: Path | str = DEFAULT_COLUMN_USAGE_INVENTORY_PATH,
    feature_type_profile_report_path: Path | str = DEFAULT_FEATURE_TYPE_PROFILE_REPORT_PATH,
    missingness_profile_report_path: Path | str = DEFAULT_MISSINGNESS_PROFILE_REPORT_PATH,
) -> pd.DataFrame:
    """
    Main workflow for inferring feature types from approved modeling columns.
    """
    print("\nLoading column usage inventory...")
    column_usage_df = load_column_usage_inventory(column_usage_inventory_path)

    modeling_columns = get_modeling_columns(column_usage_df)
    print(f"Approved modeling columns: {len(modeling_columns):,}")

    print("\nReading modeling base dataset header...")
    available_columns = get_csv_columns(modeling_base_dataset_path)
    print(f"Available modeling base columns: {len(available_columns):,}")

    print("\nValidating approved modeling columns...")
    validate_modeling_columns_available(
        available_columns=available_columns,
        modeling_columns=modeling_columns,
    )
    print("Modeling column validation passed.")

    print("\nLoading approved modeling feature columns...")
    modeling_feature_df = load_modeling_feature_dataset(
        path=modeling_base_dataset_path,
        modeling_columns=modeling_columns,
    )

    print(f"Rows loaded: {len(modeling_feature_df):,}")
    print(f"Feature columns loaded: {modeling_feature_df.shape[1]:,}")

    print("\nCreating feature type profile...")
    feature_type_profile_df = create_feature_type_profile(modeling_feature_df)

    print("\nFeature type summary:")
    print(
        feature_type_profile_df["inferred_feature_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    manual_review_count = feature_type_profile_df["needs_manual_review"].sum()
    print(f"\nColumns needing manual review: {manual_review_count:,}")

    print("\nCreating missingness profile...")
    missingness_profile_df = create_missingness_profile(feature_type_profile_df)

    print("\nSaving reports...")
    save_dataframe_to_csv(
        df=feature_type_profile_df,
        path=feature_type_profile_report_path,
    )
    print(f"Saved feature type profile to: {feature_type_profile_report_path}")

    save_dataframe_to_csv(
        df=missingness_profile_df,
        path=missingness_profile_report_path,
    )
    print(f"Saved missingness profile to: {missingness_profile_report_path}")

    print("\nFeature type inference complete.")

    return feature_type_profile_df

if __name__ == "__main__":
    create_feature_type_inference_reports()