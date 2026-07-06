# src/create_preprocessed_datasets.py

from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from column_usage import (
    DEFAULT_COLUMN_USAGE_INVENTORY_PATH,
    get_modeling_columns,
    load_column_usage_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRAIN_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "modeling_train.csv.gz"
)

DEFAULT_VALIDATION_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "modeling_validation.csv.gz"
)

DEFAULT_TEST_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "modeling_test.csv.gz"
)

DEFAULT_FEATURE_TYPE_PROFILE_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "modeling_feature_type_profile.csv"
)

DEFAULT_PREPROCESSED_TRAIN_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "preprocessed_train_features.npz"
)

DEFAULT_PREPROCESSED_VALIDATION_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "preprocessed_validation_features.npz"
)

DEFAULT_PREPROCESSED_TEST_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "preprocessed_test_features.npz"
)

DEFAULT_TRAIN_TARGET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "preprocessed_train_target.csv.gz"
)

DEFAULT_VALIDATION_TARGET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "preprocessed_validation_target.csv.gz"
)

DEFAULT_TEST_TARGET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "preprocessed_test_target.csv.gz"
)

DEFAULT_PREPROCESSING_PIPELINE_PATH = (
    PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
)

DEFAULT_PREPROCESSED_FEATURE_NAMES_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "preprocessed_feature_names.csv"
)

DEFAULT_PREPROCESSING_SUMMARY_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "preprocessing_summary.csv"
)

DEFAULT_TARGET_COLUMN = "bad_loan"

NUMERIC_FEATURE_TYPES = [
    "numeric_continuous",
    "numeric_discrete_or_count",
]

CATEGORICAL_FEATURE_TYPES = [
    "binary",
    "categorical_low_cardinality",
]

EXCLUDED_BASELINE_FEATURE_TYPES = [
    "categorical_high_cardinality_or_text",
    "date_like",
    "problem_column",
]

FEATURE_NAME_COLUMN = "column_name"
FEATURE_TYPE_COLUMN = "inferred_feature_type"

def load_split_dataset(path: Path | str) -> pd.DataFrame:
    """
    Load one modeling split dataset from disk.
    """
    path = Path(path)

    return pd.read_csv(path, low_memory=False)


def load_feature_type_profile(path: Path | str) -> pd.DataFrame:
    """
    Load the modeling feature type profile report.
    """
    path = Path(path)

    return pd.read_csv(path)


def validate_feature_type_profile(
    feature_type_profile_df: pd.DataFrame,
    feature_name_column: str = FEATURE_NAME_COLUMN,
    feature_type_column: str = FEATURE_TYPE_COLUMN,
) -> None:
    """
    Validate that the feature type profile has the required columns.
    """
    required_columns = [feature_name_column, feature_type_column]

    missing_columns = [
        column
        for column in required_columns
        if column not in feature_type_profile_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The feature type profile is missing required columns: "
            f"{missing_columns}"
        )
    
def get_baseline_feature_groups(
    feature_type_profile_df: pd.DataFrame,
    modeling_columns: list[str],
    feature_name_column: str = FEATURE_NAME_COLUMN,
    feature_type_column: str = FEATURE_TYPE_COLUMN,
) -> dict[str, list[str]]:
    """
    Group approved modeling features into baseline preprocessing groups.
    """
    validate_feature_type_profile(
        feature_type_profile_df=feature_type_profile_df,
        feature_name_column=feature_name_column,
        feature_type_column=feature_type_column,
    )

    profile_df = feature_type_profile_df.copy()

    duplicate_features = (
        profile_df[profile_df[feature_name_column].duplicated()][feature_name_column]
        .unique()
        .tolist()
    )

    if duplicate_features:
        raise ValueError(
            "The feature type profile contains duplicate feature rows: "
            f"{duplicate_features}"
        )

    profile_feature_set = set(profile_df[feature_name_column])

    missing_profile_features = [
        column for column in modeling_columns if column not in profile_feature_set
    ]

    if missing_profile_features:
        raise ValueError(
            "Some approved modeling columns are missing from the feature type profile: "
            f"{missing_profile_features}"
        )

    profile_df = profile_df[
        profile_df[feature_name_column].isin(modeling_columns)
    ].copy()

    recognized_feature_types = (
        NUMERIC_FEATURE_TYPES
        + CATEGORICAL_FEATURE_TYPES
        + EXCLUDED_BASELINE_FEATURE_TYPES
    )

    unexpected_feature_types = sorted(
        set(profile_df[feature_type_column]) - set(recognized_feature_types)
    )

    if unexpected_feature_types:
        raise ValueError(
            "The feature type profile contains unexpected feature types: "
            f"{unexpected_feature_types}"
        )

    numeric_features = profile_df.loc[
        profile_df[feature_type_column].isin(NUMERIC_FEATURE_TYPES),
        feature_name_column,
    ].tolist()

    categorical_features = profile_df.loc[
        profile_df[feature_type_column].isin(CATEGORICAL_FEATURE_TYPES),
        feature_name_column,
    ].tolist()

    excluded_features = profile_df.loc[
        profile_df[feature_type_column].isin(EXCLUDED_BASELINE_FEATURE_TYPES),
        feature_name_column,
    ].tolist()

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_features": excluded_features,
    }


def get_baseline_modeling_features(
    feature_groups: dict[str, list[str]],
) -> list[str]:
    """
    Get the features included in the baseline preprocessing pipeline.
    """
    return (
        feature_groups["numeric_features"]
        + feature_groups["categorical_features"]
    )

def validate_split_dataset_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """
    Validate that a split dataset contains all required columns.
    """
    available_columns = set(df.columns)

    missing_columns = [
        column for column in required_columns if column not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            f"The {dataset_name} dataset is missing required columns: "
            f"{missing_columns}"
        )


def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build the baseline preprocessing pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Missing"),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )

    return preprocessing_pipeline