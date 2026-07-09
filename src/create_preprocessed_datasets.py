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
    "binary_or_two_value",
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

def split_features_and_target(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataset into feature matrix X and target vector y.
    """
    required_columns = feature_columns + [target_column]

    validate_split_dataset_columns(
        df=df,
        required_columns=required_columns,
        dataset_name="split",
    )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    return X, y

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

def get_preprocessed_feature_names(
    preprocessing_pipeline: ColumnTransformer,
) -> list[str]:
    """
    Get output feature names from a fitted preprocessing pipeline.
    """
    return preprocessing_pipeline.get_feature_names_out().tolist()

def save_sparse_matrix(
    matrix: sparse.spmatrix,
    path: Path | str,
) -> None:
    """
    Save a sparse matrix to disk.
    """
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(path, matrix)


def save_target_vector(
    target: pd.Series,
    path: Path | str,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> None:
    """
    Save a target vector to disk.
    """
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    target_df = target.to_frame(name=target_column)

    target_df.to_csv(path, index=False)


def save_feature_names(
    feature_names: list[str],
    path: Path | str,
) -> None:
    """
    Save preprocessed feature names to disk.
    """
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    feature_names_df = pd.DataFrame(
        {
            "feature_name": feature_names,
        }
    )

    feature_names_df.to_csv(path, index=False)

def save_preprocessing_pipeline(
    preprocessing_pipeline: ColumnTransformer,
    path: Path | str,
) -> None:
    """
    Save the fitted preprocessing pipeline to disk.
    """
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessing_pipeline, path)

def create_preprocessing_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train_processed: sparse.spmatrix,
    X_validation_processed: sparse.spmatrix,
    X_test_processed: sparse.spmatrix,
    feature_groups: dict[str, list[str]],
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Create a summary report for preprocessing outputs.
    """
    summary_rows = [
        {
            "metric": "train_input_rows",
            "value": len(train_df),
        },
        {
            "metric": "validation_input_rows",
            "value": len(validation_df),
        },
        {
            "metric": "test_input_rows",
            "value": len(test_df),
        },
        {
            "metric": "train_preprocessed_rows",
            "value": X_train_processed.shape[0],
        },
        {
            "metric": "validation_preprocessed_rows",
            "value": X_validation_processed.shape[0],
        },
        {
            "metric": "test_preprocessed_rows",
            "value": X_test_processed.shape[0],
        },
        {
            "metric": "preprocessed_feature_count",
            "value": len(feature_names),
        },
        {
            "metric": "numeric_feature_count",
            "value": len(feature_groups["numeric_features"]),
        },
        {
            "metric": "categorical_feature_count",
            "value": len(feature_groups["categorical_features"]),
        },
        {
            "metric": "excluded_feature_count",
            "value": len(feature_groups["excluded_features"]),
        },
        {
            "metric": "train_nonzero_values",
            "value": X_train_processed.nnz,
        },
        {
            "metric": "validation_nonzero_values",
            "value": X_validation_processed.nnz,
        },
        {
            "metric": "test_nonzero_values",
            "value": X_test_processed.nnz,
        },
    ]

    return pd.DataFrame(summary_rows)

def save_dataframe_to_csv(
        df: pd.DataFrame, 
        path: Path | str, 
) -> None: 
    """
    Save a DataFrame to disk as a CSV file. 
    """
    path = Path(path)
    
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

def log_step(message: str) -> None:
    """
    Print a progress message for long-running script steps.
    """
    print(f"[create_preprocessed_datasets] {message}", flush=True)


def create_preprocessed_datasets(
    train_dataset_path: Path | str = DEFAULT_TRAIN_DATASET_PATH,
    validation_dataset_path: Path | str = DEFAULT_VALIDATION_DATASET_PATH,
    test_dataset_path: Path | str = DEFAULT_TEST_DATASET_PATH,
    column_usage_inventory_path: Path | str = DEFAULT_COLUMN_USAGE_INVENTORY_PATH,
    feature_type_profile_path: Path | str = DEFAULT_FEATURE_TYPE_PROFILE_PATH,
    preprocessed_train_features_path: Path | str = DEFAULT_PREPROCESSED_TRAIN_FEATURES_PATH,
    preprocessed_validation_features_path: Path | str = DEFAULT_PREPROCESSED_VALIDATION_FEATURES_PATH,
    preprocessed_test_features_path: Path | str = DEFAULT_PREPROCESSED_TEST_FEATURES_PATH,
    train_target_path: Path | str = DEFAULT_TRAIN_TARGET_PATH,
    validation_target_path: Path | str = DEFAULT_VALIDATION_TARGET_PATH,
    test_target_path: Path | str = DEFAULT_TEST_TARGET_PATH,
    preprocessing_pipeline_path: Path | str = DEFAULT_PREPROCESSING_PIPELINE_PATH,
    preprocessed_feature_names_path: Path | str = DEFAULT_PREPROCESSED_FEATURE_NAMES_PATH,
    preprocessing_summary_report_path: Path | str = DEFAULT_PREPROCESSING_SUMMARY_REPORT_PATH,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> None:
    """
    Create preprocessed train, validation, and test datasets.
    """
    log_step("Loading train, validation, and test split datasets.")
    train_df = load_split_dataset(train_dataset_path)
    validation_df = load_split_dataset(validation_dataset_path)
    test_df = load_split_dataset(test_dataset_path)

    log_step("Loading column usage inventory and feature type profile.")
    column_usage_df = load_column_usage_inventory(column_usage_inventory_path)
    feature_type_profile_df = load_feature_type_profile(feature_type_profile_path)

    log_step("Identifying approved modeling columns and baseline feature groups.")
    modeling_columns = get_modeling_columns(column_usage_df)

    feature_groups = get_baseline_feature_groups(
        feature_type_profile_df=feature_type_profile_df,
        modeling_columns=modeling_columns,
    )

    baseline_modeling_features = get_baseline_modeling_features(feature_groups)

    log_step("Splitting train, validation, and test into X features and y target.")
    X_train, y_train = split_features_and_target(
        df=train_df,
        feature_columns=baseline_modeling_features,
        target_column=target_column,
    )

    X_validation, y_validation = split_features_and_target(
        df=validation_df,
        feature_columns=baseline_modeling_features,
        target_column=target_column,
    )

    X_test, y_test = split_features_and_target(
        df=test_df,
        feature_columns=baseline_modeling_features,
        target_column=target_column,
    )

    log_step("Building preprocessing pipeline.")
    preprocessing_pipeline = build_preprocessing_pipeline(
        numeric_features=feature_groups["numeric_features"],
        categorical_features=feature_groups["categorical_features"],
    )

    log_step("Fitting preprocessing pipeline on train and transforming all splits.")
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_validation_processed = preprocessing_pipeline.transform(X_validation)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    log_step("Converting preprocessed outputs to CSR sparse matrices.")
    X_train_processed = sparse.csr_matrix(X_train_processed)
    X_validation_processed = sparse.csr_matrix(X_validation_processed)
    X_test_processed = sparse.csr_matrix(X_test_processed)

    log_step("Extracting final preprocessed feature names.")
    feature_names = get_preprocessed_feature_names(preprocessing_pipeline)

    log_step("Saving preprocessed sparse feature matrices.")
    save_sparse_matrix(
        matrix=X_train_processed,
        path=preprocessed_train_features_path,
    )

    save_sparse_matrix(
        matrix=X_validation_processed,
        path=preprocessed_validation_features_path,
    )

    save_sparse_matrix(
        matrix=X_test_processed,
        path=preprocessed_test_features_path,
    )

    log_step("Saving target vectors.")
    save_target_vector(
        target=y_train,
        path=train_target_path,
        target_column=target_column,
    )

    save_target_vector(
        target=y_validation,
        path=validation_target_path,
        target_column=target_column,
    )

    save_target_vector(
        target=y_test,
        path=test_target_path,
        target_column=target_column,
    )

    log_step("Saving feature names and fitted preprocessing pipeline.")
    save_feature_names(
        feature_names=feature_names,
        path=preprocessed_feature_names_path,
    )

    save_preprocessing_pipeline(
        preprocessing_pipeline=preprocessing_pipeline,
        path=preprocessing_pipeline_path,
    )

    log_step("Creating and saving preprocessing summary report.")
    preprocessing_summary_df = create_preprocessing_summary(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        X_train_processed=X_train_processed,
        X_validation_processed=X_validation_processed,
        X_test_processed=X_test_processed,
        feature_groups=feature_groups,
        feature_names=feature_names,
    )

    save_dataframe_to_csv(
        df=preprocessing_summary_df,
        path=preprocessing_summary_report_path,
    )

    log_step("Preprocessing workflow completed successfully.")

if __name__ == "__main__": 
    create_preprocessed_datasets()