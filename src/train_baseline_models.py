from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from model_evaluation import (
    calculate_probability_metrics,
    calculate_threshold_metrics,
    create_score_decile_summary,
    inspect_calibration_performance,
    log_step,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"

TRAIN_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "preprocessed_train_features.npz"
)
VALIDATION_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "preprocessed_validation_features.npz"
)

TRAIN_TARGET_PATH = (
    PROCESSED_DATA_DIR / "preprocessed_train_target.csv.gz"
)
VALIDATION_TARGET_PATH = (
    PROCESSED_DATA_DIR / "preprocessed_validation_target.csv.gz"
)

FEATURE_NAMES_PATH = (
    REPORTS_DIR / "preprocessed_feature_names.csv"
)

DUMMY_MODEL_PATH = (
    MODELS_DIR / "dummy_prior_baseline.joblib"
)
LOGISTIC_MODEL_PATH = (
    MODELS_DIR / "logistic_regression_baseline.joblib"
)

VALIDATION_PERFORMANCE_PATH = (
    REPORTS_DIR / "validation_model_performance.csv"
)
VALIDATION_THRESHOLD_PATH = (
    REPORTS_DIR / "validation_threshold_metrics.csv"
)
VALIDATION_DECILE_PATH = (
    REPORTS_DIR / "validation_score_deciles.csv"
)
VALIDATION_CALIBRATION_PATH = (
    REPORTS_DIR / "validation_calibration_summary.csv"
)
LOGISTIC_COEFFICIENT_PATH = (
    REPORTS_DIR / "logistic_regression_coefficients.csv"
)

TARGET_COLUMN = "bad_loan"
RANDOM_STATE = 42


def load_sparse_features(
    file_path: Path,
    dataset_name: str,
) -> sparse.csr_matrix:
    """Load a preprocessed sparse feature matrix from an NPZ file."""
    log_step(f"Loading {dataset_name} feature matrix")

    if not file_path.exists():
        raise FileNotFoundError(
            f"{dataset_name.capitalize()} feature file was not found: "
            f"{file_path}"
        )

    feature_matrix = sparse.load_npz(file_path).tocsr()

    if feature_matrix.shape[0] == 0:
        raise ValueError(
            f"{dataset_name.capitalize()} feature matrix contains no rows."
        )

    if feature_matrix.shape[1] == 0:
        raise ValueError(
            f"{dataset_name.capitalize()} feature matrix contains no features."
        )

    log_step(
        f"Loaded {dataset_name} features with shape "
        f"{feature_matrix.shape}"
    )

    return feature_matrix


def load_target(
    file_path: Path,
    dataset_name: str,
) -> pd.Series:
    """Load and validate a preprocessed target dataset."""
    log_step(f"Loading {dataset_name} target")

    if not file_path.exists():
        raise FileNotFoundError(
            f"{dataset_name.capitalize()} target file was not found: "
            f"{file_path}"
        )

    target_data = pd.read_csv(file_path)

    if TARGET_COLUMN not in target_data.columns:
        raise ValueError(
            f"{dataset_name.capitalize()} target file does not contain "
            f"the expected '{TARGET_COLUMN}' column."
        )

    target = target_data[TARGET_COLUMN]

    if target.empty:
        raise ValueError(
            f"{dataset_name.capitalize()} target contains no rows."
        )

    if target.isna().any():
        raise ValueError(
            f"{dataset_name.capitalize()} target contains missing values."
        )

    unexpected_values = set(target.unique()) - {0, 1}

    if unexpected_values:
        raise ValueError(
            f"{dataset_name.capitalize()} target contains unexpected "
            f"values: {sorted(unexpected_values)}"
        )

    target = target.astype("int8")

    log_step(
        f"Loaded {dataset_name} target with {len(target):,} rows "
        f"and bad-loan rate {target.mean():.4f}"
    )

    return target

def load_feature_names(file_path: Path) -> pd.Series:
    """Load and validate the post-preprocessing feature names."""
    log_step("Loading preprocessed feature names")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature-name file was not found: {file_path}"
        )

    feature_name_data = pd.read_csv(file_path)

    expected_column = "feature_name"

    if expected_column not in feature_name_data.columns:
        raise ValueError(
            f"Feature-name file does not contain the expected "
            f"'{expected_column}' column."
        )

    feature_names = feature_name_data[expected_column]

    if feature_names.empty:
        raise ValueError(
            "Feature-name file contains no feature names."
        )

    if feature_names.isna().any():
        raise ValueError(
            "Feature-name column contains missing values."
        )

    if feature_names.duplicated().any():
        duplicate_names = (
            feature_names[feature_names.duplicated()]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Feature-name column contains duplicate names: "
            f"{duplicate_names[:10]}"
        )

    feature_names = feature_names.astype("string")

    log_step(
        f"Loaded {len(feature_names):,} preprocessed feature names"
    )

    return feature_names


def validate_dataset_alignment(
    X_train: sparse.csr_matrix,
    y_train: pd.Series,
    X_validation: sparse.csr_matrix,
    y_validation: pd.Series,
    feature_names: pd.Series,
) -> None:
    """Validate alignment across features, targets, and feature names."""
    log_step("Validating dataset alignment")

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            "Training feature and target row counts do not match: "
            f"{X_train.shape[0]:,} feature rows versus "
            f"{len(y_train):,} target rows."
        )

    if X_validation.shape[0] != len(y_validation):
        raise ValueError(
            "Validation feature and target row counts do not match: "
            f"{X_validation.shape[0]:,} feature rows versus "
            f"{len(y_validation):,} target rows."
        )

    if X_train.shape[1] != X_validation.shape[1]:
        raise ValueError(
            "Training and validation feature counts do not match: "
            f"{X_train.shape[1]:,} training features versus "
            f"{X_validation.shape[1]:,} validation features."
        )

    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            "Sparse-matrix feature count does not match the number of "
            "saved feature names: "
            f"{X_train.shape[1]:,} matrix columns versus "
            f"{len(feature_names):,} feature names."
        )

    if not np.isfinite(X_train.data).all():
        raise ValueError(
            "Training feature matrix contains infinite or NaN values."
        )

    if not np.isfinite(X_validation.data).all():
        raise ValueError(
            "Validation feature matrix contains infinite or NaN values."
        )

    log_step(
        "Dataset alignment validated successfully: "
        f"{X_train.shape[1]:,} shared features"
    )


def train_dummy_baseline(
    X_train: sparse.csr_matrix,
    y_train: pd.Series,
) -> DummyClassifier:
    """Train a prior-probability dummy classifier."""
    log_step("Training prior-probability dummy baseline")

    dummy_model = DummyClassifier(strategy="prior")

    dummy_model.fit(X_train, y_train)

    log_step(
        "Dummy baseline trained successfully with training bad-loan "
        f"rate {y_train.mean():.4f}"
    )

    return dummy_model


def train_logistic_baseline(
    X_train: sparse.csr_matrix,
    y_train: pd.Series,
) -> LogisticRegression:
    """Train the baseline logistic regression model."""
    log_step("Training logistic regression baseline")

    logistic_model = LogisticRegression(
        solver="saga",
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    logistic_model.fit(X_train, y_train)

    log_step(
        "Logistic regression baseline trained successfully in "
        f"{int(logistic_model.n_iter_[0]):,} iterations"
    )

    return logistic_model

def generate_bad_loan_probabilities(
    model: DummyClassifier | LogisticRegression,
    X: sparse.csr_matrix,
    model_name: str,
    dataset_name: str,
) -> np.ndarray:
    """Generate and validate predicted bad-loan probabilities."""
    log_step(
        f"Generating {dataset_name} probabilities with {model_name}"
    )

    if not hasattr(model, "classes_"):
        raise ValueError(
            f"{model_name} has not been fitted before prediction."
        )

    class_labels = np.asarray(model.classes_)
    bad_class_positions = np.flatnonzero(class_labels == 1)

    if len(bad_class_positions) != 1:
        raise ValueError(
            f"{model_name} does not contain exactly one class labelled 1. "
            f"Fitted classes: {class_labels.tolist()}"
        )

    probability_matrix = model.predict_proba(X)

    if probability_matrix.shape[0] != X.shape[0]:
        raise ValueError(
            f"{model_name} returned {probability_matrix.shape[0]:,} "
            f"probability rows for {X.shape[0]:,} input rows."
        )

    bad_class_position = bad_class_positions[0]

    bad_loan_probabilities = probability_matrix[
        :,
        bad_class_position,
    ]

    if not np.isfinite(bad_loan_probabilities).all():
        raise ValueError(
            f"{model_name} produced non-finite probabilities."
        )

    if (
        (bad_loan_probabilities < 0).any()
        or (bad_loan_probabilities > 1).any()
    ):
        raise ValueError(
            f"{model_name} produced probabilities outside the "
            "valid range from 0 to 1."
        )

    log_step(
        f"Generated {len(bad_loan_probabilities):,} {dataset_name} "
        f"probabilities with mean "
        f"{bad_loan_probabilities.mean():.4f}"
    )

    return bad_loan_probabilities


def create_logistic_coefficient_report(
    logistic_model: LogisticRegression,
    feature_names: list[str],
) -> pd.DataFrame:
    """Create a report linking logistic coefficients to feature names."""
    log_step("Creating logistic-regression coefficient report")

    if not hasattr(logistic_model, "coef_"):
        raise ValueError(
            "The logistic-regression model has not been fitted."
        )

    coefficients = np.asarray(
        logistic_model.coef_,
        dtype=float,
    ).ravel()

    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"Received {len(feature_names):,} feature names but "
            f"{len(coefficients):,} logistic coefficients."
        )

    if not np.isfinite(coefficients).all():
        raise ValueError(
            "Logistic-regression coefficients contain non-finite values."
        )

    coefficient_report = pd.DataFrame(
        {
            "feature_name": feature_names,
            "coefficient": coefficients,
        }
    )

    coefficient_report["absolute_coefficient"] = (
        coefficient_report["coefficient"].abs()
    )

    coefficient_report["coefficient_direction"] = np.select(
        [
            coefficient_report["coefficient"] > 0,
            coefficient_report["coefficient"] < 0,
        ],
        [
            "increases_bad_loan_risk",
            "decreases_bad_loan_risk",
        ],
        default="neutral",
    )

    coefficient_report = coefficient_report.sort_values(
        [
            "absolute_coefficient",
            "feature_name",
        ],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    coefficient_report.insert(
        0,
        "coefficient_rank",
        np.arange(1, len(coefficient_report) + 1),
    )

    log_step(
        f"Created coefficient report for "
        f"{len(coefficient_report):,} features"
    )

    return coefficient_report


def save_model_artifacts(
    dummy_model: DummyClassifier,
    logistic_model: LogisticRegression,
) -> None:
    """Save the fitted baseline models to disk."""
    log_step("Saving fitted baseline model artifacts")

    if not hasattr(dummy_model, "classes_"):
        raise ValueError(
            "The dummy baseline model has not been fitted."
        )

    if not hasattr(logistic_model, "coef_"):
        raise ValueError(
            "The logistic-regression baseline has not been fitted."
        )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        dummy_model,
        DUMMY_MODEL_PATH,
    )

    joblib.dump(
        logistic_model,
        LOGISTIC_MODEL_PATH,
    )

    if not DUMMY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Dummy baseline model was not saved: "
            f"{DUMMY_MODEL_PATH}"
        )

    if not LOGISTIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Logistic-regression baseline was not saved: "
            f"{LOGISTIC_MODEL_PATH}"
        )

    log_step(
        "Saved baseline model artifacts: "
        f"{DUMMY_MODEL_PATH.name} and "
        f"{LOGISTIC_MODEL_PATH.name}"
    )

def save_report_table(
    report: pd.DataFrame,
    output_path: Path,
    report_name: str,
) -> None:
    """Save a non-empty report table to a CSV file."""
    log_step(f"Saving {report_name}")

    if report.empty:
        raise ValueError(
            f"{report_name.capitalize()} contains no rows."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        output_path,
        index=False,
    )

    if not output_path.exists():
        raise FileNotFoundError(
            f"{report_name.capitalize()} was not saved: "
            f"{output_path}"
        )

    log_step(
        f"Saved {report_name} with {len(report):,} rows: "
        f"{output_path.name}"
    )

def main() -> None:
    """Run the baseline-model training and validation workflow."""
    log_step("Starting baseline-model training workflow")

    X_train = load_sparse_features(
        TRAIN_FEATURES_PATH,
        "training",
    )

    y_train = load_target(
        TRAIN_TARGET_PATH,
        "training",
    )

    X_validation = load_sparse_features(
        VALIDATION_FEATURES_PATH,
        "validation",
    )

    y_validation = load_target(
        VALIDATION_TARGET_PATH,
        "validation",
    )

    feature_names = load_feature_names(
        FEATURE_NAMES_PATH,
    )

    validate_dataset_alignment(
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_names,
    )

    dummy_model = train_dummy_baseline(
        X_train,
        y_train,
    )

    logistic_model = train_logistic_baseline(
        X_train,
        y_train,
    )

    dummy_validation_probabilities = (
        generate_bad_loan_probabilities(
            dummy_model,
            X_validation,
            "dummy_prior_baseline",
            "validation",
        )
    )

    logistic_validation_probabilities = (
        generate_bad_loan_probabilities(
            logistic_model,
            X_validation,
            "logistic_regression_baseline",
            "validation",
        )
    )

    validation_performance_report = pd.DataFrame(
        [
            calculate_probability_metrics(
                y_validation,
                dummy_validation_probabilities,
                "dummy_prior_baseline",
                "validation",
            ),
            calculate_probability_metrics(
                y_validation,
                logistic_validation_probabilities,
                "logistic_regression_baseline",
                "validation",
            ),
        ]
    )

    validation_calibration_report = pd.concat(
        [
            inspect_calibration_performance(
                y_validation,
                dummy_validation_probabilities,
                "dummy_prior_baseline",
                "validation",
            ),
            inspect_calibration_performance(
                y_validation,
                logistic_validation_probabilities,
                "logistic_regression_baseline",
                "validation",
            ),
        ],
        ignore_index=True,
    )

    validation_threshold_report = pd.concat(
        [
            calculate_threshold_metrics(
                y_validation,
                dummy_validation_probabilities,
                "dummy_prior_baseline",
                "validation",
            ),
            calculate_threshold_metrics(
                y_validation,
                logistic_validation_probabilities,
                "logistic_regression_baseline",
                "validation",
            ),
        ],
        ignore_index=True,
    )

    validation_decile_report = pd.concat(
        [
            create_score_decile_summary(
                y_validation,
                dummy_validation_probabilities,
                "dummy_prior_baseline",
                "validation",
            ),
            create_score_decile_summary(
                y_validation,
                logistic_validation_probabilities,
                "logistic_regression_baseline",
                "validation",
            ),
        ],
        ignore_index=True,
    )

    logistic_coefficient_report = (
        create_logistic_coefficient_report(
            logistic_model,
            feature_names.tolist(),
        )
    )

    save_model_artifacts(
        dummy_model,
        logistic_model,
    )

    save_report_table(
        validation_performance_report,
        VALIDATION_PERFORMANCE_PATH,
        "validation model performance report",
    )

    save_report_table(
        validation_calibration_report,
        VALIDATION_CALIBRATION_PATH,
        "validation calibration report",
    )

    save_report_table(
        validation_threshold_report,
        VALIDATION_THRESHOLD_PATH,
        "validation threshold report",
    )

    save_report_table(
        validation_decile_report,
        VALIDATION_DECILE_PATH,
        "validation score-decile report",
    )

    save_report_table(
        logistic_coefficient_report,
        LOGISTIC_COEFFICIENT_PATH,
        "logistic-regression coefficient report",
    )

    log_step(
        "Baseline-model training workflow completed successfully"
    )


if __name__ == "__main__":
    main()