from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
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
LOGISTIC_COEFFICIENT_PATH = (
    REPORTS_DIR / "logistic_regression_coefficients.csv"
)

TARGET_COLUMN = "bad_loan"
RANDOM_STATE = 42

def log_step(message: str) -> None:
    """Print a timestamped progress message to the terminal."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)


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


def calculate_probability_metrics(
    y_true: pd.Series,
    bad_loan_probabilities: np.ndarray,
    model_name: str,
    dataset_name: str,
) -> dict[str, str | int | float]:
    """Calculate probability-based classification metrics."""
    log_step(
        f"Calculating {dataset_name} probability metrics for "
        f"{model_name}"
    )

    if len(y_true) != len(bad_loan_probabilities):
        raise ValueError(
            f"{model_name} received {len(y_true):,} observed outcomes "
            f"but {len(bad_loan_probabilities):,} probabilities."
        )

    if y_true.nunique() != 2:
        raise ValueError(
            f"{dataset_name.capitalize()} target must contain both "
            "class 0 and class 1 to calculate evaluation metrics."
        )

    metrics = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "row_count": len(y_true),
        "observed_bad_loan_rate": y_true.mean(),
        "mean_predicted_bad_loan_probability": (
            bad_loan_probabilities.mean()
        ),
        "roc_auc": roc_auc_score(
            y_true,
            bad_loan_probabilities,
        ),
        "average_precision": average_precision_score(
            y_true,
            bad_loan_probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            bad_loan_probabilities,
            labels=[0, 1],
        ),
        "brier_score": brier_score_loss(
            y_true,
            bad_loan_probabilities,
        ),
    }

    log_step(
        f"Calculated {dataset_name} probability metrics for "
        f"{model_name}: ROC AUC={metrics['roc_auc']:.4f}, "
        f"average precision={metrics['average_precision']:.4f}"
    )

    return metrics



def inspect_calibration_performance(
    y_true: pd.Series,
    bad_loan_probabilities: np.ndarray,
    model_name: str,
    dataset_name: str,
    number_of_bins: int = 10,
) -> pd.DataFrame:
    """Summarize predicted and observed bad-loan rates by probability bin."""
    log_step(
        f"Inspecting {dataset_name} calibration for {model_name}"
    )

    if len(y_true) != len(bad_loan_probabilities):
        raise ValueError(
            f"{model_name} received {len(y_true):,} observed outcomes "
            f"but {len(bad_loan_probabilities):,} probabilities."
        )

    if number_of_bins < 2:
        raise ValueError(
            "number_of_bins must be at least 2."
        )

    if not np.isfinite(bad_loan_probabilities).all():
        raise ValueError(
            f"{model_name} produced non-finite probabilities."
        )

    if (
        (bad_loan_probabilities < 0).any()
        or (bad_loan_probabilities > 1).any()
    ):
        raise ValueError(
            f"{model_name} produced probabilities outside [0, 1]."
        )

    calibration_data = pd.DataFrame(
        {
            "actual_bad_loan": y_true.to_numpy(),
            "predicted_bad_loan_probability": (
                bad_loan_probabilities
            ),
        }
    )

    probability_bin_edges = np.linspace(
        0,
        1,
        number_of_bins + 1,
    )

    calibration_data["probability_bin"] = pd.cut(
        calibration_data["predicted_bad_loan_probability"],
        bins=probability_bin_edges,
        include_lowest=True,
    )

    calibration_summary = (
        calibration_data.groupby(
            "probability_bin",
            observed=True,
        )
        .agg(
            row_count=("actual_bad_loan", "size"),
            mean_predicted_bad_loan_probability=(
                "predicted_bad_loan_probability",
                "mean",
            ),
            observed_bad_loan_rate=(
                "actual_bad_loan",
                "mean",
            ),
        )
        .reset_index()
    )

    calibration_summary["calibration_gap"] = (
        calibration_summary["observed_bad_loan_rate"]
        - calibration_summary[
            "mean_predicted_bad_loan_probability"
        ]
    )

    calibration_summary.insert(
        0,
        "dataset_name",
        dataset_name,
    )
    calibration_summary.insert(
        0,
        "model_name",
        model_name,
    )

    log_step(
        f"Created {len(calibration_summary):,} calibration bins "
        f"for {model_name} on {dataset_name}"
    )

    return calibration_summary

def calculate_threshold_metrics(
    y_true: pd.Series,
    bad_loan_probabilities: np.ndarray,
    model_name: str,
    dataset_name: str,
    thresholds: tuple[float, ...] = (
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ),
) -> pd.DataFrame:
    """Calculate classification metrics across probability thresholds."""
    log_step(
        f"Calculating {dataset_name} threshold metrics for "
        f"{model_name}"
    )

    if len(y_true) != len(bad_loan_probabilities):
        raise ValueError(
            f"{model_name} received {len(y_true):,} observed outcomes "
            f"but {len(bad_loan_probabilities):,} probabilities."
        )

    if not np.isfinite(bad_loan_probabilities).all():
        raise ValueError(
            f"{model_name} produced non-finite probabilities."
        )

    if (
        (bad_loan_probabilities < 0).any()
        or (bad_loan_probabilities > 1).any()
    ):
        raise ValueError(
            f"{model_name} produced probabilities outside [0, 1]."
        )

    threshold_values = np.asarray(
        thresholds,
        dtype=float,
    )

    if threshold_values.size == 0:
        raise ValueError(
            "At least one classification threshold is required."
        )

    if not np.isfinite(threshold_values).all():
        raise ValueError(
            "Classification thresholds must contain only finite values."
        )

    if (
        (threshold_values < 0).any()
        or (threshold_values > 1).any()
    ):
        raise ValueError(
            "Classification thresholds must fall within [0, 1]."
        )

    threshold_rows = []

    for threshold in threshold_values:
        predicted_bad_loans = (
            bad_loan_probabilities >= threshold
        ).astype("int8")

        (
            true_negative,
            false_positive,
            false_negative,
            true_positive,
        ) = confusion_matrix(
            y_true,
            predicted_bad_loans,
            labels=[0, 1],
        ).ravel()

        negative_count = true_negative + false_positive

        false_positive_rate = (
            false_positive / negative_count
            if negative_count > 0
            else 0.0
        )

        threshold_rows.append(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "threshold": threshold,
                "row_count": len(y_true),
                "observed_bad_loan_rate": y_true.mean(),
                "predicted_bad_loan_rate": (
                    predicted_bad_loans.mean()
                ),
                "true_negative": int(true_negative),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_positive": int(true_positive),
                "precision": precision_score(
                    y_true,
                    predicted_bad_loans,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predicted_bad_loans,
                    zero_division=0,
                ),
                "false_positive_rate": false_positive_rate,
            }
        )

    threshold_summary = pd.DataFrame(threshold_rows)

    log_step(
        f"Created {len(threshold_summary):,} threshold rows "
        f"for {model_name} on {dataset_name}"
    )

    return threshold_summary


def create_score_decile_summary(
    y_true: pd.Series,
    bad_loan_probabilities: np.ndarray,
    model_name: str,
    dataset_name: str,
    number_of_deciles: int = 10,
) -> pd.DataFrame:
    """Summarize model performance across predicted-risk deciles."""
    log_step(
        f"Creating {dataset_name} score-decile summary for "
        f"{model_name}"
    )

    if len(y_true) != len(bad_loan_probabilities):
        raise ValueError(
            f"{model_name} received {len(y_true):,} observed outcomes "
            f"but {len(bad_loan_probabilities):,} probabilities."
        )

    if number_of_deciles < 2:
        raise ValueError(
            "number_of_deciles must be at least 2."
        )

    if len(y_true) < number_of_deciles:
        raise ValueError(
            f"Cannot create {number_of_deciles} score groups from only "
            f"{len(y_true):,} rows."
        )

    if not np.isfinite(bad_loan_probabilities).all():
        raise ValueError(
            f"{model_name} produced non-finite probabilities."
        )

    if (
        (bad_loan_probabilities < 0).any()
        or (bad_loan_probabilities > 1).any()
    ):
        raise ValueError(
            f"{model_name} produced probabilities outside [0, 1]."
        )

    score_data = pd.DataFrame(
        {
            "actual_bad_loan": y_true.to_numpy(),
            "predicted_bad_loan_probability": (
                bad_loan_probabilities
            ),
        }
    )

    score_data = score_data.sort_values(
        "predicted_bad_loan_probability",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    score_data["score_decile"] = pd.qcut(
        score_data.index,
        q=number_of_deciles,
        labels=range(1, number_of_deciles + 1),
    )

    decile_summary = (
        score_data.groupby(
            "score_decile",
            observed=True,
        )
        .agg(
            row_count=("actual_bad_loan", "size"),
            bad_loan_count=("actual_bad_loan", "sum"),
            minimum_predicted_bad_loan_probability=(
                "predicted_bad_loan_probability",
                "min",
            ),
            mean_predicted_bad_loan_probability=(
                "predicted_bad_loan_probability",
                "mean",
            ),
            maximum_predicted_bad_loan_probability=(
                "predicted_bad_loan_probability",
                "max",
            ),
            observed_bad_loan_rate=(
                "actual_bad_loan",
                "mean",
            ),
        )
        .reset_index()
    )

    total_bad_loans = decile_summary["bad_loan_count"].sum()

    if total_bad_loans == 0:
        raise ValueError(
            f"{dataset_name.capitalize()} target contains no bad loans."
        )

    decile_summary["share_of_all_bad_loans"] = (
        decile_summary["bad_loan_count"]
        / total_bad_loans
    )

    decile_summary["cumulative_bad_loan_share"] = (
        decile_summary["share_of_all_bad_loans"].cumsum()
    )

    decile_summary.insert(
        0,
        "dataset_name",
        dataset_name,
    )
    decile_summary.insert(
        0,
        "model_name",
        model_name,
    )

    log_step(
        f"Created {len(decile_summary):,} score groups for "
        f"{model_name} on {dataset_name}"
    )

    return decile_summary


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