from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from model_evaluation import (
    evaluate_probability_predictions,
    log_step,
)
from train_baseline_models import (
    generate_bad_loan_probabilities,
    load_sparse_features,
    load_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"

VALIDATION_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "preprocessed_validation_features.npz"
)
VALIDATION_TARGET_PATH = (
    PROCESSED_DATA_DIR / "preprocessed_validation_target.csv.gz"
)

LOGISTIC_BASELINE_MODEL_PATH = (
    MODELS_DIR / "logistic_regression_baseline.joblib"
)

CALIBRATION_FIT_FRACTION = 0.50
RANDOM_STATE = 42


def load_logistic_baseline_model(
        file_path: Path, 
) -> LogisticRegression:
    """Validate and load the fitted logistic-regression baseline."""
    log_step("Loading logistic-regression baseline model")

    if not file_path.exists():
        raise FileNotFoundError(

            "Logistic-regression baseline model was not found: "
            f"{file_path}"
        )
    logistic_model = joblib.load(file_path)

    if not isinstance(logistic_model, LogisticRegression): 
        raise TypeError(
            "Saved baseline artifact is not a LogisticRegression model. "
            f"Received: {type(logistic_model).__name__}"
        )
    
    if not hasattr(logistic_model, "classes_"):
        raise ValueError(
            "Loaded logistic-regression model has not been fitted."
        )
    
    if not hasattr(logistic_model, "coef_"):
        raise ValueError(
            "Loaded logistic-regression model has no fitted coefficients."
        )
    
    class_labels = np.asarray(logistic_model.classes_)

    if set(class_labels.tolist()) != {0, 1}: 
        raise ValueError(
            "Loaded logistic-regression model must contain classes "
            f" 0 and 1. Received: {class_labels.tolist()}"
        )
    
    log_step(
        "Loaded fitted logistic-regression baseline successfully"
    )

    return logistic_model

def create_chronological_calibration_split(
    X_validation: sparse.csr_matrix,
    y_validation: pd.Series,
    calibration_fit_fraction: float,
) -> tuple[
    sparse.csr_matrix,
    pd.Series,
    sparse.csr_matrix,
    pd.Series,
]:
    """Split validation data chronologically for calibration and comparison."""
    log_step("Creating chronological calibration-data split")

    if X_validation.shape[0] != len(y_validation):
        raise ValueError(
            "Validation feature and target row counts do not match: "
            f"{X_validation.shape[0]:,} feature rows versus "
            f"{len(y_validation):,} target rows."
        )

    if not 0 < calibration_fit_fraction < 1:
        raise ValueError(
            "calibration_fit_fraction must be greater than 0 "
            "and less than 1."
        )

    split_position = int(
        len(y_validation) * calibration_fit_fraction
    )

    if split_position == 0 or split_position == len(y_validation):
        raise ValueError(
            "The calibration split must leave at least one row "
            "in both portions."
        )

    X_calibration_fit = X_validation[:split_position]
    y_calibration_fit = (
        y_validation.iloc[:split_position]
        .reset_index(drop=True)
    )

    X_calibration_comparison = X_validation[split_position:]
    y_calibration_comparison = (
        y_validation.iloc[split_position:]
        .reset_index(drop=True)
    )

    if y_calibration_fit.nunique() != 2:
        raise ValueError(
            "Calibration-fitting target must contain both classes."
        )

    if y_calibration_comparison.nunique() != 2:
        raise ValueError(
            "Calibration-comparison target must contain both classes."
        )

    log_step(
        "Created chronological calibration split: "
        f"{len(y_calibration_fit):,} fitting rows and "
        f"{len(y_calibration_comparison):,} comparison rows"
    )

    return (
        X_calibration_fit,
        y_calibration_fit,
        X_calibration_comparison,
        y_calibration_comparison,
    )

def generate_logistic_scores(
    logistic_model: LogisticRegression,
    X: sparse.csr_matrix,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate validated logistic log-odds and bad-loan probabilities."""
    log_step(
        f"Generating baseline logistic scores for {dataset_name}"
    )

    bad_loan_probabilities = generate_bad_loan_probabilities(
        model=logistic_model,
        X=X,
        model_name="logistic_regression_baseline",
        dataset_name=dataset_name,
    )

    log_odds = np.asarray(
        logistic_model.decision_function(X),
        dtype=float,
    ).ravel()

    if len(log_odds) != X.shape[0]:
        raise ValueError(
            f"Logistic model returned {len(log_odds):,} log-odds "
            f"scores for {X.shape[0]:,} input rows."
        )

    if not np.isfinite(log_odds).all():
        raise ValueError(
            f"{dataset_name.capitalize()} log-odds contain "
            "non-finite values."
        )

    reconstructed_probabilities = expit(log_odds)

    if not np.allclose(
        reconstructed_probabilities,
        bad_loan_probabilities,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Probabilities reconstructed from the logistic log-odds "
            "do not match the model probabilities."
        )

    log_step(
        f"Generated {len(log_odds):,} baseline scores for "
        f"{dataset_name}"
    )

    return log_odds, bad_loan_probabilities

def fit_intercept_adjustment(
    calibration_log_odds: np.ndarray,
    y_calibration_fit: pd.Series,
    lower_bound: float = -10.0,
    upper_bound: float = 10.0,
) -> float:
    """Find the intercept shift that minimizes calibration log loss."""
    log_step("Fitting intercept-only calibration adjustment")

    calibration_log_odds = np.asarray(
        calibration_log_odds,
        dtype=float,
    ).ravel()

    calibration_targets = np.asarray(
        y_calibration_fit,
        dtype=float,
    ).ravel()

    if len(calibration_log_odds) != len(calibration_targets):
        raise ValueError(
            "Calibration log-odds and target row counts do not match: "
            f"{len(calibration_log_odds):,} scores versus "
            f"{len(calibration_targets):,} targets."
        )

    if not np.isfinite(calibration_log_odds).all():
        raise ValueError(
            "Calibration log-odds contain non-finite values."
        )

    if not np.isfinite(calibration_targets).all():
        raise ValueError(
            "Calibration targets contain non-finite values."
        )

    if set(np.unique(calibration_targets).tolist()) != {0.0, 1.0}:
        raise ValueError(
            "Calibration targets must contain both class 0 and class 1."
        )

    if not lower_bound < upper_bound:
        raise ValueError(
            "lower_bound must be smaller than upper_bound."
        )

    def calculate_intercept_log_loss(
        intercept_shift: float,
    ) -> float:
        calibrated_probabilities = expit(
            calibration_log_odds + intercept_shift
        )

        probability_floor = np.finfo(float).eps

        calibrated_probabilities = np.clip(
            calibrated_probabilities,
            probability_floor,
            1.0 - probability_floor,
        )

        return float(
            -np.mean(
                calibration_targets
                * np.log(calibrated_probabilities)
                + (1.0 - calibration_targets)
                * np.log1p(-calibrated_probabilities)
            )
        )

    optimization_result = minimize_scalar(
        calculate_intercept_log_loss,
        bounds=(lower_bound, upper_bound),
        method="bounded",
    )

    if not optimization_result.success:
        raise RuntimeError(
            "Intercept calibration optimization failed: "
            f"{optimization_result.message}"
        )

    optimal_intercept_shift = float(optimization_result.x)

    if not np.isfinite(optimal_intercept_shift):
        raise ValueError(
            "Intercept calibration produced a non-finite adjustment."
        )

    log_step(
        "Intercept-only calibration fitted successfully with "
        f"shift {optimal_intercept_shift:.6f}"
    )

    return optimal_intercept_shift

def apply_intercept_adjustment(
    log_odds: np.ndarray,
    intercept_shift: float,
    dataset_name: str,
) -> np.ndarray:
    """Apply an intercept-only calibration adjustment to logistic scores."""
    log_odds = np.asarray(
        log_odds,
        dtype=float,
    ).ravel()

    if not np.isfinite(log_odds).all():
        raise ValueError(
            f"{dataset_name.capitalize()} log-odds contain "
            "non-finite values."
        )

    if not np.isfinite(intercept_shift):
        raise ValueError(
            "Intercept shift must be finite."
        )

    calibrated_probabilities = expit(
        log_odds + intercept_shift
    )

    if not np.isfinite(calibrated_probabilities).all():
        raise ValueError(
            f"{dataset_name.capitalize()} intercept-calibrated "
            "probabilities contain non-finite values."
        )

    if (
        (calibrated_probabilities < 0.0).any()
        or (calibrated_probabilities > 1.0).any()
    ):
        raise ValueError(
            f"{dataset_name.capitalize()} intercept-calibrated "
            "probabilities fall outside the range [0, 1]."
        )

    return calibrated_probabilities