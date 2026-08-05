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