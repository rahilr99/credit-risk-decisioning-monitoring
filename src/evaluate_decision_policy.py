from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from compare_logistic_calibration_methods import (
    apply_isotonic_calibration,
    generate_logistic_scores,
)

from create_decision_policy import (
    assign_decisions,
    calculate_breakeven_probability,
    calculate_decision_boundaries,
    calculate_expected_value_per_dollar,
    create_decision_policy_summary,
)

from train_baseline_models import (
    load_sparse_features,
    load_target,
    save_report_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"


TEST_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "preprocessed_test_features.npz"
)

TEST_TARGET_PATH = (
    PROCESSED_DATA_DIR / "preprocessed_test_target.csv.gz"
)

LOGISTIC_MODEL_PATH = (
    MODELS_DIR / "logistic_regression_baseline.joblib"
)

ISOTONIC_CALIBRATOR_PATH = (
    MODELS_DIR / "logistic_regression_isotonic_calibrator.joblib"
)

DECISION_POLICY_EVALUATION_PATH = (
    REPORTS_DIR / "decision_policy_evaluation.csv"
)

def create_decision_evaluation_table(
    y_true: pd.Series,
    calibrated_probabilities: np.ndarray,
    decisions: np.ndarray,
    expected_values: np.ndarray,
) -> pd.DataFrame:
    """
    Create a row-level table containing model, policy, and outcome information.
    """

    evaluation_table = pd.DataFrame(
        {
            "bad_loan": np.asarray(
                y_true,
                dtype=int,
            ).ravel(),
            "calibrated_probability": np.asarray(
                calibrated_probabilities,
                dtype=float,
            ).ravel(),
            "decision": np.asarray(
                decisions,
            ).ravel(),
            "expected_value_per_dollar": np.asarray(
                expected_values,
                dtype=float,
            ).ravel(),
        }
    )

    return evaluation_table

def create_decision_outcome_summary(
    evaluation_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize actual outcomes, predicted risk, and expected value by decision.
    """

    summary = (
        evaluation_table
        .groupby(
            "decision",
            as_index=False,
        )
        .agg(
            row_count=("bad_loan", "size"),
            bad_loan_count=("bad_loan", "sum"),
            observed_bad_loan_rate=("bad_loan", "mean"),
            mean_predicted_bad_loan_probability=(
                "calibrated_probability",
                "mean",
            ),
            mean_expected_value_per_dollar=(
                "expected_value_per_dollar",
                "mean",
            ),
        )
    )

    summary["good_loan_count"] = (
        summary["row_count"]
        - summary["bad_loan_count"]
    )

    summary["portfolio_share"] = (
        summary["row_count"]
        / summary["row_count"].sum()
    )

    summary["share_of_all_bad_loans"] = (
        summary["bad_loan_count"]
        / summary["bad_loan_count"].sum()
    )

    summary["share_of_all_good_loans"] = (
        summary["good_loan_count"]
        / summary["good_loan_count"].sum()
    )

    return summary