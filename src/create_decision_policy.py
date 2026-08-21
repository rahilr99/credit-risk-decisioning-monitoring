from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_baseline_models import (
    save_report_table,
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

LOGISTIC_MODEL_PATH = (
    MODELS_DIR / "logistic_regression_baseline.joblib"
)

ISOTONIC_CALIBRATOR_PATH = (
    MODELS_DIR / "logistic_regression_isotonic_calibrator.joblib"
)

DECISION_POLICY_CONFIG_PATH = (
    REPORTS_DIR / "decision_policy_summary.csv"
)


DEFAULT_GOOD_LOAN_RETURN_RATE = 0.10
DEFAULT_BAD_LOAN_LOSS_RATE = 0.60
DEFAULT_REVIEW_MARGIN = 0.015

def calculate_breakeven_probability(
    good_loan_return_rate: float,
    bad_loan_loss_rate: float,
) -> float:
    """
    Calculate the bad-loan probability where the expected value is zero.
    """

    if not 0 < good_loan_return_rate <= 1:
        raise ValueError(
            "Good-loan return rate must be greater than 0 and at most 1."
        )

    if not 0 < bad_loan_loss_rate <= 1:
        raise ValueError(
            "Bad-loan loss rate must be greater than 0 and at most 1."
        )

    breakeven_probability = (
        good_loan_return_rate
        / (good_loan_return_rate + bad_loan_loss_rate)
    )

    return breakeven_probability


def calculate_decision_boundaries(
        breakeven_probability: float, 
        review_margin: float, 
) -> tuple[float, float]:
    """
    Calculate the approve and decline probability boundaries. 
    """

    if not 0 < breakeven_probability < 1: 
        raise ValueError(
            "Breakeven probability must be greater than 0 and less than 1."
        ) 
    
    if not 0 <= review_margin < 1:
        raise ValueError(
            "Review margin must be at least 0 and less than 1."
        )
    approve_boundary = max(
        0.0, 
        breakeven_probability - review_margin
    )

    decline_boundary = min(
        1.0, 
        breakeven_probability + review_margin
    )

    return approve_boundary, decline_boundary


def assign_decisions(
    calibrated_probabilities: np.ndarray,
    approve_boundary: float,
    decline_boundary: float,
) -> np.ndarray:
    """
    Assign approve, review, or decline decisions from calibrated probabilities.
    """

    calibrated_probabilities = np.asarray(
        calibrated_probabilities,
        dtype=float,
    ).ravel()

    if not np.isfinite(calibrated_probabilities).all():
        raise ValueError(
            "Calibrated probabilities contain non-finite values."
        )

    if (
        (calibrated_probabilities < 0).any()
        or (calibrated_probabilities > 1).any()
    ):
        raise ValueError(
            "Calibrated probabilities must be between 0 and 1."
        )

    if not 0 <= approve_boundary <= decline_boundary <= 1:
        raise ValueError(
            "Decision boundaries must satisfy "
            "0 <= approve_boundary <= decline_boundary <= 1."
        )

    decisions = np.where(
        calibrated_probabilities < approve_boundary,
        "approve",
        np.where(
            calibrated_probabilities <= decline_boundary,
            "review",
            "decline",
        ),
    )

    return decisions


def calculate_expected_value_per_dollar(
    calibrated_probabilities: np.ndarray,
    good_loan_return_rate: float,
    bad_loan_loss_rate: float,
) -> np.ndarray:
    """
    Calculate expected economic value per dollar lent.
    """

    calibrated_probabilities = np.asarray(
        calibrated_probabilities,
        dtype=float,
    ).ravel()

    expected_values = (
        (1 - calibrated_probabilities) * good_loan_return_rate
        - calibrated_probabilities * bad_loan_loss_rate
    )

    return expected_values


def create_decision_policy_summary(
    y_true: pd.Series,
    calibrated_probabilities: np.ndarray,
    decisions: np.ndarray,
    expected_values: np.ndarray,
) -> pd.DataFrame:
    """
    Summarize model risk and observed outcomes by policy decision.
    """

    policy_data = pd.DataFrame(
        {
            "bad_loan": np.asarray(y_true, dtype=int).ravel(),
            "calibrated_probability": calibrated_probabilities,
            "decision": decisions,
            "expected_value_per_dollar": expected_values,
        }
    )

    summary = (
        policy_data
        .groupby(
            "decision",
            as_index=False,
        )
        .agg(
            row_count=("bad_loan", "size"),
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

    summary["portfolio_share"] = (
        summary["row_count"] / len(policy_data)
    )

    return summary

def main() -> None:
    """Create and save the baseline credit decision policy."""

    breakeven_probability = calculate_breakeven_probability(
        DEFAULT_GOOD_LOAN_RETURN_RATE,
        DEFAULT_BAD_LOAN_LOSS_RATE,
    )

    (
        approve_boundary,
        decline_boundary,
    ) = calculate_decision_boundaries(
        breakeven_probability,
        DEFAULT_REVIEW_MARGIN,
    )

    policy_configuration = pd.DataFrame(
        {
            "good_loan_return_rate": [
                DEFAULT_GOOD_LOAN_RETURN_RATE
            ],
            "bad_loan_loss_rate": [
                DEFAULT_BAD_LOAN_LOSS_RATE
            ],
            "breakeven_probability": [
                breakeven_probability
            ],
            "review_margin": [
                DEFAULT_REVIEW_MARGIN
            ],
            "approve_boundary": [
                approve_boundary
            ],
            "decline_boundary": [
                decline_boundary
            ],
        }
    )

    save_report_table(
        policy_configuration,
        DECISION_POLICY_CONFIG_PATH,
        "decision policy configuration",
    )


if __name__ == "__main__":
    main()