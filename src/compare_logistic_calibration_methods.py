from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from model_evaluation import (
    evaluate_probability_predictions,
    create_score_decile_summary, 
    calculate_probability_metrics, 
    inspect_calibration_performance,
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
        file_path: Path=LOGISTIC_BASELINE_MODEL_PATH, 
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
    calibration_fit_fraction: float = CALIBRATION_FIT_FRACTION,
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

def fit_platt_calibrator(
    calibration_log_odds: np.ndarray, 
    y_calibration_fit: pd.Series, 
) -> LogisticRegression:
    """Fit Platt Calibration using the original logistic log-odds."""
    calibration_log_odds = np.asarray(
        calibration_log_odds, 
        dtype=float,
    ).ravel()

    calibration_targets = np.asarray(
        y_calibration_fit, 
        dtype=int,
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
    
    if set(np.unique(calibration_targets).tolist()) != {0, 1}: 
        raise ValueError(
            "Calibration targets must contain both class 0 and class 1."
        )
    
    platt_calibrator = LogisticRegression(
        penalty=None, 
        solver="lbfgs",
        max_iter=1000, 
    )

    platt_calibrator.fit(
        calibration_log_odds.reshape(-1,1),
        calibration_targets,
    )

    platt_scale = float(platt_calibrator.coef_[0, 0])
    platt_intercept = float(platt_calibrator.intercept_[0])

    if not np.isfinite(platt_scale):
        raise ValueError(
            "Platt calibration produced a non-finite scale."
        )
    
    if not np.isfinite(platt_intercept):
        raise ValueError(
            "Platt calibration produced a non-finite intercept."
        )
    
    return platt_calibrator


def apply_platt_calibration(
    platt_calibrator: LogisticRegression,
    log_odds: np.ndarray,
    dataset_name: str,
) -> np.ndarray:
    """Apply a fitted Platt calibrator to logistic log-odds."""
    log_odds = np.asarray(
        log_odds,
        dtype=float,
    ).ravel()

    if not np.isfinite(log_odds).all():
        raise ValueError(
            f"{dataset_name.capitalize()} log-odds contain "
            "non-finite values."
        )

    if not hasattr(platt_calibrator, "classes_"):
        raise ValueError(
            "Platt calibrator has not been fitted."
        )

    if 1 not in platt_calibrator.classes_:
        raise ValueError(
            "Platt calibrator does not contain bad-loan class 1."
        )

    class_one_index = int(
        np.where(platt_calibrator.classes_ == 1)[0][0]
    )

    calibrated_probabilities = platt_calibrator.predict_proba(
        log_odds.reshape(-1, 1)
    )[:, class_one_index]

    if not np.isfinite(calibrated_probabilities).all():
        raise ValueError(
            f"{dataset_name.capitalize()} Platt-calibrated "
            "probabilities contain non-finite values."
        )

    if (
        (calibrated_probabilities < 0.0).any()
        or (calibrated_probabilities > 1.0).any()
    ):
        raise ValueError(
            f"{dataset_name.capitalize()} Platt-calibrated "
            "probabilities fall outside the range [0, 1]."
        )

    return calibrated_probabilities


def fit_isotonic_calibrator(
    calibration_log_odds: np.ndarray,
    y_calibration_fit: pd.Series,
) -> IsotonicRegression:
    """Fit isotonic calibration using the original logistic log-odds."""
    calibration_log_odds = np.asarray(
        calibration_log_odds,
        dtype=float,
    ).ravel()

    calibration_targets = np.asarray(
        y_calibration_fit,
        dtype=int,
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

    if set(np.unique(calibration_targets).tolist()) != {0, 1}:
        raise ValueError(
            "Calibration targets must contain both class 0 and class 1."
        )

    isotonic_calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    isotonic_calibrator.fit(
        calibration_log_odds,
        calibration_targets,
    )

    return isotonic_calibrator

def apply_isotonic_calibration(
    isotonic_calibrator: IsotonicRegression,
    log_odds: np.ndarray,
    dataset_name: str,
) -> np.ndarray:
    """Apply a fitted isotonic calibrator to logistic log-odds."""
    log_odds = np.asarray(
        log_odds,
        dtype=float,
    ).ravel()

    if not np.isfinite(log_odds).all():
        raise ValueError(
            f"{dataset_name.capitalize()} log-odds contain "
            "non-finite values."
        )

    calibrated_probabilities = isotonic_calibrator.predict(
        log_odds
    )

    if not np.isfinite(calibrated_probabilities).all():
        raise ValueError(
            f"{dataset_name.capitalize()} isotonic-calibrated "
            "probabilities contain non-finite values."
        )

    if (
        (calibrated_probabilities < 0.0).any()
        or (calibrated_probabilities > 1.0).any()
    ):
        raise ValueError(
            f"{dataset_name.capitalize()} isotonic-calibrated "
            "probabilities fall outside the range [0, 1]."
        )

    return calibrated_probabilities


def evaluate_calibration_candidates(
    y_comparison: pd.Series,
    candidate_probabilities: dict[str, np.ndarray],
    reference_method_name: str = "uncalibrated",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create calibration and discrimination comparison reports."""
    dataset_name = "validation_calibration_comparison"

    method_order = list(candidate_probabilities.keys())

    reference_probabilities = np.asarray(
        candidate_probabilities[reference_method_name],
        dtype=float,
    ).ravel()

    calibration_summary_rows = []
    calibration_band_reports = []

    discrimination_summary_rows = []
    discrimination_decile_reports = []

    for method_name, probabilities in candidate_probabilities.items():
        probabilities = np.asarray(
            probabilities,
            dtype=float,
        ).ravel()

        probability_metrics = calculate_probability_metrics(
            y_true=y_comparison,
            bad_loan_probabilities=probabilities,
            model_name=method_name,
            dataset_name=dataset_name,
        )

        calibration_report = inspect_calibration_performance(
            y_true=y_comparison,
            bad_loan_probabilities=probabilities,
            model_name=method_name,
            dataset_name=dataset_name,
        )

        calibration_report["absolute_calibration_gap"] = (
            calibration_report["calibration_gap"].abs()
        )

        weighted_mean_absolute_calibration_gap = float(
            np.average(
                calibration_report["absolute_calibration_gap"],
                weights=calibration_report["row_count"],
            )
        )

        calibration_summary_rows.append(
            {
                "calibration_method": method_name,
                "observed_bad_loan_rate": (
                    probability_metrics["observed_bad_loan_rate"]
                ),
                "mean_predicted_bad_loan_probability": (
                    probability_metrics[
                        "mean_predicted_bad_loan_probability"
                    ]
                ),
                "weighted_mean_absolute_calibration_gap": (
                    weighted_mean_absolute_calibration_gap
                ),
                "log_loss": probability_metrics["log_loss"],
                "brier_score": probability_metrics["brier_score"],
            }
        )

        calibration_report = calibration_report[
            [
                "probability_bin",
                "row_count",
                "mean_predicted_bad_loan_probability",
                "observed_bad_loan_rate",
                "calibration_gap",
                "absolute_calibration_gap",
            ]
        ].copy()

        calibration_report.insert(
            0,
            "calibration_method",
            method_name,
        )

        calibration_band_reports.append(
            calibration_report
        )

        rank_correlation = float(
            spearmanr(
                reference_probabilities,
                probabilities,
            ).statistic
        )

        gini = (
            2.0 * probability_metrics["roc_auc"]
            - 1.0
        )

        discrimination_summary_rows.append(
            {
                "calibration_method": method_name,
                "roc_auc": probability_metrics["roc_auc"],
                "gini": gini,
                "average_precision": (
                    probability_metrics["average_precision"]
                ),
                "spearman_rank_correlation_to_uncalibrated": (
                    rank_correlation
                ),
            }
        )

        decile_report = create_score_decile_summary(
            y_true=y_comparison,
            bad_loan_probabilities=probabilities,
            model_name=method_name,
            dataset_name=dataset_name,
        )

        decile_report = decile_report[
            [
                "score_decile",
                "row_count",
                "mean_predicted_bad_loan_probability",
                "observed_bad_loan_rate",
                "share_of_all_bad_loans",
                "cumulative_bad_loan_share",
            ]
        ].copy()

        decile_report.insert(
            0,
            "calibration_method",
            method_name,
        )

        discrimination_decile_reports.append(
            decile_report
        )

    calibration_summary_report = pd.DataFrame(
        calibration_summary_rows
    )

    calibration_band_report = pd.concat(
        calibration_band_reports,
        ignore_index=True,
    )

    discrimination_summary_report = pd.DataFrame(
        discrimination_summary_rows
    )

    discrimination_decile_report = pd.concat(
        discrimination_decile_reports,
        ignore_index=True,
    )

    calibration_band_report["calibration_method"] = pd.Categorical(
        calibration_band_report["calibration_method"],
        categories=method_order,
        ordered=True,
    )

    calibration_band_report = calibration_band_report.sort_values(
        [
            "probability_bin",
            "calibration_method",
        ]
    ).reset_index(drop=True)

    discrimination_decile_report[
        "calibration_method"
    ] = pd.Categorical(
        discrimination_decile_report["calibration_method"],
        categories=method_order,
        ordered=True,
    )

    discrimination_decile_report = (
        discrimination_decile_report.sort_values(
            [
                "score_decile",
                "calibration_method",
            ]
        ).reset_index(drop=True)
    )

    calibration_band_report[
        "calibration_method"
    ] = calibration_band_report[
        "calibration_method"
    ].astype(str)

    discrimination_decile_report[
        "calibration_method"
    ] = discrimination_decile_report[
        "calibration_method"
    ].astype(str)

    return (
        calibration_summary_report,
        calibration_band_report,
        discrimination_summary_report,
        discrimination_decile_report,
    )

def save_calibration_comparison_reports(
    calibration_summary_report: pd.DataFrame,
    calibration_band_report: pd.DataFrame,
    discrimination_summary_report: pd.DataFrame,
    discrimination_decile_report: pd.DataFrame,
    reports_directory: Path,
) -> None:
    """Save calibration comparison reports as CSV files."""
    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_summary_report.to_csv(
        reports_directory
        / "logistic_calibration_summary.csv",
        index=False,
    )

    calibration_band_report.to_csv(
        reports_directory
        / "logistic_calibration_band_comparison.csv",
        index=False,
    )

    discrimination_summary_report.to_csv(
        reports_directory
        / "logistic_calibration_discrimination_summary.csv",
        index=False,
    )

    discrimination_decile_report.to_csv(
        reports_directory
        / "logistic_calibration_discrimination_deciles.csv",
        index=False,
    )

def main() -> None:
    """Run the logistic-regression calibration comparison workflow."""
    log_step("Starting logistic calibration comparison workflow")

    X_validation = load_sparse_features(
        VALIDATION_FEATURES_PATH,
        "validation",
    )

    y_validation = load_target(
        VALIDATION_TARGET_PATH,
        "validation",
    )

    (
        X_calibration_fit,
        y_calibration_fit,
        X_comparison,
        y_comparison,
    ) = create_chronological_calibration_split(
        X_validation,
        y_validation,
    )

    logistic_model = load_logistic_baseline_model()

    calibration_fit_log_odds, _ = generate_logistic_scores(
        logistic_model,
        X_calibration_fit,
        "calibration fit",
    )

    (
        comparison_log_odds,
        uncalibrated_comparison_probabilities,
    ) = generate_logistic_scores(
        logistic_model,
        X_comparison,
        "calibration comparison",
    )

    intercept_shift = fit_intercept_adjustment(
        calibration_fit_log_odds,
        y_calibration_fit,
    )

    platt_calibrator = fit_platt_calibrator(
        calibration_fit_log_odds,
        y_calibration_fit,
    )

    isotonic_calibrator = fit_isotonic_calibrator(
        calibration_fit_log_odds,
        y_calibration_fit,
    )

    intercept_probabilities = apply_intercept_adjustment(
        comparison_log_odds,
        intercept_shift,
        "calibration comparison",
    )

    platt_probabilities = apply_platt_calibration(
        platt_calibrator,
        comparison_log_odds,
        "calibration comparison",
    )

    isotonic_probabilities = apply_isotonic_calibration(
        isotonic_calibrator,
        comparison_log_odds,
        "calibration comparison",
    )

    candidate_probabilities = {
        "uncalibrated": uncalibrated_comparison_probabilities,
        "intercept": intercept_probabilities,
        "platt": platt_probabilities,
        "isotonic": isotonic_probabilities,
    }

    (
        calibration_summary_report,
        calibration_band_report,
        discrimination_summary_report,
        discrimination_decile_report,
    ) = evaluate_calibration_candidates(
        y_comparison,
        candidate_probabilities,
    )

    save_calibration_comparison_reports(
        calibration_summary_report,
        calibration_band_report,
        discrimination_summary_report,
        discrimination_decile_report,
        REPORTS_DIR,
    )

    log_step(
        "Logistic calibration comparison workflow completed successfully"
    )

if __name__ == "__main__":
    main()