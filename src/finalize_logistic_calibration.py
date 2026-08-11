from pathlib import Path 

import joblib

from compare_logistic_calibration_methods import (
    apply_isotonic_calibration, 
    evaluate_calibration_candidates, 
    fit_isotonic_calibrator, 
    generate_logistic_scores,
)

from model_evaluation import (
    log_step,
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

VALIDATION_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "preprocessed_validation_features.npz"
)
VALIDATION_TARGET_PATH = (
    PROCESSED_DATA_DIR / "preprocessed_validation_target.csv.gz"
)

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

FINAL_TEST_CALIBRATION_SUMMARY_PATH = (
    REPORTS_DIR / "final_test_calibration_summary.csv"
)
FINAL_TEST_CALIBRATION_BAND_PATH = (
    REPORTS_DIR / "final_test_calibration_bands.csv"
)
FINAL_TEST_DISCRIMINATION_SUMMARY_PATH = (
    REPORTS_DIR / "final_test_discrimination_summary.csv"
)
FINAL_TEST_DECILE_PATH = (
    REPORTS_DIR / "final_test_score_deciles.csv"
)

def save_isotonic_calibrator(
    isotonic_calibrator,
    output_path: Path,
) -> None:
    """Save the fitted isotonic calibrator to disk."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        isotonic_calibrator,
        output_path,
    )

def main() -> None:
    """Fit final isotonic calibration and evaluate it on the test set."""
    log_step("Starting final logistic calibration workflow")

    X_validation = load_sparse_features(
        VALIDATION_FEATURES_PATH,
        "validation",
    )

    y_validation = load_target(
        VALIDATION_TARGET_PATH,
        "validation",
    )

    logistic_model = joblib.load(
        LOGISTIC_MODEL_PATH,
    )

    (
        validation_log_odds,
        validation_probabilities,
    ) = generate_logistic_scores(
        logistic_model,
        X_validation,
        "validation",
    )

    final_isotonic_calibrator = fit_isotonic_calibrator(
        validation_log_odds,
        y_validation,
    )

    save_isotonic_calibrator(
        final_isotonic_calibrator,
        ISOTONIC_CALIBRATOR_PATH,
    )

    X_test = load_sparse_features(
        TEST_FEATURES_PATH,
        "test",
    )

    y_test = load_target(
        TEST_TARGET_PATH,
        "test",
    )

    (
        test_log_odds,
        uncalibrated_test_probabilities,
    ) = generate_logistic_scores(
        logistic_model,
        X_test,
        "test",
    )

    isotonic_test_probabilities = apply_isotonic_calibration(
        final_isotonic_calibrator,
        test_log_odds,
        "test",
    )

    candidate_probabilities = {
        "uncalibrated": uncalibrated_test_probabilities,
        "isotonic": isotonic_test_probabilities,
    }

    (
        calibration_summary_report,
        calibration_band_report,
        discrimination_summary_report,
        discrimination_decile_report,
    ) = evaluate_calibration_candidates(
        y_test,
        candidate_probabilities,
        reference_method_name="uncalibrated",
    )

    save_report_table(
        calibration_summary_report,
        FINAL_TEST_CALIBRATION_SUMMARY_PATH,
        "final test calibration summary",
    )

    save_report_table(
        calibration_band_report,
        FINAL_TEST_CALIBRATION_BAND_PATH,
        "final test calibration band report",
    )

    save_report_table(
        discrimination_summary_report,
        FINAL_TEST_DISCRIMINATION_SUMMARY_PATH,
        "final test discrimination summary",
    )

    save_report_table(
        discrimination_decile_report,
        FINAL_TEST_DECILE_PATH,
        "final test score decile report",
    )

    log_step("Final logistic calibration workflow completed")

if __name__ == "__main__":
    main()