from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def log_step(message: str) -> None:
    """Print a timestamped progress message to the terminal."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)

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

def evaluate_probability_predictions(
    y_true: pd.Series,
    bad_loan_probabilities: np.ndarray,
    model_name: str,
    dataset_name: str,
) -> dict[
    str,
    dict[str, str | int | float] | pd.DataFrame,
]:
    """Run all standard evaluation reports for one set of probabilities."""
    log_step(
        f"Running complete {dataset_name} evaluation for "
        f"{model_name}"
    )

    evaluation_results = {
        "probability_metrics": calculate_probability_metrics(
            y_true=y_true,
            bad_loan_probabilities=bad_loan_probabilities,
            model_name=model_name,
            dataset_name=dataset_name,
        ),
        "calibration_summary": inspect_calibration_performance(
            y_true=y_true,
            bad_loan_probabilities=bad_loan_probabilities,
            model_name=model_name,
            dataset_name=dataset_name,
        ),
        "threshold_summary": calculate_threshold_metrics(
            y_true=y_true,
            bad_loan_probabilities=bad_loan_probabilities,
            model_name=model_name,
            dataset_name=dataset_name,
        ),
        "decile_summary": create_score_decile_summary(
            y_true=y_true,
            bad_loan_probabilities=bad_loan_probabilities,
            model_name=model_name,
            dataset_name=dataset_name,
        ),
    }

    log_step(
        f"Completed all {dataset_name} evaluation reports for "
        f"{model_name}"
    )

    return evaluation_results
