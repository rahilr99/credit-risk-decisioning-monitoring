from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLUMN_PROFILE_PATH = (
    PROJECT_ROOT 
    / "reports"
    / "tables"
    / "interim_column_review_inventory.csv"
)

COLUMN_REVIEW_INVENTORY_PATH = (
    PROJECT_ROOT
    / "reports" 
    / "tables"
    / "interim_column_review_inventory.csv"
)

TARGET_COLUMNS = {
    "bad_loan", 
}

TARGET_SOURCE_COLUMNS = {
    "loan_status", 
}

IDENTIFIER_COLUMNS = {
    "id", 
    "url", 
}

MONITORING_ONLY_COLUMNS = {
    "issue_d", 
}

POST_APPLICATION_LEAKAGE_PREFIXES = (
    "hardship_", 
    "settlement_", 
    "debt_settlement_", 
)

POST_APPLICATION_LEAKAGE_COLUMNS = {
    "recoveries", 
    "collection_recovery_fee", 
    "total_pymnt", 
    "total_pymnt_inv", 
    "total_rec_prncp", 
    "total_rec_int",
    "total_rec_late_fee", 
    "last_pymnt_d", 
    "last_pymnt_amnt", 
    "next_pymnt_d", 
    "out_prncp", 
    "out_prncp_inv", 
    "last_fico_range_low", 
    "last_fico_range_high", 
    "last_credit_pull_d", 
    "payment_plan_start_date", 
    "debt_settlement_flag"
}

def assign_preliminary_column_role(
        column_name: str, 
        non_null_count: int, 
        unique_count: int, 
) -> str:
    """
    Assign an initial role to a column using only obvious rules. 

    Ambiguous columns are intentionally marked as 'required_review'
    so that they acan be reviewed manually later. 

    """

    if column_name in TARGET_COLUMNS:
        return "target"
    
    if column_name in TARGET_SOURCE_COLUMNS:
        return "target_source"
    
    if column_name in IDENTIFIER_COLUMNS:
        return "identifier"
    
    if column_name in MONITORING_ONLY_COLUMNS:
        return "monitoring_only"
    
    if (
        column_name in POST_APPLICATION_LEAKAGE_COLUMNS
        or column_name.startswith(POST_APPLICATION_LEAKAGE_PREFIXES)
    ): 
        return "psot_application_leakage"
    
    if non_null_count == 0 or unique_count <=1:
        return "constant_or_empty"
    
    return "requires_review"


def assign_preliminary_action(column_role: str) -> str: 
    """
    Recommend an initial action based on the preliminary column role. 

    These actions are intentionally consservative. Columns that require
      manual interpretation remain marked as 'requires_review'. 
    """

    if column_role == "target":
        return "retain_as_target"
    
    if column_role == "monitoring_only":
        return "retain_for_monitoring"
    
    if column_role in {
        "target_source", 
        "identifier", 
        "post_application_leakage", 
        "constant_or_empty",
    }:
        return "exclude_from_model_features"
    
    if column_role == "requires_review":
        return "requires_review"
    
    raise ValueError(f"Unexpected preliminary column role: {column_role}")

def assign_preliminary_reason(column_role: str) -> str:
    """
    Explain why a preliminary role and action were assigned. 

    The reason is stored in the inventory so hat each automatic
    classification remains understandable during later manual review. 
    """

    if column_role == "target": 
        return "Outcome variable used for supervised modeling."
    
    if column_role == "target_source":
        return (
            "Source column used to construct the target. "
            "Exclude to prevent direct target leakage."
        )
    if column_role == "identifier":
        return (
            "Record identifier or record-level reference. "
            "Not suitable as a model feature. "
        )
    if column_role == "monitoring_only":
        return (
            "Retain for portfolio monitoring or cohort analysis, "
            "but do not automatically use as a model feature."
        )
    if column_role == "post_application_leakage":
        return(
            "Contains information generated afeter loan issuance or"
            " during repayment performance. Exclude to prevent leakage."
        )
    if column_role == "constant_or_empty":
        return (
            "Column is empty or contains no meaningful variation."
            "Exclude from model features. "
        )
    if column_role == "requires_review": 
        return (
            "No obvious automatic classificatioln applies. "
            "Review manually before deciding feature eligibility. "
        )
    raise ValueError(f"Unexpected preliminary column role: {column_role}")