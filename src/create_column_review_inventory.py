from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLUMN_PROFILE_PATH = (
    PROJECT_ROOT 
    / "reports"
    / "tables"
    / "interim_column_profile.csv"
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
    unique_non_missing_count: int,
    is_constant_non_missing: bool,
) -> str:
    """
    Assign an initial role to a column using only obvious rules.

    Ambiguous columns are intentionally marked as 'requires_review'
    so that they can be reviewed manually later.
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
        return "post_application_leakage"

    if unique_non_missing_count == 0 or is_constant_non_missing:
        return "constant_or_empty"

    return "requires_review"


def assign_preliminary_action(column_role: str) -> str: 
    """
    Recommend an initial action based on the preliminary column role. 

    These actions are intentionally conservative. Columns that require
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

    The reason is stored in the inventory so that each automatic
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
            "Contains information generated after loan issuance or"
            " during repayment performance. Exclude to prevent leakage."
        )
    if column_role == "constant_or_empty":
        return (
            "Column is empty or contains no meaningful variation. "
            "Exclude from model features. "
        )
    if column_role == "requires_review": 
        return (
            "No obvious automatic classification applies. "
            "Review manually before deciding feature eligibility. "
        )
    raise ValueError(f"Unexpected preliminary column role: {column_role}")

def create_column_review_inventory(
        profile_path: Path = COLUMN_PROFILE_PATH, 
        output_path: Path = COLUMN_REVIEW_INVENTORY_PATH, 
) -> pd.DataFrame:
    """
    Create and save a preliminary column-review inventory. 

    The function applies only obvious automatic classifications.
    Columns that require interpretation remain marked as 
    "requires_review" for later manual review. 
    """

    profile_df = pd.read_csv(profile_path)

    required_columns = {
        "column_name",
        "unique_non_missing_count",
        "is_constant_non_missing",
    }

    missing_columns = required_columns.difference(profile_df.columns)

    if missing_columns: 
        raise ValueError(
            "Column profile report is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    
    inventory_df = profile_df.copy()
    inventory_df["column_role"] = inventory_df.apply(
        lambda row: assign_preliminary_column_role(
            column_name=row["column_name"],
            unique_non_missing_count=row["unique_non_missing_count"],
            is_constant_non_missing=row["is_constant_non_missing"],
        ),
        axis=1,
    )

    inventory_df["recommended_action"] = inventory_df["column_role"].apply(
        assign_preliminary_action
    )

    inventory_df["reason"] = inventory_df["column_role"].apply(
        assign_preliminary_reason
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    inventory_df.to_csv(
        output_path, 
        index=False,
    )

    print(f"Saved preliminary inventory to: {output_path}")
    print(f"Columns reviewed: {len(inventory_df):,}")

    print("\nPreliminary role counts:")
    print(
        inventory_df["column_role"]
        .value_counts()
        .to_string()
    )

    requires_review_columns = inventory_df.loc[
        inventory_df["column_role"] == "requires_review",
        "column_name", 
    ].tolist()

    print("\nColumns requiring manual review:")
    print(f"Count: {len(requires_review_columns):,}")

    for column_name in requires_review_columns:
        print(f"- {column_name}")

    return inventory_df


if __name__ == "__main__":
    create_column_review_inventory()


