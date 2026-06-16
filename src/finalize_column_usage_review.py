from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRELIMINARY_INVENTORY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "interim_column_review_inventory.csv"
)

COLUMN_USAGE_INVENTORY_PATH = (
    PROJECT_ROOT 
    / "config"
    / "column_usage_inventory.csv"
    
)

EXPECTED_COLUMN_COUNT = 152

ALLOWED_PRIMARY_PURPOSES = {
    "modeling",
    "monitoring",
    "target",
    "target_source",
    "identifier",
    "lender_derived_feature",
    "conditionally_available",
    "high_cardinality_or_text",
    "post_application_leakage",
    "constant_or_empty",
    "exclude_from_model_features",
}

ALLOWED_ADDITIONAL_USE_CASES = {
    "benchmarking",
    "data_quality_check",
    "portfolio_monitoring",
    "portfolio_segmentation",
    "reporting",
    "time_split",
}

AUTO_ROLE_TO_PRIMARY_PURPOSE = {
    "target": "target",
    "target_source": "target_source",
    "identifier": "identifier",
    "monitoring_only": "monitoring",
    "post_application_leakage": "post_application_leakage",
    "constant_or_empty": "constant_or_empty",
}

AUTO_ADDITIONAL_USE_CASES = {
    "bad_loan": "portfolio_monitoring,reporting",
    "loan_status": "portfolio_monitoring,reporting",
    "id": "data_quality_check",
    "issue_d": "portfolio_segmentation,reporting,time_split",
}

MANUAL_COLUMN_USAGE = {}

# -------------------------------------------------------------------
# Batch 1 — Application and loan setup fields
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "application_type": (
            "modeling",
            "portfolio_segmentation",
        ),
        "term": (
            "modeling",
            "portfolio_segmentation",
        ),
        "verification_status": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
        "verification_status_joint": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
        "home_ownership": (
            "modeling",
            "portfolio_segmentation",
        ),
        "purpose": (
            "modeling",
            "portfolio_segmentation,reporting",
        ),
        "emp_length": (
            "modeling",
            "",
        ),
    }
)

# -------------------------------------------------------------------
# Batch 2 — Operational and timing-sensitive fields
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "disbursement_method": (
            "conditionally_available",
            "portfolio_segmentation,reporting",
        ),
        "initial_list_status": (
            "exclude_from_model_features",
            "portfolio_segmentation,reporting",
        ),
        "pymnt_plan": (
            "post_application_leakage",
            "portfolio_monitoring,reporting",
        ),
        "orig_projected_additional_accrued_interest": (
            "post_application_leakage",
            "portfolio_monitoring,reporting",
        ),
    }
)


# -------------------------------------------------------------------
# Batch 3 — Geography and high-cardinality text fields
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "addr_state": (
            "monitoring",
            "portfolio_segmentation,reporting",
        ),
        "zip_code": (
            "monitoring",
            "portfolio_segmentation",
        ),
        "title": (
            "high_cardinality_or_text",
            "",
        ),
        "desc": (
            "high_cardinality_or_text",
            "",
        ),
        "emp_title": (
            "high_cardinality_or_text",
            "",
        ),
    }
)


# -------------------------------------------------------------------
# Batch 4 — Lender-derived, pricing, and funding fields
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "grade": (
            "lender_derived_feature",
            "benchmarking,portfolio_segmentation,reporting",
        ),
        "sub_grade": (
            "lender_derived_feature",
            "benchmarking,portfolio_segmentation,reporting",
        ),
        "int_rate": (
            "lender_derived_feature",
            "benchmarking,portfolio_segmentation,reporting",
        ),
        "installment": (
            "lender_derived_feature",
            "portfolio_segmentation,reporting",
        ),
        "loan_amnt": (
            "conditionally_available",
            "portfolio_segmentation,reporting",
        ),
        "funded_amnt": (
            "monitoring",
            "portfolio_segmentation,reporting",
        ),
        "funded_amnt_inv": (
            "monitoring",
            "portfolio_segmentation,reporting",
        ),
    }
)


# -------------------------------------------------------------------
# Batch 5 — Delinquency, collections, and public-record fields
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "num_tl_30dpd": (
            "conditionally_available",
            "data_quality_check,portfolio_segmentation",
        ),
        "num_tl_120dpd_2m": (
            "conditionally_available",
            "data_quality_check,portfolio_segmentation",
        ),
        "acc_now_delinq": (
            "modeling",
            "portfolio_segmentation",
        ),
        "chargeoff_within_12_mths": (
            "modeling",
            "portfolio_segmentation",
        ),
        "pub_rec_bankruptcies": (
            "modeling",
            "portfolio_segmentation",
        ),
        "collections_12_mths_ex_med": (
            "modeling",
            "portfolio_segmentation",
        ),
        "num_tl_90g_dpd_24m": (
            "modeling",
            "portfolio_segmentation",
        ),
        "delinq_2yrs": (
            "modeling",
            "portfolio_segmentation",
        ),
        "tax_liens": (
            "modeling",
            "portfolio_segmentation",
        ),
        "num_accts_ever_120_pd": (
            "modeling",
            "portfolio_segmentation",
        ),
        "pub_rec": (
            "modeling",
            "portfolio_segmentation",
        ),
        "delinq_amnt": (
            "modeling",
            "portfolio_segmentation",
        ),
        "tot_coll_amt": (
            "modeling",
            "portfolio_segmentation",
        ),
    }
)


# -------------------------------------------------------------------
# Batch 6 — Credit inquiries and recently opened accounts
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "open_acc_6m": (
            "modeling",
            "portfolio_segmentation",
        ),
        "open_il_12m": (
            "modeling",
            "portfolio_segmentation",
        ),
        "mths_since_recent_inq": (
            "modeling",
            "portfolio_segmentation",
        ),
        "open_rv_12m": (
            "modeling",
            "portfolio_segmentation",
        ),
        "inq_last_6mths": (
            "modeling",
            "portfolio_segmentation",
        ),
        "inq_fi": (
            "modeling",
            "portfolio_segmentation",
        ),
        "open_il_24m": (
            "modeling",
            "portfolio_segmentation",
        ),
        "num_tl_op_past_12m": (
            "modeling",
            "portfolio_segmentation",
        ),
        "inq_last_12m": (
            "modeling",
            "portfolio_segmentation",
        ),
        "open_rv_24m": (
            "modeling",
            "portfolio_segmentation",
        ),
        "acc_open_past_24mths": (
            "modeling",
            "portfolio_segmentation",
        ),
        "mo_sin_rcnt_tl": (
            "modeling",
            "portfolio_segmentation",
        ),
        "mo_sin_rcnt_rev_tl_op": (
            "modeling",
            "portfolio_segmentation",
        ),
        "mths_since_rcnt_il": (
            "modeling",
            "portfolio_segmentation",
        ),
        "mths_since_recent_bc": (
            "modeling",
            "portfolio_segmentation",
        ),
    }
)


# -------------------------------------------------------------------
# Batch 7 — Account counts and credit mix
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "num_actv_bc_tl": (
            "modeling",
            "portfolio_segmentation",
        ),
        "mort_acc": (
            "modeling",
            "portfolio_segmentation",
        ),
        "num_rev_tl_bal_gt_0": (
            "modeling",
            "portfolio_segmentation",
        ),
        "open_act_il": (
            "modeling",
            "portfolio_segmentation",
        ),
        "num_actv_rev_tl": (
            "modeling",
            "portfolio_segmentation",
        ),
        "num_bc_sats": (
            "modeling",
            "portfolio_segmentation",
        ),
        "total_cu_tl": (
            "modeling",
            "portfolio_segmentation",
        ),
        "num_bc_tl": (
            "modeling",
            "portfolio_segmentation",
        ),
        "num_op_rev_tl": (
            "modeling",
            "portfolio_segmentation",
        ),
        "num_sats": (
            "modeling",
            "portfolio_segmentation",
        ),
        "open_acc": (
            "modeling",
            "portfolio_segmentation",
        ),
        "num_rev_accts": (
            "modeling",
            "portfolio_segmentation",
        ),
        "num_il_tl": (
            "modeling",
            "portfolio_segmentation",
        ),
        "total_acc": (
            "modeling",
            "portfolio_segmentation",
        ),
    }
)


# -------------------------------------------------------------------
# Batch 8 — Credit-history age, derogatory-event recency, and FICO range
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "fico_range_high": (
            "modeling",
            "benchmarking,portfolio_segmentation",
        ),
        "fico_range_low": (
            "modeling",
            "benchmarking,portfolio_segmentation",
        ),
        "mths_since_last_record": (
            "modeling",
            "portfolio_segmentation",
        ),
        "mths_since_last_delinq": (
            "modeling",
            "portfolio_segmentation",
        ),
        "mths_since_recent_bc_dlq": (
            "modeling",
            "portfolio_segmentation",
        ),
        "mths_since_recent_revol_delinq": (
            "modeling",
            "portfolio_segmentation",
        ),
        "mths_since_last_major_derog": (
            "modeling",
            "portfolio_segmentation",
        ),
        "mo_sin_old_il_acct": (
            "modeling",
            "portfolio_segmentation",
        ),
        "earliest_cr_line": (
            "modeling",
            "data_quality_check,portfolio_segmentation",
        ),
        "mo_sin_old_rev_tl_op": (
            "modeling",
            "portfolio_segmentation",
        ),
        "pct_tl_nvr_dlq": (
            "modeling",
            "portfolio_segmentation",
        ),
    }
)


# -------------------------------------------------------------------
# Batch 9 — Utilization and revolving-credit measures
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "all_util": (
            "modeling",
            "portfolio_segmentation",
        ),
        "il_util": (
            "modeling",
            "portfolio_segmentation",
        ),
        "percent_bc_gt_75": (
            "modeling",
            "portfolio_segmentation",
        ),
        "revol_util": (
            "modeling",
            "portfolio_segmentation",
        ),
        "bc_util": (
            "modeling",
            "portfolio_segmentation",
        ),
        "bc_open_to_buy": (
            "modeling",
            "portfolio_segmentation",
        ),
        "revol_bal": (
            "modeling",
            "portfolio_segmentation",
        ),
        "total_bc_limit": (
            "modeling",
            "portfolio_segmentation",
        ),
        "total_rev_hi_lim": (
            "modeling",
            "portfolio_segmentation",
        ),
        "max_bal_bc": (
            "modeling",
            "portfolio_segmentation",
        ),
    }
)


# -------------------------------------------------------------------
# Batch 10 — Income, debt burden, and aggregate balances
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "dti": (
            "modeling",
            "portfolio_segmentation",
        ),
        "annual_inc": (
            "modeling",
            "data_quality_check,portfolio_segmentation",
        ),
        "avg_cur_bal": (
            "modeling",
            "portfolio_segmentation",
        ),
        "total_bal_il": (
            "modeling",
            "portfolio_segmentation",
        ),
        "total_il_high_credit_limit": (
            "modeling",
            "portfolio_segmentation",
        ),
        "total_bal_ex_mort": (
            "modeling",
            "portfolio_segmentation",
        ),
        "tot_cur_bal": (
            "modeling",
            "portfolio_segmentation",
        ),
        "tot_hi_cred_lim": (
            "modeling",
            "portfolio_segmentation",
        ),
    }
)


# -------------------------------------------------------------------
# Batch 11 — Secondary-applicant bureau fields
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "sec_app_inq_last_6mths": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
        "sec_app_collections_12_mths_ex_med": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
        "sec_app_chargeoff_within_12_mths": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
        "sec_app_mort_acc": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
        "sec_app_open_act_il": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
        "sec_app_open_acc": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
        "sec_app_fico_range_high": (
            "conditionally_available",
            "benchmarking,portfolio_segmentation",
        ),
        "sec_app_fico_range_low": (
            "conditionally_available",
            "benchmarking,portfolio_segmentation",
        ),
        "sec_app_num_rev_accts": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
        "sec_app_mths_since_last_major_derog": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
        "sec_app_earliest_cr_line": (
            "conditionally_available",
            "data_quality_check,portfolio_segmentation",
        ),
        "sec_app_revol_util": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
    }
)


# -------------------------------------------------------------------
# Batch 12 — Joint-financial fields
# -------------------------------------------------------------------

MANUAL_COLUMN_USAGE.update(
    {
        "dti_joint": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
        "annual_inc_joint": (
            "conditionally_available",
            "data_quality_check,portfolio_segmentation",
        ),
        "revol_bal_joint": (
            "conditionally_available",
            "portfolio_segmentation",
        ),
    }
)

def normalize_additional_use_cases(raw_value: str) -> str:
    """
    Return a standardized comma-separated list of approved
    additional use cases. 
    """

    if pd.isna(raw_value) or not str(raw_value).strip():
        return ""
    
    values = [
        value.strip()
        for value in str(raw_value).split(",")
        if value.strip()
    ]

    unknown_values = set(values).difference(
        ALLOWED_ADDITIONAL_USE_CASES
    )

    if unknown_values:
        raise ValueError(
            "Unknown additional use cases: "
            f"{sorted(unknown_values)}"
        )
    
    if len(values) != len(set(values)):
        raise ValueError(
            f"Duplicate addtional use case found in: {raw_value}"
        )
    
    return ",".join(sorted(values))


def validate_manual_review_coverage(
        preliminary_df: pd.DataFrame, 
) -> None:
    """
    Confirm that the manual mapping covers the unresolved columns exactly. 
    """

    requires_review_columns= set(
        preliminary_df.loc[
            preliminary_df["column_role"] == "requires_review",
            "column_name",
        ]
    )

    manually_mapped_columns = set(
        MANUAL_COLUMN_USAGE
    )

    missing_manual_decisions = requires_review_columns.difference(
        manually_mapped_columns
    )

    extra_manual_decisions = manually_mapped_columns.difference(
        requires_review_columns
    )

    if missing_manual_decisions:
        raise ValueError(
            "Manual decisions are missing for these columns: "
            f"{sorted(missing_manual_decisions)}"
        )
    
    print(
        "Manual-review coverage passed: "
        f"{len(requires_review_columns):,} unresolved columns mapped. "
    )


def assign_column_usage(row: pd.Series) -> pd.Series:
    """
    Assign the final primary purpose and additional use cases
    for one row in the preliminary inventory. 
    """

    column_name = row["column_name"]
    preliminary_role = row["column_role"]

    if preliminary_role == "requires_review": 
        primary_purpose, additional_use_cases = MANUAL_COLUMN_USAGE[
            column_name
        ]
    
    else:
        primary_purpose = AUTO_ROLE_TO_PRIMARY_PURPOSE.get(
            preliminary_role
        )

        if primary_purpose is None:
            raise ValueError(
                "Unexpexted automatic preliminary role: "
                f"{preliminary_role}"
            )
        
        additional_use_cases = AUTO_ADDITIONAL_USE_CASES.get(
            column_name, 
            "",
        )
    
    return pd.Series(
        {
            "primary_purpose": primary_purpose, 
            "additional_use_cases": normalize_additional_use_cases(
                additional_use_cases
            ),
        }
    )

def validate_column_usage_inventory(
        inventory_df: pd.DataFrame, 
        preliminary_df: pd.DataFrame, 
) -> None:
    """
    Validate the completed machine-readable column-usage inventory. 

    """

    expected_columns = {
        "column_name", 
        "primary_purpose", 
        "additional_use_cases", 
    }

    if set(inventory_df.columns) != expected_columns:
        raise ValueError(
            "Final inventory columns do not match the required schema:"
        )
    
    if len(inventory_df) != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COLUMN_COUNT:,} row but found "
            f"{len(inventory_df):,}."
        )
    
    if inventory_df["column_name"].duplicated().any():
        duplicated_columns = inventory_df.loc[
            inventory_df["column_name"].duplicated(keep=False),
            "column_name", 
        ].tolist()

        raise ValueError(
            "Duplicate column names found in final inventory: "
            f"{duplicated_columns}"
        )
    preliminary_columns = set(
        preliminary_df["column_name"]
    )

    final_columns = set(
        inventory_df["column_name"]
    )

    missing_columns = preliminary_columns.difference(
        final_columns
    )

    unknown_columns = final_columns.difference(
        preliminary_columns
    )

    if missing_columns:
        raise ValueError(
            "Columns missing from final inventory: "
            f"{sorted(missing_columns)}"
        )
    
    if unknown_columns:
        raise ValueError(
            "Unknown column found in final inventory: "
            f"{sorted(unknown_columns)}"
        )
    
    unknown_primary_purpose = set(
        inventory_df["primary_purpose"]
    ).difference(
        ALLOWED_PRIMARY_PURPOSES
    )

    if unknown_primary_purpose:
        raise ValueError(
            "Unknown primary purposes found: "
            f"{sorted(unknown_primary_purpose)}"
        )
    
    for raw_value in inventory_df["additional_use_cases"]:
         normalize_additional_use_cases(raw_value)

    print("Final inventory validation passed.")
    print(f"Columns validated: {len(inventory_df):,}")


def create_column_usage_inventory(
        preliminary_path: Path = PRELIMINARY_INVENTORY_PATH, 
        output_path: Path = COLUMN_USAGE_INVENTORY_PATH, 
) -> pd.DataFrame:
    """
    Create, validate and save the lightweight column-usage inventory. 
    """

    preliminary_df = pd.read_csv(preliminary_path)

    required_columns = {
        "column_name", 
        "column_role", 
    }

    missing_columns = required_columns.difference(preliminary_df.columns)

    if missing_columns:
        raise ValueError(
            "Preliminary inventory is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    
    validate_manual_review_coverage(preliminary_df)

    usage_df = preliminary_df[["column_name"]].copy()

    assigned_usage_df = preliminary_df.apply(
        assign_column_usage, 
        axis=1, 
    )

    usage_df = pd.concat(
        [
            usage_df, 
            assigned_usage_df, 
        ], 
        axis=1,
    )

    validate_column_usage_inventory(
        inventory_df=usage_df, 
        preliminary_df = preliminary_df, 
    )

    output_path.parent.mkdir(
        parents=True, 
        exist_ok = True, 
    )

    usage_df.to_csv(
        output_path, 
        index=False, 
    )

    print(f"\nSaved column-usage inventory to: {output_path}")

    print("\nPrimary-purpose counts: ")
    print(
        usage_df["primary_purpose"]
        .value_counts()
        .to_string()
    )

    return usage_df

if __name__ == "__main__":
    create_column_usage_inventory()