"""
Prepare Reonic training dataset: Clean status quo features and merge with processed targets.

This module loads customer household data (projects_status_quo.csv), applies domain-aware
imputation and feature engineering, then merges with pre-processed target variables
(project_options_processed.csv) to create the final training dataset.

Output: data/reonic_training_ready.csv
- One row per project_id (inner join on existing projects with targets)
- Features (X): Cleaned household characteristics
- Targets (Y): Solar PV, Battery Storage, Inverter Power predictions
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(status_quo_path: str, targets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load status quo features and processed target variables.
    
    Args:
        status_quo_path: Path to projects_status_quo.csv (customer household data)
        targets_path: Path to project_options_processed.csv (pre-processed targets)
        
    Returns:
        Tuple of (status_quo_df, targets_df)
    """
    logger.info(f"Loading status quo data from {status_quo_path}")
    df_status_quo = pd.read_csv(status_quo_path)
    logger.info(f"  Loaded {len(df_status_quo)} rows, {len(df_status_quo.columns)} columns")
    
    logger.info(f"Loading processed targets from {targets_path}")
    df_targets = pd.read_csv(targets_path)
    logger.info(f"  Loaded {len(df_targets)} rows, {len(df_targets.columns)} columns")
    
    return df_status_quo, df_targets


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-aware imputation for missing values.
    
    Strategy:
        - energy_demand_wh: Fill with median (representative energy consumption)
        - heating_existing_electricity_demand_kwh: Fill with 0 (assume no existing heating)
        - house_size_sqm: Fill with median (representative house size)
        
    Args:
        df: Status quo dataframe
        
    Returns:
        Dataframe with imputed values
    """
    df = df.copy()
    
    logger.info("Applying domain-aware imputation...")
    
    # Log missing values before imputation
    missing_before = df[['energy_demand_wh', 'heating_existing_electricity_demand_kwh', 
                         'house_size_sqm']].isna().sum()
    logger.info(f"Missing values before imputation:")
    for col, count in missing_before.items():
        if count > 0:
            logger.info(f"  {col}: {count}")
    
    # Impute energy_demand_wh with median
    if df['energy_demand_wh'].isna().any():
        median_energy = df['energy_demand_wh'].median()
        logger.info(f"  Imputing energy_demand_wh with median: {median_energy:.0f}")
        df['energy_demand_wh'].fillna(median_energy, inplace=True)
    
    # Impute heating_existing_electricity_demand_kwh with 0
    if df['heating_existing_electricity_demand_kwh'].isna().any():
        count = df['heating_existing_electricity_demand_kwh'].isna().sum()
        logger.info(f"  Imputing heating_existing_electricity_demand_kwh with 0 ({count} rows)")
        df['heating_existing_electricity_demand_kwh'].fillna(0, inplace=True)
    
    # Impute house_size_sqm with median
    if df['house_size_sqm'].isna().any():
        median_size = df['house_size_sqm'].median()
        logger.info(f"  Imputing house_size_sqm with median: {median_size:.1f}")
        df['house_size_sqm'].fillna(median_size, inplace=True)
    
    logger.info("Imputation complete")
    return df


def encode_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert boolean columns to integers (0/1).
    
    Converts: has_ev, has_solar, has_storage, has_wallbox
    
    Args:
        df: Dataframe with boolean columns
        
    Returns:
        Dataframe with integer-encoded booleans
    """
    df = df.copy()
    
    boolean_cols = ['has_ev', 'has_solar', 'has_storage', 'has_wallbox']
    
    logger.info(f"Encoding boolean columns to integers (0/1)...")
    for col in boolean_cols:
        if col in df.columns:
            before_type = df[col].dtype
            # Convert boolean or string 'True'/'False' to 0/1
            df[col] = df[col].astype(int)
            logger.info(f"  {col}: {before_type} → int")
    
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific derived features for improved model prediction.
    
    Features:
        - total_estimated_demand_kwh: Baseline demand + EV load + heating
            = (energy_demand_wh / 1000) + (has_ev * 2500 kWh/year) + heating_electricity_demand_kwh
        
    Rationale:
        - EV adds ~2500 kWh/year average charging demand
        - Heating electricity demand already in dataset (existing baselines)
        - Total demand determines optimal battery storage capacity
    
    Args:
        df: Dataframe with required columns
        
    Returns:
        Dataframe with new derived columns
    """
    df = df.copy()
    
    logger.info("Creating derived features...")
    
    # Calculate total estimated annual demand
    ev_demand_kwh = 2500  # Average EV charging demand (kWh/year)
    
    df['total_estimated_demand_kwh'] = (
        (df['energy_demand_wh'] / 1000) +  # Convert Wh to kWh
        (df['has_ev'] * ev_demand_kwh) +   # Add EV load if present
        df['heating_existing_electricity_demand_kwh']  # Add existing heating demand
    )
    
    logger.info(f"  total_estimated_demand_kwh created")
    logger.info(f"    Range: {df['total_estimated_demand_kwh'].min():.0f} - "
                f"{df['total_estimated_demand_kwh'].max():.0f} kWh")
    logger.info(f"    Mean: {df['total_estimated_demand_kwh'].mean():.0f} kWh")
    
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-feature columns and sparse columns with minimal signal.
    
    Removes:
        - Administrative IDs: load_profile_editor_id, customer_contact_id
        - Sparse detail columns: solar_*, storage_*, wallbox_*, heating_* details
          (Keep only boolean flags: has_solar, has_storage, has_wallbox)
        - Cost/price columns (not relevant for system sizing)
        - Request metadata (not features)
    
    Args:
        df: Dataframe with all columns
        
    Returns:
        Dataframe with useless/sparse columns removed
    """
    df = df.copy()
    
    logger.info(f"Removing non-feature and sparse columns...")
    
    # Administrative IDs
    id_cols = ['load_profile_editor_id', 'customer_contact_id']
    
    # Sparse detail columns (>90% missing for non-solar/storage/wallbox projects)
    sparse_cols = [
        'solar_size_kwp', 'solar_angle', 'solar_orientation', 'solar_built_year', 
        'solar_feedin_renumeration', 'solar_feedin_renumeration_post_eeg',
        'storage_size_kwh', 'storage_built_year',
        'wallbox_charge_speed_kw',
        'heating_existing_type', 'heating_existing_cost_per_year', 
        'heating_existing_cost_increase_per_year', 'heating_existing_heating_demand_wh'
    ]
    
    # Cost/price columns (not relevant for system sizing)
    cost_cols = [
        'energy_price_with_flexible_tariff', 'base_price_per_month', 
        'base_price_increase', 'ev_annual_drive_distance_km'
    ]
    
    # Request metadata
    metadata_cols = ['request_created_at']
    
    # House metadata (not predictive)
    house_meta = ['house_built_year']
    
    # Household size (already captured by energy demand)
    household_cols = ['num_inhabitants']
    
    all_drop_cols = id_cols + sparse_cols + cost_cols + metadata_cols + house_meta + household_cols
    
    dropped = []
    for col in all_drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            dropped.append(col)
    
    logger.info(f"  Dropped {len(dropped)} non-predictive columns:")
    for col in dropped:
        logger.info(f"    {col}")
    
    return df


def merge_features_and_targets(df_features: pd.DataFrame, df_targets: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join features (X) with targets (Y) on project_id.
    
    Only keeps projects that exist in BOTH datasets. This ensures clean alignment
    for training: every row has features and corresponding target values.
    
    Args:
        df_features: Cleaned status quo dataframe (X variables)
        df_targets: Processed options dataframe (Y variables)
        
    Returns:
        Merged dataframe indexed by project_id
    """
    logger.info("Merging features and targets on project_id...")
    logger.info(f"  Features: {len(df_features)} projects")
    logger.info(f"  Targets: {len(df_targets)} projects")
    
    # Inner join: keep only projects with both features and targets
    df_merged = df_features.merge(
        df_targets,
        on='project_id',
        how='inner'
    )
    
    logger.info(f"  After inner join: {len(df_merged)} projects (complete pairs)")
    
    # Check for alignment. Use raise (not assert) so the check still runs
    # under `python -O` and survives bytecode optimization.
    if len(df_merged) == 0:
        raise ValueError("No projects matched between features and targets!")
    
    n_feat_only = len(df_features) - len(df_merged)
    n_targ_only = len(df_targets) - len(df_merged)
    
    if n_feat_only > 0:
        logger.warning(f"  {n_feat_only} feature-only projects (no targets)")
    if n_targ_only > 0:
        logger.warning(f"  {n_targ_only} target-only projects (no features)")
    
    return df_merged


def validate_training_data(df: pd.DataFrame) -> None:
    """
    Validate merged training dataframe for quality and completeness.
    
    Checks:
        - Critical columns have no missing values
        - Correct dtypes
        - Positive demand values
        - Project_id uniqueness
        
    Args:
        df: Merged training dataframe
        
    Raises:
        ValueError: If validation fails.
    """
    logger.info("Validating training data...")

    # Check uniqueness
    if df['project_id'].nunique() != len(df):
        raise ValueError("Duplicate project_ids in training data!")
    logger.info(f"  ✓ All project_ids are unique")

    # Check critical columns for missing values (the ones we explicitly imputed + targets)
    critical_cols = ['energy_demand_wh', 'house_size_sqm', 'heating_existing_electricity_demand_kwh',
                     'has_ev', 'has_solar', 'has_storage', 'has_wallbox',
                     'total_estimated_demand_kwh',
                     'total_pv_kwp', 'total_inverter_kw', 'total_battery_kwh']

    missing_critical = df[critical_cols].isna().sum()
    if missing_critical.sum() != 0:
        raise ValueError(
            "Critical columns have missing values:\n"
            f"{missing_critical[missing_critical > 0]}"
        )
    logger.info(f"  ✓ All critical features and targets have no missing values")

    # Check demand is positive
    demand_col = 'total_estimated_demand_kwh'
    if not (df[demand_col] >= 0).all():
        raise ValueError("Negative demand values detected!")
    logger.info(f"  ✓ All demand values are non-negative")

    # Check targets are non-negative
    target_cols = ['total_pv_kwp', 'total_inverter_kw', 'total_battery_kwh']
    for col in target_cols:
        if not (df[col] >= 0).all():
            raise ValueError(f"Negative values in {col}!")
    logger.info(f"  ✓ All target values are non-negative")
    
    # Log any remaining missing values (informational)
    remaining_missing = df.isna().sum()
    if remaining_missing.sum() > 0:
        logger.warning(f"  ⚠ {remaining_missing.sum()} missing values in non-critical columns:")
        for col, count in remaining_missing[remaining_missing > 0].items():
            logger.warning(f"    {col}: {count}")
    
    logger.info("Validation passed!")



def prepare_training_data(
    status_quo_path: str,
    targets_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    End-to-end pipeline: Clean features, create derived variables, merge with targets.
    
    Pipeline:
        1. Load features and targets
        2. Impute missing values (domain-aware)
        3. Encode boolean columns
        4. Create derived features
        5. Drop useless ID columns
        6. Merge features and targets on project_id
        7. Validate and save
        
    Args:
        status_quo_path: Path to projects_status_quo.csv
        targets_path: Path to project_options_processed.csv
        output_path: Path to save merged CSV
        
    Returns:
        Merged training dataframe
    """
    logger.info("=" * 80)
    logger.info("PREPARE: Customer Features + System Targets → Training Dataset")
    logger.info("=" * 80)
    
    # Step 1: Load
    df_features, df_targets = load_data(status_quo_path, targets_path)
    
    # Step 2-5: Clean features
    df_features = impute_missing_values(df_features)
    df_features = encode_boolean_columns(df_features)
    df_features = create_derived_features(df_features)
    df_features = drop_useless_columns(df_features)
    
    # Step 6: Merge
    df_training = merge_features_and_targets(df_features, df_targets)
    
    # Step 7: Validate
    validate_training_data(df_training)
    
    # Save
    df_training.to_csv(output_path, index=False)
    logger.info(f"Saved merged training data to {output_path}")
    
    logger.info("=" * 80)
    logger.info("COMPLETE: Training dataset ready for model input")
    logger.info("=" * 80)
    
    return df_training


if __name__ == '__main__':
    # Define paths
    project_root = Path(__file__).parent.parent
    status_quo_path = project_root / 'CSV' / 'projects_status_quo.csv'
    targets_path = project_root / 'data' / 'project_options_processed.csv'
    output_path = project_root / 'data' / 'reonic_training_ready.csv'
    
    # Prepare training data
    df_training = prepare_training_data(
        status_quo_path=str(status_quo_path),
        targets_path=str(targets_path),
        output_path=str(output_path)
    )
    
    # Display summary
    print("\n" + "=" * 80)
    print("TRAINING DATASET SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(df_training)}")
    print(f"Total features: {len(df_training.columns)}")
    print(f"\nShape: {df_training.shape}")
    
    print(f"\nColumn breakdown:")
    print(f"  Feature columns: {df_training.shape[1] - 5}")  # Minus 5 target/id columns
    print(f"  Target columns: 3 (total_pv_kwp, total_inverter_kw, total_battery_kwh)")
    print(f"  ID columns: 2 (project_id, option_id)")
    
    print(f"\nTarget Variable Statistics:")
    print(df_training[['total_pv_kwp', 'total_inverter_kw', 'total_battery_kwh']].describe())
    
    print(f"\nFeature Sample (first 5 rows):")
    feature_cols = [col for col in df_training.columns 
                   if col not in ['project_id', 'option_id', 'total_pv_kwp', 'total_inverter_kw', 'total_battery_kwh']]
    print(df_training[feature_cols].head())
    
    print("\n✓ Training dataset ready at: " + str(output_path))
