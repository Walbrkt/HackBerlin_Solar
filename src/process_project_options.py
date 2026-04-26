"""
Process project_options_parts.csv to extract ground truth renewable energy system designs.

This module aggregates line-item components (PV modules, batteries, inverters) into
project-level system specifications, filtering for the maximum viable design per project.

Key Concept:
    Each project_id may have multiple option_ids representing different design proposals
    (e.g., "Good", "Better", "Best"). We retain only the option with maximum PV capacity
    to train the model on "maximized designs" that the UI can later scale down.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_project_options(csv_path: str) -> pd.DataFrame:
    """
    Load project options CSV file.
    
    Args:
        csv_path: Path to project_options_parts.csv
        
    Returns:
        DataFrame with columns: project_id, option_id, option_number, technology,
                                line_item_function, component_type, component_name,
                                quantity, and technical specifications.
    """
    logger.info(f"Loading project options from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def filter_hardware_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to include only hardware components, excluding labor and service fees.
    
    Retained component types:
        - Module: PV modules
        - Inverter: Inverter equipment
        - BatteryStorage: Battery storage units
        
    Excluded component types:
        - InstallationFee, ServiceFee: Labor and service costs
        - ModuleFrameConstruction, AccessoryToModule, etc.: Installation components
        
    Args:
        df: DataFrame with all line items
        
    Returns:
        Filtered DataFrame containing only hardware components
    """
    hardware_types = ['Module', 'Inverter', 'BatteryStorage']
    df_filtered = df[df['component_type'].isin(hardware_types)].copy()
    
    logger.info(f"Filtered to {len(df_filtered)} hardware rows from {len(df)} total rows")
    logger.info(f"Hardware breakdown: {df_filtered['component_type'].value_counts().to_dict()}")
    
    return df_filtered


def extract_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract technical metrics for each hardware component.
    
    Data mapping (note: some column names appear swapped in source data):
        - Module rows: module_watt_peak → PV capacity in W
        - Inverter rows: battery_capacity_kwh → Inverter capacity in Wh (convert to kW)
        - BatteryStorage rows: inverter_power_kw → Battery capacity in Wh (convert to kWh)
    
    Args:
        df: Filtered DataFrame with only hardware components
        
    Returns:
        DataFrame with normalized columns: project_id, option_id, component_type,
                                          pv_w, inverter_w, battery_wh
    """
    df_metrics = df[['project_id', 'option_id', 'quantity', 'component_type',
                     'module_watt_peak', 'battery_capacity_kwh', 'inverter_power_kw']].copy()
    
    # Initialize metric columns
    df_metrics['pv_w'] = 0.0
    df_metrics['inverter_w'] = 0.0
    df_metrics['battery_wh'] = 0.0
    
    # Extract metrics per component type (accounting for swapped column names in source)
    pv_mask = df_metrics['component_type'] == 'Module'
    df_metrics.loc[pv_mask, 'pv_w'] = (
        df_metrics.loc[pv_mask, 'quantity'] * 
        df_metrics.loc[pv_mask, 'module_watt_peak']
    )
    logger.info(f"Extracted PV metrics from {pv_mask.sum()} Module rows")
    
    inv_mask = df_metrics['component_type'] == 'Inverter'
    df_metrics.loc[inv_mask, 'inverter_w'] = (
        df_metrics.loc[inv_mask, 'battery_capacity_kwh']  # Column contains Wh values
    )
    logger.info(f"Extracted Inverter metrics from {inv_mask.sum()} Inverter rows")
    
    bat_mask = df_metrics['component_type'] == 'BatteryStorage'
    df_metrics.loc[bat_mask, 'battery_wh'] = (
        df_metrics.loc[bat_mask, 'inverter_power_kw']  # Column contains Wh values
    )
    logger.info(f"Extracted Battery metrics from {bat_mask.sum()} BatteryStorage rows")
    
    return df_metrics


def aggregate_by_option(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics to option level (project_id + option_id).
    
    For each project option, sums all component quantities and capacities:
        - total_pv_kwp: Sum of PV module wattages in kW-peak
        - total_inverter_kw: Sum of inverter capacities in kW
        - total_battery_kwh: Sum of battery capacities in kWh
    
    Args:
        df: DataFrame with extracted metrics
        
    Returns:
        Aggregated DataFrame with one row per (project_id, option_id)
    """
    agg_spec = {
        'pv_w': 'sum',
        'inverter_w': 'sum',
        'battery_wh': 'sum'
    }
    
    df_agg = df.groupby(['project_id', 'option_id'], as_index=False).agg(agg_spec)
    
    # Convert units: W → kW, Wh → kWh
    df_agg['total_pv_kwp'] = df_agg['pv_w'] / 1000.0
    df_agg['total_inverter_kw'] = df_agg['inverter_w'] / 1000.0
    df_agg['total_battery_kwh'] = df_agg['battery_wh'] / 1000.0
    
    # Drop intermediate columns
    df_agg = df_agg.drop(columns=['pv_w', 'inverter_w', 'battery_wh'])
    
    logger.info(f"Aggregated to {len(df_agg)} option-level records")
    logger.info(f"PV capacity range: {df_agg['total_pv_kwp'].min():.2f} - "
                f"{df_agg['total_pv_kwp'].max():.2f} kWp")
    logger.info(f"Battery capacity range: {df_agg['total_battery_kwh'].min():.2f} - "
                f"{df_agg['total_battery_kwh'].max():.2f} kWh")
    logger.info(f"Inverter capacity range: {df_agg['total_inverter_kw'].min():.2f} - "
                f"{df_agg['total_inverter_kw'].max():.2f} kW")
    
    return df_agg


def filter_maximum_pv_option(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only the option with maximum PV capacity per project.
    
    Rationale:
        Projects may have multiple design proposals (Good/Better/Best). We train the model
        on the "maximized viable system" representing the largest capacity the roof can support.
        The UI later offers scaled-down variants to customers at different price points.
    
    Args:
        df: Aggregated option-level DataFrame
        
    Returns:
        DataFrame with one row per project_id (the option with max total_pv_kwp)
    """
    # Find the option_id with max PV capacity for each project
    idx = df.groupby('project_id')['total_pv_kwp'].idxmax()
    df_max = df.loc[idx].reset_index(drop=True)
    
    # Verify we have one row per project. Use raise (not assert) so the check
    # still runs under `python -O` and survives bytecode optimization.
    n_projects = len(df['project_id'].unique())
    if len(df_max) != n_projects:
        raise ValueError(
            f"Expected {n_projects} rows after filtering max options, got {len(df_max)}"
        )
    
    logger.info(f"Filtered to {len(df_max)} projects (max PV option per project)")
    
    # Show statistics on discarded options
    df_discarded = df.loc[~df.index.isin(idx)]
    logger.info(f"Discarded {len(df_discarded)} alternative options")
    if len(df_discarded) > 0:
        logger.info(f"Discarded PV capacity: mean={df_discarded['total_pv_kwp'].mean():.2f} kWp, "
                    f"median={df_discarded['total_pv_kwp'].median():.2f} kWp")
    
    return df_max


def validate_output(df: pd.DataFrame) -> None:
    """
    Validate output DataFrame for data quality.
    
    Checks:
        - One row per unique project_id
        - No missing values in target metrics
        - Positive metric values
    
    Args:
        df: Output DataFrame
        
    Raises:
        ValueError: If validation fails.
    """
    n_rows = len(df)
    n_unique_projects = len(df['project_id'].unique())

    if n_rows != n_unique_projects:
        raise ValueError(
            f"Expected {n_unique_projects} unique projects, got {n_rows} rows"
        )

    metric_cols = ['total_pv_kwp', 'total_inverter_kw', 'total_battery_kwh']

    for col in metric_cols:
        missing = df[col].isna().sum()
        if missing != 0:
            raise ValueError(f"Column '{col}' has {missing} missing values")

        # Warnings for zero or very small values (may indicate data quality issues)
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            logger.warning(f"Column '{col}' has {zero_count} zero values")

        # Check for negative values
        negative = (df[col] < 0).sum()
        if negative != 0:
            raise ValueError(f"Column '{col}' has {negative} negative values")
    
    logger.info("Validation passed: clean output with one row per project")


def process_project_options(csv_path: str, output_path: str | None = None) -> pd.DataFrame:
    """
    End-to-end pipeline to process project options into training data.
    
    Pipeline:
        1. Load CSV
        2. Filter to hardware components only
        3. Extract technical metrics
        4. Aggregate by option (project_id + option_id)
        5. Keep maximum PV option per project
        6. Validate output
        7. Optionally save to CSV
    
    Args:
        csv_path: Path to project_options_parts.csv
        output_path: Optional path to save processed dataframe
        
    Returns:
        Processed DataFrame with columns: project_id, option_id, total_pv_kwp,
                                          total_inverter_kw, total_battery_kwh
    """
    logger.info("=" * 70)
    logger.info("PROCESS: Project Options → Ground Truth System Designs")
    logger.info("=" * 70)
    
    # Execute pipeline
    df = load_project_options(csv_path)
    df_hardware = filter_hardware_components(df)
    df_metrics = extract_metrics(df_hardware)
    df_agg = aggregate_by_option(df_metrics)
    df_final = filter_maximum_pv_option(df_agg)
    
    # Validate
    validate_output(df_final)
    
    # Save if requested
    if output_path:
        df_final.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
    
    logger.info("=" * 70)
    logger.info("COMPLETE: Project options processing pipeline")
    logger.info("=" * 70)
    
    return df_final


if __name__ == '__main__':
    # Define paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / 'CSV' / 'project_options_parts.csv'
    output_path = project_root / 'data' / 'project_options_processed.csv'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process
    df_targets = process_project_options(
        csv_path=str(csv_path),
        output_path=str(output_path)
    )
    
    # Display summary
    print("\n" + "=" * 70)
    print("OUTPUT SUMMARY")
    print("=" * 70)
    print(f"Total projects: {len(df_targets)}")
    print(f"\nTarget Variable Statistics:\n")
    print(df_targets[['total_pv_kwp', 'total_inverter_kw', 'total_battery_kwh']].describe())
    print(f"\nFirst 5 rows:\n")
    print(df_targets.head())
