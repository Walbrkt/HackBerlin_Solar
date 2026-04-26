"""
Reonic System Designer: Multi-Output Regression Model Training

This module trains a RandomForest-based multi-output regressor to predict
optimal renewable energy system components (PV capacity, battery storage, inverter power)
from customer household features.

Pipeline:
    1. Load training data (reonic_training_ready.csv)
    2. Select features and targets
    3. Train/test split (80/20, random_state=42)
    4. StandardScaler + RandomForestRegressor pipeline
    5. Evaluate with per-target metrics (MAE, RMSE, R²)
    6. Extract and visualize feature importances
    7. Save trained model and scaler

Target Distribution (note sparse PV/Inverter):
    - total_battery_kwh: 99.6% non-zero (primary target, well-populated)
    - total_pv_kwp: 0.3% non-zero (sparse, only 3 projects)
    - total_inverter_kw: 0.5% non-zero (sparse, only 6 projects)

The sparse distribution is intentional: customer base is almost entirely
energy storage-focused with minimal PV/Inverter adoption.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
TRAINING_DATA_PATH = DATA_DIR / "reonic_training_ready.csv"
MODEL_OUTPUT_PATH = MODEL_DIR / "reonic_system_designer.joblib"
SCALER_OUTPUT_PATH = MODEL_DIR / "feature_scaler.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.txt"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "battery_feature_importance.png"

# Create model directory if it doesn't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data(path: str) -> pd.DataFrame:
    """
    Load the merged training dataset.
    
    Args:
        path: Path to reonic_training_ready.csv
        
    Returns:
        DataFrame with customer features and system targets
    """
    logger.info(f"Loading training data from {path}")
    df = pd.read_csv(path)
    logger.info(f"  Loaded {len(df)} samples × {len(df.columns)} columns")
    return df


def prepare_features_and_targets(
    df: pd.DataFrame,
    feature_names: list,
    target_names: list
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Extract feature matrix (X) and target matrix (Y) from training data.
    
    Args:
        df: Training dataframe
        feature_names: List of feature column names to use
        target_names: List of target column names (Y variables)
        
    Returns:
        Tuple of (X, Y, feature_names) ready for modeling
        
    Raises:
        ValueError: If any required columns are missing
    """
    logger.info(f"Preparing features and targets...")
    
    # Verify all required columns exist
    missing_cols = [col for col in feature_names + target_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in training data: {missing_cols}")
    
    X = df[feature_names].copy()
    Y = df[target_names].copy()
    
    logger.info(f"  Features (X): {X.shape} - {feature_names}")
    logger.info(f"  Targets (Y): {Y.shape}")
    logger.info(f"    total_battery_kwh: non-zero={Y['total_battery_kwh'].gt(0).sum()} ({100*Y['total_battery_kwh'].gt(0).sum()/len(Y):.1f}%)")
    logger.info(f"    total_pv_kwp: non-zero={Y['total_pv_kwp'].gt(0).sum()} ({100*Y['total_pv_kwp'].gt(0).sum()/len(Y):.1f}%)")
    logger.info(f"    total_inverter_kw: non-zero={Y['total_inverter_kw'].gt(0).sum()} ({100*Y['total_inverter_kw'].gt(0).sum()/len(Y):.1f}%)")
    
    return X, Y, feature_names


def split_data(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Uses battery capacity as stratification variable to ensure
    both train/test have representative battery size distributions.
    
    Args:
        X: Feature matrix
        Y: Target matrix
        test_size: Proportion for test set (default 0.2 = 80/20)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, Y_train, Y_test)
    """
    logger.info(f"Splitting data (test_size={test_size}, random_state={random_state})...")
    
    # Stratify by battery capacity quartiles to ensure balanced distribution
    # (99.6% have battery, so this primarily ensures good battery distribution)
    Y_battery_quartiles = pd.qcut(Y['total_battery_kwh'], q=4, labels=False, duplicates='drop')
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=random_state,
        stratify=Y_battery_quartiles
    )
    
    logger.info(f"  Train set: {X_train.shape}")
    logger.info(f"  Test set: {X_test.shape}")
    
    return X_train, X_test, Y_train, Y_test


def train_model(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    random_state: int = 42
) -> Tuple[StandardScaler, RandomForestRegressor]:
    """
    Train a multi-output regression model with feature scaling.
    
    Pipeline:
        1. StandardScaler: Scale features to mean=0, std=1
        2. RandomForestRegressor: Multi-output ensemble learner
        
    RandomForest is ideal for this task because:
        - Natively handles multi-output targets
        - Robust to sparse target distribution (PV/Inverter sparsity)
        - Captures non-linear relationships in demand→system mapping
        - Provides feature importance rankings
    
    Args:
        X_train: Training features
        Y_train: Training targets
        random_state: Random seed
        
    Returns:
        Tuple of (scaler, trained_model)
    """
    logger.info("=" * 80)
    logger.info("TRAINING: StandardScaler + RandomForestRegressor")
    logger.info("=" * 80)
    
    # Step 1: Feature Scaling
    logger.info("Step 1: Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logger.info(f"  Scaled features to mean=0, std=1")
    
    # Step 2: Random Forest Training
    logger.info("Step 2: Training RandomForestRegressor (n_estimators=100)...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=0
    )
    model.fit(X_train_scaled, Y_train)
    logger.info(f"  RandomForest trained successfully")
    
    logger.info("=" * 80)
    
    return scaler, model


def evaluate_model(
    model: RandomForestRegressor,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    target_names: list
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance on test set with per-target metrics.
    
    Calculates:
        - Mean Absolute Error (MAE): Average absolute deviation
        - Root Mean Squared Error (RMSE): Penalizes large errors
        - R² Score: Proportion of variance explained (1.0 = perfect)
    
    Args:
        model: Trained RandomForestRegressor
        scaler: Fitted StandardScaler
        X_test: Test features
        Y_test: Test targets
        target_names: List of target variable names
        
    Returns:
        Dictionary with per-target metrics:
        {
            'total_battery_kwh': {'mae': X.XX, 'rmse': X.XX, 'r2': X.XX},
            'total_pv_kwp': {...},
            'total_inverter_kw': {...}
        }
    """
    logger.info("=" * 80)
    logger.info("EVALUATION: Per-Target Metrics on Test Set")
    logger.info("=" * 80)
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Generate predictions
    Y_pred = model.predict(X_test_scaled)
    
    # Calculate per-target metrics
    metrics = {}
    
    for idx, target_name in enumerate(target_names):
        y_true = Y_test[target_name].values
        y_pred = Y_pred[:, idx]
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics[target_name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        # Log results with context for sparse targets
        logger.info(f"\n{target_name}:")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²:   {r2:.4f}")
        
        # Add warning for sparse targets
        non_zero_count = (y_true > 0).sum()
        zero_pct = 100 * (1 - non_zero_count / len(y_true))
        if zero_pct > 90:
            logger.warning(f"  ⚠ {zero_pct:.1f}% of test values are zero (sparse target)")
    
    logger.info("=" * 80)
    
    return metrics


def extract_feature_importance(
    model: RandomForestRegressor,
    feature_names: list
) -> Dict[str, np.ndarray]:
    """
    Extract feature importances from the Random Forest model.
    
    Random Forests provide feature importance as the mean decrease in impurity
    (Gini importance) across all trees and splits.
    
    Args:
        model: Trained RandomForestRegressor
        feature_names: List of feature names (in same order as training)
        
    Returns:
        Dictionary mapping target names to importance arrays
    """
    logger.info("Extracting feature importances from Random Forest...")
    
    # RandomForestRegressor for multi-output returns n_outputs importance arrays
    importances = model.feature_importances_
    
    # If multi-output, importances is 2D (n_features, n_outputs)
    # Average across outputs for overall feature importance
    if importances.ndim > 1:
        feature_importance_avg = importances.mean(axis=1)
    else:
        feature_importance_avg = importances
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance_avg
    }).sort_values('importance', ascending=True)
    
    logger.info(f"  Top 5 features by importance:")
    for idx, row in importance_df.tail(5).iterrows():
        logger.info(f"    {row['feature']}: {row['importance']:.4f}")
    
    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: str,
    title: str = "Feature Importance for Battery Storage Sizing"
) -> None:
    """
    Create and save a horizontal bar chart of feature importances.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        output_path: Path to save the figure
        title: Chart title
    """
    logger.info(f"Creating feature importance chart...")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x='importance',
        y='feature',
        palette='viridis'
    )
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  Saved to {output_path}")
    plt.close()


def save_model_artifacts(
    model: RandomForestRegressor,
    scaler: StandardScaler,
    feature_names: list,
    model_path: str,
    scaler_path: str,
    feature_list_path: str
) -> None:
    """
    Save trained model and preprocessing artifacts.
    
    Saves:
        1. Trained RandomForestRegressor (joblib)
        2. Fitted StandardScaler (joblib)
        3. Feature names list (text file for reference)
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature names
        model_path: Output path for model
        scaler_path: Output path for scaler
        feature_list_path: Output path for feature names
    """
    logger.info("Saving model artifacts...")
    
    # Save model
    joblib.dump(model, model_path)
    logger.info(f"  Saved model to {model_path}")
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Saved scaler to {scaler_path}")
    
    # Save feature list
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(feature_names))
    logger.info(f"  Saved feature list to {feature_list_path}")


def main():
    """
    Execute complete training pipeline:
    Load → Prepare → Split → Train → Evaluate → Save
    """
    logger.info("\n" + "=" * 80)
    logger.info("REONIC SYSTEM DESIGNER: Multi-Output Regression Training Pipeline")
    logger.info("=" * 80 + "\n")
    
    # Define features and targets (as specified in requirements)
    feature_names = [
        'total_estimated_demand_kwh',
        'energy_demand_wh',
        'has_ev',
        'has_solar',
        'has_storage',
        'has_wallbox',
        'house_size_sqm',
        'heating_existing_electricity_demand_kwh'
    ]
    
    target_names = [
        'total_battery_kwh',
        'total_pv_kwp',
        'total_inverter_kw'
    ]
    
    # Load data
    df = load_training_data(str(TRAINING_DATA_PATH))
    
    # Prepare X and Y
    X, Y, features = prepare_features_and_targets(df, feature_names, target_names)
    
    # Train/test split
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.2, random_state=42)
    
    # Train model
    scaler, model = train_model(X_train, Y_train, random_state=42)
    
    # Evaluate
    metrics = evaluate_model(model, scaler, X_test, Y_test, target_names)
    
    # Feature importance
    importance_df = extract_feature_importance(model, features)
    plot_feature_importance(importance_df, str(FEATURE_IMPORTANCE_PATH))
    
    # Save artifacts
    save_model_artifacts(model, scaler, features, 
                        str(MODEL_OUTPUT_PATH), 
                        str(SCALER_OUTPUT_PATH),
                        str(FEATURE_LIST_PATH))
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ TRAINING COMPLETE: Model ready for inference")
    logger.info("=" * 80)
    logger.info(f"\nModel artifacts saved to: {MODEL_DIR}")
    logger.info(f"  - Model: {MODEL_OUTPUT_PATH.name}")
    logger.info(f"  - Scaler: {SCALER_OUTPUT_PATH.name}")
    logger.info(f"  - Features: {FEATURE_LIST_PATH.name}")
    logger.info(f"  - Chart: {FEATURE_IMPORTANCE_PATH.name}\n")
    
    return model, scaler, metrics


if __name__ == "__main__":
    model, scaler, metrics = main()
