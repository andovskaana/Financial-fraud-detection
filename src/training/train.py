"""
Training Pipeline for Fraud Detection Model

This script:
1. Loads and explores the dataset
2. Performs feature engineering
3. Splits data (60% train, 20% test, 20% streaming holdout)
4. Trains XGBoost/LightGBM model with class imbalance handling
5. Evaluates and finds optimal threshold
6. Saves model and config
"""
import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import XGBoost or LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.features import (
    FeatureConfig, ColumnDetector, BatchFeatureEngineer
)
from src.training.evaluate import FraudModelEvaluator, print_class_distribution
from src.training.ensemble import EnsembleClassifier  # import ensemble

warnings.filterwarnings('ignore')


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load dataset with automatic column detection."""
    print(f"\n{'='*60}")
    print(" Loading Dataset")
    print(f"{'='*60}")

    # Determine file type and load
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


def explore_dataset(df: pd.DataFrame, config: FeatureConfig):
    """Perform basic EDA."""
    print(f"\n{'='*60}")
    print(" Exploratory Data Analysis")
    print(f"{'='*60}")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': missing_pct
    }).sort_values('missing_count', ascending=False)

    print("\nMissing values:")
    print(missing_df[missing_df['missing_count'] > 0])

    # Target distribution
    if config.target_col in df.columns:
        print(f"\nTarget variable distribution ({config.target_col}):")
        target = df[config.target_col].fillna(0).astype(int)
        print(target.value_counts())
        print(f"Fraud rate: {target.mean()*100:.4f}%")

    # Amount statistics
    if config.amount_col in df.columns:
        print(f"\nAmount statistics:")
        print(df[config.amount_col].describe())

    # Timestamp range
    if config.timestamp_col in df.columns:
        df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], errors='coerce')
        print(f"\nTimestamp range:")
        print(f"  Min: {df[config.timestamp_col].min()}")
        print(f"  Max: {df[config.timestamp_col].max()}")

    # Unique users
    if config.sender_col in df.columns:
        print(f"\nUnique senders: {df[config.sender_col].nunique():,}")

    if config.receiver_col and config.receiver_col in df.columns:
        print(f"Unique receivers: {df[config.receiver_col].nunique():,}")


def prepare_target(df: pd.DataFrame, config: FeatureConfig) -> pd.Series:
    """Prepare target variable."""
    target = df[config.target_col].fillna(0)

    # Handle different possible formats
    if target.dtype == 'object':
        target = target.map({'True': 1, 'False': 0, 'true': 1, 'false': 0,
                             'yes': 1, 'no': 0, 'Yes': 1, 'No': 0,
                             '1': 1, '0': 0}).fillna(0)

    return target.astype(int)


def split_data(
    df: pd.DataFrame,
    y: pd.Series,
    config: FeatureConfig,
    test_size: float = 0.2,
    holdout_size: float = 0.2,
    random_state: int = 42,
    use_time_split: bool = True
) -> tuple:
    """
    Split data into train, test, and streaming holdout sets.

    Split ratios:
    - 60% train
    - 20% test
    - 20% streaming holdout (for simulation)
    """
    print(f"\n{'='*60}")
    print(" Splitting Data")
    print(f"{'='*60}")

    if use_time_split and config.timestamp_col in df.columns:
        # Time-based split
        print("Using time-based split...")
        df_sorted = df.sort_values(config.timestamp_col).reset_index(drop=True)
        y_sorted = y.iloc[df_sorted.index].reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(n * 0.6)
        test_end = int(n * 0.8)

        df_train = df_sorted.iloc[:train_end].copy()
        y_train = y_sorted.iloc[:train_end].copy()

        df_test = df_sorted.iloc[train_end:test_end].copy()
        y_test = y_sorted.iloc[train_end:test_end].copy()

        df_holdout = df_sorted.iloc[test_end:].copy()
        y_holdout = y_sorted.iloc[test_end:].copy()
    else:
        # Random split
        print("Using random split...")

        # First split: train + temp (test + holdout)
        df_train, df_temp, y_train, y_temp = train_test_split(
            df, y,
            test_size=(test_size + holdout_size),
            random_state=random_state,
            stratify=y
        )

        # Second split: test and holdout
        relative_holdout = holdout_size / (test_size + holdout_size)
        df_test, df_holdout, y_test, y_holdout = train_test_split(
            df_temp, y_temp,
            test_size=relative_holdout,
            random_state=random_state,
            stratify=y_temp
        )

    print(f"\nTrain set:    {len(df_train):,} samples ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Test set:     {len(df_test):,} samples ({len(df_test)/len(df)*100:.1f}%)")
    print(f"Holdout set:  {len(df_holdout):,} samples ({len(df_holdout)/len(df)*100:.1f}%)")

    print_class_distribution(y_train.values, "Train")
    print_class_distribution(y_test.values, "Test")
    print_class_distribution(y_holdout.values, "Holdout")

    return df_train, df_test, df_holdout, y_train, y_test, y_holdout


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'xgboost',
    scale_pos_weight: float = None
):
    """
    Train the fraud detection model.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: 'xgboost', 'lightgbm', 'random_forest', or 'ensemble'
        scale_pos_weight: Weight for positive class (computed if None)
    """
    print(f"\n{'='*60}")
    print(f" Training {model_type.upper()} Model")
    print(f"{'='*60}")

    # Compute class weight if not provided
    if scale_pos_weight is None:
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / max(1, pos_count)
        print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")

    if model_type == 'xgboost' and HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    elif model_type == 'lightgbm' and HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
    elif model_type == 'ensemble':
        # Ensemble: train multiple base models and average their probabilities
        models = []
        # Train XGBoost if available
        if HAS_XGB:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)
            models.append(xgb_model)
        # Train LightGBM if available
        if HAS_LGB:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            models.append(lgb_model)
        # Always include RandomForest as fallback
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models.append(rf_model)
        # Wrap base models into ensemble classifier
        model = EnsembleClassifier(models)
    else:
        print(f"Falling back to RandomForest...")
        # For RandomForest, use class_weight
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

    print("Training complete!")
    return model


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from model."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return pd.DataFrame()

    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return fi_df


def save_artifacts(
    model,
    config: FeatureConfig,
    output_dir: str,
    model_name: str = 'fraud_model',
    extra_metadata: dict = None
):
    """Save model and configuration artifacts."""
    print(f"\n{'='*60}")
    print(" Saving Artifacts")
    print(f"{'='*60}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_path / f'{model_name}.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save config
    config_path = output_path / 'feature_config.json'
    config.save(str(config_path))
    print(f"Config saved to: {config_path}")

    # Save metadata
    metadata = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'feature_count': len(config.feature_columns),
        'anomaly_threshold': config.anomaly_threshold,
        'trained_at': datetime.now().isoformat(),
        'feature_columns': config.feature_columns
    }

    # If ensemble, record base model names
    if hasattr(model, 'models'):
        metadata['base_models'] = [type(m).__name__ for m in model.models]

    # Merge extra metadata (e.g., amount statistics)
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = output_path / 'model_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    return model_path, config_path


def save_holdout_for_streaming(
    df_holdout: pd.DataFrame,
    output_dir: str,
    filename: str = 'streaming_holdout.csv'
):
    """Save holdout set for streaming simulation."""
    output_path = Path(output_dir) / filename
    df_holdout.to_csv(output_path, index=False)
    print(f"Holdout set saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Train Fraud Detection Model')
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='models/data/financial_fraud_detection_dataset.csv',
        help='Path to dataset CSV'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Output directory for model artifacts'
    )
    parser.add_argument(
        '--model-type', '-m',
        type=str,
        choices=['xgboost', 'lightgbm', 'random_forest', 'ensemble'],
        default='xgboost',
        help='Model type to train'
    )
    parser.add_argument(
        '--threshold-method',
        type=str,
        choices=['f2', 'cost', 'recall_at_precision'],
        default='f2',
        help='Method for finding optimal threshold'
    )
    parser.add_argument(
        '--min-precision',
        type=float,
        default=0.5,
        help='Minimum precision for recall_at_precision method'
    )
    parser.add_argument(
        '--skip-velocity',
        action='store_true',
        help='Skip velocity feature computation (faster but less accurate)'
    )

    args = parser.parse_args()

    # Load data
    df = load_dataset(args.data)

    # Detect columns
    config = ColumnDetector.detect_columns(df)
    print(f"\nDetected columns:")
    print(f"  Transaction ID: {config.transaction_id_col}")
    print(f"  Timestamp: {config.timestamp_col}")
    print(f"  Sender: {config.sender_col}")
    print(f"  Amount: {config.amount_col}")
    print(f"  Target: {config.target_col}")
    if config.receiver_col:
        print(f"  Receiver: {config.receiver_col}")
    if config.transaction_type_col:
        print(f"  Transaction Type: {config.transaction_type_col}")
    if config.categorical_cols:
        print(f"  Additional categoricals: {config.categorical_cols}")

    # EDA
    explore_dataset(df, config)

    # Prepare target
    y = prepare_target(df, config)

    # Split data BEFORE feature engineering (to avoid leakage)
    df_train, df_test, df_holdout, y_train, y_test, y_holdout = split_data(
        df, y, config
    )

    # Feature engineering
    print(f"\n{'='*60}")
    print(" Feature Engineering")
    print(f"{'='*60}")

    feature_engineer = BatchFeatureEngineer(config)

    # Fit on training data
    df_train_fe, feature_cols = feature_engineer.fit_transform(df_train)

    # Transform test and holdout (no fitting)
    df_test_fe = feature_engineer.transform(df_test)
    df_holdout_fe = feature_engineer.transform(df_holdout)

    # Prepare feature matrices
    X_train = df_train_fe[feature_cols].values
    X_test = df_test_fe[feature_cols].values

    print(f"\nFeature matrix shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")

    # Train model
    model = train_model(X_train, y_train.values, args.model_type)

    # Evaluate
    print(f"\n{'='*60}")
    print(" Model Evaluation")
    print(f"{'='*60}")

    evaluator = FraudModelEvaluator(
        false_negative_cost=10.0,
        false_positive_cost=1.0
    )

    # Get predictions
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    print("\nFinding optimal threshold...")
    optimal_threshold, optimal_metrics = evaluator.find_optimal_threshold(
        y_test.values,
        y_test_pred,
        method=args.threshold_method,
        min_precision=args.min_precision if args.threshold_method == 'recall_at_precision' else None
    )

    print(f"Optimal threshold ({args.threshold_method}): {optimal_threshold:.3f}")

    # Update config with threshold
    config.anomaly_threshold = float(optimal_threshold)

    # Print evaluation reports
    evaluator.print_evaluation_report(
        y_train.values, y_train_pred,
        threshold=optimal_threshold,
        title="Training Set Evaluation"
    )

    evaluator.print_evaluation_report(
        y_test.values, y_test_pred,
        threshold=optimal_threshold,
        title="Test Set Evaluation"
    )

    # Feature importance
    print(f"\n{'='*60}")
    print(" Feature Importance (Top 20)")
    print(f"{'='*60}")

    fi_df = get_feature_importance(model, feature_cols)
    if not fi_df.empty:
        print(fi_df.head(20).to_string(index=False))

    # Save plots
    plots_dir = Path(args.output) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    evaluator.plot_precision_recall_curve(
        y_test.values, y_test_pred,
        save_path=str(plots_dir / 'pr_curve.png')
    )

    evaluator.plot_threshold_analysis(
        y_test.values, y_test_pred,
        save_path=str(plots_dir / 'threshold_analysis.png')
    )

    evaluator.plot_confusion_matrix(
        y_test.values, y_test_pred,
        threshold=optimal_threshold,
        save_path=str(plots_dir / 'confusion_matrix.png')
    )

    # Compute amount statistics for metadata (average, min, max)
    amount_stats = {}
    if config.amount_col in df.columns:
        try:
            amounts = df[config.amount_col].astype(float)
            amount_stats = {
                'average_amount': float(amounts.mean()),
                'min_amount': float(amounts.min()),
                'max_amount': float(amounts.max())
            }
        except Exception:
            amount_stats = {}

    # Save artifacts
    save_artifacts(model, config, args.output, extra_metadata=amount_stats)

    # Save holdout for streaming simulation (original data, not feature-engineered)
    data_dir = Path(args.output) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    save_holdout_for_streaming(df_holdout, str(data_dir))

    print(f"\n{'='*60}")
    print(" Training Complete!")
    print(f"{'='*60}")
    print(f"\nArtifacts saved to: {args.output}")
    print(f"Use threshold {optimal_threshold:.3f} for anomaly detection")
    print(f"Holdout set ({len(df_holdout):,} samples) ready for streaming simulation")


if __name__ == '__main__':
    main()