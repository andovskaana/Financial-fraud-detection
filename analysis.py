"""
analysis.py

EDA/statistics script for the financial fraud detection project.

Covers R2:
- Fraud / non-fraud counts
- Fraud rate %
- Amount avg/min/max
- Transaction type distribution
- Feature usefulness:
    1. Mutual information scores
    2. Correlation with fraud target
- Saves plots and CSV reports
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

from src.training.features import ColumnDetector


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError("Unsupported dataset format. Use CSV or Parquet.")


def prepare_target(series: pd.Series) -> pd.Series:
    """
    Converts different fraud label formats into 0/1.
    Handles booleans, ints, floats, and strings.
    """
    target = series.copy()

    if target.dtype == "object":
        target = (
            target.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "fraud": 1,
                "not_fraud": 0,
                "normal": 0,
                "1": 1,
                "0": 0,
            })
        )

    return target.fillna(0).astype(int)


def save_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def encode_features_for_analysis(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Creates an analysis-friendly feature matrix.

    Numeric columns are kept as numeric.
    Boolean columns are converted to 0/1.
    Low-cardinality categorical columns are factorized.
    High-cardinality ID-like columns should already be excluded before this.
    """
    X = pd.DataFrame(index=df.index)

    for col in feature_cols:
        s = df[col]

        if pd.api.types.is_bool_dtype(s):
            X[col] = s.astype(int)

        elif pd.api.types.is_numeric_dtype(s):
            X[col] = pd.to_numeric(s, errors="coerce")

        elif pd.api.types.is_datetime64_any_dtype(s):
            X[f"{col}_hour"] = s.dt.hour
            X[f"{col}_day_of_week"] = s.dt.dayofweek
            X[f"{col}_month"] = s.dt.month

        else:
            # Factorize only reasonable categorical columns
            nunique = s.nunique(dropna=True)

            if nunique <= 100:
                X[f"{col}_encoded"] = pd.factorize(s.fillna("missing").astype(str))[0]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def run_analysis(data_path: str, output_dir: str, max_mi_rows: int):
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "analysis_reports"

    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    config = ColumnDetector.detect_columns(df)

    print("=" * 70)
    print("FINANCIAL FRAUD DATASET ANALYSIS")
    print("=" * 70)

    print(f"\nDataset path: {data_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nDetected columns:")
    print(f"  Transaction ID:   {config.transaction_id_col}")
    print(f"  Timestamp:        {config.timestamp_col}")
    print(f"  Sender/User:      {config.sender_col}")
    print(f"  Receiver:         {config.receiver_col}")
    print(f"  Amount:           {config.amount_col}")
    print(f"  Transaction type: {config.transaction_type_col}")
    print(f"  Target:           {config.target_col}")

    # -----------------------------------------------------------------
    # 1. Missing values
    # -----------------------------------------------------------------
    missing_summary = (
        df.isna()
        .sum()
        .to_frame(name="missing_count")
        .assign(missing_percent=lambda x: (x["missing_count"] / len(df)) * 100)
        .sort_values("missing_count", ascending=False)
    )

    print("\n" + "=" * 70)
    print("1. Missing values")
    print("=" * 70)
    print(missing_summary)

    missing_summary.to_csv(reports_dir / "missing_values.csv")

    # -----------------------------------------------------------------
    # 2. Fraud / non-fraud counts and fraud rate
    # -----------------------------------------------------------------
    if config.target_col not in df.columns:
        raise ValueError(f"Target column not found: {config.target_col}")

    y = prepare_target(df[config.target_col])

    fraud_counts = y.value_counts().rename(index={0: "non_fraud", 1: "fraud"})
    fraud_percent = y.value_counts(normalize=True).mul(100).rename(index={0: "non_fraud_%", 1: "fraud_%"})

    print("\n" + "=" * 70)
    print("2. Fraud / non-fraud statistics")
    print("=" * 70)
    print("\nCounts:")
    print(fraud_counts)
    print("\nPercentages:")
    print(fraud_percent.round(4))
    print(f"\nFraud rate: {y.mean() * 100:.4f}%")

    fraud_summary = pd.DataFrame({
        "count": fraud_counts,
        "percentage": fraud_percent.values,
    })
    fraud_summary.to_csv(reports_dir / "fraud_distribution.csv")

    fraud_counts.plot(kind="bar")
    plt.title("Fraud vs Non-Fraud Count")
    plt.xlabel("Class")
    plt.ylabel("Number of transactions")
    save_plot(plots_dir / "fraud_distribution.png")

    # -----------------------------------------------------------------
    # 3. Amount statistics
    # -----------------------------------------------------------------
    if config.amount_col in df.columns:
        amounts = pd.to_numeric(df[config.amount_col], errors="coerce")

        amount_stats = pd.Series({
            "average_amount": amounts.mean(),
            "min_amount": amounts.min(),
            "max_amount": amounts.max(),
            "median_amount": amounts.median(),
            "std_amount": amounts.std(),
        })

        print("\n" + "=" * 70)
        print("3. Amount statistics")
        print("=" * 70)
        print(amount_stats.round(4))

        amount_stats.to_csv(reports_dir / "amount_statistics.csv", header=["value"])

        # Also useful: amount stats by fraud class
        amount_by_class = (
            df.assign(_fraud_label=y, _amount=amounts)
            .groupby("_fraud_label")["_amount"]
            .agg(["count", "mean", "min", "max", "median", "std"])
            .rename(index={0: "non_fraud", 1: "fraud"})
        )

        print("\nAmount statistics by class:")
        print(amount_by_class.round(4))

        amount_by_class.to_csv(reports_dir / "amount_statistics_by_class.csv")

    # -----------------------------------------------------------------
    # 4. Transaction type distribution
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. Transaction type distribution")
    print("=" * 70)

    if config.transaction_type_col and config.transaction_type_col in df.columns:
        type_counts = df[config.transaction_type_col].fillna("missing").value_counts()
        type_percent = df[config.transaction_type_col].fillna("missing").value_counts(normalize=True).mul(100)

        type_distribution = pd.DataFrame({
            "count": type_counts,
            "percentage": type_percent.round(4),
        })

        print(type_distribution)

        type_distribution.to_csv(reports_dir / "transaction_type_distribution.csv")

        type_percent.sort_values(ascending=False).plot(kind="bar")
        plt.title("Transaction Type Distribution")
        plt.xlabel("Transaction type")
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=30, ha="right")
        save_plot(plots_dir / "transaction_type_distribution.png")

        # Bonus: transaction type vs fraud rate
        type_fraud_rate = (
            df.assign(_fraud_label=y)
            .groupby(config.transaction_type_col)["_fraud_label"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "fraud_rate"})
            .sort_values("fraud_rate", ascending=False)
        )

        type_fraud_rate["fraud_rate_percent"] = type_fraud_rate["fraud_rate"] * 100

        print("\nFraud rate by transaction type:")
        print(type_fraud_rate[["count", "fraud_rate_percent"]].round(4))

        type_fraud_rate.to_csv(reports_dir / "transaction_type_fraud_rate.csv")

        type_fraud_rate["fraud_rate_percent"].plot(kind="bar")
        plt.title("Fraud Rate by Transaction Type")
        plt.xlabel("Transaction type")
        plt.ylabel("Fraud rate (%)")
        plt.xticks(rotation=30, ha="right")
        save_plot(plots_dir / "transaction_type_fraud_rate.png")

    else:
        print("No transaction type column was detected.")

    # -----------------------------------------------------------------
    # 5. Basic user statistics
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. User-level transaction statistics")
    print("=" * 70)

    if config.sender_col in df.columns:
        user_stats = (
            df.groupby(config.sender_col)
            .agg(
                transaction_count=(config.sender_col, "count"),
                avg_transaction_amount=(config.amount_col, "mean") if config.amount_col in df.columns else (config.sender_col, "count"),
            )
            .sort_values("transaction_count", ascending=False)
        )

        print(user_stats.head(10))
        user_stats.head(100).to_csv(reports_dir / "top_user_statistics.csv")

    # -----------------------------------------------------------------
    # 6. Datetime features for analysis
    # -----------------------------------------------------------------
    if config.timestamp_col in df.columns:
        df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], errors="coerce")

        df["hour"] = df[config.timestamp_col].dt.hour
        df["day_of_week"] = df[config.timestamp_col].dt.dayofweek
        df["month"] = df[config.timestamp_col].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)

        print("\nDatetime range:")
        print(f"  Min timestamp: {df[config.timestamp_col].min()}")
        print(f"  Max timestamp: {df[config.timestamp_col].max()}")

    # -----------------------------------------------------------------
    # 7. Feature usefulness: mutual information + correlation
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. Feature usefulness")
    print("=" * 70)

    excluded_cols = {
        config.target_col,
        config.transaction_id_col,
        config.sender_col,
    }

    if config.receiver_col:
        excluded_cols.add(config.receiver_col)

    # Avoid using obvious target leakage / IDs.
    feature_cols = [
        c for c in df.columns
        if c not in excluded_cols
    ]

    X = encode_features_for_analysis(df, feature_cols)

    # Remove constant columns
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant_cols, errors="ignore")

    if X.empty:
        print("No usable features found for usefulness analysis.")
        return

    # Sampling keeps mutual information fast on 1M+ rows
    if len(X) > max_mi_rows:
        sample_idx = X.sample(n=max_mi_rows, random_state=42).index
        X_mi = X.loc[sample_idx]
        y_mi = y.loc[sample_idx]
        print(f"Using sampled {max_mi_rows:,} rows for mutual information calculation.")
    else:
        X_mi = X
        y_mi = y

    mi_scores = mutual_info_classif(
        X_mi,
        y_mi,
        discrete_features="auto",
        random_state=42
    )

    mi_series = (
        pd.Series(mi_scores, index=X_mi.columns, name="mutual_information")
        .sort_values(ascending=False)
    )

    print("\nTop 15 most useful features by mutual information:")
    print(mi_series.head(15).round(6))

    mi_series.to_csv(reports_dir / "feature_usefulness_mutual_information.csv")

    mi_series.head(15).sort_values().plot(kind="barh")
    plt.title("Feature Usefulness - Mutual Information with Fraud Target")
    plt.xlabel("Mutual information score")
    plt.ylabel("Feature")
    save_plot(plots_dir / "feature_usefulness_mutual_information.png")

    # Correlation with target
    corr_with_target = (
        X.assign(_target=y)
        .corr(numeric_only=True)["_target"]
        .drop("_target")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    print("\nTop 15 features by absolute correlation with fraud target:")
    print(corr_with_target.head(15).round(6))

    corr_with_target.to_csv(reports_dir / "feature_usefulness_correlation.csv")

    corr_with_target.head(15).sort_values().plot(kind="barh")
    plt.title("Feature Usefulness - Correlation with Fraud Target")
    plt.xlabel("Correlation")
    plt.ylabel("Feature")
    save_plot(plots_dir / "feature_usefulness_correlation.png")

    # Combined feature usefulness report
    usefulness_report = pd.DataFrame({
        "mutual_information": mi_series,
        "correlation_with_fraud": corr_with_target,
        "abs_correlation_with_fraud": corr_with_target.abs(),
    }).sort_values(
        ["mutual_information", "abs_correlation_with_fraud"],
        ascending=False
    )

    usefulness_report.to_csv(reports_dir / "feature_usefulness_combined.csv")

    print("\nCombined feature usefulness report saved to:")
    print(f"  {reports_dir / 'feature_usefulness_combined.csv'}")

    print("\nPlots saved to:")
    print(f"  {plots_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fraud dataset analysis for R2 statistics.")

    parser.add_argument(
        "--data",
        default="models/data/financial_fraud_detection_dataset.csv",
        help="Path to dataset CSV or Parquet file."
    )

    parser.add_argument(
        "--output",
        default="models",
        help="Output directory for plots and analysis reports."
    )

    parser.add_argument(
        "--max-mi-rows",
        type=int,
        default=200_000,
        help="Maximum number of rows used for mutual information calculation."
    )

    args = parser.parse_args()

    run_analysis(
        data_path=args.data,
        output_dir=args.output,
        max_mi_rows=args.max_mi_rows
    )