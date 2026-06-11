"""
Rule Integration Comparison
============================

Compares two fraud-detection strategies on the same train/test split:

  Strategy A — current approach
      Standard model features.  After the model scores a transaction, two
      hard rules are OR'd in as post-prediction overrides:
        • geo_anomaly  : sender country changed within the last hour
        • seq_anomaly  : user had >= 2 consecutive flagged transactions

  Strategy B — rules as training features
      The same two rules are encoded as binary input features
      (geo_velocity_flag, seq_fraud_flag) so the model can learn their
      predictive signal directly.  No post-hoc overrides are applied.

Outputs (saved to <output_dir>/plots/):
  rule_comparison_cm.png       — side-by-side confusion matrices
  rule_comparison_metrics.png  — metrics comparison bar chart

Usage:
    python -m src.training.compare_rules \\
        --data models/data/financial_fraud_detection_dataset.csv \\
        --output models
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.features import ColumnDetector, BatchFeatureEngineer
from src.training.train import (
    load_dataset,
    prepare_target,
    split_data,
    train_model,
)
from src.training.evaluate import FraudModelEvaluator

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray,
                    evaluator: FraudModelEvaluator) -> dict:
    """Compute full metric set given binary predictions and raw scores."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "f2":        fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc":    average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total_cost": int(fp) * evaluator.fp_cost + int(fn) * evaluator.fn_cost,
    }


def evaluate_strategy_a(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    geo_flag_test: np.ndarray,
    seq_flag_test: np.ndarray,
    threshold: float,
    evaluator: FraudModelEvaluator,
) -> dict:
    """Model score + post-hoc rules OR'd in."""
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (
        (y_scores >= threshold) | (geo_flag_test == 1) | (seq_flag_test == 1)
    ).astype(int)
    return _binary_metrics(y_test, y_pred, y_scores, evaluator)


def evaluate_strategy_b(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    evaluator: FraudModelEvaluator,
) -> dict:
    """Model prediction only — rules baked into features, no post-hoc overrides."""
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
    return _binary_metrics(y_test, y_pred, y_scores, evaluator)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    metrics_a: dict,
    metrics_b: dict,
    y_test: np.ndarray,
    save_path: str,
):
    """Save side-by-side confusion matrix heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metrics, title in zip(
        axes,
        [metrics_a, metrics_b],
        [
            "Strategy A\n(Model + post-hoc rules)",
            "Strategy B\n(Rules as training features)",
        ],
    ):
        cm = np.array([
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Normal", "Predicted Fraud"],
            yticklabels=["Actual Normal", "Actual Fraud"],
            ax=ax,
        )
        ax.set_title(
            f"{title}\n"
            f"Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}  "
            f"F2={metrics['f2']:.3f}",
            fontsize=10,
        )

    fig.suptitle("Confusion Matrix Comparison: Rules Post-hoc vs Rules as Features", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices → {save_path}")


def plot_metrics_comparison(
    metrics_a: dict,
    metrics_b: dict,
    save_path: str,
):
    """Save a grouped bar chart comparing key metrics for both strategies."""
    metric_keys = ["precision", "recall", "f1", "f2", "roc_auc", "pr_auc"]
    labels = ["Precision", "Recall", "F1", "F2", "ROC-AUC", "PR-AUC"]

    vals_a = [metrics_a[k] for k in metric_keys]
    vals_b = [metrics_b[k] for k in metric_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: score metrics
    ax = axes[0]
    bars_a = ax.bar(x - width / 2, vals_a, width, label="A: Model + post-hoc rules", color="#4C72B0")
    bars_b = ax.bar(x + width / 2, vals_b, width, label="B: Rules as features",      color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics")
    ax.legend()
    ax.bar_label(bars_a, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars_b, fmt="%.3f", padding=2, fontsize=8)

    # Right: confusion matrix counts + cost
    count_keys = ["tp", "fp", "fn", "tn", "total_cost"]
    count_labels = ["TP", "FP", "FN", "TN", "Total Cost"]
    vals_a_c = [metrics_a[k] for k in count_keys]
    vals_b_c = [metrics_b[k] for k in count_keys]

    ax2 = axes[1]
    x2 = np.arange(len(count_labels))
    bars_a2 = ax2.bar(x2 - width / 2, vals_a_c, width, label="A: Model + post-hoc rules", color="#4C72B0")
    bars_b2 = ax2.bar(x2 + width / 2, vals_b_c, width, label="B: Rules as features",      color="#DD8452")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(count_labels)
    ax2.set_ylabel("Count / Cost")
    ax2.set_title("Confusion Matrix Counts & Cost")
    ax2.legend()
    ax2.bar_label(bars_a2, fmt="%d", padding=2, fontsize=8)
    ax2.bar_label(bars_b2, fmt="%d", padding=2, fontsize=8)

    fig.suptitle("Strategy Comparison: Rules Post-hoc vs Rules as Training Features", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics comparison → {save_path}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_comparison_report(metrics_a: dict, metrics_b: dict):
    print(f"\n{'='*70}")
    print(" Rule Integration Comparison Report")
    print(f"{'='*70}")

    header = f"{'Metric':<22} {'Strategy A (post-hoc)':>22} {'Strategy B (features)':>22} {'Delta (B-A)':>12}"
    print(header)
    print("-" * 70)

    rows = [
        ("Precision",   "precision"),
        ("Recall",      "recall"),
        ("F1",          "f1"),
        ("F2",          "f2"),
        ("ROC-AUC",     "roc_auc"),
        ("PR-AUC",      "pr_auc"),
        ("TP",          "tp"),
        ("FP",          "fp"),
        ("FN",          "fn"),
        ("TN",          "tn"),
        ("Total Cost",  "total_cost"),
    ]

    for label, key in rows:
        a, b = metrics_a[key], metrics_b[key]
        if isinstance(a, float):
            delta_sign = "+" if b > a else ""
            print(f"  {label:<20} {a:>22.4f} {b:>22.4f} {delta_sign}{b-a:>11.4f}")
        else:
            delta = b - a
            delta_sign = "+" if delta > 0 else ""
            print(f"  {label:<20} {a:>22d} {b:>22d} {delta_sign}{delta:>11d}")

    print(f"{'='*70}")

    # Winner summary
    better_b = sum(
        1 for k in ["recall", "f1", "f2", "pr_auc"]
        if metrics_b[k] > metrics_a[k]
    )
    if better_b >= 3:
        verdict = "Strategy B (rules as features) outperforms on recall-oriented metrics."
    elif better_b <= 1:
        verdict = "Strategy A (post-hoc rules) is competitive; baking rules in did not help."
    else:
        verdict = "Mixed results — neither strategy clearly dominates."

    print(f"\n  Verdict: {verdict}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare post-hoc rules vs rules baked into model features"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="models/data/financial_fraud_detection_dataset.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--model-type", "-m",
        type=str,
        choices=["xgboost", "lightgbm", "random_forest", "ensemble"],
        default="xgboost",
        help="Model type to train for both strategies",
    )
    parser.add_argument(
        "--threshold-method",
        type=str,
        choices=["f2", "cost", "recall_at_precision"],
        default="f2",
        help="Threshold selection method",
    )
    args = parser.parse_args()

    evaluator = FraudModelEvaluator(false_negative_cost=10.0, false_positive_cost=1.0)

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    df = load_dataset(args.data)
    config = ColumnDetector.detect_columns(df)
    y = prepare_target(df, config)

    # ------------------------------------------------------------------
    # 2. Compute rule features on the FULL sorted dataset BEFORE splitting.
    #    Chronological order is preserved so that each row's rule features
    #    only reflect transactions that came before it.
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(" Computing rule features (full dataset, chronological)")
    print(f"{'='*60}")

    feat_eng_full = BatchFeatureEngineer(config)
    df_with_rules = feat_eng_full.create_rule_features(df)

    # Carry the rule columns through the split alongside the raw data
    rule_cols = ["geo_velocity_flag", "seq_fraud_flag"]

    # ------------------------------------------------------------------
    # 3. Split (same 60/20/20 as train.py)
    # ------------------------------------------------------------------
    df_train, df_test, df_holdout, y_train, y_test, _ = split_data(df_with_rules, y, config)

    # Extract rule columns for test set (already computed chronologically)
    geo_flag_test = df_test["geo_velocity_flag"].values.astype(np.int32)
    seq_flag_test = df_test["seq_fraud_flag"].values.astype(np.int32)

    print(f"\nTest set rule signal rates:")
    print(f"  geo_velocity_flag: {geo_flag_test.mean()*100:.2f}% flagged")
    print(f"  seq_fraud_flag:    {seq_flag_test.mean()*100:.2f}% flagged")

    # ------------------------------------------------------------------
    # 4. Feature engineering
    #    - Model A: standard features only (fit on train without rule cols)
    #    - Model B: standard features + rule features
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(" Feature Engineering — Strategy A (standard features)")
    print(f"{'='*60}")

    # Drop rule cols from train/test so feat engineer sees a clean dataframe
    df_train_no_rules = df_train.drop(columns=rule_cols, errors="ignore")
    df_test_no_rules  = df_test.drop(columns=rule_cols, errors="ignore")

    feat_eng_a = BatchFeatureEngineer(config)
    df_train_fe_a, feature_cols_a = feat_eng_a.fit_transform(df_train_no_rules)
    df_test_fe_a = feat_eng_a.transform(df_test_no_rules)

    X_train_a = df_train_fe_a[feature_cols_a].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)
    X_test_a  = df_test_fe_a[feature_cols_a].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)

    print(f"\n{'='*60}")
    print(" Feature Engineering — Strategy B (standard + rule features)")
    print(f"{'='*60}")

    feat_eng_b = BatchFeatureEngineer(config)
    df_train_fe_b, feature_cols_b = feat_eng_b.fit_transform(df_train)  # rule cols present in df_train
    df_test_fe_b = feat_eng_b.transform(df_test)

    # Ensure rule columns end up in the feature matrix
    for rc in rule_cols:
        if rc not in df_test_fe_b.columns:
            df_test_fe_b[rc] = df_test[rc].values

    X_train_b = df_train_fe_b[feature_cols_b].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)
    X_test_b  = df_test_fe_b[feature_cols_b].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)

    print(f"\nFeature counts — A: {X_train_a.shape[1]}  B: {X_train_b.shape[1]}")

    # ------------------------------------------------------------------
    # 5. Train both models
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(" Training Strategy A model")
    print(f"{'='*60}")
    model_a = train_model(X_train_a, y_train.values, model_type=args.model_type)

    print(f"\n{'='*60}")
    print(" Training Strategy B model")
    print(f"{'='*60}")
    model_b = train_model(X_train_b, y_train.values, model_type=args.model_type)

    # ------------------------------------------------------------------
    # 6. Find optimal thresholds on the test set for each model
    # ------------------------------------------------------------------
    print("\nFinding optimal thresholds...")
    thresh_a, _ = evaluator.find_optimal_threshold(
        y_test.values, model_a.predict_proba(X_test_a)[:, 1], method=args.threshold_method
    )
    thresh_b, _ = evaluator.find_optimal_threshold(
        y_test.values, model_b.predict_proba(X_test_b)[:, 1], method=args.threshold_method
    )
    print(f"  Threshold A: {thresh_a:.3f}")
    print(f"  Threshold B: {thresh_b:.3f}")

    # ------------------------------------------------------------------
    # 7. Evaluate both strategies
    # ------------------------------------------------------------------
    metrics_a = evaluate_strategy_a(
        model_a, X_test_a, y_test.values,
        geo_flag_test, seq_flag_test,
        thresh_a, evaluator,
    )
    metrics_b = evaluate_strategy_b(
        model_b, X_test_b, y_test.values,
        thresh_b, evaluator,
    )

    print_comparison_report(metrics_a, metrics_b)

    # ------------------------------------------------------------------
    # 8. Save plots
    # ------------------------------------------------------------------
    plots_dir = Path(args.output) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrices(
        metrics_a, metrics_b, y_test.values,
        save_path=str(plots_dir / "rule_comparison_cm.png"),
    )
    plot_metrics_comparison(
        metrics_a, metrics_b,
        save_path=str(plots_dir / "rule_comparison_metrics.png"),
    )

    print(f"\nAll comparison artifacts saved to: {plots_dir}")


if __name__ == "__main__":
    main()
