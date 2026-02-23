"""
Evaluation utilities for Fraud Detection Model

Focus on:
- Imbalanced class metrics (PR-AUC, Recall@Precision)
- Cost-sensitive evaluation (false negatives are more expensive)
- Threshold optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class FraudModelEvaluator:
    """Comprehensive evaluation for fraud detection models."""

    def __init__(
        self,
        false_negative_cost: float = 10.0,
        false_positive_cost: float = 1.0
    ):
        """
        Initialize evaluator with cost parameters.

        Args:
            false_negative_cost: Cost of missing a fraud (typically higher)
            false_positive_cost: Cost of false alarm
        """
        self.fn_cost = false_negative_cost
        self.fp_cost = false_positive_cost

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {}

        # Basic metrics
        metrics['accuracy'] = np.mean(y_true == y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['f2'] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)  # Weighted towards recall

        # ROC-AUC and PR-AUC
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)

        # Rates
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # Miss rate

        # Cost-sensitive metric
        total_cost = fp * self.fp_cost + fn * self.fn_cost
        metrics['total_cost'] = total_cost
        metrics['avg_cost_per_sample'] = total_cost / len(y_true)

        # Class distribution
        metrics['fraud_rate'] = np.mean(y_true)
        metrics['predicted_fraud_rate'] = np.mean(y_pred)

        return metrics

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        method: str = 'f2',
        min_precision: float = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold based on specified method.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            method: 'f2' (F2 score), 'cost' (minimize cost), 'recall_at_precision'
            min_precision: Minimum precision constraint for recall_at_precision method

        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        thresholds = np.arange(0.01, 0.99, 0.01)
        best_threshold = 0.5
        best_score = -np.inf

        results = []

        for thresh in thresholds:
            metrics = self.compute_metrics(y_true, y_pred_proba, thresh)

            if method == 'f2':
                score = metrics['f2']
            elif method == 'cost':
                score = -metrics['total_cost']  # Negative because we minimize cost
            elif method == 'recall_at_precision':
                if min_precision and metrics['precision'] >= min_precision:
                    score = metrics['recall']
                else:
                    score = -np.inf
            else:
                score = metrics['f1']

            results.append({
                'threshold': thresh,
                'score': score,
                **metrics
            })

            if score > best_score:
                best_score = score
                best_threshold = thresh

        # Get metrics at optimal threshold
        optimal_metrics = self.compute_metrics(y_true, y_pred_proba, best_threshold)

        return best_threshold, optimal_metrics

    def recall_at_precision(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        target_precision: float
    ) -> Tuple[float, float, float]:
        """
        Find recall at a target precision level.

        Returns:
            Tuple of (recall, actual_precision, threshold)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        # Find threshold where precision >= target
        valid_idx = precision >= target_precision
        if not valid_idx.any():
            return 0.0, 0.0, 1.0

        # Get highest recall among valid precisions
        best_idx = np.argmax(recall[:-1][valid_idx[:-1]])
        actual_idx = np.where(valid_idx[:-1])[0][best_idx]

        return recall[actual_idx], precision[actual_idx], thresholds[actual_idx]

    def print_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        title: str = "Model Evaluation"
    ):
        """Print comprehensive evaluation report."""
        metrics = self.compute_metrics(y_true, y_pred_proba, threshold)

        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")

        print(f"\nThreshold: {threshold:.3f}")
        print(f"Dataset size: {len(y_true):,}")
        print(f"Fraud rate: {metrics['fraud_rate']*100:.2f}%")

        print(f"\n--- Classification Metrics ---")
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        print(f"F1 Score:   {metrics['f1']:.4f}")
        print(f"F2 Score:   {metrics['f2']:.4f}")

        print(f"\n--- AUC Metrics ---")
        print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:     {metrics['pr_auc']:.4f}")

        print(f"\n--- Confusion Matrix ---")
        print(f"True Positives:   {metrics['true_positives']:,}")
        print(f"True Negatives:   {metrics['true_negatives']:,}")
        print(f"False Positives:  {metrics['false_positives']:,}")
        print(f"False Negatives:  {metrics['false_negatives']:,}")

        print(f"\n--- Rates ---")
        print(f"True Positive Rate (Recall):  {metrics['tpr']:.4f}")
        print(f"False Positive Rate:          {metrics['fpr']:.4f}")
        print(f"False Negative Rate:          {metrics['fnr']:.4f}")

        print(f"\n--- Cost Analysis ---")
        print(f"FN Cost Weight: {self.fn_cost}, FP Cost Weight: {self.fp_cost}")
        print(f"Total Cost:     {metrics['total_cost']:,.2f}")
        print(f"Avg Cost/Sample:{metrics['avg_cost_per_sample']:.4f}")

        print(f"\n{'='*60}\n")

        return metrics

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot precision-recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP={ap:.3f})')
        ax.axhline(y=np.mean(y_true), color='r', linestyle='--', label=f'Baseline (fraud rate={np.mean(y_true):.3f})')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved PR curve to {save_path}")

        plt.close()

    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot metrics across different thresholds."""
        thresholds = np.arange(0.01, 0.99, 0.01)

        precisions = []
        recalls = []
        f1_scores = []
        f2_scores = []
        costs = []

        for thresh in thresholds:
            metrics = self.compute_metrics(y_true, y_pred_proba, thresh)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1'])
            f2_scores.append(metrics['f2'])
            costs.append(metrics['total_cost'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Precision, Recall, F-scores
        axes[0].plot(thresholds, precisions, 'b-', label='Precision')
        axes[0].plot(thresholds, recalls, 'r-', label='Recall')
        axes[0].plot(thresholds, f1_scores, 'g-', label='F1')
        axes[0].plot(thresholds, f2_scores, 'm-', label='F2')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Metrics vs Threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Cost
        axes[1].plot(thresholds, costs, 'r-', linewidth=2)
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Total Cost')
        axes[1].set_title('Cost vs Threshold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved threshold analysis to {save_path}")

        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix heatmap."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'],
            ax=ax
        )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (threshold={threshold:.2f})')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")

        plt.close()


def print_class_distribution(y: np.ndarray, name: str = "Dataset"):
    """Print class distribution statistics."""
    total = len(y)
    fraud_count = np.sum(y)
    normal_count = total - fraud_count

    print(f"\n--- {name} Class Distribution ---")
    print(f"Total samples:   {total:,}")
    print(f"Normal (0):      {normal_count:,} ({normal_count/total*100:.2f}%)")
    print(f"Fraud (1):       {fraud_count:,} ({fraud_count/total*100:.2f}%)")
    print(f"Imbalance ratio: 1:{normal_count/max(1, fraud_count):.1f}")
