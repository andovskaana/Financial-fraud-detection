"""
Ensemble classifier for fraud detection.

This module defines a simple ensemble classifier that averages the
probabilities of multiple base models.  It expects each base model to
implement a `predict_proba` method returning class probabilities.

The ensemble classifier returns a probability for each class by taking
the mean of the probabilities predicted by the base models.  This allows
the model to combine diverse classifiers such as XGBoost, LightGBM and
RandomForest.  See the training script for usage.
"""

from typing import List
import numpy as np


class EnsembleClassifier:
    """A simple ensemble classifier that averages probabilities from base models."""

    def __init__(self, models: List):
        # List of fitted estimators implementing predict_proba
        self.models = models

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities by averaging the positive-class probability."""
        probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            # take the positive class (index 1)
            probas.append(proba[:, 1])
        avg = np.mean(probas, axis=0)
        return np.vstack([1 - avg, avg]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using the supplied threshold."""
        pos = self.predict_proba(X)[:, 1]
        return (pos >= threshold).astype(int)