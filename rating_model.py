"""Inference adapters serialized with the enhanced scoring model."""
import numpy as np


class TextModel:
    def __init__(self, vectorizer, estimators, classes, mode, offsets=None):
        self.vectorizer = vectorizer
        self.estimators = estimators
        self.classes_ = np.asarray(classes)
        self.mode = mode
        self.offsets = offsets

    def probabilities(self, matrix):
        if self.mode == 'ordinal':
            above = np.column_stack([model.predict_proba(matrix)[:, 1]
                                     for model in self.estimators])
            # Enforce non-increasing cumulative probabilities across star thresholds.
            above = np.sort(above, axis=1)[:, ::-1]
            bounds = np.column_stack([np.ones(len(above)), above, np.zeros(len(above))])
            return -np.diff(bounds, axis=1)
        if self.mode == 'blend':
            return np.mean([model.probabilities(matrix) for model in self.estimators], axis=0)
        if self.mode == 'collapsed':
            stars = self.estimators[0].probabilities(matrix)
            return np.column_stack([stars[:, :2].sum(axis=1), stars[:, 2], stars[:, 3:].sum(axis=1)])
        return self.estimators[0].predict_proba(matrix)

    def predict_proba(self, text):
        probabilities = self.probabilities(self.vectorizer.transform(text))
        if self.offsets is None:
            return probabilities
        weighted = probabilities * np.exp(self.offsets)
        return weighted / weighted.sum(axis=1, keepdims=True)
