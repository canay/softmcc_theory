"""scikit-learn compatible scorer for SoftMCC.

SoftMCC is the standard MCC formula evaluated on soft confusion-matrix counts
TP = sum(p*y), FP = sum(p*(1-y)), FN = sum((1-p)*y), TN = sum((1-p)*(1-y)),
i.e. a threshold-free, probability-based Matthews correlation coefficient.

Usage:
    from softmcc_scorer import soft_mcc, softmcc_scorer
    softmcc_scorer(model, X_val, y_val)            # as a callable scorer
    GridSearchCV(model, grid, scoring=softmcc_scorer)
"""
import numpy as np
from sklearn.metrics import make_scorer


def soft_mcc(y_true, y_prob, eps=1e-12):
    """Threshold-free probabilistic MCC on soft confusion-matrix counts."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    tp = np.sum(p * y)
    fp = np.sum(p * (1 - y))
    fn = np.sum((1 - p) * y)
    tn = np.sum((1 - p) * (1 - y))
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    return num / den


softmcc_scorer = make_scorer(soft_mcc, response_method="predict_proba")
