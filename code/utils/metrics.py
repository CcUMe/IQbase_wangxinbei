"""Single-label classification metrics from a confusion matrix (rows are truth)."""
import numpy as np


def classification_metrics(matrix, labels=None):
    matrix = np.asarray(matrix, dtype=np.int64)
    support, predicted = matrix.sum(1), matrix.sum(0)
    tp = matrix.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(len(matrix)), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(matrix)), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(matrix)), where=support + predicted > 0)
    total = int(support.sum())
    accuracy = float(tp.sum() / max(total, 1))
    return {'accuracy': accuracy, 'micro_f1': accuracy, 'macro_precision': float(precision.mean()),
            'macro_recall': float(recall.mean()), 'balanced_accuracy': float(recall.mean()),
            'macro_f1': float(f1.mean()), 'weighted_f1': float((f1 * support).sum() / max(total, 1)),
            'support': total, 'labels': list(labels) if labels is not None else [str(i) for i in range(len(matrix))],
            'class_support': support.tolist(), 'precision': precision.tolist(), 'recall': recall.tolist(),
            'f1': f1.tolist(), 'confusion_matrix': matrix.tolist()}
