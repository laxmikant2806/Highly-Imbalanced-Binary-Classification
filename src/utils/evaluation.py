import numpy as np
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

class ImbalancedEvaluator:
    """A class for evaluating models on imbalanced datasets."""
    @staticmethod
    def compute_metrics(y_true, y_pred_proba, threshold=0.5):
        """Computes a comprehensive set of metrics for imbalanced classification."""
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        # F1 at fixed recall (95%)
        target_recall = 0.95
        recall_mask = recall >= target_recall
        f1_at_recall = 0
        if np.any(recall_mask):
            recall_idx = np.where(recall_mask)[0][0]
            if precision[recall_idx] > 0 and recall[recall_idx] > 0:
                f1_at_recall = 2 * (precision[recall_idx] * recall[recall_idx]) / (precision[recall_idx] + recall[recall_idx])

        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

        # Cost calculation (FN cost = 100x FP cost)
        total_cost = fp + 100 * fn

        return {
            'pr_auc': pr_auc,
            'f1_at_95_recall': f1_at_recall,
            'total_cost': total_cost,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        }
