"""
Enhanced evaluation metrics for imbalanced classification.
Includes comprehensive metrics, threshold optimization, and visualization support.
"""
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    confusion_matrix, classification_report,
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score
)


class ImbalancedEvaluator:
    """
    Comprehensive evaluator for imbalanced classification problems.
    Provides metrics optimized for fraud detection.
    """
    
    @staticmethod
    def compute_metrics(y_true, y_pred_proba, threshold=0.5):
        """
        Compute comprehensive metrics for fraud detection.
        
        Returns:
            dict with all relevant metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # PR-AUC (most important for imbalanced data)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall curve for F1 at thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # F1 at various recall levels
        f1_at_recall = {}
        for target_recall in [0.90, 0.95, 0.99]:
            recall_mask = recall >= target_recall
            if np.any(recall_mask):
                idx = np.where(recall_mask)[0][-1]  # Last index with recall >= target
                if precision[idx] > 0 and recall[idx] > 0:
                    f1 = 2 * (precision[idx] * recall[idx]) / (precision[idx] + recall[idx])
                    f1_at_recall[f'f1_at_{int(target_recall*100)}_recall'] = f1
        
        # Confusion matrix at threshold
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        
        # Cost calculation (FN cost = 100x FP cost, typical for fraud)
        fn_cost = 100
        fp_cost = 1
        total_cost = fp * fp_cost + fn * fn_cost
        
        metrics = {
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'total_cost': total_cost,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'threshold': threshold
        }
        
        metrics.update(f1_at_recall)
        
        return metrics
    
    @staticmethod
    def find_optimal_threshold(y_true, y_pred_proba, method='f1'):
        """
        Find optimal classification threshold.
        
        Args:
            method: 'f1', 'f2', 'cost', 'precision_target', 'recall_target'
        
        Returns:
            optimal_threshold, metrics_at_threshold
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Handle edge case
        if len(thresholds) == 0:
            return 0.5, {}
        
        # Ensure arrays are same length
        thresholds = np.append(thresholds, 1.0)
        
        if method == 'f1':
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            
        elif method == 'f2':
            # F2 weights recall higher than precision
            beta = 2
            f2_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-8)
            optimal_idx = np.argmax(f2_scores)
            
        elif method == 'cost':
            # Minimize cost: FN is 100x more costly than FP
            fn_cost = 100
            fp_cost = 1
            n_pos = y_true.sum()
            n_neg = len(y_true) - n_pos
            
            # Calculate costs at each threshold
            costs = []
            for i, thresh in enumerate(thresholds):
                tp = recall[i] * n_pos
                fn = n_pos - tp
                prec = precision[i] if precision[i] > 0 else 1e-8
                fp = (tp / prec) - tp if prec > 0 else n_neg
                fp = min(fp, n_neg)
                
                cost = fn * fn_cost + fp * fp_cost
                costs.append(cost)
            
            optimal_idx = np.argmin(costs)
            
        elif method == 'precision_target':
            # Find threshold for minimum precision of 20%
            target_precision = 0.20
            valid_mask = precision >= target_precision
            if np.any(valid_mask):
                # Get highest recall while maintaining precision
                valid_recall = recall.copy()
                valid_recall[~valid_mask] = -1
                optimal_idx = np.argmax(valid_recall)
            else:
                optimal_idx = np.argmax(precision)
                
        elif method == 'recall_target':
            # Find threshold for minimum recall of 95%
            target_recall = 0.95
            valid_mask = recall >= target_recall
            if np.any(valid_mask):
                # Get highest precision while maintaining recall
                valid_precision = precision.copy()
                valid_precision[~valid_mask] = -1
                optimal_idx = np.argmax(valid_precision)
            else:
                optimal_idx = np.argmax(recall)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return optimal_threshold
    
    @staticmethod
    def get_curves(y_true, y_pred_proba):
        """
        Get PR and ROC curves for visualization.
        """
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        
        return {
            'precision': precision,
            'recall': recall,
            'pr_thresholds': pr_thresholds,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds,
            'pr_auc': auc(recall, precision),
            'roc_auc': auc(fpr, tpr)
        }
    
    @staticmethod
    def compare_models(y_true, model_predictions, threshold=0.5):
        """
        Compare multiple models side by side.
        
        Args:
            model_predictions: dict of {model_name: y_pred_proba}
        
        Returns:
            DataFrame with comparison
        """
        results = []
        for name, preds in model_predictions.items():
            metrics = ImbalancedEvaluator.compute_metrics(y_true, preds, threshold)
            metrics['model'] = name
            results.append(metrics)
        
        return results


def print_evaluation_report(y_true, y_pred_proba, model_name="Model"):
    """
    Print a comprehensive evaluation report.
    """
    evaluator = ImbalancedEvaluator()
    
    # Find optimal thresholds using different methods
    print(f"\n{'='*60}")
    print(f"Evaluation Report: {model_name}")
    print(f"{'='*60}")
    
    # Default threshold metrics
    metrics_05 = evaluator.compute_metrics(y_true, y_pred_proba, threshold=0.5)
    print(f"\nMetrics at threshold 0.5:")
    print(f"  PR-AUC:     {metrics_05['pr_auc']:.4f}")
    print(f"  ROC-AUC:    {metrics_05['roc_auc']:.4f}")
    print(f"  Precision:  {metrics_05['precision']:.4f}")
    print(f"  Recall:     {metrics_05['recall']:.4f}")
    print(f"  F1:         {metrics_05['f1']:.4f}")
    
    # Optimal thresholds
    for method in ['f1', 'f2', 'cost']:
        opt_thresh = evaluator.find_optimal_threshold(y_true, y_pred_proba, method=method)
        metrics = evaluator.compute_metrics(y_true, y_pred_proba, threshold=opt_thresh)
        print(f"\nOptimal threshold ({method}): {opt_thresh:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    
    # Confusion matrix at optimal F1
    opt_thresh = evaluator.find_optimal_threshold(y_true, y_pred_proba, method='f1')
    y_pred = (y_pred_proba >= opt_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nConfusion Matrix (threshold={opt_thresh:.4f}):")
    print(f"  TN: {cm[0,0]:>6}  FP: {cm[0,1]:>6}")
    print(f"  FN: {cm[1,0]:>6}  TP: {cm[1,1]:>6}")
    
    return metrics_05
