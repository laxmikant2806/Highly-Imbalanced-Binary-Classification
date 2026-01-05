"""
Enhanced prediction pipeline for fraud detection.
Supports multiple models and provides unified interface.
"""
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
from src.utils.evaluation import ImbalancedEvaluator
from src.modeling.calibration import ProbabilityCalibrator


class FraudDetectionPipeline:
    """
    Production-ready fraud detection pipeline.
    Supports PyTorch neural networks and tree-based models.
    """
    
    def __init__(self, model, scaler, threshold=0.5, model_type='pytorch', calibrator=None):
        """
        Args:
            model: Trained model (PyTorch or tree wrapper)
            scaler: Feature scaler
            threshold: Classification threshold
            model_type: 'pytorch' or 'tree'
            calibrator: Optional probability calibrator
        """
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.model_type = model_type
        self.calibrator = calibrator
        self.evaluator = ImbalancedEvaluator()

    def predict_proba(self, X, scale=True):
        """
        Generate fraud probabilities.
        
        Args:
            X: Input features
            scale: Whether to apply scaling (False if already scaled)
        
        Returns:
            numpy array of fraud probabilities
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Scale features if needed
        X_processed = self.scaler.transform(X) if scale else X
        
        if self.model_type == 'pytorch':
            X_tensor = torch.tensor(X_processed, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(X_tensor)
                probs = torch.sigmoid(logits).numpy()
        else:
            # Tree model
            probs = self.model.predict_proba(X_processed)
        
        # Apply calibration if available
        if self.calibrator is not None:
            probs = self.calibrator.transform(probs)
        
        return probs

    def predict(self, X, scale=True, threshold=None):
        """
        Generate binary fraud predictions.
        
        Args:
            X: Input features
            scale: Whether to apply scaling
            threshold: Optional threshold override
        
        Returns:
            numpy array of binary predictions (0 or 1)
        """
        thresh = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X, scale=scale) >= thresh).astype(int)

    def optimize_threshold(self, X_val, y_val, method='cost', scale=True):
        """
        Optimize classification threshold using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            method: 'cost', 'f1', 'f2', 'precision_target', 'recall_target'
            scale: Whether to apply scaling
        
        Returns:
            Optimal threshold value
        """
        probs = self.predict_proba(X_val, scale=scale)
        
        optimal_threshold = self.evaluator.find_optimal_threshold(
            y_val, probs, method=method
        )
        
        self.threshold = optimal_threshold
        return optimal_threshold

    def evaluate(self, X, y, scale=True, threshold=None):
        """
        Comprehensive evaluation on a dataset.
        
        Returns:
            dict of metrics
        """
        probs = self.predict_proba(X, scale=scale)
        thresh = threshold if threshold is not None else self.threshold
        
        return self.evaluator.compute_metrics(y, probs, threshold=thresh)

    def set_calibrator(self, calibrator):
        """Set probability calibrator."""
        self.calibrator = calibrator

    def get_feature_importance(self):
        """Get feature importance (for tree models)."""
        if self.model_type == 'tree':
            return self.model.feature_importance()
        return None


def create_ensemble_predictions(pipelines, X, weights=None):
    """
    Create ensemble predictions from multiple pipelines.
    
    Args:
        pipelines: dict of {name: FraudDetectionPipeline}
        X: Input features
        weights: Optional dict of {name: weight}
    
    Returns:
        Ensemble probabilities
    """
    if weights is None:
        weights = {name: 1.0 for name in pipelines}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {name: w / total_weight for name, w in weights.items()}
    
    ensemble_probs = np.zeros(len(X))
    
    for name, pipeline in pipelines.items():
        probs = pipeline.predict_proba(X, scale=True)
        ensemble_probs += weights.get(name, 1.0 / len(pipelines)) * probs
    
    return ensemble_probs
