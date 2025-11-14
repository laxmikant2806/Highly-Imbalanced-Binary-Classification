import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
from src.utils.evaluation import ImbalancedEvaluator

class FraudDetectionPipeline:
    """A pipeline for fraud detection, including prediction and threshold optimization."""
    def __init__(self, model, scaler, threshold=0.5):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.evaluator = ImbalancedEvaluator()

    def predict_proba(self, X):
        """Generates probability predictions for the input data."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            return torch.sigmoid(logits).numpy()

    def predict(self, X):
        """Generates binary predictions based on the optimized threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def optimize_threshold(self, X_val, y_val, method='cost'):
        """Optimizes the prediction threshold based on the specified method."""
        probabilities = self.predict_proba(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, probabilities)

        if method == 'cost':
            costs = (1 - precision) + 100 * (1 - recall)
            optimal_idx = np.argmin(costs)
        elif method == 'f1':
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
        else:
            raise ValueError("Invalid optimization method. Choose 'cost' or 'f1'.")

        self.threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        return self.threshold
