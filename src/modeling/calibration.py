"""
Probability calibration for fraud detection models.
Ensures model outputs reflect true fraud likelihood.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """
    Calibrates model probabilities using Platt scaling or isotonic regression.
    """
    
    def __init__(self, method='platt'):
        """
        Args:
            method: 'platt' (sigmoid) or 'isotonic'
        """
        self.method = method
        self.calibrator = None
        
    def fit(self, y_true, y_pred_proba):
        """
        Fit the calibrator on validation data.
        
        Args:
            y_true: True labels
            y_pred_proba: Uncalibrated probabilities
        """
        if self.method == 'platt':
            # Platt scaling: fit logistic regression on log-odds
            self.calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
            # Reshape for sklearn
            X = y_pred_proba.reshape(-1, 1)
            self.calibrator.fit(X, y_true)
            
        elif self.method == 'isotonic':
            # Isotonic regression: monotonic piecewise-constant fit
            self.calibrator = IsotonicRegression(
                y_min=0, y_max=1, 
                out_of_bounds='clip'
            )
            self.calibrator.fit(y_pred_proba, y_true)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def transform(self, y_pred_proba):
        """
        Apply calibration to probabilities.
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.method == 'platt':
            X = y_pred_proba.reshape(-1, 1)
            return self.calibrator.predict_proba(X)[:, 1]
        else:
            return self.calibrator.transform(y_pred_proba)
    
    def fit_transform(self, y_true, y_pred_proba):
        """Fit and transform in one step."""
        self.fit(y_true, y_pred_proba)
        return self.transform(y_pred_proba)


def compute_calibration_metrics(y_true, y_pred_proba, n_bins=10):
    """
    Compute calibration metrics and reliability diagram data.
    
    Returns:
        dict with ECE (Expected Calibration Error), MCE (Max Calibration Error),
        and binned data for reliability diagram
    """
    # Get calibration curve
    fraction_positives, mean_predicted = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )
    
    # Calculate ECE and MCE
    bin_sizes = np.histogram(y_pred_proba, bins=n_bins, range=(0, 1))[0]
    bin_sizes = bin_sizes / len(y_pred_proba)
    
    # ECE: weighted average of |accuracy - confidence| per bin
    ece = np.sum(bin_sizes[:len(fraction_positives)] * 
                 np.abs(fraction_positives - mean_predicted))
    
    # MCE: maximum calibration error
    mce = np.max(np.abs(fraction_positives - mean_predicted))
    
    return {
        'ece': ece,
        'mce': mce,
        'fraction_positives': fraction_positives,
        'mean_predicted': mean_predicted,
        'n_bins': n_bins
    }


def calibrate_model_predictions(y_val, val_proba, y_test, test_proba, method='platt'):
    """
    Calibrate model predictions using validation set.
    
    Args:
        y_val: Validation labels
        val_proba: Validation probabilities (for fitting)
        y_test: Test labels (for evaluation)
        test_proba: Test probabilities (to calibrate)
        method: 'platt' or 'isotonic'
    
    Returns:
        calibrated_test_proba, calibrator
    """
    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(y_val, val_proba)
    
    calibrated_proba = calibrator.transform(test_proba)
    
    # Print calibration improvement
    before_metrics = compute_calibration_metrics(y_test, test_proba)
    after_metrics = compute_calibration_metrics(y_test, calibrated_proba)
    
    print(f"\nCalibration ({method}):")
    print(f"  ECE: {before_metrics['ece']:.4f} -> {after_metrics['ece']:.4f}")
    print(f"  MCE: {before_metrics['mce']:.4f} -> {after_metrics['mce']:.4f}")
    
    return calibrated_proba, calibrator
