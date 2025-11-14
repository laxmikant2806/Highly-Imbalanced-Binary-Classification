import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from collections import Counter
import warnings
from src.utils.utils import load_config
from src.etl.data_loader import load_data
from src.features.feature_engineering import split_data, scale_features, resample_data
from src.modeling.model import FraudDetector
from src.modeling.train import train_model
from src.modeling.predict import FraudDetectionPipeline
from src.utils.evaluation import ImbalancedEvaluator

warnings.filterwarnings('ignore')

# Load configuration
config = load_config()


# Main Execution
def main():
    print("Fraud Detection with Severe Class Imbalance")

    data = load_data(config)
    if data is None:
        return

    X = data.drop('Class', axis=1).values
    y = data['Class'].values

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    X_resampled, y_resampled = resample_data(X_train_scaled, y_train, config)

    model = FraudDetector(input_dim=X_train_scaled.shape[1], hidden_dims=config['model']['hidden_dims'])
    trained_model = train_model(model, X_resampled, y_resampled, X_val_scaled, y_val, config)

    pipeline = FraudDetectionPipeline(trained_model, scaler)
    optimal_threshold = pipeline.optimize_threshold(X_val_scaled, y_val, method=config['evaluation']['optimize_threshold_method'])
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    print("\nFinal Evaluation:")
    test_probs = pipeline.predict_proba(X_test_scaled)
    evaluator = ImbalancedEvaluator()

    metrics_default = evaluator.compute_metrics(y_test, test_probs, threshold=config['evaluation']['default_threshold'])
    print(f"Metrics at {config['evaluation']['default_threshold']} threshold: {metrics_default}")

    metrics_optimal = evaluator.compute_metrics(y_test, test_probs, threshold=optimal_threshold)
    print(f"Metrics at optimal threshold ({optimal_threshold:.4f}): {metrics_optimal}")

    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred_proba': test_probs,
        'y_pred_default': (test_probs >= config['evaluation']['default_threshold']).astype(int),
        'y_pred_optimal': (test_probs >= optimal_threshold).astype(int)
    })

    results_df.to_csv(config['data']['results_path'], index=False)

    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'scaler': scaler,
        'threshold': optimal_threshold,
        'model_architecture': config['model']
    }, config['model']['pipeline_save_path'])

    print(f"Results and pipeline saved.")

if __name__ == "__main__":
    main()
