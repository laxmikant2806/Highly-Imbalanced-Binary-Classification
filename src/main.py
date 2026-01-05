"""
Optimized Fraud Detection Pipeline
===================================
Production-ready pipeline with:
- Feature engineering (time features, amount features)
- Multiple sampling strategies (SMOTE variants)
- Multiple models (Neural Net, LightGBM)
- Advanced loss functions (Focal, Weighted BCE)
- Threshold optimization
- Probability calibration
"""
import os
import pandas as pd
import numpy as np
import torch
import warnings

from src.utils.utils import load_config
from src.etl.data_loader import load_data
from src.features.feature_engineering import (
    preprocess_features, split_data, scale_features, resample_data
)
from src.modeling.model import get_model
from src.modeling.train import train_model, train_all_models
from src.modeling.predict import FraudDetectionPipeline, create_ensemble_predictions
from src.utils.evaluation import ImbalancedEvaluator, print_evaluation_report
from src.modeling.calibration import calibrate_model_predictions

warnings.filterwarnings('ignore')


def main():
    print("="*60)
    print("Optimized Fraud Detection Pipeline")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Load data
    print("\n[1/7] Loading data...")
    data = load_data(config)
    if data is None:
        return
    
    # Clean data types
    if data['Class'].dtype == 'object':
        data['Class'] = data['Class'].str.strip("'").astype(int)
    
    for col in data.columns:
        if col != 'Class':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna()
    
    print(f"Dataset shape: {data.shape}")
    n_fraud = data['Class'].sum()
    print(f"Fraud cases: {n_fraud} ({n_fraud/len(data)*100:.2f}%)")
    
    # Feature engineering
    print("\n[2/7] Feature engineering...")
    if config.get('feature_engineering', {}).get('enable_time_features', True):
        data = preprocess_features(data, config)
        print(f"Enhanced features: {data.shape[1]} columns")
    
    # Prepare features and labels
    X = data.drop('Class', axis=1).values.astype(np.float32)
    y = data['Class'].values.astype(np.float32)
    
    feature_names = [col for col in data.columns if col != 'Class']
    print(f"Features: {len(feature_names)}")
    
    # Split data
    print("\n[3/7] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale features
    print("\n[4/7] Scaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test, method='standard'
    )
    
    # Resample training data
    print("\n[5/7] Resampling training data...")
    X_resampled, y_resampled = resample_data(X_train_scaled, y_train, config)
    
    # Train models
    print("\n[6/7] Training models...")
    
    results = {}
    pipelines = {}
    
    # Train Neural Network
    if config.get('train_nn', True):
        print("\n" + "-"*50)
        print("Training Neural Network...")
        print("-"*50)
        
        nn_model = get_model(config, input_dim=X_train_scaled.shape[1])
        nn_model = train_model(nn_model, X_resampled, y_resampled, X_val_scaled, y_val, config)
        
        nn_pipeline = FraudDetectionPipeline(
            nn_model, scaler, threshold=0.5, model_type='pytorch'
        )
        
        # Optimize threshold
        opt_thresh = nn_pipeline.optimize_threshold(
            X_val_scaled, y_val, 
            method=config['evaluation']['optimize_threshold_method'],
            scale=False
        )
        print(f"Neural Net optimal threshold: {opt_thresh:.4f}")
        
        # Evaluate
        nn_probs = nn_pipeline.predict_proba(X_test_scaled, scale=False)
        nn_metrics = print_evaluation_report(y_test, nn_probs, "Neural Network")
        
        results['neural_net'] = nn_metrics
        pipelines['neural_net'] = nn_pipeline
    
    # Train LightGBM
    if config.get('train_lightgbm', True):
        print("\n" + "-"*50)
        print("Training LightGBM...")
        print("-"*50)
        
        try:
            from src.modeling.tree_models import train_lightgbm, TreeModelWrapper
            
            lgb_model, _ = train_lightgbm(X_resampled, y_resampled, X_val_scaled, y_val, config)
            
            if lgb_model is not None:
                lgb_wrapper = TreeModelWrapper(lgb_model, 'lightgbm')
                lgb_pipeline = FraudDetectionPipeline(
                    lgb_wrapper, scaler, threshold=0.5, model_type='tree'
                )
                
                # Optimize threshold
                opt_thresh = lgb_pipeline.optimize_threshold(
                    X_val_scaled, y_val,
                    method=config['evaluation']['optimize_threshold_method'],
                    scale=False
                )
                print(f"LightGBM optimal threshold: {opt_thresh:.4f}")
                
                # Evaluate
                lgb_probs = lgb_pipeline.predict_proba(X_test_scaled, scale=False)
                lgb_metrics = print_evaluation_report(y_test, lgb_probs, "LightGBM")
                
                results['lightgbm'] = lgb_metrics
                pipelines['lightgbm'] = lgb_pipeline
        except ImportError as e:
            print(f"LightGBM not available: {e}")
    
    # Ensemble predictions
    if len(pipelines) > 1:
        print("\n" + "-"*50)
        print("Creating Ensemble...")
        print("-"*50)
        
        # Create ensemble with equal weights
        ensemble_probs = create_ensemble_predictions(
            pipelines, X_test_scaled, weights=None
        )
        
        ensemble_metrics = print_evaluation_report(y_test, ensemble_probs, "Ensemble")
        results['ensemble'] = ensemble_metrics
    
    # Probability calibration
    if config.get('calibration', {}).get('enabled', True):
        print("\n" + "-"*50)
        print("Probability Calibration...")
        print("-"*50)
        
        best_model_name = max(results.keys(), key=lambda k: results[k]['pr_auc'])
        best_pipeline = pipelines.get(best_model_name)
        
        if best_pipeline:
            val_probs = best_pipeline.predict_proba(X_val_scaled, scale=False)
            test_probs = best_pipeline.predict_proba(X_test_scaled, scale=False)
            
            calibrated_probs, calibrator = calibrate_model_predictions(
                y_val, val_probs, y_test, test_probs,
                method=config.get('calibration', {}).get('method', 'platt')
            )
            
            best_pipeline.set_calibrator(calibrator)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['pr_auc'])
    best_pr_auc = results[best_model]['pr_auc']
    print(f"\nBest Model: {best_model} with PR-AUC = {best_pr_auc:.4f}")
    
    # Save results
    print("\n[7/7] Saving results...")
    
    # Save predictions
    best_pipeline = pipelines.get(best_model)
    if best_pipeline:
        test_probs = best_pipeline.predict_proba(X_test_scaled, scale=False)
        results_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred_proba': test_probs,
            'y_pred': (test_probs >= best_pipeline.threshold).astype(int)
        })
        results_df.to_csv(config['data']['results_path'], index=False)
        print(f"Results saved to {config['data']['results_path']}")
    
    # Save best model
    if 'neural_net' in pipelines:
        nn_pipeline = pipelines['neural_net']
        os.makedirs(os.path.dirname(config['model']['pipeline_save_path']), exist_ok=True)
        torch.save({
            'model_state_dict': nn_pipeline.model.state_dict(),
            'scaler': scaler,
            'threshold': nn_pipeline.threshold,
            'model_architecture': config['model'],
            'feature_names': feature_names
        }, config['model']['pipeline_save_path'])
        print(f"Pipeline saved to {config['model']['pipeline_save_path']}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    
    return results, pipelines


if __name__ == "__main__":
    main()
