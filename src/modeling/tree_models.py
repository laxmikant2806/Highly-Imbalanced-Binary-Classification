"""
Tree-based models for fraud detection.
LightGBM typically achieves best performance on tabular fraud data.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score


def train_lightgbm(X_train, y_train, X_val, y_val, config):
    """
    Train a LightGBM model for fraud detection.
    
    Returns:
        model: Trained LightGBM model
        best_iteration: Best iteration based on validation
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Install with: pip install lightgbm")
        return None, None
    
    # Calculate class weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / (n_pos + 1e-8)
    
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'boosting_type': 'gbdt',
        'num_leaves': config.get('lightgbm', {}).get('num_leaves', 31),
        'max_depth': config.get('lightgbm', {}).get('max_depth', -1),
        'learning_rate': config.get('lightgbm', {}).get('learning_rate', 0.05),
        'feature_fraction': config.get('lightgbm', {}).get('feature_fraction', 0.8),
        'bagging_fraction': config.get('lightgbm', {}).get('bagging_fraction', 0.8),
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'min_child_samples': config.get('lightgbm', {}).get('min_child_samples', 20),
        'reg_alpha': config.get('lightgbm', {}).get('reg_alpha', 0.1),
        'reg_lambda': config.get('lightgbm', {}).get('reg_lambda', 0.1),
        'verbose': -1,
        'seed': config['data_processing']['random_state'],
        'n_jobs': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    print(f"Training LightGBM with scale_pos_weight={scale_pos_weight:.2f}")
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=config.get('lightgbm', {}).get('num_boost_round', 1000),
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_pr_auc = average_precision_score(y_val, val_preds)
    print(f"LightGBM Validation PR-AUC: {val_pr_auc:.4f}")
    
    return model, model.best_iteration


def train_xgboost(X_train, y_train, X_val, y_val, config):
    """
    Train an XGBoost model for fraud detection.
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Install with: pip install xgboost")
        return None, None
    
    # Calculate class weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / (n_pos + 1e-8)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': config.get('xgboost', {}).get('max_depth', 6),
        'learning_rate': config.get('xgboost', {}).get('learning_rate', 0.05),
        'subsample': config.get('xgboost', {}).get('subsample', 0.8),
        'colsample_bytree': config.get('xgboost', {}).get('colsample_bytree', 0.8),
        'scale_pos_weight': scale_pos_weight,
        'reg_alpha': config.get('xgboost', {}).get('reg_alpha', 0.1),
        'reg_lambda': config.get('xgboost', {}).get('reg_lambda', 1.0),
        'seed': config['data_processing']['random_state'],
        'n_jobs': -1,
        'verbosity': 0
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    print(f"Training XGBoost with scale_pos_weight={scale_pos_weight:.2f}")
    
    evals = [(dtrain, 'train'), (dval, 'valid')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config.get('xgboost', {}).get('num_boost_round', 1000),
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    # Evaluate
    val_preds = model.predict(dval)
    val_pr_auc = average_precision_score(y_val, val_preds)
    print(f"XGBoost Validation PR-AUC: {val_pr_auc:.4f}")
    
    return model, model.best_iteration


class TreeModelWrapper:
    """
    Wrapper to provide consistent interface for tree models.
    """
    def __init__(self, model, model_type='lightgbm'):
        self.model = model
        self.model_type = model_type
    
    def predict_proba(self, X):
        """Return fraud probabilities."""
        if self.model_type == 'lightgbm':
            return self.model.predict(X)
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X)
            return self.model.predict(dmatrix)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict(self, X, threshold=0.5):
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def feature_importance(self, importance_type='gain'):
        """Return feature importance."""
        if self.model_type == 'lightgbm':
            return self.model.feature_importance(importance_type=importance_type)
        elif self.model_type == 'xgboost':
            return list(self.model.get_score(importance_type='gain').values())
