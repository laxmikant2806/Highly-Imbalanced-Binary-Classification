"""
Enhanced training module for fraud detection.
Supports multiple model types and advanced training techniques.
"""
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import numpy as np

from src.modeling.losses import get_loss_function, HardNegativeMiner


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # min
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


def train_model(model, X_train, y_train, X_val, y_val, cfg):
    """
    Enhanced training function with multiple improvements:
    - Configurable loss functions
    - Learning rate scheduling (OneCycle, Cosine, ReduceOnPlateau)
    - Gradient clipping
    - Hard negative mining
    - Early stopping based on PR-AUC
    """
    # Get loss function
    criterion = get_loss_function(cfg, y_train)
    print(f"Using loss function: {type(criterion).__name__}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler_type = cfg['training'].get('scheduler', 'reduce_on_plateau')
    epochs = cfg['training']['epochs']
    
    if scheduler_type == 'one_cycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg['training']['learning_rate'] * 10,
            epochs=epochs,
            steps_per_epoch=1,
            pct_start=0.3
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=cfg['training']['learning_rate'] * 0.01
        )
    else:  # reduce_on_plateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=cfg['training']['patience'],
            factor=cfg['training']['factor'],
            min_lr=1e-7
        )
    
    # Hard negative mining
    hard_miner = None
    if cfg['training'].get('use_hard_mining', False):
        hard_miner = HardNegativeMiner(neg_pos_ratio=cfg['training']['neg_pos_ratio'])
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Training state
    best_pr_auc = 0
    early_stopping = EarlyStopping(
        patience=cfg['training'].get('early_stopping_patience', 15),
        min_delta=0.001
    )
    best_model_state = None
    
    # Batch training for large datasets
    batch_size = cfg['training'].get('batch_size', None)
    use_batches = batch_size is not None and len(X_train) > batch_size
    
    for epoch in range(epochs):
        model.train()
        
        if use_batches:
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X_train_tensor[batch_idx]
                y_batch = y_train_tensor[batch_idx]
                
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=cfg['training']['max_grad_norm']
                )
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            epoch_loss = total_loss / n_batches
        else:
            # Full batch training
            optimizer.zero_grad()
            logits = model(X_train_tensor)
            
            if hard_miner:
                selected_idx = hard_miner.mine_hard_negatives(logits, y_train_tensor)
                loss = criterion(logits[selected_idx], y_train_tensor[selected_idx])
            else:
                loss = criterion(logits, y_train_tensor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=cfg['training']['max_grad_norm']
            )
            optimizer.step()
            epoch_loss = loss.item()
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_probs = torch.sigmoid(val_logits).numpy()
                
                pr_auc = average_precision_score(y_val, val_probs)
                
                # Update scheduler
                if scheduler_type == 'reduce_on_plateau':
                    scheduler.step(pr_auc)
                else:
                    scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}: Loss={epoch_loss:.4f}, Val PR-AUC={pr_auc:.4f}, LR={current_lr:.6f}')
                
                if pr_auc > best_pr_auc:
                    best_pr_auc = pr_auc
                    best_model_state = model.state_dict().copy()
                    
                    # Save checkpoint
                    save_path = cfg['model'].get('model_save_path', 'models/best_fraud_model.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                
                if early_stopping(pr_auc):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with PR-AUC: {best_pr_auc:.4f}")
    
    return model


def train_all_models(X_train, y_train, X_val, y_val, config):
    """
    Train all configured models and return them.
    """
    from src.modeling.model import get_model
    from src.modeling.tree_models import train_lightgbm, train_xgboost, TreeModelWrapper
    
    models = {}
    
    # Neural Network
    if config.get('train_nn', True):
        print("\n" + "="*50)
        print("Training Neural Network...")
        print("="*50)
        
        nn_model = get_model(config, input_dim=X_train.shape[1])
        nn_model = train_model(nn_model, X_train, y_train, X_val, y_val, config)
        models['neural_net'] = nn_model
    
    # LightGBM
    if config.get('train_lightgbm', True):
        print("\n" + "="*50)
        print("Training LightGBM...")
        print("="*50)
        
        lgb_model, _ = train_lightgbm(X_train, y_train, X_val, y_val, config)
        if lgb_model is not None:
            models['lightgbm'] = TreeModelWrapper(lgb_model, 'lightgbm')
    
    # XGBoost
    if config.get('train_xgboost', False):
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        xgb_model, _ = train_xgboost(X_train, y_train, X_val, y_val, config)
        if xgb_model is not None:
            models['xgboost'] = TreeModelWrapper(xgb_model, 'xgboost')
    
    return models
