import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Weighted BCE Loss
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        return nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight
        )

# Hard Negative Mining
class HardNegativeMiner:
    def __init__(self, neg_pos_ratio=3):
        self.neg_pos_ratio = neg_pos_ratio
    
    def mine_hard_negatives(self, logits, targets):
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        if pos_mask.sum() == 0:
            return torch.arange(len(targets))
        
        num_pos = pos_mask.sum()
        num_neg = min(neg_mask.sum(), num_pos * self.neg_pos_ratio)
        
        if num_neg == 0:
            return torch.where(pos_mask)[0]
        
        neg_losses = nn.functional.binary_cross_entropy_with_logits(
            logits[neg_mask], targets[neg_mask].float(), reduction='none'
        )
        _, hard_neg_idx = torch.topk(neg_losses, num_neg)
        
        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0][hard_neg_idx]
        
        return torch.cat([pos_idx, neg_idx])

# Neural Network Model
class FraudDetector(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()

# Text Data Augmentation
class TextAugmenter:
    def __init__(self):
        self.replacements = {
            'payment': ['transaction', 'transfer', 'purchase'],
            'card': ['credit', 'debit', 'account'],
            'merchant': ['vendor', 'retailer', 'store']
        }
        
    def synonym_replacement(self, text, n=1):
        import random
        words = text.split()
        new_words = words.copy()
        
        for i, word in enumerate(words):
            if word.lower() in self.replacements:
                new_words[i] = random.choice(self.replacements[word.lower()])
                
        return ' '.join(new_words)
    
    def back_translation(self, text, prob=0.1):
        import random
        words = text.split()
        for i in range(len(words)):
            if random.random() < prob:
                words[i] = words[i][::-1]
        return ' '.join(words)

# Comprehensive Evaluation Metrics
class ImbalancedEvaluator:
    @staticmethod
    def compute_metrics(y_true, y_pred_proba, threshold=0.5):
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # F1 at fixed recall (95%)
        target_recall = 0.95
        recall_mask = recall >= target_recall
        if np.any(recall_mask):
            recall_idx = np.where(recall_mask)[0][0]
            if precision[recall_idx] > 0 and recall[recall_idx] > 0:
                f1_at_recall = 2 * (precision[recall_idx] * recall[recall_idx]) / (precision[recall_idx] + recall[recall_idx])
            else:
                f1_at_recall = 0
        else:
            f1_at_recall = 0
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            if len(np.unique(y_pred)) == 1:
                if y_pred[0] == 0:
                    tp, fp = 0, 0
                    fn, tn = np.sum(y_true), np.sum(1 - y_true)
                else:
                    tn, fn = 0, 0
                    fp, tp = np.sum(1 - y_true), np.sum(y_true)
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        
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

# Training Function
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, use_hard_mining=True):
    # Calculate class weights
    class_counts = Counter(y_train)
    pos_weight = torch.tensor(class_counts[0] / class_counts[1], dtype=torch.float32)
    
    # Choose loss function
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # FIXED: Removed verbose parameter from ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    hard_miner = HardNegativeMiner(neg_pos_ratio=3) if use_hard_mining else None
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    best_pr_auc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_train_tensor)
        
        if use_hard_mining and hard_miner:
            selected_idx = hard_miner.mine_hard_negatives(logits, y_train_tensor)
            loss = criterion(logits[selected_idx], y_train_tensor[selected_idx])
        else:
            loss = criterion(logits, y_train_tensor)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_probs = torch.sigmoid(val_logits).numpy()
                
                precision, recall, _ = precision_recall_curve(y_val, val_probs)
                pr_auc = auc(recall, precision)
                
                scheduler.step(pr_auc)
                
                if pr_auc > best_pr_auc:
                    best_pr_auc = pr_auc
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), 'best_fraud_model.pth')
                else:
                    patience_counter += 1
                
                print(f'Epoch {epoch}: Loss={loss.item():.4f}, Val PR-AUC={pr_auc:.4f}')
                
                if patience_counter >= 10:
                    print("Early stopping triggered")
                    break
    
    # Load best model if it exists
    if os.path.exists('best_fraud_model.pth'):
        model.load_state_dict(torch.load('best_fraud_model.pth', weights_only=True))
    
    return model

# Production Pipeline
class FraudDetectionPipeline:
    def __init__(self, model, scaler, threshold=0.5):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.evaluator = ImbalancedEvaluator()
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.sigmoid(logits).numpy()
        
        return probabilities
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)
    
    def optimize_threshold(self, X_val, y_val, method='cost'):
        probabilities = self.predict_proba(X_val)
        
        precision, recall, thresholds = precision_recall_curve(y_val, probabilities)
        
        if method == 'cost':
            # Minimize cost (FN cost = 100x FP cost)
            fp_rates = 1 - precision
            fn_rates = 1 - recall
            costs = fp_rates + 100 * fn_rates
            optimal_idx = np.argmin(costs)
        elif method == 'f1':
            # Maximize F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
        
        if optimal_idx < len(thresholds):
            self.threshold = thresholds[optimal_idx]
        else:
            self.threshold = 0.5
        
        return self.threshold

# Main Execution
def main():
    print("Fraud Detection with Severe Class Imbalance")
    
    # Load or create dataset
    try:
        data = pd.read_csv('./data/creditcard.csv')
        print("Loaded real Kaggle dataset!")
    except FileNotFoundError:
        print("Creating synthetic fraud dataset...")
        # Create realistic synthetic data
        np.random.seed(42)
        n_samples = 50000
        fraud_rate = 0.0017
        
        n_fraud = int(n_samples * fraud_rate)
        n_normal = n_samples - n_fraud
        
        # Create features (V1-V28 like PCA components, Time, Amount)
        normal_features = np.random.normal(0, 1, (n_normal, 28))
        fraud_features = np.random.normal(1.5, 1.2, (n_fraud, 28))
        
        # Time and Amount features
        normal_time = np.random.uniform(0, 172800, n_normal)
        fraud_time = np.random.uniform(0, 172800, n_fraud)
        normal_amount = np.random.lognormal(3, 1.5, n_normal)
        fraud_amount = np.random.lognormal(5, 2, n_fraud)
        
        # Combine
        normal_data = np.column_stack([normal_features, normal_time, normal_amount])
        fraud_data = np.column_stack([fraud_features, fraud_time, fraud_amount])
        
        X = np.vstack([normal_data, fraud_data])
        y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
        
        # Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        
        # Create DataFrame
        columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        data = pd.DataFrame(X, columns=columns)
        data['Class'] = y
    
    print(f"Dataset shape: {data.shape}")
    print(f"Fraud rate: {data['Class'].mean():.4f}")
    print(f"Class distribution:\n{data['Class'].value_counts()}")
    
    # Prepare features and target
    X = data.drop('Class', axis=1).values
    y = data['Class'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    print("Applying SMOTE + Tomek Links...")
    try:
        smote_tomek = SMOTETomek(
            smote=SMOTE(sampling_strategy=0.1, random_state=42),
            tomek=TomekLinks(sampling_strategy='majority')
        )
        X_resampled, y_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)
        print(f"After resampling: {Counter(y_resampled)}")
    except Exception as e:
        print(f"SMOTE+Tomek failed: {e}")
        print("Using SMOTE only...")
        smote = SMOTE(sampling_strategy=0.1, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE: {Counter(y_resampled)}")
    
    # Train model
    print("Training fraud detection model...")
    model = FraudDetector(input_dim=X_train_scaled.shape[1])
    trained_model = train_model(model, X_resampled, y_resampled, X_val_scaled, y_val)
    
    # Create pipeline
    pipeline = FraudDetectionPipeline(trained_model, scaler)
    
    # Optimize threshold
    optimal_threshold = pipeline.optimize_threshold(X_val, y_val, method='cost')
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    test_probs = pipeline.predict_proba(X_test)
    evaluator = ImbalancedEvaluator()
    
    # Metrics at default threshold (0.5)
    metrics_default = evaluator.compute_metrics(y_test, test_probs, threshold=0.5)
    print(f"Metrics at 0.5 threshold:")
    print(f"  PR-AUC: {metrics_default['pr_auc']:.4f}")
    print(f"  Precision: {metrics_default['precision']:.4f}")
    print(f"  Recall: {metrics_default['recall']:.4f}")
    print(f"  Total Cost: {metrics_default['total_cost']}")
    
    # Metrics at optimal threshold
    metrics_optimal = evaluator.compute_metrics(y_test, test_probs, threshold=optimal_threshold)
    print(f"\nMetrics at optimal threshold ({optimal_threshold:.4f}):")
    print(f"  PR-AUC: {metrics_optimal['pr_auc']:.4f}")
    print(f"  Precision: {metrics_optimal['precision']:.4f}")
    print(f"  Recall: {metrics_optimal['recall']:.4f}")
    print(f"  F1 at 95% recall: {metrics_optimal['f1_at_95_recall']:.4f}")
    print(f"  Total Cost: {metrics_optimal['total_cost']}")
    
    # Save results
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred_proba': test_probs,
        'y_pred_default': (test_probs >= 0.5).astype(int),
        'y_pred_optimal': (test_probs >= optimal_threshold).astype(int)
    })
    
    results_df.to_csv('fraud_detection_results.csv', index=False)
    print("\nResults saved to 'fraud_detection_results.csv'")
    
    # Save model and pipeline
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'scaler': scaler,
        'threshold': optimal_threshold,
        'model_architecture': {
            'input_dim': X_train_scaled.shape[1],
            'hidden_dims': [256, 128, 64]
        }
    }, 'fraud_detection_pipeline.pth')
    
    print("Model and pipeline saved to 'fraud_detection_pipeline.pth'")
    
    return pipeline, metrics_optimal

if __name__ == "__main__":
    pipeline, final_metrics = main()
