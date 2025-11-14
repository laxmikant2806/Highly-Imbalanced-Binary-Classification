import os
import torch
import torch.optim as optim
from sklearn.metrics import precision_recall_curve, auc
from src.modeling.losses import FocalLoss, HardNegativeMiner

def train_model(model, X_train, y_train, X_val, y_val, cfg):
    """Trains the fraud detection model."""
    criterion = FocalLoss(**cfg['loss']['focal_loss'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=cfg['training']['patience'],
        factor=cfg['training']['factor']
    )

    hard_miner = HardNegativeMiner(neg_pos_ratio=cfg['training']['neg_pos_ratio']) if cfg['training']['use_hard_mining'] else None

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    best_pr_auc = 0
    patience_counter = 0

    for epoch in range(cfg['training']['epochs']):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_tensor)

        if cfg['training']['use_hard_mining'] and hard_miner:
            selected_idx = hard_miner.mine_hard_negatives(logits, y_train_tensor)
            loss = criterion(logits[selected_idx], y_train_tensor[selected_idx])
        else:
            loss = criterion(logits, y_train_tensor)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['training']['max_grad_norm'])
        optimizer.step()

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
                    torch.save(model.state_dict(), cfg['model']['model_save_path'])
                else:
                    patience_counter += 1

                print(f'Epoch {epoch}: Loss={loss.item():.4f}, Val PR-AUC={pr_auc:.4f}')

                if patience_counter >= cfg['training']['patience']:
                    print("Early stopping triggered")
                    break

    if os.path.exists(cfg['model']['model_save_path']):
        model.load_state_dict(torch.load(cfg['model']['model_save_path']))

    return model
