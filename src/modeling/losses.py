import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
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

class HardNegativeMiner:
    """Hard Negative Mining for training."""
    def __init__(self, neg_pos_ratio=3):
        self.neg_pos_ratio = neg_pos_ratio

    def mine_hard_negatives(self, logits, targets):
        pos_mask = targets == 1
        neg_mask = targets == 0

        if pos_mask.sum() == 0:
            return torch.arange(len(targets))

        num_pos = pos_mask.sum()
        num_neg = min(neg_mask.sum(), int(num_pos * self.neg_pos_ratio))

        if num_neg == 0:
            return torch.where(pos_mask)[0]

        neg_losses = nn.functional.binary_cross_entropy_with_logits(
            logits[neg_mask], targets[neg_mask].float(), reduction='none'
        )
        _, hard_neg_idx = torch.topk(neg_losses, num_neg)

        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0][hard_neg_idx]

        return torch.cat([pos_idx, neg_idx])
