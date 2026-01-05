"""
Advanced loss functions for imbalanced classification.
Includes Weighted BCE, Focal Loss, and Asymmetric Focal Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss.
    Automatically calculates pos_weight from class distribution.
    """
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    @staticmethod
    def calculate_pos_weight(y_train):
        """Calculate pos_weight from training labels."""
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        return n_neg / (n_pos + 1e-8)
    
    def forward(self, inputs, targets):
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=inputs.device, dtype=inputs.dtype)
        else:
            pos_weight = None
        
        return F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=pos_weight,
            reduction=self.reduction
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance.
    Down-weights easy negatives and focuses on hard examples.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (default 0.25)
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard examples
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Get probabilities
        p = torch.sigmoid(inputs)
        
        # Compute focal weights
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weights
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply focal and alpha weights
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss - higher penalty for false negatives.
    Useful when missing fraud is much more costly than false alarms.
    
    Args:
        gamma_pos: Focusing parameter for positive class (default 0)
        gamma_neg: Focusing parameter for negative class (default 4)
        clip: Probability clipping threshold
    """
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, reduction='mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        
        # Clip probabilities for numerical stability
        p_clip = torch.clamp(p, min=self.clip, max=1 - self.clip)
        
        # Positive samples (fraud)
        pos_loss = targets * torch.log(p_clip) * ((1 - p_clip) ** self.gamma_pos)
        
        # Negative samples (non-fraud)
        neg_loss = (1 - targets) * torch.log(1 - p_clip) * (p_clip ** self.gamma_neg)
        
        loss = -(pos_loss + neg_loss)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Combines multiple losses with configurable weights.
    """
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)
    
    def forward(self, inputs, targets):
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        return total_loss


class HardNegativeMiner:
    """Hard Negative Mining for training - selects hardest negative samples."""
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

        neg_losses = F.binary_cross_entropy_with_logits(
            logits[neg_mask], targets[neg_mask].float(), reduction='none'
        )
        _, hard_neg_idx = torch.topk(neg_losses, num_neg)

        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0][hard_neg_idx]

        return torch.cat([pos_idx, neg_idx])


def get_loss_function(config, y_train=None):
    """
    Factory function to get the appropriate loss function.
    
    Args:
        config: Configuration dictionary
        y_train: Training labels for calculating class weights
    """
    loss_type = config['loss'].get('type', 'focal')
    
    if loss_type == 'weighted_bce':
        pos_weight = config['loss'].get('pos_weight', None)
        if pos_weight is None and y_train is not None:
            pos_weight = WeightedBCELoss.calculate_pos_weight(y_train)
            print(f"Calculated pos_weight: {pos_weight:.2f}")
        return WeightedBCELoss(pos_weight=pos_weight)
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=config['loss']['focal_loss'].get('alpha', 0.25),
            gamma=config['loss']['focal_loss'].get('gamma', 2.0)
        )
    
    elif loss_type == 'asymmetric_focal':
        return AsymmetricFocalLoss(
            gamma_pos=config['loss'].get('gamma_pos', 0),
            gamma_neg=config['loss'].get('gamma_neg', 4)
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
