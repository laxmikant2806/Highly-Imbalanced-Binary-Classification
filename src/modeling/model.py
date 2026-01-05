"""
Enhanced neural network models for fraud detection.
Includes improved MLP with BatchNorm, residual connections, and attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection (with projection if dimensions differ)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        out = self.fc(x)
        out = self.bn(out)
        out = F.leaky_relu(out, 0.1)
        out = self.dropout(out)
        return out + identity


class FraudDetector(nn.Module):
    """
    Improved neural network model for fraud detection.
    Features:
    - Input BatchNorm for continuous features
    - Residual connections
    - LeakyReLU activation
    - Proper weight initialization
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3, use_residual=True):
        super().__init__()
        
        self.use_residual = use_residual
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            if use_residual and i > 0:  # Use residual blocks after first layer
                layers.append(ResidualBlock(prev_dim, hidden_dim, dropout))
            else:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout)
                ])
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming initialization for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.input_bn(x)
        x = self.hidden_layers(x)
        return self.output(x).squeeze(-1)


class FraudDetectorWithAttention(nn.Module):
    """
    Advanced fraud detector with self-attention mechanism.
    Can learn importance of different features.
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3, num_heads=4):
        super().__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Feature attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP backbone
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.input_bn(x)
        
        # Apply self-attention (treat features as sequence)
        x_unsq = x.unsqueeze(1)  # [B, 1, D]
        attn_out, _ = self.attention(x_unsq, x_unsq, x_unsq)
        x = attn_out.squeeze(1) + x  # Residual connection
        
        x = self.backbone(x)
        return self.output(x).squeeze(-1)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple neural network models.
    Averages predictions from different architectures.
    """
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)


def get_model(config, input_dim):
    """
    Factory function to create the appropriate model.
    
    Args:
        config: Configuration dictionary
        input_dim: Number of input features
    """
    model_type = config['model'].get('type', 'mlp')
    hidden_dims = config['model'].get('hidden_dims', [256, 128, 64])
    dropout = config['model'].get('dropout', 0.3)
    
    if model_type == 'mlp':
        return FraudDetector(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_residual=config['model'].get('use_residual', True)
        )
    
    elif model_type == 'attention':
        return FraudDetectorWithAttention(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_heads=config['model'].get('num_heads', 4)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
