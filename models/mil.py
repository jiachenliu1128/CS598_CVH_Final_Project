import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    def __init__(self, feature_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, D] - Batch of image features
            mask: [B, N] - Mask indicating valid views (True) vs padding (False)
            
        Returns:
            [B, D] - Aggregated breast-level feature vector
        """
        # Calculate attention scores: [B, N, 1]
        scores = self.attention(features)
        
        # Squeeze to [B, N]
        scores = scores.squeeze(-1)
        
        # Mask out padding (set to -inf so softmax makes them 0)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax: [B, N]
        # Handle case where all are masked (though dataset should prevent this) safely
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum: [B, N, 1] * [B, N, D] -> [B, N, D] -> sum(dim=1) -> [B, D]
        weights = weights.unsqueeze(-1)
        weighted_features = torch.sum(weights * features, dim=1)
        
        return weighted_features

