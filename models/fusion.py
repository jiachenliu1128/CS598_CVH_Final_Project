import torch
import torch.nn as nn

class MetadataOnlyClassifier(nn.Module):
    """Simple MLP classifier for metadata features"""
    def __init__(self, metadata_feature_dim=64, hidden_dim=64, dropout=0.3, use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.fc = nn.Sequential(
            nn.Linear(metadata_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.fc(x)
        if self.use_sigmoid:
            return torch.sigmoid(logits)
        return logits
