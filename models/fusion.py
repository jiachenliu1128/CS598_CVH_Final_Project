"""
Late Fusion Module for Breast Cancer Detection.

Combines image features (from CNN backbone) with metadata features
for final cancer prediction.
"""

import torch
import torch.nn as nn
from typing import Optional


class GatedFusionClassifier(nn.Module):
    """
    Gated Fusion Classifier using metadata to gate image features.
    
    Mechanism:
    1. Metadata features -> Attention Gate (0-1)
    2. Image features * Gate -> Gated Image features
    3. Concatenate(Gated Image, Metadata) -> Classifier
    """
    
    def __init__(
        self,
        image_feature_dim: int = 1024,
        metadata_feature_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_sigmoid: bool = False
    ):
        super().__init__()
        
        self.image_feature_dim = image_feature_dim
        self.metadata_feature_dim = metadata_feature_dim
        self.use_sigmoid = use_sigmoid
        
        # Gating Mechanism
        # Metadata -> Gate for Image Features
        self.gate_layer = nn.Sequential(
            nn.Linear(metadata_feature_dim, image_feature_dim),
            nn.Sigmoid()
        )
        
        # Combined feature dimension
        combined_dim = image_feature_dim + metadata_feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Optional sigmoid for inference
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()
        
    def forward(
        self,
        image_features: torch.Tensor,
        metadata_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with gating.
        """
        # 1. Compute Gate from metadata
        gate = self.gate_layer(metadata_features)  # [B, image_dim]
        
        # 2. Gate image features
        gated_image = image_features * gate  # Element-wise multiplication
        
        # 3. Concatenate features
        combined = torch.cat([gated_image, metadata_features], dim=-1)
        
        # 4. Classification
        logits = self.classifier(combined)
        
        return self.sigmoid(logits)


# For backward compatibility, aliasing LateFusionClassifier to standard one
# But we should swap to GatedFusionClassifier in BreastCancerModel
class LateFusionClassifier(nn.Module):
    """Standard Late Fusion (Concatenation)"""
    # ... existing implementation ...
    
    def __init__(
        self,
        image_feature_dim: int = 1024,
        metadata_feature_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_sigmoid: bool = False
    ):
        """
        Initialize the Late Fusion Classifier.
        
        Args:
            image_feature_dim: Dimension of image features from CNN backbone
            metadata_feature_dim: Dimension of metadata features from MetadataBranch
            hidden_dim: Hidden layer dimension (default: 256)
            dropout: Dropout probability (default: 0.3)
            use_sigmoid: Whether to apply sigmoid at output (default: False)
                         Set to False when using FocalLoss (expects logits)
        """
        super().__init__()
        
        self.image_feature_dim = image_feature_dim
        self.metadata_feature_dim = metadata_feature_dim
        self.use_sigmoid = use_sigmoid
        
        # Combined feature dimension
        combined_dim = image_feature_dim + metadata_feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Optional sigmoid for inference
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize linear layer weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        image_features: torch.Tensor,
        metadata_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through fusion classifier.
        
        Args:
            image_features: Image features from CNN [batch_size, image_feature_dim]
            metadata_features: Metadata features [batch_size, metadata_feature_dim]
            
        Returns:
            Cancer prediction logits (or probabilities if use_sigmoid=True)
            Shape: [batch_size, 1]
        """
        # Concatenate features
        combined = torch.cat([image_features, metadata_features], dim=-1)
        
        # Classification
        logits = self.classifier(combined)
        
        # Optional sigmoid
        output = self.sigmoid(logits)
        
        return output


class ImageOnlyClassifier(nn.Module):
    """
    Classifier using only image features (no metadata).
    Useful for ablation studies.
    """
    
    def __init__(
        self,
        image_feature_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_sigmoid: bool = False
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(image_features)
        return self.sigmoid(logits)


class MetadataOnlyClassifier(nn.Module):
    """
    Classifier using only metadata features (no images).
    Useful for ablation studies and understanding metadata contribution.
    """
    
    def __init__(
        self,
        metadata_feature_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        use_sigmoid: bool = False
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(metadata_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()
    
    def forward(self, metadata_features: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(metadata_features)
        return self.sigmoid(logits)

