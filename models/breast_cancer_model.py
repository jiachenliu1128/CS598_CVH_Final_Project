"""
Full Breast Cancer Detection Model (Fusion Only).

Combines:
- Image Features (Precomputed 768-dim from ConvNeXt)
- Metadata Branch (MLP for patient metadata)
- Late Fusion Classifier
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from .metadata_branch import MetadataBranch
from .fusion import GatedFusionClassifier
from .mil import AttentionMIL


class BreastCancerModel(nn.Module):
    """
    End-to-end Breast Cancer Detection Model with Gated Fusion.
    
    Architecture:
        ┌─────────────────┐     ┌─────────────────┐
        │  Image Features │     │    Metadata     │
        │  (Precomputed)  │     │   (age, etc.)   │
        │    [B, N, 768]  │     │                 │
        └────────┬────────┘     └────────┬────────┘
                 │                       │
                 ▼                       ▼
        ┌─────────────────┐     ┌─────────────────┐
        │  Attention MIL  │     │ Metadata Branch │
        │    (Aggregation)│     │     (MLP)       │
        └────────┬────────┘     └────────┬────────┘
                 │                       │
                 │       768-dim         │    64-dim
                 │                       │
                 └───────────┬───────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Gated Fusion   │
                    │  Classifier     │
                    └────────┬────────┘
                             │
                             ▼
                    Cancer Probability
    """
    
    def __init__(
        self,
        image_feature_dim: int = 768,
        metadata_output_dim: int = 64,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        """
        Initialize the Breast Cancer Detection Model.
        """
        super().__init__()
        
        self.image_feature_dim = image_feature_dim
        self.metadata_output_dim = metadata_output_dim
        
        # MIL Module
        self.mil_module = AttentionMIL(feature_dim=image_feature_dim)
        
        # Metadata branch
        self.metadata_branch = MetadataBranch(output_dim=metadata_output_dim)
        
        # Gated Fusion classifier
        self.fusion_classifier = GatedFusionClassifier(
            image_feature_dim=image_feature_dim,
            metadata_feature_dim=metadata_output_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
            use_sigmoid=False
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the fusion model.
        
        Args:
            image_features: Precomputed image features [batch_size, num_views, image_feature_dim]
            metadata: Dictionary with encoded metadata
            mask: [batch_size, num_views] - True for real views, False for padding
                
        Returns:
            Cancer prediction logits [batch_size, 1]
        """
        # Apply MIL if input is a sequence of views
        if image_features.dim() == 3:
            if mask is None:
                # Assume all valid if no mask provided
                mask = torch.ones(image_features.shape[:2], dtype=torch.bool, device=image_features.device)
            image_features = self.mil_module(image_features, mask)
        
        # Metadata branch
        metadata_features = self.metadata_branch.forward_dict(metadata)  # [B, 64]
        
        # Fusion and classification
        logits = self.fusion_classifier(image_features, metadata_features)  # [B, 1]
        
        return logits
    
    def forward_features(
        self,
        image_features: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass that also returns intermediate features.
        """
        if image_features.dim() == 3:
            if mask is None:
                mask = torch.ones(image_features.shape[:2], dtype=torch.bool, device=image_features.device)
            image_features = self.mil_module(image_features, mask)
            
        metadata_features = self.metadata_branch.forward_dict(metadata)
        logits = self.fusion_classifier(image_features, metadata_features)
        probabilities = torch.sigmoid(logits)
        
        return {
            'image_features': image_features,
            'metadata_features': metadata_features,
            'logits': logits,
            'probabilities': probabilities
        }


def test_breast_cancer_model():
    """Test the full Breast Cancer Model."""
    print("Testing BreastCancerModel (Fusion + MIL)...")
    
    # Create model
    model = BreastCancerModel(
        image_feature_dim=768,
        metadata_output_dim=64,
        classifier_hidden_dim=256
    )
    
    # Create dummy batch
    batch_size = 4
    num_views = 3
    # [B, N, D]
    image_features = torch.randn(batch_size, num_views, 768)
    # Mask: last view is padding for 2nd and 4th sample
    mask = torch.ones(batch_size, num_views, dtype=torch.bool)
    mask[1, 2] = False
    mask[3, 2] = False
    
    metadata = {
        'age': torch.tensor([55.0, 62.0, 48.0, 70.0]),
        'implant': torch.tensor([0, 1, 0, 0]),
        'density': torch.tensor([0, 1, 4, 2]),  # A, B, Missing, C
        'site_id': torch.tensor([0, 1, 0, 1])
    }
    
    # Forward pass
    print("\n1. Basic forward pass (with MIL):")
    logits = model(image_features, metadata, mask)
    print(f"   Input: images={image_features.shape}, mask={mask.shape}")
    print(f"   Output logits: {logits.shape}")
    assert logits.shape == (batch_size, 1)
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_breast_cancer_model()

