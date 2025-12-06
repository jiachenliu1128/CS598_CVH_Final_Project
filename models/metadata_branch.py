"""
Metadata Branch for Breast Cancer Detection Model.

This module processes patient metadata (age, implant, density, site_id) 
and outputs a fixed-dimensional feature vector for late fusion with image features.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class MetadataBranch(nn.Module):
    """
    Processes patient metadata and outputs a feature vector.
    
    Features:
        - age: Continuous, normalized with z-score
        - implant: Binary (0/1)
        - density: Categorical (A=0, B=1, C=2, D=3, Missing=4)
        - site_id: Categorical (0, 1)
    
    Architecture:
        Embeddings + normalization -> Concat -> MLP -> 64-dim output
    """
    
    # Constants for normalization (computed from training data)
    AGE_MEAN = 58.0
    AGE_STD = 10.0
    
    # Mapping for density categories
    DENSITY_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    DENSITY_MISSING_IDX = 4
    NUM_DENSITY_CATEGORIES = 5  # A, B, C, D, Missing
    
    # Number of sites in dataset
    NUM_SITES = 2
    
    def __init__(
        self,
        output_dim: int = 64,
        density_embed_dim: int = 8,
        site_embed_dim: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize the MetadataBranch.
        
        Args:
            output_dim: Dimension of output feature vector (default: 64)
            density_embed_dim: Embedding dimension for density (default: 8)
            site_embed_dim: Embedding dimension for site_id (default: 4)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()
        
        self.output_dim = output_dim
        
        # Embeddings for categorical features
        self.density_embedding = nn.Embedding(
            num_embeddings=self.NUM_DENSITY_CATEGORIES,
            embedding_dim=density_embed_dim
        )
        self.site_embedding = nn.Embedding(
            num_embeddings=self.NUM_SITES,
            embedding_dim=site_embed_dim
        )
        
        # Calculate input dimension to MLP
        # age (1) + implant (1) + density_embed (8) + site_embed (4) = 14
        mlp_input_dim = 1 + 1 + density_embed_dim + site_embed_dim
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding and linear layer weights."""
        # Initialize embeddings with small random values
        nn.init.normal_(self.density_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.site_embedding.weight, mean=0, std=0.02)
        
        # Initialize linear layers
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def normalize_age(self, age: torch.Tensor) -> torch.Tensor:
        """
        Normalize age using z-score normalization.
        
        Args:
            age: Tensor of ages [batch_size]
            
        Returns:
            Normalized ages [batch_size, 1]
        """
        normalized = (age - self.AGE_MEAN) / self.AGE_STD
        return normalized.unsqueeze(-1)
    
    @staticmethod
    def encode_density(density_values: list) -> torch.Tensor:
        """
        Encode density values to indices.
        
        Args:
            density_values: List of density strings ('A', 'B', 'C', 'D') or NaN
            
        Returns:
            Tensor of density indices [batch_size]
        """
        indices = []
        for val in density_values:
            if val in MetadataBranch.DENSITY_MAP:
                indices.append(MetadataBranch.DENSITY_MAP[val])
            else:
                # Missing or unknown -> MISSING category
                indices.append(MetadataBranch.DENSITY_MISSING_IDX)
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(
        self,
        age: torch.Tensor,
        implant: torch.Tensor,
        density: torch.Tensor,
        site_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through metadata branch.
        
        Args:
            age: Patient ages [batch_size], float
            implant: Implant status [batch_size], binary (0/1)
            density: Density category indices [batch_size], int (0-4)
            site_id: Site indices [batch_size], int (0-1)
            
        Returns:
            Metadata feature vector [batch_size, output_dim]
        """
        # Normalize age
        age_norm = self.normalize_age(age)  # [B, 1]
        
        # Implant as float
        implant_feat = implant.float().unsqueeze(-1)  # [B, 1]
        
        # Get embeddings
        density_feat = self.density_embedding(density)  # [B, density_embed_dim]
        site_feat = self.site_embedding(site_id)  # [B, site_embed_dim]
        
        # Concatenate all features
        combined = torch.cat([
            age_norm,
            implant_feat,
            density_feat,
            site_feat
        ], dim=-1)  # [B, 14]
        
        # Pass through MLP
        output = self.mlp(combined)  # [B, output_dim]
        
        return output
    
    def forward_dict(self, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using a dictionary of metadata.
        
        Args:
            metadata: Dictionary with keys 'age', 'implant', 'density', 'site_id'
            
        Returns:
            Metadata feature vector [batch_size, output_dim]
        """
        return self.forward(
            age=metadata['age'],
            implant=metadata['implant'],
            density=metadata['density'],
            site_id=metadata['site_id']
        )


def test_metadata_branch():
    """Test the MetadataBranch module."""
    print("Testing MetadataBranch...")
    
    # Create module
    branch = MetadataBranch(output_dim=64)
    print(f"Created MetadataBranch with output_dim=64")
    
    # Create dummy batch
    batch_size = 4
    age = torch.tensor([55.0, 62.0, 48.0, 70.0])
    implant = torch.tensor([0, 1, 0, 0])
    density = torch.tensor([0, 1, 4, 2])  # A, B, Missing, C
    site_id = torch.tensor([0, 1, 0, 1])
    
    # Forward pass
    output = branch(age, implant, density, site_id)
    print(f"Input shapes: age={age.shape}, implant={implant.shape}, density={density.shape}, site_id={site_id.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 64), f"Expected (4, 64), got {output.shape}"
    
    # Test with dict input (use eval mode to disable dropout randomness)
    branch.eval()
    metadata = {
        'age': age,
        'implant': implant,
        'density': density,
        'site_id': site_id
    }
    output_eval = branch(age, implant, density, site_id)
    output_dict = branch.forward_dict(metadata)
    assert torch.allclose(output_eval, output_dict), "Dict forward should match regular forward"
    branch.train()  # Back to train mode
    
    # Test density encoding
    density_strings = ['A', 'B', None, 'D', 'C']
    encoded = MetadataBranch.encode_density(density_strings)
    expected = torch.tensor([0, 1, 4, 3, 2])
    assert torch.equal(encoded, expected), f"Density encoding failed: {encoded} vs {expected}"
    
    print("All tests passed!")
    print(f"\nModel summary:")
    print(f"  - Total parameters: {sum(p.numel() for p in branch.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in branch.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    test_metadata_branch()

