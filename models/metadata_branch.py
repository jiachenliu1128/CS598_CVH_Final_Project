import torch
import torch.nn as nn
from typing import Dict, Optional

class MetadataBranch(nn.Module):
    """
    Enhanced Metadata Branch for Breast Cancer Classification.
    
    Features:
        - age: Continuous, normalized with z-score
        - implant: Binary (0/1)
        - density: Categorical (A=0, B=1, C=2, D=3, Missing=4)
        - site_id: Categorical (0, 1)
        - laterality: Categorical (L=0, R=1)
        - view: Categorical (CC=0, MLO=1, other=2, missing=3)
        - machine_id: Categorical (variable)
    
    Architecture:
        Embeddings + normalization + optional feature interactions -> Deep MLP -> 64-dim output
    """
    
    # Constants for normalization
    AGE_MEAN = 58.0
    AGE_STD = 10.0
    
    # Categorical mappings
    DENSITY_MAP = {'A':0, 'B':1, 'C':2, 'D':3}
    DENSITY_MISSING_IDX = 4
    NUM_DENSITY_CATEGORIES = 5
    
    LATERALITY_MAP = {'L':0, 'R':1}
    NUM_LATERALITY = 2
    
    VIEW_MAP = {'CC':0, 'MLO':1}
    NUM_VIEWS = 4  # CC, MLO, other, missing
    
    NUM_SITES = 2
    
    def __init__(
        self,
        output_dim: int = 64,
        density_embed_dim: int = 16,
        site_embed_dim: int = 8,
        laterality_embed_dim: int = 2,
        view_embed_dim: int = 4,
        machine_embed_dim: int = 8,
        dropout: float = 0.2,
        num_machines: int = 10
    ):
        super().__init__()
        self.output_dim = output_dim
        
        # Embeddings
        self.density_embedding = nn.Embedding(self.NUM_DENSITY_CATEGORIES, density_embed_dim)
        self.site_embedding = nn.Embedding(self.NUM_SITES, site_embed_dim)
        self.laterality_embedding = nn.Embedding(self.NUM_LATERALITY, laterality_embed_dim)
        self.view_embedding = nn.Embedding(self.NUM_VIEWS, view_embed_dim)
        self.machine_embedding = nn.Embedding(num_machines, machine_embed_dim)
        
        # MLP input dim
        mlp_input_dim = 1 + 1 + density_embed_dim + site_embed_dim + laterality_embed_dim + view_embed_dim + machine_embed_dim
        
        # Deep MLP
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.density_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.site_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.laterality_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.view_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.machine_embedding.weight, mean=0, std=0.02)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def normalize_age(self, age: torch.Tensor) -> torch.Tensor:
        return ((age - self.AGE_MEAN) / self.AGE_STD).unsqueeze(-1)
    
    @staticmethod
    def encode_density(density_values: list) -> torch.Tensor:
        indices = [MetadataBranch.DENSITY_MAP.get(val, MetadataBranch.DENSITY_MISSING_IDX) for val in density_values]
        return torch.tensor(indices, dtype=torch.long)
    
    @staticmethod
    def encode_laterality(values: list) -> torch.Tensor:
        indices = [MetadataBranch.LATERALITY_MAP.get(v, 0) for v in values]
        return torch.tensor(indices, dtype=torch.long)
    
    @staticmethod
    def encode_view(values: list) -> torch.Tensor:
        indices = [MetadataBranch.VIEW_MAP.get(v, 2) for v in values]  # 2=other/missing
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(
        self,
        age: torch.Tensor,
        implant: torch.Tensor,
        density: torch.Tensor,
        site_id: torch.Tensor,
        laterality: Optional[torch.Tensor] = None,
        view: Optional[torch.Tensor] = None,
        machine_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        age_feat = self.normalize_age(age)
        implant_feat = implant.float().unsqueeze(-1)
        density_feat = self.density_embedding(density)
        site_feat = self.site_embedding(site_id)
        laterality_feat = self.laterality_embedding(laterality) if laterality is not None else torch.zeros((age.shape[0], self.laterality_embedding.embedding_dim), device=age.device)
        view_feat = self.view_embedding(view) if view is not None else torch.zeros((age.shape[0], self.view_embedding.embedding_dim), device=age.device)
        machine_feat = self.machine_embedding(machine_id) if machine_id is not None else torch.zeros((age.shape[0], self.machine_embedding.embedding_dim), device=age.device)
        
        combined = torch.cat([age_feat, implant_feat, density_feat, site_feat, laterality_feat, view_feat, machine_feat], dim=-1)
        return self.mlp(combined)
    
    def forward_dict(self, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(
            age=metadata['age'],
            implant=metadata['implant'],
            density=metadata['density'],
            site_id=metadata['site_id'],
            laterality=metadata.get('laterality'),
            view=metadata.get('view'),
            machine_id=metadata.get('machine_id')
        )


def test_metadata_branch():
    """Test and validate the enhanced MetadataBranch."""
    print("Testing Enhanced MetadataBranch...")
    branch = MetadataBranch(output_dim=64)
    
    batch_size = 4
    age = torch.tensor([55.0, 62.0, 48.0, 70.0])
    implant = torch.tensor([0, 1, 0, 0])
    density = torch.tensor([0, 1, 4, 2])  # A, B, Missing, C
    site_id = torch.tensor([0, 1, 0, 1])
    laterality = torch.tensor([0, 1, 0, 1])
    view = torch.tensor([0, 1, 2, 3])
    machine_id = torch.tensor([0, 1, 2, 3])
    
    output = branch(age, implant, density, site_id, laterality, view, machine_id)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 64)
    
    metadata = {'age': age, 'implant': implant, 'density': density, 'site_id': site_id,
                'laterality': laterality, 'view': view, 'machine_id': machine_id}
    output_dict = branch.forward_dict(metadata)
    assert torch.allclose(output, output_dict), "Dict forward mismatch"
    
    density_strings = ['A', 'B', None, 'D', 'C']
    encoded = MetadataBranch.encode_density(density_strings)
    expected = torch.tensor([0, 1, 4, 3, 2])
    assert torch.equal(encoded, expected)
    
    print("All tests passed!")
    total_params = sum(p.numel() for p in branch.parameters())
    trainable_params = sum(p.numel() for p in branch.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable: {trainable_params}")


if __name__ == "__main__":
    test_metadata_branch()
