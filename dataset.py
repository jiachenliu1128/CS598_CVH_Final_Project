"""
Dataset classes for RSNA Breast Cancer Detection.

Handles:
- Loading and preprocessing metadata from CSV
- Breast-level grouping (patient_id + laterality)
- Missing value handling for metadata
- Image loading (placeholder for now)
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class RSNABreastCancerDataset(Dataset):
    """
    Dataset for RSNA Breast Cancer Detection.
    
    Groups data by breast (patient_id + laterality) for breast-level predictions.
    Each sample represents one breast with potentially multiple images.
    
    Returns:
        - images: Placeholder tensor (real implementation will load DICOM)
        - metadata: Dictionary with encoded metadata
        - label: Cancer label (0 or 1)
        - breast_id: String identifier (patient_id_laterality)
    """
    
    # Mapping for density categories
    DENSITY_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    DENSITY_MISSING_IDX = 4
    
    # Site ID mapping
    SITE_MAP = {1: 0, 2: 1}
    
    # Age statistics for normalization (from training data)
    AGE_MEAN = 58.0
    AGE_MEDIAN = 58.0
    
    def __init__(
        self,
        csv_path: str,
        image_dir: Optional[str] = None,
        transform: Optional[Any] = None,
        return_image_ids: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to the CSV file (train.csv, val.csv, or test.csv)
            image_dir: Directory containing DICOM images (optional for now)
            transform: Image transforms to apply (optional)
            return_image_ids: Whether to return image IDs for each breast
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.transform = transform
        self.return_image_ids = return_image_ids
        
        # Load and process CSV
        self.df = pd.read_csv(csv_path)
        
        # Create breast-level grouping
        self.breast_df = self._create_breast_level_df()
        
        # Create index mapping
        self.breast_ids = self.breast_df['breast_id'].tolist()
        
        print(f"Loaded {len(self.df)} images from {len(self.breast_ids)} breasts")
    
    def _create_breast_level_df(self) -> pd.DataFrame:
        """
        Aggregate image-level data to breast-level.
        
        Groups by (patient_id, laterality) and aggregates:
        - cancer: max (if any image has cancer=1, breast has cancer)
        - age, density, site_id, implant: first value (same for all images of a breast)
        - image_ids: list of all image IDs for this breast
        """
        # Create breast identifier
        self.df['breast_id'] = (
            self.df['patient_id'].astype(str) + '_' + self.df['laterality']
        )
        
        # Aggregate to breast level
        breast_df = self.df.groupby('breast_id').agg({
            'patient_id': 'first',
            'laterality': 'first',
            'cancer': 'max',  # If any image has cancer, breast has cancer
            'age': 'first',
            'density': 'first',
            'site_id': 'first',
            'implant': 'first',
            'image_id': list,  # Keep all image IDs
            'view': list  # Keep all views
        }).reset_index()
        
        # Rename image_id column
        breast_df = breast_df.rename(columns={'image_id': 'image_ids', 'view': 'views'})
        
        return breast_df
    
    def _encode_density(self, density: Any) -> int:
        """Encode density value to index."""
        if pd.isna(density) or density not in self.DENSITY_MAP:
            return self.DENSITY_MISSING_IDX
        return self.DENSITY_MAP[density]
    
    def _encode_site(self, site_id: int) -> int:
        """Encode site_id to index."""
        return self.SITE_MAP.get(site_id, 0)
    
    def _handle_missing_age(self, age: float) -> float:
        """Handle missing age values."""
        if pd.isna(age):
            return self.AGE_MEDIAN
        return float(age)
    
    def __len__(self) -> int:
        """Return number of breasts in dataset."""
        return len(self.breast_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single breast sample.
        
        Args:
            idx: Index of the breast
            
        Returns:
            Dictionary with:
                - 'images': Placeholder tensor [1, 3, 224, 224]
                - 'metadata': Dict with encoded metadata tensors
                - 'label': Cancer label tensor
                - 'breast_id': String identifier
                - 'image_ids': List of image IDs (if return_image_ids=True)
        """
        row = self.breast_df.iloc[idx]
        
        # Create placeholder image (will be replaced with DICOM loading)
        # Shape: [num_images, C, H, W] for MIL, but using [1, 3, 224, 224] for now
        images = torch.randn(1, 3, 224, 224)
        
        # Encode metadata
        metadata = {
            'age': torch.tensor(self._handle_missing_age(row['age']), dtype=torch.float32),
            'implant': torch.tensor(int(row['implant']), dtype=torch.long),
            'density': torch.tensor(self._encode_density(row['density']), dtype=torch.long),
            'site_id': torch.tensor(self._encode_site(row['site_id']), dtype=torch.long)
        }
        
        # Label
        label = torch.tensor(float(row['cancer']), dtype=torch.float32)
        
        # Build output
        output = {
            'images': images,
            'metadata': metadata,
            'label': label,
            'breast_id': row['breast_id']
        }
        
        if self.return_image_ids:
            output['image_ids'] = row['image_ids']
            output['views'] = row['views']
        
        return output


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader.
    
    Stacks tensors and preserves metadata structure.
    """
    # Stack images
    images = torch.stack([item['images'] for item in batch])
    
    # Stack metadata tensors
    metadata = {
        'age': torch.stack([item['metadata']['age'] for item in batch]),
        'implant': torch.stack([item['metadata']['implant'] for item in batch]),
        'density': torch.stack([item['metadata']['density'] for item in batch]),
        'site_id': torch.stack([item['metadata']['site_id'] for item in batch])
    }
    
    # Stack labels
    labels = torch.stack([item['label'] for item in batch])
    
    # Collect breast IDs
    breast_ids = [item['breast_id'] for item in batch]
    
    output = {
        'images': images,
        'metadata': metadata,
        'labels': labels,
        'breast_ids': breast_ids
    }
    
    # Optional fields
    if 'image_ids' in batch[0]:
        output['image_ids'] = [item['image_ids'] for item in batch]
        output['views'] = [item['views'] for item in batch]
    
    return output


def get_class_weights(dataset: RSNABreastCancerDataset) -> Tuple[float, float]:
    """
    Calculate class weights for handling imbalance.
    
    Returns:
        Tuple of (weight_negative, weight_positive)
    """
    labels = dataset.breast_df['cancer'].values
    num_positive = labels.sum()
    num_negative = len(labels) - num_positive
    
    # Weight inversely proportional to class frequency
    weight_positive = len(labels) / (2 * num_positive)
    weight_negative = len(labels) / (2 * num_negative)
    
    return weight_negative, weight_positive


def get_dataset_stats(dataset: RSNABreastCancerDataset) -> Dict[str, Any]:
    """
    Get statistics about the dataset.
    """
    df = dataset.breast_df
    
    stats = {
        'num_breasts': len(df),
        'num_patients': df['patient_id'].nunique(),
        'cancer_count': int(df['cancer'].sum()),
        'cancer_rate': float(df['cancer'].mean()),
        'age_mean': float(df['age'].mean()),
        'age_std': float(df['age'].std()),
        'density_distribution': df['density'].value_counts(dropna=False).to_dict(),
        'site_distribution': df['site_id'].value_counts().to_dict(),
        'laterality_distribution': df['laterality'].value_counts().to_dict(),
        'images_per_breast': {
            'mean': float(df['image_ids'].apply(len).mean()),
            'min': int(df['image_ids'].apply(len).min()),
            'max': int(df['image_ids'].apply(len).max())
        }
    }
    
    return stats


def test_dataset():
    """Test the dataset class."""
    print("Testing RSNABreastCancerDataset...")
    
    # Use the label/train.csv file
    csv_path = "label/train.csv"
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        print("Skipping dataset test.")
        return
    
    # Create dataset
    dataset = RSNABreastCancerDataset(
        csv_path=csv_path,
        return_image_ids=True
    )
    
    # Get stats
    print("\n1. Dataset Statistics:")
    stats = get_dataset_stats(dataset)
    print(f"   Breasts: {stats['num_breasts']}")
    print(f"   Patients: {stats['num_patients']}")
    print(f"   Cancer rate: {stats['cancer_rate']:.2%}")
    print(f"   Age: {stats['age_mean']:.1f} ± {stats['age_std']:.1f}")
    print(f"   Images per breast: {stats['images_per_breast']}")
    
    # Test single item
    print("\n2. Single Item:")
    item = dataset[0]
    print(f"   breast_id: {item['breast_id']}")
    print(f"   images shape: {item['images'].shape}")
    print(f"   label: {item['label']}")
    print(f"   metadata:")
    for key, value in item['metadata'].items():
        print(f"      {key}: {value}")
    print(f"   image_ids: {item['image_ids'][:3]}... ({len(item['image_ids'])} total)")
    
    # Test DataLoader
    print("\n3. DataLoader Test:")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Images shape: {batch['images'].shape}")
    print(f"   Labels shape: {batch['labels'].shape}")
    print(f"   Metadata shapes:")
    for key, value in batch['metadata'].items():
        print(f"      {key}: {value.shape}")
    
    # Test class weights
    print("\n4. Class Weights:")
    weight_neg, weight_pos = get_class_weights(dataset)
    print(f"   Negative class weight: {weight_neg:.4f}")
    print(f"   Positive class weight: {weight_pos:.4f}")
    print(f"   Ratio (pos/neg): {weight_pos/weight_neg:.2f}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_dataset()

