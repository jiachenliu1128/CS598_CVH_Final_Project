"""
Training script for metadata-only breast cancer detection baseline.

Trains MetadataBranch + MetadataOnlyClassifier without image features.
Provides baseline performance for comparison with full multimodal model.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from dataset import RSNABreastCancerDataset, collate_fn
from models.metadata_branch import MetadataBranch
from models.fusion import MetadataOnlyClassifier
from utils.losses import FocalLoss


class MetadataOnlyModel(nn.Module):
    """Combined model: MetadataBranch + MetadataOnlyClassifier"""
    
    def __init__(self, metadata_dim=64, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.metadata_branch = MetadataBranch(output_dim=metadata_dim)
        self.classifier = MetadataOnlyClassifier(
            metadata_feature_dim=metadata_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_sigmoid=False  # Use logits for FocalLoss
        )
    
    def forward(self, metadata):
        features = self.metadata_branch.forward_dict(metadata)
        logits = self.classifier(features)
        return logits


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move to device
        metadata = {k: v.to(device) for k, v in batch['metadata'].items()}
        labels = batch['labels'].to(device).unsqueeze(1)
        
        # Forward
        optimizer.zero_grad()
        logits = model(metadata)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Validation')
    for batch in pbar:
        # Move to device
        metadata = {k: v.to(device) for k, v in batch['metadata'].items()}
        labels = batch['labels'].to(device).unsqueeze(1)
        
        # Forward
        logits = model(metadata)
        loss = criterion(logits, labels)
        
        # Metrics
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, threshold=0.5):
    """Compute classification metrics"""
    preds_binary = (preds > threshold).astype(int).flatten()
    labels_binary = labels.astype(int).flatten()
    
    # Confusion matrix
    tp = ((preds_binary == 1) & (labels_binary == 1)).sum()
    tn = ((preds_binary == 0) & (labels_binary == 0)).sum()
    fp = ((preds_binary == 1) & (labels_binary == 0)).sum()
    fn = ((preds_binary == 0) & (labels_binary == 1)).sum()
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = RSNABreastCancerDataset(args.train_csv)
    val_dataset = RSNABreastCancerDataset(args.val_csv)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Train: {len(train_dataset)} breasts, Val: {len(val_dataset)} breasts")
    
    # Create model
    model = MetadataOnlyModel(
        metadata_dim=args.metadata_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_metrics = compute_metrics(train_preds, train_labels, threshold=args.threshold)
        
        # Validate
        val_loss, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        val_metrics = compute_metrics(val_preds, val_labels, threshold=args.threshold)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train - Acc: {train_metrics['accuracy']:.4f}, Prec: {train_metrics['precision']:.4f}, "
              f"Rec: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val   - Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, "
              f"Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, args.output_dir / 'best_loss_model.pth')
            print(f"Saved best loss model (val_loss: {val_loss:.4f})")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, args.output_dir / 'best_f1_model.pth')
            print(f"Saved best F1 model (val_f1: {val_metrics['f1']:.4f})")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train metadata-only baseline')
    
    # Data
    parser.add_argument('--train_csv', type=str, default='label/train.csv')
    parser.add_argument('--val_csv', type=str, default='label/val.csv')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    
    # Model
    parser.add_argument('--metadata_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Loss
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    # Metrics
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)

