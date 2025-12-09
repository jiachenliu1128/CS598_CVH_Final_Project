"""
Training script for full multimodal breast cancer detection model.

Trains image branch + metadata branch + fusion classifier jointly.
Supports loading pretrained weights from separate models.
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
from models import BreastCancerModel
from utils.losses import FocalLoss


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move to device
        images = batch['images'].to(device)
        metadata = {k: v.to(device) for k, v in batch['metadata'].items()}
        labels = batch['labels'].to(device).unsqueeze(1)
        
        # Forward
        optimizer.zero_grad()
        logits = model(images, metadata)
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
        images = batch['images'].to(device)
        metadata = {k: v.to(device) for k, v in batch['metadata'].items()}
        labels = batch['labels'].to(device).unsqueeze(1)
        
        # Forward
        logits = model(images, metadata)
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
    
    # Prediction statistics
    pred_stats = {
        'pred_min': float(preds.min()),
        'pred_max': float(preds.max()),
        'pred_mean': float(preds.mean()),
        'pred_std': float(preds.std()),
        'num_pred_positive': int((preds > threshold).sum())
    }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        **pred_stats
    }


def load_pretrained_weights(model, metadata_path=None, image_path=None):
    """Load pretrained weights for branches"""
    if metadata_path and os.path.exists(metadata_path):
        print(f"Loading pretrained metadata weights from {metadata_path}")
        metadata_state = torch.load(metadata_path, map_location='cpu')
        # Handle different save formats
        if 'model_state_dict' in metadata_state:
            metadata_state = metadata_state['model_state_dict']
        # Extract just metadata_branch weights
        metadata_weights = {k.replace('metadata_branch.', ''): v 
                           for k, v in metadata_state.items() 
                           if 'metadata_branch' in k}
        model.metadata_branch.load_state_dict(metadata_weights, strict=False)
        print("✓ Loaded metadata weights")
    
    if image_path and os.path.exists(image_path):
        print(f"Loading pretrained image weights from {image_path}")
        image_state = torch.load(image_path, map_location='cpu')
        if 'model_state_dict' in image_state:
            image_state = image_state['model_state_dict']
        # Extract just image_branch weights
        image_weights = {k.replace('image_branch.', ''): v 
                        for k, v in image_state.items() 
                        if 'image_branch' in k}
        model.image_branch.load_state_dict(image_weights, strict=False)
        print("✓ Loaded image weights")


def freeze_branch(branch, freeze=True):
    """Freeze or unfreeze a branch"""
    for param in branch.parameters():
        param.requires_grad = not freeze


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
    model = BreastCancerModel(
        image_feature_dim=args.image_dim,
        metadata_output_dim=args.metadata_dim,
        classifier_hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_placeholder_image_branch=args.use_placeholder
    ).to(device)
    
    # Load pretrained weights
    if args.pretrained_metadata or args.pretrained_image:
        load_pretrained_weights(
            model, 
            args.pretrained_metadata, 
            args.pretrained_image
        )
    
    # Freeze branches if specified
    if args.freeze_metadata:
        print("Freezing metadata branch")
        freeze_branch(model.metadata_branch, freeze=True)
    if args.freeze_image:
        print("Freezing image branch")
        freeze_branch(model.image_branch, freeze=True)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
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
        print(f"Train Preds - Min: {train_metrics['pred_min']:.4f}, Max: {train_metrics['pred_max']:.4f}, "
              f"Mean: {train_metrics['pred_mean']:.4f}, #Pos: {train_metrics['num_pred_positive']}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val   - Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, "
              f"Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"Val Preds - Min: {val_metrics['pred_min']:.4f}, Max: {val_metrics['pred_max']:.4f}, "
              f"Mean: {val_metrics['pred_mean']:.4f}, #Pos: {val_metrics['num_pred_positive']}")
        
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
    parser = argparse.ArgumentParser(description='Train full multimodal model')
    
    # Data
    parser.add_argument('--train_csv', type=str, default='label/train.csv')
    parser.add_argument('--val_csv', type=str, default='label/val.csv')
    parser.add_argument('--output_dir', type=str, default='checkpoints_full')
    
    # Model
    parser.add_argument('--image_dim', type=int, default=1024)
    parser.add_argument('--metadata_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--use_placeholder', action='store_true', 
                       help='Use placeholder image branch (for testing)')
    
    # Pretrained weights
    parser.add_argument('--pretrained_metadata', type=str, default=None,
                       help='Path to pretrained metadata weights')
    parser.add_argument('--pretrained_image', type=str, default=None,
                       help='Path to pretrained image weights')
    parser.add_argument('--freeze_metadata', action='store_true',
                       help='Freeze metadata branch (train fusion only)')
    parser.add_argument('--freeze_image', action='store_true',
                       help='Freeze image branch (train fusion only)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Loss
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    # Metrics
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)

