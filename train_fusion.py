"""
Training script for Multimodal Fusion (Precomputed Features).

Trains ONLY the fusion classifier and metadata branch.
Inputs:
- Image features: Precomputed 768-dim vectors (from .pt files)
- Metadata: Raw values from CSV (processed by MetadataBranch)

This training is very fast (no CNN backprop).
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
        images = batch['images'].to(device)  # [B, N, 768]
        mask = batch['mask'].to(device)      # [B, N]
        metadata = {k: v.to(device) for k, v in batch['metadata'].items()}
        labels = batch['labels'].to(device).unsqueeze(1)
        
        # Forward
        optimizer.zero_grad()
        logits = model(images, metadata, mask)
        
        # Soft labeling: 1.0 -> 0.9, 0.0 -> 0.1 (Regularization)
        # This prevents the model from being "too sure" and overfitting
        smooth_labels = labels * 0.9 + (1 - labels) * 0.1
        loss = criterion(logits, smooth_labels)
        
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
        mask = batch['mask'].to(device)
        metadata = {k: v.to(device) for k, v in batch['metadata'].items()}
        labels = batch['labels'].to(device).unsqueeze(1)
        
        # Forward
        logits = model(images, metadata, mask)
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
    # Dynamic threshold optimization (sweep)
    best_f1 = 0.0
    best_thresh = 0.5
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    for t in thresholds:
        p_binary = (preds > t).astype(int)
        # Fast F1 calc
        tp = ((p_binary == 1) & (labels.astype(int) == 1)).sum()
        fp = ((p_binary == 1) & (labels.astype(int) == 0)).sum()
        fn = ((p_binary == 0) & (labels.astype(int) == 1)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    # Use best threshold for reporting
    preds_binary = (preds > best_thresh).astype(int).flatten()
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
    
    # Prediction statistics
    pred_stats = {
        'pred_min': float(preds.min()),
        'pred_max': float(preds.max()),
        'pred_mean': float(preds.mean()),
        'pred_std': float(preds.std()),
        'num_pred_positive': int((preds > best_thresh).sum()),
        'best_threshold': float(best_thresh)
    }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': best_f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        **pred_stats
    }


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = RSNABreastCancerDataset(
        csv_path=args.train_csv,
        image_dir=args.features_dir
    )
    val_dataset = RSNABreastCancerDataset(
        csv_path=args.val_csv,
        image_dir=args.features_dir
    )
    
    # Create sampler for imbalanced training
    from torch.utils.data import WeightedRandomSampler
    
    # Calculate weights for each sample
    labels = train_dataset.breast_df['cancer'].values
    class_counts = np.bincount(labels.astype(int))
    weight_per_class = 1.0 / class_counts
    samples_weights = weight_per_class[labels.astype(int)]
    samples_weights = torch.from_numpy(samples_weights).double()
    
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
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
        dropout=args.dropout
    ).to(device)
    
    # Load pretrained metadata weights (optional)
    if args.pretrained_metadata and os.path.exists(args.pretrained_metadata):
        print(f"Loading pretrained metadata weights from {args.pretrained_metadata}")
        state = torch.load(args.pretrained_metadata, map_location='cpu', weights_only=False)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        # Extract metadata_branch weights
        weights = {k.replace('metadata_branch.', ''): v 
                  for k, v in state.items() if 'metadata_branch' in k}
        model.metadata_branch.load_state_dict(weights, strict=False)
        print("✓ Loaded metadata weights")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    # Use BCEWithLogitsLoss with label smoothing instead of Focal Loss for better calibration
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device)) 
    # Note: pos_weight helps with recall, but too high kills precision. 
    # Soft labels are handled manually in the training loop if needed, 
    # but BCEWithLogitsLoss supports label_smoothing in newer PyTorch versions.
    # We will implement manual soft labels in the loop.
    
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
              f"Rec: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f} (@ {train_metrics['best_threshold']:.2f})")
        print(f"Train Preds - Min: {train_metrics['pred_min']:.4f}, Max: {train_metrics['pred_max']:.4f}, "
              f"Mean: {train_metrics['pred_mean']:.4f}, #Pos: {train_metrics['num_pred_positive']}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val   - Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, "
              f"Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f} (@ {val_metrics['best_threshold']:.2f})")
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
    parser = argparse.ArgumentParser(description='Train fusion model on precomputed features')
    
    # Data
    parser.add_argument('--train_csv', type=str, default='label/train.csv')
    parser.add_argument('--val_csv', type=str, default='label/val.csv')
    parser.add_argument('--features_dir', type=str, default='features/')
    parser.add_argument('--output_dir', type=str, default='checkpoints_fusion')
    
    # Model
    parser.add_argument('--image_dim', type=int, default=768)
    parser.add_argument('--metadata_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Pretraining (optional)
    parser.add_argument('--pretrained_metadata', type=str, default=None)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
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

