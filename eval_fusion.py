"""
Evaluation script for Fusion Model.

Loads a trained model and evaluates it on the validation set
across multiple decision thresholds to find the optimal operating point.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from dataset import RSNABreastCancerDataset, collate_fn
from models import BreastCancerModel


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Get all predictions and labels"""
    model.eval()
    all_probs = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc='Evaluating'):
        # Move to device
        images = batch['images'].to(device)
        metadata = {k: v.to(device) for k, v in batch['metadata'].items()}
        labels = batch['labels'].numpy()
        
        # Forward
        logits = model(images, metadata)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        all_probs.extend(probs)
        all_labels.extend(labels)
        
    return np.array(all_probs), np.array(all_labels)


def compute_metrics_at_threshold(probs, labels, threshold):
    """Compute metrics for a specific threshold"""
    preds = (probs > threshold).astype(int)
    
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading validation set: {args.val_csv}")
    dataset = RSNABreastCancerDataset(
        csv_path=args.val_csv,
        image_dir=args.features_dir
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Create model
    print("Creating model...")
    model = BreastCancerModel(
        image_feature_dim=args.image_dim,
        metadata_output_dim=args.metadata_dim,
        classifier_hidden_dim=args.hidden_dim
    ).to(device)
    
    # Load weights
    print(f"Loading weights from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Get predictions
    probs, labels = get_predictions(model, dataloader, device)
    
    # 1. AUC-ROC
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    print(f"\nAUC-ROC: {roc_auc:.4f}")
    
    # 2. Optimal Threshold Analysis
    print("\nThreshold Analysis:")
    print(f"{'Thresh':<10} {'F1':<10} {'Prec':<10} {'Rec':<10} {'Acc':<10}")
    print("-" * 50)
    
    results = []
    best_f1 = 0
    best_thresh = 0
    
    # Sweep thresholds
    thresholds = np.arange(0.1, 0.95, 0.05)
    for t in thresholds:
        metrics = compute_metrics_at_threshold(probs, labels, t)
        results.append(metrics)
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_thresh = t
            
        print(f"{t:<10.2f} {metrics['f1']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['accuracy']:.4f}")
    
    print("-" * 50)
    print(f"Best F1: {best_f1:.4f} at threshold {best_thresh:.2f}")
    
    # 3. Probabilistic F1 (pKaggle metric approximation)
    # The competition used pF1, which is F1 optimized on probabilities
    # Here we just show standard F1 at best threshold
    
    # 4. Save predictions
    if args.save_preds:
        df = pd.DataFrame({
            'breast_id': dataset.breast_ids,
            'label': labels,
            'probability': probs
        })
        df.to_csv('val_predictions.csv', index=False)
        print("\nPredictions saved to val_predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fusion model')
    
    parser.add_argument('--val_csv', type=str, default='label/val.csv')
    parser.add_argument('--features_dir', type=str, default='features/')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Model config (must match training)
    parser.add_argument('--image_dim', type=int, default=768)
    parser.add_argument('--metadata_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_preds', action='store_true')
    
    args = parser.parse_args()
    main(args)

