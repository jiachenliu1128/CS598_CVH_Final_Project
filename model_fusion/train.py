import os
import torch
import numpy as np

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from dataloader import BreastDataset, collate_breast
from model_fusion import BreastClassifier
    
    
    
################################################################################
# training script
################################################################################   
# config
train_csv_path = "label/train.csv"
valid_csv_path = "label/val.csv"
features_dir = "features"
meta_cols = ["laterality", "age", "biopsy", "invasive", "BIRADS", "implant", "density", "machine_id", "difficult_negative_case"]  # adjust
feat_dim = torch.load(os.path.join(features_dir, os.listdir(features_dir)[0])).numel()
meta_dim = len(meta_cols)
epochs = 10

# datasets and dataloaders
train_dataset = BreastDataset(
    csv_path=train_csv_path,
    features_dir=features_dir,
    meta_cols=meta_cols
)

valid_dataset = BreastDataset(
    csv_path=valid_csv_path,
    features_dir=features_dir,
    meta_cols=meta_cols
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    collate_fn=collate_breast
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=collate_breast
)

# model and optimizer
model = BreastClassifier(feat_dim=feat_dim, meta_dim=meta_dim)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Lower learning rate
print("Starting training...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Get device from model parameters
device = next(model.parameters()).device
print(f"Using device: {device}")

# training loop
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch} ---------------------")
    
    # training
    model.train()
    trained_count = 0
    train_losses = []
    
    for feats, mask, metas, labels in train_loader:
        print(f"Training samples: {trained_count}/{len(train_dataset)}", end='\r')
        trained_count += feats.size(0)
        feats, mask, metas, labels = feats.cuda(), mask.cuda(), metas.cuda(), labels.cuda()
        
        # Check for NaN in inputs
        if torch.isnan(feats).any() or torch.isnan(metas).any():
            print(f"\nWARNING: NaN in training inputs at sample {trained_count}")
            continue
        
        # Forward pass
        logits = model(feats, mask, metas)
        
        # Check for NaN in logits
        if torch.isnan(logits).any():
            print(f"\nWARNING: NaN in training logits at sample {trained_count}")
            print(f"Feats range: [{feats.min():.4f}, {feats.max():.4f}]")
            print(f"Metas range: [{metas.min():.4f}, {metas.max():.4f}]")
            continue
        
        # Compute loss 
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"\nWARNING: NaN loss at sample {trained_count}")
            continue
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update model parameters
        optimizer.step()
        train_losses.append(loss.item())

    # validation
    print("\nValidating...")
    model.eval()
    val_losses = []
    preds = []
    trues = []
    valid_count = 0
    
    with torch.no_grad():
        for feats, mask, metas, labels in valid_loader:
            print(f"Validating samples: {valid_count}/{len(valid_dataset)}", end='\r')
            valid_count += feats.size(0)
            
            feats = feats.cuda()
            mask = mask.cuda()
            metas = metas.cuda()
            labels = labels.cuda()
            
            # Debug: Check input ranges
            if torch.isnan(feats).any() or torch.isinf(feats).any():
                print(f"\nWARNING: Invalid values in validation feats")
                print(f"NaN count: {torch.isnan(feats).sum()}, Inf count: {torch.isinf(feats).sum()}")
            if torch.isnan(metas).any() or torch.isinf(metas).any():
                print(f"\nWARNING: Invalid values in validation metas")
                print(f"NaN count: {torch.isnan(metas).sum()}, Inf count: {torch.isinf(metas).sum()}")

            logits = model(feats, mask, metas)
            
            # Debug: Check if logits are valid
            if torch.isnan(logits).any():
                print(f"\nERROR: NaN in validation logits at count {valid_count}")
                print(f"Feats stats - min: {feats.min():.4f}, max: {feats.max():.4f}, mean: {feats.mean():.4f}")
                print(f"Metas stats - min: {metas.min():.4f}, max: {metas.max():.4f}, mean: {metas.mean():.4f}")
                # Check model weights
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"NaN found in parameter: {name}")
                breakpoint()
            
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            val_losses.append(loss.item())

            preds.extend(torch.sigmoid(logits).cpu().tolist())
            trues.extend(labels.cpu().tolist())

    # summarize epoch results
    avg_train_loss = np.mean(train_losses) if train_losses else 0.0
    avg_valid_loss = np.mean(val_losses) if val_losses else 0.0
    
    # Find optimal threshold using PPV (Precision)
    best_threshold = 0.5
    best_ppv = 0.0
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    for thresh in thresholds:
        preds_binary_temp = [1 if p >= thresh else 0 for p in preds]
        ppv_temp = precision_score(trues, preds_binary_temp, zero_division=0)
        if ppv_temp > best_ppv:
            best_ppv = ppv_temp
            best_threshold = thresh
    
    # Calculate metrics with optimal threshold
    preds_binary = [1 if p >= best_threshold else 0 for p in preds]
    precision = precision_score(trues, preds_binary, zero_division=0)  # PPV = Precision
    recall = recall_score(trues, preds_binary, zero_division=0)
    f1 = f1_score(trues, preds_binary, zero_division=0)
    f1_macro = f1_score(trues, preds_binary, average='macro', zero_division=0)  # Macro F1
    
    # Calculate specificity (True Negative Rate)
    from sklearn.metrics import confusion_matrix
    if len(set(trues)) > 1:  # Only if both classes exist
        tn, fp, fn, tp = confusion_matrix(trues, preds_binary).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    else:
        specificity = 0.0
        npv = 0.0
    
    # Calculate AUC if there are both classes
    try:
        auc = roc_auc_score(trues, preds)
    except ValueError:
        auc = 0.0
    
    print(f"\nEpoch {epoch}: Train Loss={avg_train_loss:.4f}, Valid Loss={avg_valid_loss:.4f}")
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"PPV (Precision): {precision:.4f}, Recall (Sensitivity): {recall:.4f}, Specificity: {specificity:.4f}")
    print(f"NPV: {npv:.4f}, F1: {f1:.4f}, Macro F1: {f1_macro:.4f}, AUC: {auc:.4f}")
    

# test 
print("\n" + "="*50)
print("Testing on test set...")
print("="*50)

test_csv_path = "label/test.csv"
test_dataset = BreastDataset(
    csv_path=test_csv_path,
    features_dir=features_dir,
    meta_cols=meta_cols
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=collate_breast
)

model.eval()
test_preds = []
test_trues = []
test_count = 0

with torch.no_grad():
    for feats, mask, metas, labels in test_loader:
        print(f"Testing samples: {test_count}/{len(test_dataset)}", end='\r')
        test_count += feats.size(0)
        
        feats, mask, metas, labels = feats.cuda(), mask.cuda(), metas.cuda(), labels.cuda()
        logits = model(feats, mask, metas)
        test_preds.extend(torch.sigmoid(logits).cpu().tolist())
        test_trues.extend(labels.cpu().tolist())

# Calculate test metrics with optimal threshold
test_preds_binary = [1 if p >= best_threshold else 0 for p in test_preds]
test_precision = precision_score(test_trues, test_preds_binary, zero_division=0)
test_recall = recall_score(test_trues, test_preds_binary, zero_division=0)
test_f1 = f1_score(test_trues, test_preds_binary, zero_division=0)
test_f1_macro = f1_score(test_trues, test_preds_binary, average='macro', zero_division=0)

tn, fp, fn, tp = confusion_matrix(test_trues, test_preds_binary).ravel()
test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
test_npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

try:
    test_auc = roc_auc_score(test_trues, test_preds)
except ValueError:
    test_auc = 0.0

print(f"\nTest Results:")
print(f"PPV (Precision): {test_precision:.4f}, Recall (Sensitivity): {test_recall:.4f}, Specificity: {test_specificity:.4f}")
print(f"NPV: {test_npv:.4f}, F1: {test_f1:.4f}, Macro F1: {test_f1_macro:.4f}, AUC: {test_auc:.4f}")