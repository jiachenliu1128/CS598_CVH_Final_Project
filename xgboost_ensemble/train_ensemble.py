import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, classification_report

def load_and_prep_data():
    """
    Loads colleague's image predictions and aligns them with our splits.
    """
    print("Loading data...")
    # 1. Load Colleague's Preds (The "Super Feature")
    preds_df = pd.read_csv("../data/image_level_predictions.csv")
    
    # 2. Aggregate to Breast Level
    # Use MAX pooling for predictions (standard for cancer detection)
    breast_preds = preds_df.groupby(['patient_id', 'laterality']).agg({
        'preds': 'max', 
        'age': 'first',
        'density': 'first',
        'implant': 'first',
        'site_id': 'first',
        'cancer': 'max'
    }).reset_index()
    
    # Construct breast_id if missing or use index
    breast_preds['breast_id'] = breast_preds['patient_id'].astype(str) + "_" + breast_preds['laterality']
    
    # Rename 'preds' to 'image_score'
    breast_preds = breast_preds.rename(columns={'preds': 'image_score'})
    
    # Map density
    density_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    breast_preds['density'] = breast_preds['density'].map(density_map).fillna(-1)
    
    # 3. Load Our Splits
    train_split = pd.read_csv("../label/train.csv")
    val_split = pd.read_csv("../label/val.csv")
    test_split = pd.read_csv("../label/test.csv")
    
    train_pids = set(train_split['patient_id'].unique())
    val_pids = set(val_split['patient_id'].unique())
    test_pids = set(test_split['patient_id'].unique())
    
    # 4. Filter
    train_df = breast_preds[breast_preds['patient_id'].isin(train_pids)].copy()
    val_df = breast_preds[breast_preds['patient_id'].isin(val_pids)].copy()
    test_df = breast_preds[breast_preds['patient_id'].isin(test_pids)].copy()
    
    print(f"Splits created:")
    print(f"Train: {len(train_df)} breasts")
    print(f"Val:   {len(val_df)} breasts")
    print(f"Test:  {len(test_df)} breasts")
    
    return train_df, val_df, test_df

def train_xgb(train_df, val_df, test_df):
    features = ['image_score', 'age', 'density', 'implant', 'site_id']
    target = 'cancer'
    
    print(f"\nTraining XGBoost on features: {features}")
    
    dtrain = xgb.DMatrix(train_df[features], label=train_df[target])
    dval = xgb.DMatrix(val_df[features], label=val_df[target])
    dtest = xgb.DMatrix(test_df[features], label=test_df[target])
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    # Save Model
    model.save_model("models/xgboost_ensemble.json")
    print("\nModel saved to models/xgboost_ensemble.json")
    
    # --- Validation ---
    val_probs = model.predict(dval)
    val_auc = roc_auc_score(val_df[target], val_probs)
    
    # Find Optimal Threshold
    precisions, recalls, thresholds = precision_recall_curve(val_df[target], val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_idx = np.nanargmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\n--- Validation Results ---")
    print(f"AUC: {val_auc:.4f}")
    print(f"Best F1: {best_f1:.4f} (@ threshold {best_thresh:.4f})")
    
    # --- Test ---
    test_probs = model.predict(dtest)
    test_auc = roc_auc_score(test_df[target], test_probs)
    test_preds = (test_probs > best_thresh).astype(int)
    test_f1 = f1_score(test_df[target], test_preds)
    
    print(f"\n--- Test Results ---")
    print(f"AUC: {test_auc:.4f}")
    print(f"F1:  {test_f1:.4f} (using val threshold)")
    print("\nClassification Report (Test):")
    print(classification_report(test_df[target], test_preds))
    
    # Save Predictions
    test_df['prob'] = test_probs
    test_df['pred'] = test_preds
    output_cols = ['patient_id', 'laterality', 'cancer', 'prob', 'pred']
    test_df[output_cols].to_csv("../results/predictions/ensemble_test_predictions.csv", index=False)
    print("Test predictions saved to ../results/predictions/ensemble_test_predictions.csv")

    # --- Baseline: Image Only ---
    # What if we just used the raw image score?
    # We use the same 'best_thresh' just for fairness, or optimize it separately?
    # Let's optimize it separately on Val to be fair.
    
    # Val Optimization for Baseline
    prec, rec, threshs = precision_recall_curve(val_df[target], val_df['image_score'])
    f1s = 2 * (prec * rec) / (prec + rec)
    base_idx = np.nanargmax(f1s)
    base_thresh = threshs[base_idx]
    base_val_f1 = f1s[base_idx]
    
    # Test Baseline
    base_test_preds = (test_df['image_score'] > base_thresh).astype(int)
    base_test_f1 = f1_score(test_df[target], base_test_preds)
    base_test_auc = roc_auc_score(test_df[target], test_df['image_score'])
    
    print("\n" + "="*40)
    print("      IMPACT OF METADATA FUSION")
    print("="*40)
    print(f"Baseline (Image Only):")
    print(f"  AUC: {base_test_auc:.4f}")
    print(f"  F1:  {base_test_f1:.4f} (@ {base_thresh:.4f})")
    print("-" * 20)
    print(f"Ensemble (Image + Meta):")
    print(f"  AUC: {test_auc:.4f} ({test_auc - base_test_auc:+.4f})")
    print(f"  F1:  {test_f1:.4f} ({test_f1 - base_test_f1:+.4f})")
    print("="*40)

    # --- Subgroup Analysis ---
    print("\n" + "="*40)
    print("      SUBGROUP ANALYSIS (AUC)")
    print("="*40)
    
    # By Density
    print("\n[By Density]")
    for density in sorted(test_df['density'].unique()):
        subset = test_df[test_df['density'] == density]
        if len(subset) < 10 or subset[target].nunique() < 2:
            continue
            
        # Get subset metrics
        sub_base_auc = roc_auc_score(subset[target], subset['image_score'])
        sub_ens_probs = model.predict(xgb.DMatrix(subset[features], label=subset[target]))
        sub_ens_auc = roc_auc_score(subset[target], sub_ens_probs)
        
        # Map back to name
        d_name = {0: 'A (Fatty)', 1: 'B (Scattered)', 2: 'C (Hetero)', 3: 'D (Dense)', -1: 'Missing'}.get(density, str(density))
        print(f"  {d_name:<15} (n={len(subset)}): {sub_base_auc:.4f} -> {sub_ens_auc:.4f} ({sub_ens_auc - sub_base_auc:+.4f})")

    # By Age Group
    print("\n[By Age]")
    test_df['age_bin'] = pd.cut(test_df['age'], bins=[0, 40, 50, 60, 70, 100], labels=['<40', '40-50', '50-60', '60-70', '70+'])
    for age_group in sorted(test_df['age_bin'].unique()):
        subset = test_df[test_df['age_bin'] == age_group]
        if len(subset) < 10 or subset[target].nunique() < 2:
            continue
            
        sub_base_auc = roc_auc_score(subset[target], subset['image_score'])
        sub_ens_probs = model.predict(xgb.DMatrix(subset[features], label=subset[target]))
        sub_ens_auc = roc_auc_score(subset[target], sub_ens_probs)
        
        print(f"  {age_group:<15} (n={len(subset)}): {sub_base_auc:.4f} -> {sub_ens_auc:.4f} ({sub_ens_auc - sub_base_auc:+.4f})")

    # Feature Importance
    importance = model.get_score(importance_type='gain')
    print("\nFeature Importance (Gain):")
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.2f}")

if __name__ == "__main__":
    train_df, val_df, test_df = load_and_prep_data()
    train_xgb(train_df, val_df, test_df)
