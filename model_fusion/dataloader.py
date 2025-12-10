import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


# Read breast cancer dataset with image features and metadata
class BreastDataset(Dataset):
    def __init__(self, csv_path, features_dir, meta_cols):
        self.df = pd.read_csv(csv_path)

        # group rows by breast (patient_id + laterality)
        self.groups = []
        for (pid, lat), g in self.df.groupby(["patient_id", "laterality"]):
            img_ids = g["image_id"].tolist()
            labels = g["cancer"].unique()
            assert len(labels) == 1              # breast-level label
            label = float(labels[0])
            
            # Fill missing metadata with a default value (e.g., 0)
            self.df[meta_cols] = self.df[meta_cols].fillna(0)

            # metadata
            g.loc[g.index[0], "laterality"] = 0.0 if lat == "L" else 1.0
            
            density_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
            g.loc[g.index[0], "density"] = density_mapping.get(g.loc[g.index[0], "density"], -1)
            
            birads_value = g.loc[g.index[0], "BIRADS"]
            if pd.isna(birads_value):
                g.loc[g.index[0], "BIRADS"] = -1  # Assign a default value for missing BIRADS
            
            meta = g.iloc[0][meta_cols].values.astype("float32")
            
            # Replace any remaining NaN or inf values in metadata
            meta = np.nan_to_num(meta, nan=0.0, posinf=0.0, neginf=0.0)

            
            self.groups.append({
                "patient_id": pid,
                "laterality": lat,
                "image_ids": img_ids,
                "meta": meta,
                "label": label,
            })

        self.features_dir = features_dir
        self.meta_cols = meta_cols

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        g = self.groups[idx]
        feats = []
        for img_id in g["image_ids"]:
            fname = f"{g['patient_id']}@{img_id}.pt"
            path = os.path.join(self.features_dir, fname)
            f = torch.load(path)          # shape: (feat_dim,)
            feats.append(f)

        feats = torch.stack(feats, dim=0)  # (num_views, feat_dim)
        meta = torch.tensor(g["meta"], dtype=torch.float32)  # (meta_dim,)
        label = torch.tensor(g["label"], dtype=torch.float32)  # scalar

        return feats, meta, label
    
    
    
    
# Collate function to pad variable number of views per breast 
def collate_breast(batch):
    """
    batch: list of (feats, meta, label)
      feats: (num_views, D)
    """
    feat_list, meta_list, label_list = zip(*batch)
    max_views = max(f.size(0) for f in feat_list)
    D = feat_list[0].size(1)

    B = len(batch)
    feats_padded = torch.zeros(B, max_views, D, dtype=torch.float32)
    mask = torch.zeros(B, max_views, dtype=torch.bool)

    for i, f in enumerate(feat_list):
        n = f.size(0)
        feats_padded[i, :n] = f
        mask[i, :n] = True  # True = real view, False = padding

    metas = torch.stack(meta_list, dim=0)   # (B, meta_dim)
    labels = torch.stack(label_list, dim=0) # (B,)

    return feats_padded, mask, metas, labels




