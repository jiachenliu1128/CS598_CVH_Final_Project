import torch
import torch.nn as nn
import torch.nn.functional as F




# Attention MIL module for images fusion
class AttentionMIL(nn.Module):
    def __init__(self, feat_dim, meta_dim=None, hidden_dim=128):
        super().__init__()
        self.meta_dim = meta_dim

        in_dim = feat_dim if meta_dim is None else feat_dim + meta_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # scalar score per view
        
        # Better initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feats, mask, meta=None):
        """
        feats: (B, N, D)
        mask: (B, N) bool, True = keep, False = padding
        meta: (B, M) or None
        returns: (B, D) breast-level feature
        """
        B, N, D = feats.shape
        
        # Check for NaN in inputs
        if torch.isnan(feats).any():
            print("WARNING: NaN detected in feats")
            feats = torch.nan_to_num(feats, nan=0.0)

        if meta is not None:
            if torch.isnan(meta).any():
                print("WARNING: NaN detected in meta")
                meta = torch.nan_to_num(meta, nan=0.0)
            # expand metadata to each view: (B, N, M)
            M = meta.size(1)
            meta_exp = meta.unsqueeze(1).expand(B, N, M)
            x = torch.cat([feats, meta_exp], dim=-1)  # (B, N, D+M)
        else:
            x = feats  # (B, N, D)

        # compute scores
        h = torch.tanh(self.fc1(x))           # (B, N, H)
        scores = self.fc2(h).squeeze(-1)      # (B, N)

        # mask out padded views before softmax (use smaller value to avoid overflow)
        scores = scores.masked_fill(~mask, -1e4)
        
        # Add numerical stability check
        if torch.isnan(scores).any():
            print("WARNING: NaN detected in scores before softmax")
            scores = torch.nan_to_num(scores, nan=0.0)

        attn = F.softmax(scores, dim=1)       # (B, N)
        attn = attn.unsqueeze(-1)             # (B, N, 1)

        # weighted sum of feats
        breast_feat = torch.sum(attn * feats, dim=1)  # (B, D)
        return breast_feat, attn.squeeze(-1)          # return attn optionally
    
    
    
    
    
    

    
    
    
    
# Breast cancer classifier model 
class BreastClassifier(nn.Module):
    def __init__(self, feat_dim, meta_dim, hidden_mil=128, hidden_meta=64, hidden_cls=128):
        super().__init__()
        # attention MIL fusion
        self.mil = AttentionMIL(feat_dim=feat_dim,
                                meta_dim=None,
                                hidden_dim=hidden_mil)

        # metadata branch - use LayerNorm instead of BatchNorm for stability
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_meta),
            nn.ReLU(),
            nn.LayerNorm(hidden_meta),
            nn.Dropout(0.1),
        )

        # classifier head on [breast_feat || meta_feat]
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + hidden_meta, hidden_cls),
            nn.ReLU(),
            nn.LayerNorm(hidden_cls),
            nn.Dropout(0.1),
            nn.Linear(hidden_cls, 1),
        )
        
        # Initialize classifier weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats, mask, meta):
        """
        feats: (B, N, D)
        mask: (B, N)
        meta: (B, meta_dim)
        """
        breast_feat, _ = self.mil(feats, mask, meta=None)    # (B, D)
        meta_feat = self.meta_mlp(meta)                 # (B, H_meta)
        x = torch.cat([breast_feat, meta_feat], dim=-1) # (B, D+H_meta)
        logit = self.classifier(x).squeeze(-1)          # (B,)
        return logit
    
    
    
 








