"""
Custom loss functions for Breast Cancer Detection.

Implements Focal Loss to handle severe class imbalance (97.88% negative, 2.12% positive).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for imbalanced binary classification.
    
    Focal Loss reduces the contribution of easy examples and focuses
    training on hard negatives. Particularly useful when one class
    dominates (e.g., 97.88% negative in breast cancer detection).
    
    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
    where:
        p_t = p if y=1 else (1-p)
        alpha_t = alpha if y=1 else (1-alpha)
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Binary Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class (default: 0.25)
                   Higher alpha increases weight of positive samples.
                   For 2.12% positive rate, try alpha=0.25 to 0.75.
            gamma: Focusing parameter (default: 2.0)
                   Higher gamma reduces weight of easy examples more.
                   gamma=0 is equivalent to standard cross-entropy.
            reduction: 'mean', 'sum', or 'none' (default: 'mean')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            inputs: Predicted probabilities [batch_size] or [batch_size, 1]
                    Should be sigmoid outputs (0-1 range)
            targets: Ground truth labels [batch_size] or [batch_size, 1]
                     Binary values (0 or 1)
                     
        Returns:
            Focal loss value (scalar if reduction='mean' or 'sum')
        """
        # Flatten inputs if needed
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # Clamp for numerical stability
        inputs = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)
        
        # Compute binary cross entropy components
        bce = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)
        
        # Compute p_t (probability of correct class)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        
        # Compute alpha_t (class weight)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class FocalLoss(nn.Module):
    """
    Focal Loss that accepts logits (pre-sigmoid outputs).
    
    This is a convenience wrapper that applies sigmoid internally,
    making it compatible with models that output raw logits.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss (logits version).
        
        Args:
            alpha: Weighting factor for positive class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean', 'sum', or 'none' (default: 'mean')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss from logits.
        
        Args:
            logits: Raw model outputs (pre-sigmoid) [batch_size] or [batch_size, 1]
            targets: Ground truth labels [batch_size] or [batch_size, 1]
                     Binary values (0 or 1)
                     
        Returns:
            Focal loss value
        """
        # Flatten
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # Compute probabilities using numerically stable sigmoid
        probs = torch.sigmoid(logits)
        
        # Clamp for numerical stability
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Use BCE with logits for numerical stability
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def test_focal_loss():
    """Test Focal Loss implementations."""
    print("Testing Focal Loss implementations...")
    
    # Create test data
    batch_size = 8
    
    # Simulated predictions (logits)
    logits = torch.randn(batch_size)
    
    # Imbalanced targets (mostly 0s, few 1s - like our dataset)
    targets = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float32)
    
    # Test FocalLoss (logits version)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"FocalLoss (logits): {loss.item():.4f}")
    
    # Test BinaryFocalLoss (probabilities version)
    probs = torch.sigmoid(logits)
    binary_focal_loss = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    loss_binary = binary_focal_loss(probs, targets)
    print(f"BinaryFocalLoss (probs): {loss_binary.item():.4f}")
    
    # Compare with standard BCE
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    print(f"Standard BCE: {bce_loss.item():.4f}")
    
    # Test gamma effect
    print("\nEffect of gamma (focusing parameter):")
    for gamma in [0.0, 1.0, 2.0, 5.0]:
        fl = FocalLoss(alpha=0.25, gamma=gamma)
        loss = fl(logits, targets)
        print(f"  gamma={gamma}: loss={loss.item():.4f}")
    
    # Test alpha effect
    print("\nEffect of alpha (class weight):")
    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
        fl = FocalLoss(alpha=alpha, gamma=2.0)
        loss = fl(logits, targets)
        print(f"  alpha={alpha}: loss={loss.item():.4f}")
    
    # Test with extreme predictions (easy examples)
    print("\nEasy examples (confident correct predictions):")
    easy_logits = torch.tensor([5.0, -5.0, -5.0, -5.0])  # High confidence
    easy_targets = torch.tensor([1.0, 0.0, 0.0, 0.0])     # Correct labels
    
    fl = FocalLoss(alpha=0.25, gamma=2.0)
    bce = F.binary_cross_entropy_with_logits(easy_logits, easy_targets)
    focal = fl(easy_logits, easy_targets)
    print(f"  BCE loss: {bce.item():.6f}")
    print(f"  Focal loss: {focal.item():.6f}")
    print(f"  Focal/BCE ratio: {focal.item()/bce.item():.4f} (lower = more downweighting)")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_focal_loss()

