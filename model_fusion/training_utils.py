import os
import json
from typing import List

import numpy as np
import pandas as pd
import torch


def impute_age_inplace(dataset, age_idx: int, fill_value: float) -> None:
    """Impute NaN age values in-place for a dataset's metadata array."""
    meta = dataset.meta_array.copy()
    nan_mask = np.isnan(meta[:, age_idx])
    meta[nan_mask, age_idx] = fill_value
    dataset.set_meta_array(meta)


def save_epoch_metrics(history: List[dict], results_csv_path: str) -> None:
    """Persist the accumulated epoch metrics to a CSV file."""
    try:
        pd.DataFrame(history).to_csv(results_csv_path, index=False)
    except Exception as e:
        print(f"WARNING: failed to write training CSV: {e}")


def save_best_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    valid_loss: float,
    feat_dim: int,
    meta_dim: int,
    best_threshold: float,
    best_ckpt_path: str,
    best_info_path: str,
) -> None:
    """Save best model checkpoint along with companion info JSON."""
    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "valid_loss": valid_loss,
            },
            best_ckpt_path,
        )
        with open(best_info_path, "w") as f:
            json.dump(
                {
                    "best_epoch": epoch,
                    "best_valid_loss": float(valid_loss),
                    "best_threshold": float(best_threshold),
                    "feat_dim": int(feat_dim),
                    "meta_dim": int(meta_dim),
                },
                f,
            )
    except Exception as e:
        print(f"WARNING: failed to save best checkpoint: {e}")


def save_test_outputs(
    best_threshold: float,
    test_trues: List[int],
    test_preds: List[float],
    test_metrics_path: str,
    test_preds_path: str,
) -> None:
    """Save test metrics JSON and predictions CSV."""
    try:
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
        )

        test_preds_binary = [1 if p >= best_threshold else 0 for p in test_preds]
        precision = precision_score(test_trues, test_preds_binary, zero_division=0)
        recall = recall_score(test_trues, test_preds_binary, zero_division=0)
        f1 = f1_score(test_trues, test_preds_binary, zero_division=0)
        f1_macro = f1_score(
            test_trues, test_preds_binary, average="macro", zero_division=0
        )

        # Specificity and NPV
        tn, fp, fn, tp = confusion_matrix(test_trues, test_preds_binary).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        try:
            auc = roc_auc_score(test_trues, test_preds)
        except ValueError:
            auc = 0.0

        with open(test_metrics_path, "w") as f:
            json.dump(
                {
                    "threshold": float(best_threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "specificity": float(specificity),
                    "npv": float(npv),
                    "f1": float(f1),
                    "f1_macro": float(f1_macro),
                    "auc": float(auc),
                },
                f,
            )

        pd.DataFrame({"pred": test_preds, "true": test_trues}).to_csv(
            test_preds_path, index=False
        )
        print(f"Saved test metrics to: {test_metrics_path}")
        print(f"Saved test predictions to: {test_preds_path}")
    except Exception as e:
        print(f"WARNING: failed to save test outputs: {e}")
