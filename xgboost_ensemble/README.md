# XGBoost Ensemble for Breast Cancer Classification

Two-stage stacking approach combining high-resolution CNN predictions with patient metadata.

## Architecture

```
Stage 1: ConvNeXt (1024x2048) → Per-image predictions
Stage 2: Aggregate to breast-level → XGBoost (image + metadata) → Final prediction
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
cd xgboost_ensemble
python train_ensemble.py
```

**Inputs:**
- `../data/image_level_predictions.csv`: Pre-computed CNN predictions
- `../label/{train,val,test}.csv`: Data splits

**Outputs:**
- `models/xgboost_ensemble.json`: Trained model
- `../results/predictions/ensemble_test_predictions.csv`: Test predictions
- `../results/metrics/ensemble_metrics.txt`: Performance summary

### Generate SHAP Explanations

```bash
python explain_shap.py
```

**Outputs:**
- `../results/figures/shap_summary.png`: Feature importance
- `../results/figures/shap_dependence_*.png`: Feature interactions
- `../results/figures/shap_force_positive.png`: Individual case explanation

## Features

- `image_score`: Aggregated CNN predictions (MAX pooling across views)
- `age`: Patient age in years
- `density`: Breast density (0=A, 1=B, 2=C, 3=D, -1=Missing)
- `implant`: Binary indicator for implants
- `site_id`: Hospital/imaging site identifier

## Expected Results

| Metric | Value |
|--------|-------|
| Test AUC | 0.9873 |
| Test F1 | 0.7640 |
| Improvement over Image-Only | +1.4% F1 |

## Citation

```
@article{agarwal2025breast,
  title={Breast Cancer Classification with ConvNeXt and Metadata Integration},
  author={Agarwal, Kushal and Liu, Jiachen and Xu, Maojie and Raje, Kalika},
  journal={CS598 CVH Final Project},
  year={2025}
}
```

