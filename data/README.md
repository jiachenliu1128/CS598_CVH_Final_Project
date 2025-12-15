# Data Directory

## image_level_predictions.csv

Pre-computed cancer probabilities from high-resolution ConvNeXt model (trained at 1024x2048 resolution).

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | int | Patient identifier |
| `image_id` | int | Image identifier |
| `laterality` | str | Breast side (L/R) |
| `preds` | float | CNN cancer probability (0-1) |
| `age` | float | Patient age in years |
| `density` | str | Breast density (A/B/C/D or NaN) |
| `implant` | int | Implant indicator (0/1) |
| `site_id` | int | Hospital/imaging site |
| `cancer` | int | Ground truth label (0/1) |

### Aggregation

Multiple images per breast (CC + MLO views) are aggregated using MAX pooling:
```python
breast_score = max(image_predictions)
```

### Source

Original RSNA dataset: https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data

