# WESAD Reference Models (Research Artifacts)

⚠️ **This directory contains research artifacts and training pipeline reference code.**

For the **production Python SDK**, see: [`sdks/python/`](../../sdks/python/)

## Purpose

This directory contains pre-trained models from the WESAD dataset for research and comparison purposes. These are NOT the production SDK models.

## Contents

- **SDK-aligned pre-trained ML models** from the WESAD dataset
- Torch and scikit-learn checkpoints (`*.pth`, `*.pkl`)
- Feature scaler (`scaler_wrist_all.pkl`) and metadata
- Reference inference code (`inference.py`)

## Emotion Labels (WESAD)

- **0** → Baseline (Calm)
- **1** → Stress
- **2** → Amusement

## Input Data

Input is a Pandas DataFrame (or dict/list/NumPy array) containing the HRV summary
features consumed by `inference.py`. Typical pipeline:
1. Clean ECG with `nk.ecg_clean()`
2. Detect R-peaks via `nk.ecg_peaks()`
3. Compute HRV metrics with `nk.hrv()` (2-minute sliding windows)
4. Assemble the following columns in the order defined in `FEATURE_NAMES` of
   `inference.py`: `SDNN`, `RMSSD`, `pNN50`, `Mean_RR`, `HR_mean`
5. Missing values are imputed with the median before scaling

Scaling is applied automatically using `scaler_wrist_all.pkl` when calling
`prepare_input()` or any of the public helpers in `inference.py`.

## Usage (Research Only)

```python
import pandas as pd
from inference import predict, list_available_models

print(list_available_models())
# ['attention_wrist_all', 'deep_mlp_wrist_all', 'extratrees_wrist_all', ...]

sample_features = pd.DataFrame([
    {
        "SDNN": 87.4,
        "RMSSD": 62.1,
        "pNN50": 0.31,
        "Mean_RR": 812.5,
        "HR_mean": 73.9,
    }
])

predictions = predict(sample_features, model_name="attention")
print(predictions)
# [{'numeric': 2, 'label': 'Amusement', 'probabilities': {'Baseline': 0.02, ...}}]
```

## Available Models

Models live in `sdk models/` (see `MODEL_DIR` inside `inference.py`). Aliases such as
`attention`, `deepmlp`, `randomforest`, and `xgb` are supported; use
`list_available_models()` to see the canonical names. Example inventory:

| Alias | Checkpoint | Type |
|-------|------------|------|
| `attention` | `attention_wrist_all.pth` | PyTorch (AttentionClassifier) |
| `deepmlp` | `deep_mlp_wrist_all.pth` | PyTorch (DeepMLPClassifier) |
| `simple` | `simple_wrist_all.pth` | PyTorch (SimpleDeepNN) |
| `randomforest` | `rf_wrist_all.pkl` | Scikit-learn |
| `extratrees` | `extratrees_wrist_all.pkl` | Scikit-learn |
| `logreg` | `logreg_wrist_all.pkl` | Scikit-learn |
| `linearsvm` | `linearsvm_wrist_all.pkl` | Scikit-learn |
| `xgb` | `xgb_wrist_all.pkl` | XGBoost (via joblib) |

Additional experimental checkpoints may be present (for example
`transformer_wrist_all.pth`); availability depends on the local checkout.

## Files

```
wesad-reference-models/
├── inference.py              # Reference inference code
├── models/
│   ├── *.pth                # PyTorch attention/MLP checkpoints
│   ├── *.pkl                # Scikit-learn models & scaler
│   └── *.xgb                # XGBoost models (joblib serialized)
└── README.md                # This file
```

## Differences from Production SDK

| Aspect | This (Research) | Production SDK |
|--------|----------------|----------------|
| **Location** | `tools/wesad-reference-models/` | `sdks/python/` |
| **Purpose** | Research/training reference | Production deployment |
| **Models** | 14 pre-trained models | 1 embedded model |
| **Input** | DataFrame with many features | Raw HR + RR intervals |
| **API** | Function-based | Class-based engine |
| **Architecture** | Stateless | Stateful sliding window |
| **Installation** | Not pip-installable | `pip install synheart-emotion` |

## For Production Use

👉 **Use the production SDK instead**: [`sdks/python/`](../../sdks/python/)

The production SDK:
- ✅ Pip-installable
- ✅ Matches Flutter/Android/iOS APIs
- ✅ Real-time sliding window processing
- ✅ Works with raw biosignal data
- ✅ Thread-safe
- ✅ Comprehensive tests and examples

```bash
cd sdks/python
pip install -e .
```

## Dependencies

```bash
pip install numpy pandas joblib torch scikit-learn xgboost
```

## Training Pipeline Reference

This code represents the **output** of a training pipeline:
1. ECG data collected from WESAD dataset
2. Feature extraction with NeuroKit2
3. Model training with cross-validation
4. Model serialization to joblib/xgb

For production deployment, the simplified LinearSVM model in `sdks/python/` is used.

## Citation

WESAD Dataset:
```bibtex
@article{schmidt2018introducing,
  title={Introducing WESAD, a multimodal dataset for wearable stress and affect detection},
  author={Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marberger, Claus and Van Laerhoven, Kristof},
  journal={ICMI 2018},
  year={2018}
}
```

## License

Research artifacts - See main repository LICENSE.
