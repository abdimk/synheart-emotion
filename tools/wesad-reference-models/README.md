# WESAD Reference Models (Research Artifacts)

⚠️ **This directory contains research artifacts and training pipeline reference code.**

For the **production Python SDK**, see: [`sdks/python/`](../../sdks/python/)

## Purpose

This directory contains pre-trained models from the WESAD dataset for research and comparison purposes. These are NOT the production SDK models.

## Contents

- **Pre-trained ML models** from the WESAD dataset organized by window configuration
- ONNX, PyTorch, and scikit-learn checkpoints (`*.onnx`, `*.pth`, `*.pkl`)
- Model metadata JSON files with feature schemas and performance metrics
- Reference inference code (`inference.py`) with random data generation

## Model Configurations

Models are organized by window configuration:

- **`w60s5_binary`**: 60-second windows, 5-second steps
- **`w120s5_binary`**: 120-second windows, 5-second steps
- **`w120s60_binary`**: 120-second windows, 60-second steps

Each configuration directory contains multiple model types (ExtraTrees, RandomForest, LogisticRegression, etc.)

## Emotion Labels (Binary Classification)

- **0** → Baseline (Calm)
- **1** → Stress

## Input Features

Models expect 14 HRV (Heart Rate Variability) features in the following order:

1. `RMSSD` (ms) - Root Mean Square of Successive Differences
2. `Mean_RR` (ms) - Mean RR interval
3. `HRV_SDNN` (ms) - Standard Deviation of NN intervals
4. `pNN50` (%) - Percentage of NN intervals > 50ms
5. `HRV_HF` (ms²) - High Frequency power
6. `HRV_LF` (ms²) - Low Frequency power
7. `HRV_HF_nu` (normalized) - Normalized HF power
8. `HRV_LF_nu` (normalized) - Normalized LF power
9. `HRV_LFHF` (ratio) - LF/HF ratio
10. `HRV_TP` (ms²) - Total Power
11. `HRV_SD1SD2` (ratio) - Poincaré plot SD1/SD2 ratio
12. `HRV_Sampen` (entropy) - Sample Entropy
13. `HRV_DFA_alpha1` (alpha) - Detrended Fluctuation Analysis alpha1
14. `HR` (bpm) - Heart Rate

## Usage (Research Only)

### Basic Usage

```python
from inference import predict, list_available_models, generate_random_features
import pandas as pd

# List available models
available = list_available_models()
print(available)
# {
#   'w60s5_binary': ['extratrees', 'rf', 'logreg', 'xgb', ...],
#   'w120s5_binary': ['extratrees', 'rf', 'logreg', ...],
#   'w120s60_binary': ['extratrees', 'rf', 'logreg', ...]
# }

# Generate random test data
baseline_data = generate_random_features(emotion="baseline", n_samples=1, seed=42)
print(baseline_data)

# Run inference
results = predict(
    data=baseline_data,
    config_name="w60s5_binary",
    model_name="extratrees",
    return_probabilities=True
)

print(results)
# [{
#   'numeric': 0,
#   'label': 'Baseline',
#   'probabilities': {'Baseline': 0.85, 'Stress': 0.15}
# }]
```

### Using Custom Data

```python
import pandas as pd
from inference import predict

# Create custom feature data
custom_features = pd.DataFrame([{
    "RMSSD": 45.2,
    "Mean_RR": 850.3,
    "HRV_SDNN": 52.1,
    "pNN50": 12.5,
    "HRV_HF": 1200.0,
    "HRV_LF": 800.0,
    "HRV_HF_nu": 0.6,
    "HRV_LF_nu": 0.4,
    "HRV_LFHF": 0.67,
    "HRV_TP": 2000.0,
    "HRV_SD1SD2": 0.5,
    "HRV_Sampen": 1.2,
    "HRV_DFA_alpha1": 1.0,
    "HR": 70.5
}])

results = predict(
    data=custom_features,
    config_name="w120s60_binary",
    model_name="extratrees"
)
```

### Using Different Model Types

```python
from inference import predict, generate_random_features

data = generate_random_features(emotion="stress", n_samples=1)

# Try different models
for model_name in ["extratrees", "rf", "logreg"]:
    results = predict(
        data=data,
        config_name="w60s5_binary",
        model_name=model_name
    )
    print(f"{model_name}: {results[0]['label']}")
```

## Available Models

Models are organized by configuration directory. Each contains:

| Model Type | File Format | Description |
|------------|-------------|-------------|
| `extratrees` | `.onnx` or `.pkl` | ExtraTrees Classifier |
| `rf` | `.onnx` or `.pkl` | Random Forest |
| `logreg` | `.onnx` or `.pkl` | Logistic Regression |
| `xgb` | `.pkl` | XGBoost Classifier |
| `linearsvm` | `.pkl` | Linear SVM |

Note: Some models are available in ONNX format (with built-in normalization), others as scikit-learn pickle files.

## Files

```
wesad-reference-models/
├── inference.py              # Reference inference code
├── models/
│   ├── w60s5_binary/        # 60s window, 5s step models
│   │   ├── ExtraTrees.pkl
│   │   ├── ExtraTrees_metadata.json
│   │   ├── RF.pkl
│   │   ├── RF_metadata.json
│   │   └── ...
│   ├── w120s5_binary/       # 120s window, 5s step models
│   │   ├── ExtraTrees.onnx
│   │   ├── ExtraTrees_metadata.json
│   │   └── ...
│   └── w120s60_binary/      # 120s window, 60s step models
│       ├── ExtraTrees.onnx
│       ├── ExtraTrees_metadata.json
│       └── ...
└── README.md                # This file
```

## Random Data Generation

The `inference.py` module includes a `generate_random_features()` function for testing:

```python
from inference import generate_random_features

# Generate baseline (calm) features
baseline = generate_random_features(emotion="baseline", n_samples=5, seed=42)

# Generate stress features
stress = generate_random_features(emotion="stress", n_samples=5, seed=123)
```

This generates realistic HRV features based on typical physiological patterns:
- **Baseline**: Lower HR (60-75 bpm), higher HRV metrics
- **Stress**: Higher HR (75-100 bpm), lower HRV metrics

## Differences from Production SDK

| Aspect | This (Research) | Production SDK |
|--------|----------------|----------------|
| **Location** | `tools/wesad-reference-models/` | `sdks/python/` |
| **Purpose** | Research/training reference | Production deployment |
| **Models** | Multiple models per config | Single embedded model |
| **Input** | Pre-computed HRV features | Raw HR + RR intervals |
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

## Setup and Installation

### Using Virtual Environment (Recommended)

```bash
cd tools/wesad-reference-models

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Required packages (included in `requirements.txt`):
```bash
pip install numpy pandas joblib onnxruntime scikit-learn
```

Optional (for PyTorch models):
```bash
pip install torch
```

### Testing

After setup, run the test script to verify everything works:

```bash
python test_inference.py
```

Or run the example code directly:

```bash
python inference.py
```

## Training Pipeline Reference

This code represents the **output** of a training pipeline:
1. ECG data collected from WESAD dataset
2. Feature extraction with NeuroKit2 (14 HRV features)
3. Model training with cross-validation
4. Model serialization to ONNX/joblib/pickle

For production deployment, the simplified ExtraTrees ONNX model in `sdks/python/` is used.

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
