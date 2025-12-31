# Tools Directory

This directory contains development tools and utilities for the Synheart Emotion project.

## Available Tools

### 1. Synthetic Data Generator (`synthetic-data-generator/`)

**Purpose**: Generate realistic biosignal data for testing all SDKs

Generate synthetic heart rate and RR interval data for testing emotion inference across Python, Android, iOS, and Flutter platforms.

**Features**:
- 🎯 Realistic physiological data
- 🎭 3 emotion scenarios (Calm, Stressed, Amused)
- 🔄 Smooth transitions between emotions
- 📊 Export to CSV, JSON, Python, Kotlin, Swift
- 🔁 Reproducible with random seeds

**Quick Start**:
```bash
cd synthetic-data-generator
python cli.py --emotion Calm --duration 60 --output ./data
```

**Documentation**: See [synthetic-data-generator/README.md](synthetic-data-generator/README.md)

---

### 2. WESAD Reference Models (`wesad-reference-models/`)

**Purpose**: Research artifacts and training pipeline reference

Pre-trained models from the WESAD dataset for research and model comparison. Includes multiple model configurations and types for binary emotion classification (Baseline vs Stress).

**Contains**:
- Pre-trained ML models organized by window configuration:
  - `w60s5_binary`: 60-second windows, 5-second steps
  - `w120s5_binary`: 120-second windows, 5-second steps
  - `w120s60_binary`: 120-second windows, 60-second steps
- Multiple model types per configuration (ExtraTrees, RandomForest, LogisticRegression, XGBoost, etc.)
- Model metadata with performance metrics
- Reference inference code with random data generation
- Support for ONNX, scikit-learn, and PyTorch models

**Features**:
- 🔬 Research-grade models for comparison
- 📊 Multiple window configurations
- 🤖 Multiple model architectures
- 🎲 Built-in random data generation for testing
- 📈 Performance metrics included

**⚠️ Not for Production**: This is research code. For production, use [`sdks/python/`](../sdks/python/)

**Quick Start**:
```python
from tools.wesad_reference_models.inference import predict, generate_random_features

# Generate random test data
data = generate_random_features(emotion="baseline", n_samples=1, seed=42)

# Run inference
results = predict(
    data=data,
    config_name="w60s5_binary",
    model_name="extratrees",
    return_probabilities=True
)

print(results[0]['label'])  # "Baseline" or "Stress"
```

**Documentation**: See [wesad-reference-models/README.md](wesad-reference-models/README.md)

---

## Tool Comparison

| Tool | Purpose | Output | Use Case |
|------|---------|--------|----------|
| **synthetic-data-generator** | Generate test data | Biosignal time series | SDK testing |
| **wesad-reference-models** | Research reference | Model predictions | Research/comparison |

## For SDK Development

If you're developing with the SDKs, you likely want:

1. **Testing SDKs** → Use `synthetic-data-generator/`
2. **Research/comparison** → See `wesad-reference-models/`
3. **Production deployment** → Use `sdks/python/`, `sdks/android/`, or `sdks/ios/`

## Directory Structure

```
tools/
├── README.md                      # This file
├── synthetic-data-generator/      # Test data generation tool
│   ├── syndata/                   # Generator package
│   ├── examples/                  # Usage examples
│   ├── cli.py                     # Command-line interface
│   └── README.md                  # Full documentation
└── wesad-reference-models/        # Research artifacts
    ├── inference.py               # Reference inference with random data gen
    ├── models/                    # Pre-trained models by configuration
    │   ├── w60s5_binary/          # 60s window, 5s step models
    │   ├── w120s5_binary/         # 120s window, 5s step models
    │   └── w120s60_binary/        # 120s window, 60s step models
    └── README.md                  # Documentation
```

## Contributing

To add a new tool:

1. Create a new directory under `tools/`
2. Add a descriptive README.md
3. Update this file with a summary
4. Consider making it pip-installable if appropriate
