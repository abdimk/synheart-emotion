# Synheart Emotion

**On-device affective state inference from physiological signals (HR/RR) for Dart, Python, Kotlin, and Swift applications**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform Support](https://img.shields.io/badge/platforms-Dart%20%7C%20Python%20%7C%20Kotlin%20%7C%20Swift-blue.svg)](#-sdks)

Synheart Emotion is a comprehensive SDK ecosystem for inferring momentary affective state tendencies from physiological signals (e.g., heart rate and RR intervals) directly on device, ensuring privacy and real-time performance.

## 🚀 Features

- **📱 Multi-Platform**: Dart/Flutter, Python, Kotlin, Swift
- **🔄 Real-Time Inference**: Low-latency affective state inference from HR/RR-derived features
- **🧠 On-Device Processing**: All computations happen locally for privacy
- **📊 Unified API**: Consistent API across all platforms
- **🔒 Privacy-First**: No raw biometric data leaves your device
- **⚡ High Performance**: < 10ms inference latency (ONNX models)
- **🎓 Research-Based**: Models trained on WESAD dataset with 78.4% accuracy (72.6% F1 score)
- **🧬 14 HRV Features**: Comprehensive feature extraction (time-domain, frequency-domain, non-linear)
- **🤖 ExtraTrees Models**: ONNX-optimized classifiers for on-device inference
- **🧪 Thread-Safe**: Concurrent data ingestion supported on all platforms
- **🏗️ HSI-Compatible**: Output schema validated against Synheart Core HSI specification

## 📦 SDKs

All SDKs provide **identical functionality** with platform-idiomatic APIs. Each SDK is maintained in its own repository:

### Dart/Flutter SDK
```yaml
dependencies:
  synheart_emotion: ^0.2.3
```
📖 **Repository**: [synheart-emotion-dart](https://github.com/synheart-ai/synheart-emotion-dart)

### Python SDK [![PyPI](https://img.shields.io/badge/PyPI-pip%20installable-blue.svg)](https://pypi.org/project/synheart-emotion/)
```bash
pip install synheart-emotion
```
📖 **Repository**: [synheart-emotion-python](https://github.com/synheart-ai/synheart-emotion-python)

### Kotlin SDK
```kotlin
dependencies {
    implementation("ai.synheart:emotion:0.1.0")
}
```
📖 **Repository**: [synheart-emotion-kotlin](https://github.com/synheart-ai/synheart-emotion-kotlin)

### Swift SDK
**Swift Package Manager:**
```swift
dependencies: [
    .package(url: "https://github.com/synheart-ai/synheart-emotion-swift.git", from: "0.1.0")
]
```

**CocoaPods:**
```ruby
pod 'SynheartEmotion', '~> 0.1.0'
```
📖 **Repository**: [synheart-emotion-swift](https://github.com/synheart-ai/synheart-emotion-swift)

## 🏗️ Relationship with Synheart Core (HSI)

Synheart Emotion serves **two deployment modes**:

### 1. **Standalone SDK** (Direct Integration)
Use synheart-emotion directly for emotion-only applications:

```dart
import 'package:synheart_emotion/synheart_emotion.dart';

final engine = EmotionEngine.fromPretrained(EmotionConfig());
engine.push(hr: 72.0, rrIntervalsMs: [...], timestamp: DateTime.now());
final results = await engine.consumeReadyAsync();
```

**Use when:** Your app only needs emotion detection, not full human state intelligence.

### 2. **Via Synheart Core** (HSI Integration)
Use emotion as part of a complete Human State Interface with focus, behavior, and context:

```dart
import 'package:synheart_core/synheart_core.dart';

// Initialize synheart-core (includes emotion capability)
await Synheart.initialize(
  userId: 'user_123',
  config: SynheartConfig(enableWear: true),
);

// Enable emotion interpretation layer
await Synheart.enableEmotion();

// Get emotion updates (powered by synheart-emotion under the hood)
Synheart.onEmotionUpdate.listen((emotion) {
  print('Stress: ${emotion.stress}, Calm: ${emotion.calm}');
});
```

**Use when:** You want emotion as part of a unified human state representation (HSV).

### Architecture & Dependencies

```
┌─────────────────────────────────────────────────────┐
│          Synheart Core (HSI Runtime)                │
│                                                     │
│  EmotionHead Module                                 │
│    └─► depends on synheart-emotion package         │
│         (runtime dependency)                        │
└─────────────────────────────────────────────────────┘
                      ▲
                      │
                      │ runtime: package dependency
                      │ schema: validates against HSI spec
                      │
┌─────────────────────────────────────────────────────┐
│          synheart-emotion (this repo)               │
│                                                     │
│  • Standalone emotion inference SDK                 │
│  • NO code dependency on synheart-core              │
│  • Output schema validated against:                 │
│    ../synheart-core/docs/HSI_SPECIFICATION.md       │
│                                                     │
│  EmotionEngine → EmotionResult                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Key Principles:**
- ✅ **Standalone**: synheart-emotion works independently, no core dependency
- ✅ **HSI-Compatible**: Output schema matches HSI EmotionState specification
- ✅ **Schema Validation**: CI enforces compatibility with HSI spec
- ✅ **Used by Core**: synheart-core's EmotionHead uses synheart-emotion as implementation
- ✅ **Backward Compatible**: Existing standalone users unaffected

## 📂 Repository Structure

This repository serves as the **source of truth** for shared resources across all SDK implementations:

```
synheart-emotion/                  # Source of truth repository
├── models/                        # ML model definitions and assets
│   ├── wesad_emotion_v1_0.json    # Model configuration
│   └── *.onnx                     # Pre-trained model weights
│
├── docs/                          # Technical documentation
│   └── MODEL_CARD.md              # Model details and performance
│   ├── RFC-Emotion-0001-spec.md        # Formal specification 
│   ├── RFC-Emotion-0002-guide.md   # Implementation guide 
│
├── tools/                         # Development tools
│   ├── synthetic-data-generator/  # Generate test biosignal data
│   └── wesad-reference-models/   # Research artifacts (pre-trained ML models)
│       ├── inference.py           # Reference inference code with ONNX support
│       ├── requirements.txt       # Python dependencies
│       ├── test_inference.py      # Test suite
│       └── models/                # Pre-trained models by configuration
│           ├── w60s5_binary/      # 60s window, 5s step models
│           ├── w120s5_binary/     # 120s window, 5s step models
│           └── w120s60_binary/   # 120s window, 60s step models
│
├── examples/                      # Cross-platform example applications
├── scripts/                       # Build and deployment scripts
├── .github/workflows/             # CI/CD including HSI schema checks
└── CONTRIBUTING.md                # Contribution guidelines for all SDKs
```

**Platform-specific SDK repositories** (maintained separately):
- [synheart-emotion-dart](https://github.com/synheart-ai/synheart-emotion-dart) - Dart/Flutter SDK
- [synheart-emotion-python](https://github.com/synheart-ai/synheart-emotion-python) - Python SDK
- [synheart-emotion-kotlin](https://github.com/synheart-ai/synheart-emotion-kotlin) - Kotlin SDK
- [synheart-emotion-swift](https://github.com/synheart-ai/synheart-emotion-swift) - Swift SDK

## 🎯 Quick Start

### Python (Recommended for Testing)

```python
from datetime import datetime
from synheart_emotion import EmotionEngine, EmotionConfig

# Initialize engine
config = EmotionConfig()
engine = EmotionEngine.from_pretrained(config)

# Push biosignal data
engine.push(
    hr=72.0,
    rr_intervals_ms=[850.0, 820.0, 830.0, 845.0, 825.0],
    timestamp=datetime.now()
)

# Get inference results
results = engine.consume_ready()
for result in results:
    print(f"Emotion: {result.emotion} ({result.confidence:.1%})")
```

### Dart/Flutter

```dart
import 'package:synheart_emotion/synheart_emotion.dart';

// Initialize the emotion engine
final engine = EmotionEngine.fromPretrained(
  const EmotionConfig(
    window: Duration(seconds: 60),
    step: Duration(seconds: 5),
  ),
);

// Push biometric data
engine.push(
  hr: 72.0,
  rrIntervalsMs: [850.0, 820.0, 830.0, 845.0, 825.0],
  timestamp: DateTime.now(),
);

// Get results
final results = engine.consumeReady();
for (final result in results) {
  print('Emotion: ${result.emotion} (${result.confidence})');
}
```

### Kotlin

```kotlin
import ai.synheart.emotion.*

val config = EmotionConfig()
val engine = EmotionEngine.fromPretrained(config)

engine.push(
    hr = 72.0,
    rrIntervalsMs = listOf(850.0, 820.0, 830.0, 845.0, 825.0),
    timestamp = Date()
)

val results = engine.consumeReady()
results.forEach { result ->
    println("Emotion: ${result.emotion} (${result.confidence})")
}
```

### Swift

```swift
import SynheartEmotion

let config = EmotionConfig()
let engine = try! EmotionEngine.fromPretrained(config: config)

engine.push(
    hr: 72.0,
    rrIntervalsMs: [850.0, 820.0, 830.0, 845.0, 825.0],
    timestamp: Date()
)

let results = engine.consumeReady()
results.forEach { result in
    print("Emotion: \(result.emotion) (\(result.confidence))")
}
```

## 📊 Supported Affective Categories

> These categories represent inferred affective states based on physiological patterns, not definitive emotional labels.

The library currently supports three emotion categories:

- **😊 Amused**: Positive, engaged emotional state
- **😌 Calm**: Relaxed, peaceful emotional state
- **😰 Stressed**: Anxious, tense emotional state

## 🛠️ Development Tools

### Synthetic Data Generator

Generate realistic biosignal data for testing all SDKs:

```bash
cd tools/synthetic-data-generator

# Generate test data
python cli.py --emotion Calm --duration 60 --output ./data

# Generate session with transitions
python cli.py --session Calm Stressed Amused --transitions --output ./data
```

Exports to: CSV, JSON, Python, Kotlin, Swift

📖 [Data Generator Documentation](tools/synthetic-data-generator/README.md)

### WESAD Reference Models

Research artifacts with pre-trained ML models from WESAD dataset organized by window configuration:

- **Model Configurations**: w60s5_binary, w120s5_binary, w120s60_binary
- **Model Types**: ExtraTrees, RandomForest, LogisticRegression, XGBoost, LinearSVM
- **Formats**: ONNX (with built-in normalization) and scikit-learn pickle files
- **Features**: 14 HRV features (RMSSD, Mean_RR, HRV_SDNN, pNN50, etc.)
- **Random Data Generation**: Built-in function for testing with realistic HRV features
- For research and model comparison only
- **Not for production use** (use SDKs instead)

**Quick Start:**
```python
from tools.wesad_reference_models.inference import predict, generate_random_features

# Generate random test data
data = generate_random_features(emotion="baseline", n_samples=1, seed=42)

# Run inference
results = predict(data, config_name="w60s5_binary", model_name="extratrees")
```

📖 [Research Models Documentation](tools/wesad-reference-models/README.md)

## 🏗️ Architecture

### Standalone Mode

All SDKs implement the same architecture for standalone usage:

```
Wearable / Sensor
   └─(HR bpm, RR ms)──► Your App
                           │
                           ▼
                   Synheart Emotion SDK
            [Ring Buffer] → [Feature Extraction] → [Normalization]
                                     │
                                  [Model]
                                     │
                              EmotionResult
```

### HSI Integration Mode

When used via Synheart Core:

```
Synheart Core SDK
├── Wear Module (collects HR/RR from wearable)
│   └── HSI Runtime (processes biosignals, extracts HRV features)
│       └── EmotionHead Module
│           └── synheart-emotion EmotionEngine
│               [Ring Buffer] → [Feature Extraction] → [Normalization]
│                                     │
│                                  [Model]
│                                     │
│                              EmotionResult
│                                     │
│                          mapped to HSV.emotion
│                                     │
│                                     ▼
│                         Complete Human State Vector
│                         ├─ Emotion (stress, calm, engagement)
│                         ├─ Focus (cognitive load, clarity)
│                         ├─ Behavior (interaction patterns)
│                         └─ Context (activity, environment)
```

**Components:**
- **Ring Buffer**: Holds last 60s of HR/RR data (configurable)
- **Feature Extractor**: Computes HR mean, SDNN, RMSSD
- **Scaler**: Standardizes features using training μ/σ
- **Model**: Linear SVM (One-vs-Rest) with softmax
- **Emitter**: Throttles outputs (default: every 5s)

## 🎨 API Parity

All SDKs expose identical functionality:

| Feature | Python | Kotlin | Swift | Dart |
|---------|--------|--------|-------|------|
| EmotionConfig | ✅ | ✅ | ✅ | ✅ |
| EmotionEngine | ✅ | ✅ | ✅ | ✅ |
| EmotionResult | ✅ | ✅ | ✅ | ✅ |
| EmotionError | ✅ | ✅ | ✅ | ✅ |
| Feature Extraction | ✅ | ✅ | ✅ | ✅ |
| Linear SVM Model | ✅ | ✅ | ✅ | ✅ |
| Thread-Safe | ✅ | ✅ | ✅ | ✅ |
| Sliding Window | ✅ | ✅ | ✅ | ✅ |

## 🧪 Test Results

### Python SDK
- ✅ **16/16 tests passing** (100%)
- ✅ All examples working
- ✅ CLI demo functional

### Kotlin SDK
- ✅ All modules compile successfully
- ✅ 6 Kotlin source files
- ✅ API parity verified
- ✅ Gradle build and tests passing

### Swift SDK
- ✅ Swift build successful
- ✅ 6 Swift source files
- ✅ Multi-platform support (iOS, macOS, watchOS, tvOS)
- ✅ Swift Package Manager integration

## 🔬 Model Details

>The model outputs probabilistic class scores with confidence estimates over a rolling time window; predictions should be interpreted as state tendencies, not ground-truth emotional labels.

**Model Type**: ExtraTrees Classifier (ONNX-optimized)
**Task**: Binary emotion recognition (Baseline vs Stress) from HR/RR-derived HRV features
**Input Features**: 14 HRV features (`RMSSD`, `Mean_RR`, `HRV_SDNN`, `pNN50`, `HRV_HF`, `HRV_LF`, `HRV_HF_nu`, `HRV_LF_nu`, `HRV_LFHF`, `HRV_TP`, `HRV_SD1SD2`, `HRV_Sampen`, `HRV_DFA_alpha1`, `HR`) over configurable rolling windows (60s or 120s)
**Performance**:
- Accuracy: ~78.4% (LOSO CV)
- Macro-F1: ~72.6% (LOSO CV)
- Latency: < 10ms on modern mid-range devices (ONNX models)

The models are trained on WESAD-derived binary classification (Baseline vs Stress) with artifact rejection and normalization. Multiple window configurations available (60s/5s, 120s/5s, 120s/60s).

📖 [Model Card](docs/MODEL_CARD.md) | [RFC E1.1](docs/RFC-E1.1.md)

## 🔒 Privacy & Security

- **On-Device Processing**: All emotion inference happens locally
- **No Data Retention**: Raw biometric data is not retained after processing
- **No Network Calls**: No data is sent to external servers
- **Privacy-First Design**: No built-in storage - you control what gets persisted
- **Not a Medical Device**: This library is for wellness and research purposes only

⚠️ **Important**: The default model weights are trained on the WESAD dataset and achieve 78% accuracy. For production use, consider training on your own data if needed.

## 📚 Documentation

### SDK Documentation
- [Dart SDK](https://github.com/synheart-ai/synheart-emotion-dart) - Dart/Flutter implementation
- [Python SDK](https://github.com/synheart-ai/synheart-emotion-python) - Python implementation
- [Kotlin SDK](https://github.com/synheart-ai/synheart-emotion-kotlin) - Kotlin implementation
- [Swift SDK](https://github.com/synheart-ai/synheart-emotion-swift) - Swift implementation

### Tools Documentation
- [Synthetic Data Generator](tools/synthetic-data-generator/README.md) - Test data generation
- [WESAD Reference Models](tools/wesad-reference-models/README.md) - Research artifacts with ONNX support and random data generation

### Technical Documentation
- [RFC 0001](docs/RFC-Emotion-0001-spec.md) - Formal specification 
- [RFC 0002](docs/RFC-Emotion-0002-guide.md) - Implementation guide
 - [Model Card](docs/MODEL_CARD.md) - Model details and performance
- [Contributing Guide](CONTRIBUTING.md) - How to contribute (covers all SDKs)
- [Changelog](CHANGELOG.md) - Version history for all SDKs

## 🔧 Development

### Requirements

- **Dart SDK**: Flutter >= 3.10.0, Dart >= 3.0.0
- **Python SDK**: Python >= 3.8
- **Kotlin SDK**: Kotlin 1.8+, Android API 21+ (if targeting Android)
- **Swift SDK**: Swift 5.9+, iOS 13+ / macOS 11+ (if targeting Apple platforms)

### Running Tests

For SDK-specific tests, see the individual SDK repositories:
- [Dart Tests](https://github.com/synheart-ai/synheart-emotion-dart#testing)
- [Python Tests](https://github.com/synheart-ai/synheart-emotion-python#running-tests)
- [Kotlin Tests](https://github.com/synheart-ai/synheart-emotion-kotlin#testing)
- [Swift Tests](https://github.com/synheart-ai/synheart-emotion-swift#testing)

**Generate test data for all SDKs:**
```bash
cd tools/synthetic-data-generator
python cli.py --emotion Calm --duration 60 --output ./test_data
```

## 🔗 Integration Examples

### With Custom Data Source

```python
# Python example
from synheart_emotion import EmotionEngine, EmotionConfig
from your_sensor import get_biosignal_stream

engine = EmotionEngine.from_pretrained(EmotionConfig())

for data_point in get_biosignal_stream():
    engine.push(
        hr=data_point.heart_rate,
        rr_intervals_ms=data_point.rr_intervals,
        timestamp=data_point.timestamp
    )

    results = engine.consume_ready()
    if results:
        print(f"Current emotion: {results[0].emotion}")
```

### With Apple HealthKit (Swift)

See [Swift SDK Examples](https://github.com/synheart-ai/synheart-emotion-swift#healthkit-integration) for HealthKit integration.

## 📈 Performance Targets

**Target Performance (mid-range device):**
- **Latency**: < 5ms per inference
- **Model Size**: < 100 KB
- **CPU Usage**: < 2% during active streaming
- **Memory**: < 3 MB (engine + buffers)
- **Accuracy**: 78% on WESAD dataset (3-class emotion recognition)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and conventions
- Testing requirements
- Pull request process
- Development setup

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Synheart AI**: [synheart.ai](https://synheart.ai)
- **Documentation**: [Full Documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/synheart-ai/synheart-emotion/issues)
- **Discussions**: [GitHub Discussions](https://github.com/synheart-ai/synheart-emotion/discussions)

## 📖 Citation

If you use this SDK in your research:

```bibtex
@software{synheart_emotion,
  title = {Synheart Emotion: Multi-platform SDK for on-device emotion inference from biosignals},
  author = {Synheart AI Team},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/synheart-ai/synheart-emotion}
}
```

WESAD Dataset:
```bibtex
@article{schmidt2018introducing,
  title={Introducing WESAD, a multimodal dataset for wearable stress and affect detection},
  author={Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marberger, Claus and Van Laerhoven, Kristof},
  journal={ICMI 2018},
  year={2018}
}
```

## 👥 Authors

- **Israel Goytom** - _Initial work_, _RFC Design & Architecture_
- **Synheart AI Team** - _Development & Research_

---

**Made with ❤️ by the Synheart AI Team**

_Technology with a heartbeat._
