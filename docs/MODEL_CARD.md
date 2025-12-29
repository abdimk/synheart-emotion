# Model Card Template: synheart-emotion

This document describes the model card structure and requirements for models used with the synheart-emotion SDK. Each model implementation should provide its own model card following this template.

---

## Model Identification

**Model ID:** `{type}_{features}_{version}` (e.g., `svm_linear_wrist_sdnn_v1_0`)  
**Model Type:** [e.g., Linear SVM, Neural Network, etc.]  
**Task:** Momentary emotion recognition from HR/RR biosignals  
**Labels:** `Amused`, `Calm`, `Stressed` (as per RFC-Emotion-0001)  
**Minimum Required Features:** `[hr_mean, sdnn, rmssd]` over a 60s rolling window  
**Export Format:** [e.g., Embedded arrays, TFLite, ONNX, etc.]

---

## Intended Use

Models in synheart-emotion are designed for:
- On-device inference in applications via `synheart_emotion` package
- Fusion into SWIP Score within `swip-core`
- Integration into Synheart Core's Human State Interface (HSI) via EmotionHead module

### Deployment Contexts

Models are deployed in **two contexts**:

#### 1. Standalone SDK
Direct integration via synheart-emotion package for applications requiring emotion-only detection.

#### 2. Synheart Core (HSI Integration)
Consumed by synheart-core's EmotionHead module for unified human state representation.

**Performance Requirements:** Models MUST meet the performance targets specified in RFC-Emotion-0001:
- Latency: < 5 ms on mid-range devices
- Model size: < 100 KB
- CPU usage: < 2% during active streaming
- Memory: < 3 MB (engine + buffers)

**Schema Compatibility:** Model outputs MUST be mappable to HSI EmotionState schema as specified in RFC-Emotion-0001 section 9.1.

---

## Model Requirements

All models MUST comply with the following requirements (as per RFC-Emotion-0001):

### Constraints
- Model MUST run fully on-device
- Model size SHOULD be < 100 KB
- Inference latency SHOULD be < 5 ms on mid-range devices
- No network calls during inference

### Feature Support
- Models MUST support the minimum required feature set: `hr_mean`, `SDNN`, `RMSSD`
- Additional features MAY be added but MUST be backward compatible
- Features MUST be computed over the inference window (default 60 seconds)
- Features MUST be normalized using training statistics (μ, σ) bundled with the model

### Model Identification
- Models MUST have unique model IDs following format: `{type}_{features}_{version}`
- Normalization parameters (μ, σ) MUST be bundled with the model
- Model metadata MUST include: model ID, version, type, and feature order

---

## Limitations

All models share the following limitations:
- **Not a medical device** — intended for wellness, UX, and research use only
- **Sensitive to RR quality** — requires minimally 30 RR intervals over the inference window for stable feature extraction
- **Population generalization** — models trained on specific datasets may not generalize to all populations or sensor conditions without calibration
- **Input constraints** — RR intervals MUST be pre-filtered for artifacts (< 300 ms or > 2000 ms; jumps > 250 ms)

---

## Training Data

Model cards SHOULD document:
- Training dataset(s) used
- Data preprocessing steps (artifact rejection, normalization, etc.)
- Population characteristics of training data
- Any data augmentation or synthetic data generation techniques

**Note:** Training data details are model-specific and should be documented in individual model cards.

---

## Performance Metrics

Model cards SHOULD report the following metrics (as applicable):
- **Accuracy:** Classification accuracy on held-out test set
- **Macro-F1:** Macro-averaged F1 score across emotion categories
- **Per-class F1:** F1 score for each emotion category (Amused, Calm, Stressed)
- **Calibration:** Expected Calibration Error (ECE) if probability calibration is performed
- **Latency:** Inference latency on target devices
- **Confusion Matrix:** Per-class classification performance

**Evaluation Protocol:** Models SHOULD be evaluated using the offline validation protocol specified in RFC-Emotion-0002-guide section 13.

---

## Ethical Considerations

All models MUST adhere to the following ethical guidelines:
- **On-device processing only** — no raw biometric data leaves the device by default
- **Consent requirements** — host applications MUST obtain user consent before using emotion inference
- **Privacy by design** — no raw RR intervals persisted unless explicitly enabled by host app
- **Transparency** — model limitations and intended use MUST be clearly communicated
- **Avoid high-stakes decisions** — emotion outputs should not be used as the sole basis for critical decisions

See RFC-Emotion-0001 section 13 for detailed privacy and security requirements.

---

## Versioning

Models MUST follow the versioning scheme:
- **Model ID format:** `{type}_{features}_{version}`
- **Version format:** `v{major}_{minor}` (e.g., `v1_0`, `v1_1`)
- **Compatibility:** Models with the same major version MUST maintain backward compatibility for feature dimensions and output schema
- **Breaking changes:** Require a new major version

### Version Information

Model cards SHOULD document:
- Current model version
- Feature order and dimensions
- Scaler statistics (μ, σ) for normalization
- Compatibility requirements with engine configuration

---

## Changelog

Model cards SHOULD maintain a changelog documenting:
- Version history
- Changes to model architecture, features, or training data
- Performance improvements or regressions
- Breaking changes and migration notes

---

## References

- [RFC-Emotion-0001](RFC-Emotion-0001-spec.md) - Formal specification
- [RFC-Emotion-0002](RFC-Emotion-0002-guide.md) - Implementation guide
- [Synheart Core HSI Specification](../synheart-core/docs/HSI_SPECIFICATION.md) - Schema compatibility requirements

---

## Model-Specific Documentation

Individual model implementations should extend this template with:
- Specific model architecture details
- Training dataset information
- Model-specific performance metrics
- Deployment instructions
- Known limitations or edge cases
