# RFC-Emotion-0001: Synheart Emotion — On-Device Affective State Inference Specification

**Status:** Accepted  
**RFC Type:** Capability / Inference Module  
**Target Version:** Emotion SDK v0.1+  
**Last Updated:** 2025-12-29  
**Authors:** Synheart AI Team

---

## 1. Abstract

This RFC defines Synheart Emotion, a cross-platform, on-device inference module for estimating momentary affective state tendencies from physiological signals, specifically heart rate (HR) and RR intervals.

Synheart Emotion is designed to operate:
- Fully on-device
- With strict privacy guarantees
- With consistent behavior across platforms
- As both a standalone SDK and a runtime dependency of Synheart Core

The module produces probabilistic, confidence-scoped outputs that are compatible with the Synheart Human State Interface (HSI) and suitable for integration into higher-level human state representations.

⸻

## 2. Motivation

Emotion is a critical component of human state, but direct measurement of emotion is not possible from biosignals alone. Instead, physiological signals such as HR and RR provide indirect, noisy proxies for arousal, stress, and engagement-related affective dimensions.

Existing emotion SDKs typically suffer from one or more of the following issues:
- Cloud-based inference with privacy risks
- Platform-specific implementations with inconsistent semantics
- Undocumented or opaque models
- Emotion "detection" claims that exceed scientific validity

Synheart Emotion addresses these gaps by providing:
- On-device affective state inference
- Explicit probabilistic outputs with confidence
- Time-windowed semantics
- Transparent architecture and documentation
- Formal compatibility with Synheart's HSI contract

---

## 3. Non-Goals

Synheart Emotion explicitly does not aim to:
- Diagnose medical or mental health conditions
- Provide ground-truth emotional labels
- Infer emotions from non-physiological modalities (e.g. text, audio, video)
- Replace psychological or clinical assessments
- Persist or transmit raw biometric data

---

## 4. Scope

This RFC specifies:
- Input signal requirements and constraints
- Inference semantics and timing behavior
- Output structure and guarantees
- Integration modes (standalone vs Synheart Core)
- Model constraints and performance targets
- Privacy and security requirements

It does not mandate a specific ML framework, training pipeline, or hardware sensor.

---

## 5. Terminology

- **Affective State:** A short-term physiological-emotional tendency (e.g. stress, calm)
- **Inference Window:** A rolling time window over which features are computed
- **Emission Step:** Minimum time between successive outputs
- **Confidence:** Model-estimated certainty of the output distribution
- **HSI:** Human State Interface, Synheart's canonical interchange contract
- **HSV:** Human State Vector, internal runtime representation

---

## 6. Input Signals

### 6.1 Required Inputs

Synheart Emotion consumes the following inputs:

| Signal | Description | Unit |
|--------|-------------|------|
| hr | Instantaneous heart rate | bpm |
| rr_intervals | Beat-to-beat intervals | milliseconds |
| timestamp | Sample time | UTC |

### 6.2 Input Constraints

- RR intervals MUST be pre-filtered for obvious artifacts where possible
- Timestamps MUST be monotonic per data stream
- Input cadence MAY be irregular
- Missing data MUST NOT crash the engine
- Minimum RR count: 30 intervals required for stable feature extraction
- RR outliers SHOULD be filtered: < 300 ms or > 2000 ms
- RR jumps > 250 ms vs previous SHOULD be treated as artifacts

---

## 7. Inference Semantics

### 7.1 Windowing

- Default inference window: 60 seconds
- Default emission step: 5 seconds
- Window and step MUST be configurable

### 7.2 Feature Extraction

Minimum required features:
- Mean heart rate (hr_mean): Average of HR samples over inference window
- SDNN: Standard deviation of RR intervals (sample std; N-1 denominator)
- RMSSD: Root mean square of successive differences, sqrt(mean(diff(rr)^2))

Feature computation:
- Features MUST be computed over the inference window (default 60 seconds)
- Features MUST be normalized using training statistics (μ, σ) bundled with the model
- Normalization formula: (x - μ) / σ

Additional features MAY be added in future versions but MUST be backward compatible.

---

## 8. Output Semantics

### 8.1 Affective Categories

Version 1.0 supports the following affective categories:
- Calm
- Stressed
- Amused

These categories represent affective tendencies, not definitive emotional states.

### 8.2 Output Structure

Each inference produces an EmotionResult containing:
- A probability distribution over affective categories
- A confidence score
- A timestamp and window reference

Outputs MUST be:
- Probabilistic (not single-label only)
- Confidence-scoped
- Time-windowed

---

## 9. HSI Compatibility

Synheart Emotion outputs MUST be mappable to the HSI EmotionState schema.

Requirements:
- Field names and semantics MUST remain compatible
- New categories MUST NOT be added without HSI review
- Schema validation MUST be enforced in CI

Synheart Emotion has no code dependency on synheart-core, but synheart-core depends on synheart-emotion at runtime.

### 9.1 HSI Mapping Table

EmotionResult fields MUST map to HSI EmotionState as follows:

| EmotionResult Field | HSI EmotionState Field | Mapping |
|---------------------|------------------------|---------|
| `probabilities["Stressed"]` | `emotion.stress` | Direct (0.0-1.0) |
| `probabilities["Calm"]` | `emotion.calm` | Direct (0.0-1.0) |
| `probabilities["Amused"]` | `emotion.engagement` | Direct (0.0-1.0) |
| `confidence` | `emotion.activation` | Derived formula |
| `features` | `emotion.valence` | Derived formula |

Schema validation MUST be enforced in CI against the HSI specification.

---

## 10. Deployment Modes

### 10.1 Standalone Mode

- Used directly by applications
- No dependency on Synheart Core
- Outputs EmotionResult objects only

### 10.2 Core Integration Mode

- Used as EmotionHead within Synheart Core
- Outputs mapped into HSV.emotion
- Participates in unified human state computation

---

## 11. Model Requirements

### 11.1 Constraints

- Model MUST run fully on-device
- Model size SHOULD be < 100 KB
- Inference latency SHOULD be < 5 ms on mid-range devices
- No network calls during inference

### 11.2 Reference Model

The reference implementation uses:
- Linear SVM (One-vs-Rest)
- HRV-derived features
- Trained on a WESAD-derived 3-class subset

Model implementations MUST:
- Identify models with unique model IDs (format: `{type}_{features}_{version}`)
- Bundle normalization parameters (μ, σ) with the model
- Support the minimum required feature set (hr_mean, SDNN, RMSSD)

Model details are documented separately in the Model Card.

### 11.3 Package Naming

Platform-specific package names:
- Dart/Flutter: `synheart_emotion`
- Python: `synheart-emotion` (future)
- Node.js: `@synheart/emotion` (future)

The package name MUST be consistent across platforms for discoverability.

---

## 12. Performance Targets

| Metric | Target |
|--------|--------|
| Latency | < 5 ms |
| CPU Usage | < 2% |
| Memory | < 3 MB |
| Accuracy | ~78% (WESAD subset) |

These are targets, not guarantees.

---

## 13. Privacy & Security

Synheart Emotion MUST adhere to:
- On-device processing only
- No raw data persistence by default
- No implicit network activity
- No logging of biometric data
- Explicit developer control over storage

### 13.1 Storage Policy

Storage policies MUST be explicitly configurable:
- `none` (default): No persistence
- `ephemeral`: Session-only storage, cleared on app termination
- `local_persist`: Encrypted local storage with explicit user consent

When storage is enabled:
- Only EmotionResult objects and aggregated features MAY be persisted
- Raw RR intervals MUST NOT be persisted unless explicitly enabled by host app
- Storage MUST use platform-provided encryption (e.g., encrypted box/Isolate)
- User consent MUST be obtained and managed by the host application

### 13.2 Consent Requirements

The module MUST provide:
- ConsentState check mechanism
- Requirement for host app to surface user consent UI
- No implicit data collection or transmission

The module is not a medical device and is intended for wellness, UX, and research use.

---

## 14. Extensibility

Future extensions MAY include:
- Additional affective categories
- Personalization layers
- Alternative models
- Sensor fusion (subject to separate RFCs)

Any extension MUST preserve:
- Backward compatibility
- HSI schema stability
- Cross-platform parity

---

## 15. Open Questions

- How personalization interacts with HSI confidence semantics
- How multi-session baselines should be represented
- Whether adaptive windows should be standardized

These are intentionally deferred.

---

## 16. Error Handling

### 16.1 Error Types

The module MUST define the following error types:
- `tooFewRR`: Insufficient RR intervals for feature extraction (minimum required: 30)
- `badInput`: Invalid input data (e.g., NaN, out-of-range values)
- `modelIncompatible`: Model feature dimensions mismatch

### 16.2 Error Behavior

Errors MUST be handled gracefully:
- Non-throwing by default in streaming mode
- Engine MUST skip emission and log via callback when errors occur
- Errors MUST NOT crash the engine
- Optional error callback (`onWarn`) for host application logging

---

## 17. References

- Schmidt et al., Introducing WESAD, ICMI 2018
- Synheart Core HSI Specification
- Synheart Emotion Model Card

---

## 18. Summary

Synheart Emotion defines a careful, honest, and scalable approach to affective state inference from physiological signals.

By constraining scope, enforcing privacy, and integrating formally with HSI, it provides a reliable building block for emotion-aware applications without overstating scientific claims.

---