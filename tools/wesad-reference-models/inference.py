from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


_THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = _THIS_DIR / "models"

# Feature names matching the metadata schema (14 features)
FEATURE_NAMES = [
    "RMSSD",
    "Mean_RR",
    "HRV_SDNN",
    "pNN50",
    "HRV_HF",
    "HRV_LF",
    "HRV_HF_nu",
    "HRV_LF_nu",
    "HRV_LFHF",
    "HRV_TP",
    "HRV_SD1SD2",
    "HRV_Sampen",
    "HRV_DFA_alpha1",
    "HR",
]

# Binary classification labels
LABEL_MAP_BINARY = {0: "Baseline", 1: "Stress"}

_MODEL_CACHE: Dict[str, "LoadedModel"] = {}


@dataclass
class LoadedModel:
    name: str
    kind: str  # "onnx", "pkl", "pth"
    model: Any
    path: Path
    metadata: Dict[str, Any]
    feature_names: List[str]
    window_config: Optional[Dict[str, int]] = None


def _discover_models() -> Dict[str, Dict[str, Path]]:
    """Discover all available models organized by configuration."""
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    models_by_config: Dict[str, Dict[str, Path]] = {}

    for config_dir in MODEL_DIR.iterdir():
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name
        models_by_config[config_name] = {}

        for model_file in config_dir.iterdir():
            if not model_file.is_file():
                continue

            if model_file.suffix.lower() in {".pkl", ".pth", ".onnx"}:
                model_name = model_file.stem.lower()
                models_by_config[config_name][model_name] = model_file

    if not models_by_config:
        raise FileNotFoundError(f"No model files discovered in {MODEL_DIR}")
    return models_by_config


_MODEL_FILES = _discover_models()


def _load_metadata(model_path: Path) -> Optional[Dict[str, Any]]:
    """Load metadata JSON file if it exists."""
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return None


def _load_onnx_model(path: Path, name: str) -> LoadedModel:
    """Load an ONNX model."""
    if not ONNX_AVAILABLE:
        raise ImportError(
            "onnxruntime is required for ONNX models. Install with: pip install onnxruntime"
        )

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    metadata = _load_metadata(path) or {}
    feature_names = metadata.get("schema", {}).get("input_names", FEATURE_NAMES)
    window_config = metadata.get("window_config")

    return LoadedModel(
        name=name,
        kind="onnx",
        model=session,
        path=path,
        metadata=metadata,
        feature_names=feature_names,
        window_config=window_config,
    )


def _load_pkl_model(path: Path, name: str) -> LoadedModel:
    """Load a joblib/pickle model."""
    model = joblib.load(path)

    metadata = _load_metadata(path) or {}
    feature_names = metadata.get("schema", {}).get("input_names", FEATURE_NAMES)
    window_config = metadata.get("window_config")

    return LoadedModel(
        name=name,
        kind="pkl",
        model=model,
        path=path,
        metadata=metadata,
        feature_names=feature_names,
        window_config=window_config,
    )


def _build_torch_model(model_type: str, input_dim: int) -> nn.Module:
    """Build a PyTorch model (legacy support)."""
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for PyTorch models")

    # Simplified model builders (matching original structure)
    if model_type == "attention":
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    elif model_type == "deep_mlp":
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    elif model_type == "simple":
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
    else:
        raise ValueError(f"Unsupported torch model type '{model_type}'")


def _load_torch_model(path: Path, name: str) -> LoadedModel:
    """Load a PyTorch model (legacy support)."""
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for PyTorch models")

    checkpoint = torch.load(path, map_location="cpu")
    model_type = checkpoint.get("model_type")
    input_dim = checkpoint.get("input_dim")

    if model_type is None or input_dim is None:
        raise ValueError(
            f"Torch model '{path.name}' is missing required metadata (model_type/input_dim)"
        )

    model = _build_torch_model(model_type, input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {
        "model_type": model_type,
        "input_dim": input_dim,
        "test_acc": checkpoint.get("test_acc"),
        "test_f1": checkpoint.get("test_f1"),
    }

    return LoadedModel(
        name=name,
        kind="pth",
        model=model,
        path=path,
        metadata=metadata,
        feature_names=FEATURE_NAMES,
        window_config=None,
    )


def load_model(config_name: str, model_name: str) -> LoadedModel:
    """
    Load a model from a specific configuration directory.

    Args:
        config_name: Configuration name (e.g., 'w60s5_binary', 'w120s5_binary', 'w120s60_binary')
        model_name: Model name (e.g., 'extratrees', 'rf', 'logreg')

    Returns:
        LoadedModel instance
    """
    cache_key = f"{config_name}/{model_name}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if config_name not in _MODEL_FILES:
        raise ValueError(
            f"Configuration '{config_name}' not found. Available: {list(_MODEL_FILES.keys())}"
        )

    config_models = _MODEL_FILES[config_name]
    model_name_lower = model_name.lower()

    # Try exact match first
    if model_name_lower in config_models:
        path = config_models[model_name_lower]
    else:
        # Try partial match
        matching = [k for k in config_models.keys() if model_name_lower in k]
        if not matching:
            raise ValueError(
                f"Model '{model_name}' not found in '{config_name}'. "
                f"Available: {list(config_models.keys())}"
            )
        if len(matching) > 1:
            raise ValueError(
                f"Multiple models match '{model_name}': {matching}. Be more specific."
            )
        path = config_models[matching[0]]

    # Load based on file extension
    if path.suffix.lower() == ".onnx":
        loaded = _load_onnx_model(path, cache_key)
    elif path.suffix.lower() == ".pth":
        loaded = _load_torch_model(path, cache_key)
    else:
        loaded = _load_pkl_model(path, cache_key)

    _MODEL_CACHE[cache_key] = loaded
    return loaded


def list_available_models() -> Dict[str, List[str]]:
    """List all available models organized by configuration."""
    return {
        config: sorted(models.keys()) for config, models in _MODEL_FILES.items()
    }


def prepare_input(
    data: Union[pd.DataFrame, Dict[str, Any], np.ndarray, List[Any]],
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Prepare input data for inference.

    Args:
        data: Input data (DataFrame, dict, array, or list)
        feature_names: Optional list of feature names (uses model's feature names if not provided)

    Returns:
        Prepared numpy array
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        if len(data) != len(feature_names):
            raise ValueError(
                f"List input must have {len(feature_names)} values in the order {feature_names}"
            )
        df = pd.DataFrame([data], columns=feature_names)
    elif isinstance(data, np.ndarray):
        arr = np.asarray(data)
        if arr.ndim == 1:
            if arr.shape[0] != len(feature_names):
                raise ValueError(
                    f"Numpy array input must have {len(feature_names)} values"
                )
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(feature_names):
            raise ValueError(
                f"Numpy array input must have shape (n_samples, {len(feature_names)})"
            )
        df = pd.DataFrame(arr, columns=feature_names)
    else:
        raise TypeError(
            "Input data must be a pandas DataFrame, dict, list, or numpy array"
        )

    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

    df = df[feature_names].copy()
    df = df.fillna(df.median(numeric_only=True))

    # Handle inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    X = df.values.astype(np.float32)
    return X


def _format_probabilities(probs: np.ndarray) -> Dict[str, float]:
    """Format probability array as dictionary."""
    return {LABEL_MAP_BINARY[i]: float(probs[i]) for i in range(probs.shape[0])}


def predict(
    data: Union[pd.DataFrame, Dict[str, Any], np.ndarray, List[Any]],
    config_name: str,
    model_name: str = "extratrees",
    return_probabilities: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run inference using a model from a specific configuration.

    Args:
        data: Input data (DataFrame, dict, array, or list)
        config_name: Configuration name (e.g., 'w60s5_binary', 'w120s5_binary', 'w120s60_binary')
        model_name: Model name (default: 'extratrees')
        return_probabilities: Whether to return probability distributions

    Returns:
        List of prediction dictionaries, one per row of input
    """
    model_wrapper = load_model(config_name, model_name)
    X = prepare_input(data, model_wrapper.feature_names)

    start = time.perf_counter()

    if model_wrapper.kind == "onnx":
        # ONNX inference
        input_name = model_wrapper.model.get_inputs()[0].name
        outputs = model_wrapper.model.run(None, {input_name: X})
        probs = outputs[0]  # ONNX models output probabilities directly
        preds = np.argmax(probs, axis=1)

    elif model_wrapper.kind == "pth":
        # PyTorch inference
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for PyTorch models")
        with torch.no_grad():
            tensor = torch.from_numpy(X).float()
            logits = model_wrapper.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

    else:
        # Scikit-learn inference
        model = model_wrapper.model
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        else:
            probs = None
        preds = model.predict(X)

    elapsed = time.perf_counter() - start
    print(f"{config_name}/{model_name} inference: {elapsed:.4f} seconds")

    results: List[Dict[str, Any]] = []
    for idx, pred_class in enumerate(preds):
        prob_row = probs[idx] if probs is not None else None
        result = {
            "numeric": int(pred_class),
            "label": LABEL_MAP_BINARY[int(pred_class)],
        }
        if return_probabilities and prob_row is not None:
            result["probabilities"] = _format_probabilities(prob_row)
        results.append(result)

    return results


# ========================================
# Random Data Generation (for examples)
# ========================================


def generate_random_features(
    emotion: str = "baseline", n_samples: int = 1, seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate random HRV features for testing.

    Args:
        emotion: Target emotion ('baseline', 'stress')
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with generated features
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    features_list = []

    for _ in range(n_samples):
        if emotion.lower() == "baseline":
            # Calm/baseline state: lower HR, higher HRV
            hr = random.uniform(60, 75)
            rmssd = random.uniform(40, 80)
            mean_rr = 60000 / hr + random.uniform(-50, 50)
            sdnn = random.uniform(50, 90)
            pnn50 = random.uniform(15, 35)
            hf = random.uniform(800, 2000)
            lf = random.uniform(500, 1500)
        else:  # stress
            # Stress state: higher HR, lower HRV
            hr = random.uniform(75, 100)
            rmssd = random.uniform(20, 50)
            mean_rr = 60000 / hr + random.uniform(-30, 30)
            sdnn = random.uniform(30, 60)
            pnn50 = random.uniform(5, 20)
            hf = random.uniform(300, 1000)
            lf = random.uniform(400, 1200)

        # Normalized features
        hf_nu = hf / (hf + lf) if (hf + lf) > 0 else 0.5
        lf_nu = lf / (hf + lf) if (hf + lf) > 0 else 0.5
        lfhf = lf / hf if hf > 0 else 1.0
        tp = hf + lf

        # Nonlinear features
        sd1sd2 = random.uniform(0.3, 0.7)
        sampen = random.uniform(0.8, 1.5)
        dfa_alpha1 = random.uniform(0.8, 1.2)

        features = {
            "RMSSD": rmssd,
            "Mean_RR": mean_rr,
            "HRV_SDNN": sdnn,
            "pNN50": pnn50,
            "HRV_HF": hf,
            "HRV_LF": lf,
            "HRV_HF_nu": hf_nu,
            "HRV_LF_nu": lf_nu,
            "HRV_LFHF": lfhf,
            "HRV_TP": tp,
            "HRV_SD1SD2": sd1sd2,
            "HRV_Sampen": sampen,
            "HRV_DFA_alpha1": dfa_alpha1,
            "HR": hr,
        }

        features_list.append(features)

    return pd.DataFrame(features_list)


# ========================================
# Example Usage
# ========================================

if __name__ == "__main__":
    print("=" * 80)
    print("WESAD Reference Models - Inference Example")
    print("=" * 80)
    print()

    # Example 1: List available models
    print("[1] Available models:")
    available = list_available_models()
    for config, models in available.items():
        print(f"   {config}: {models}")
    print()

    # Example 2: Generate random data and run inference
    print("[2] Generating random baseline features and running inference...")
    baseline_data = generate_random_features(emotion="baseline", n_samples=3, seed=42)
    print("\nGenerated features (first sample):")
    print(baseline_data.iloc[0].to_dict())
    print()

    # Try different configurations
    configs_to_test = ["w60s5_binary", "w120s5_binary", "w120s60_binary"]

    for config in configs_to_test:
        if config not in available:
            continue

        print(f"\n{'='*60}")
        print(f"Testing {config}")
        print(f"{'='*60}")

        try:
            results = predict(
                baseline_data,
                config_name=config,
                model_name="extratrees",
                return_probabilities=True,
            )

            for i, result in enumerate(results):
                print(f"\nSample {i+1}:")
                print(f"  Predicted: {result['label']}")
                if "probabilities" in result:
                    print(f"  Probabilities:")
                    for label, prob in result["probabilities"].items():
                        print(f"    {label}: {prob:.3f}")

        except Exception as e:
            print(f"  Error: {e}")

    # Example 3: Stress prediction
    print(f"\n\n{'='*80}")
    print("[3] Generating random stress features and running inference...")
    stress_data = generate_random_features(emotion="stress", n_samples=2, seed=123)
    print("\nGenerated features (first sample):")
    print(stress_data.iloc[0].to_dict())
    print()

    config = "w60s5_binary"
    if config in available:
        try:
            results = predict(
                stress_data,
                config_name=config,
                model_name="extratrees",
                return_probabilities=True,
            )

            for i, result in enumerate(results):
                print(f"\nSample {i+1}:")
                print(f"  Predicted: {result['label']}")
                if "probabilities" in result:
                    print(f"  Probabilities:")
                    for label, prob in result["probabilities"].items():
                        print(f"    {label}: {prob:.3f}")

        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'='*80}")
    print("✅ Examples completed!")
    print("=" * 80)
