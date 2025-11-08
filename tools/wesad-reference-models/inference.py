from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


_THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = _THIS_DIR / "sdk models"
SCALER_PATH = MODEL_DIR / "scaler_wrist_all.pkl"

FEATURE_NAMES = ["SDNN", "RMSSD", "pNN50", "Mean_RR", "HR_mean"]
LABEL_MAP = {0: "Baseline", 1: "Stress", 2: "Amusement"}

_MODEL_CACHE: Dict[str, "LoadedModel"] = {}
_SCALER = None


class AttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")

        self.feature_embed = nn.Linear(input_dim, hidden_dims[0])
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0], num_heads=4, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dims[0])

        layers: List[nn.Module] = []
        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        self.ffn = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_embed(x)
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)
        x = x.squeeze(1)
        x = self.ffn(x)
        return self.output(x)


class DeepMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (256, 128, 64),
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        hidden_dims = list(hidden_dims)
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])

        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "linear1": nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        "bn1": nn.BatchNorm1d(hidden_dims[i + 1]),
                        "linear2": nn.Linear(hidden_dims[i + 1], hidden_dims[i + 1]),
                        "bn2": nn.BatchNorm1d(hidden_dims[i + 1]),
                        "dropout": nn.Dropout(dropout),
                        "shortcut": nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                        if hidden_dims[i] != hidden_dims[i + 1]
                        else nn.Identity(),
                    }
                )
            )

        self.output = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_bn(self.input_proj(x)))
        for block in self.blocks:
            identity = block["shortcut"](x)
            out = F.relu(block["bn1"](block["linear1"](x)))
            out = block["bn2"](block["linear2"](out))
            out = block["dropout"](out)
            x = F.relu(out + identity)
        return self.output(x)


class SimpleDeepNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64, 32),
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        hidden_dims = list(hidden_dims)
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _get_scaler():
    global _SCALER
    if _SCALER is None:
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Expected scaler file at {SCALER_PATH}")
        _SCALER = joblib.load(SCALER_PATH)
    return _SCALER


def prepare_input(data: Union[pd.DataFrame, Dict[str, Any], np.ndarray, List[Any]]) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        if len(data) != len(FEATURE_NAMES):
            raise ValueError(
                f"List input must have {len(FEATURE_NAMES)} values in the order {FEATURE_NAMES}"
            )
        df = pd.DataFrame([data], columns=FEATURE_NAMES)
    elif isinstance(data, np.ndarray):
        arr = np.asarray(data)
        if arr.ndim == 1:
            if arr.shape[0] != len(FEATURE_NAMES):
                raise ValueError(
                    f"Numpy array input must have {len(FEATURE_NAMES)} values"
                )
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Numpy array input must have shape (n_samples, {len(FEATURE_NAMES)})"
            )
        df = pd.DataFrame(arr, columns=FEATURE_NAMES)
    else:
        raise TypeError(
            "Input data must be a pandas DataFrame, dict, list, or numpy array"
        )

    missing = [name for name in FEATURE_NAMES if name not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

    df = df[FEATURE_NAMES].copy()
    df = df.fillna(df.median(numeric_only=True))

    scaler = _get_scaler()
    X_scaled = scaler.transform(df.values.astype(np.float32))
    return X_scaled


@dataclass
class LoadedModel:
    name: str
    kind: str
    model: Any
    path: Path
    metadata: Dict[str, Any]


def _discover_models() -> Dict[str, Path]:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    model_files: Dict[str, Path] = {}
    for path in MODEL_DIR.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("scaler"):
            continue
        if path.suffix.lower() not in {".pkl", ".pth"}:
            continue
        model_files[path.stem.lower()] = path
    if not model_files:
        raise FileNotFoundError(f"No model files discovered in {MODEL_DIR}")
    return model_files


_MODEL_FILES = _discover_models()

_MODEL_ALIASES = {
    "attention": "attention_wrist_all",
    "deepmlp": "deep_mlp_wrist_all",
    "simple": "simple_wrist_all",
    "randomforest": "rf_wrist_all",
    "extratrees": "extratrees_wrist_all",
    "xgb": "xgb_wrist_all",
    "logreg": "logreg_wrist_all",
    "linearsvm": "linearsvm_wrist_all",
}


def _resolve_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    if key in _MODEL_FILES:
        return key
    if key in _MODEL_ALIASES:
        alias = _MODEL_ALIASES[key]
        if alias in _MODEL_FILES:
            return alias
    raise ValueError(
        f"Model '{model_name}' is not available. Try one of: {list_available_models()}"
    )


def _build_torch_model(model_type: str, input_dim: int) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "attention":
        return AttentionClassifier(input_dim=input_dim)
    if model_type == "deep_mlp":
        return DeepMLPClassifier(input_dim=input_dim)
    if model_type == "simple":
        return SimpleDeepNN(input_dim=input_dim)
    raise ValueError(f"Unsupported torch model type '{model_type}'")


def _load_torch_model(path: Path, name: str) -> LoadedModel:
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

    return LoadedModel(name=name, kind="torch", model=model, path=path, metadata=metadata)


def _load_sklearn_model(path: Path, name: str) -> LoadedModel:
    model = joblib.load(path)
    return LoadedModel(name=name, kind="sklearn", model=model, path=path, metadata={})


def load_model(model_name: str) -> LoadedModel:
    resolved_name = _resolve_model_name(model_name)
    if resolved_name in _MODEL_CACHE:
        return _MODEL_CACHE[resolved_name]

    path = _MODEL_FILES[resolved_name]
    if path.suffix.lower() == ".pth":
        loaded = _load_torch_model(path, resolved_name)
    else:
        loaded = _load_sklearn_model(path, resolved_name)

    _MODEL_CACHE[resolved_name] = loaded
    return loaded


def list_available_models() -> List[str]:
    return sorted(_MODEL_FILES.keys())


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def _format_probabilities(probs: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if probs is None:
        return None
    return {LABEL_MAP[i]: float(probs[i]) for i in range(probs.shape[0])}


def predict(
    data: Union[pd.DataFrame, Dict[str, Any], np.ndarray, List[Any]],
    model_name: str,
    return_probabilities: bool = True,
) -> List[Dict[str, Any]]:
    """Run inference using a single model.

    Returns a list of prediction dictionaries, one per row of input.
    """

    model_wrapper = load_model(model_name)
    X = prepare_input(data)

    start = time.perf_counter()

    if model_wrapper.kind == "torch":
        with torch.no_grad():
            tensor = torch.from_numpy(X).float()
            logits = model_wrapper.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
    else:
        model = model_wrapper.model
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            probs = _softmax_np(scores)
        else:
            probs = None
        preds = model.predict(X)

    elapsed = time.perf_counter() - start
    print(f"{model_wrapper.name} inference: {elapsed:.4f} seconds")

    results: List[Dict[str, Any]] = []
    for idx, pred_class in enumerate(preds):
        prob_row = probs[idx] if probs is not None else None
        result = {
            "numeric": int(pred_class),
            "label": LABEL_MAP[int(pred_class)],
        }
        if return_probabilities and prob_row is not None:
            result["probabilities"] = _format_probabilities(prob_row)
        results.append(result)

    return results

