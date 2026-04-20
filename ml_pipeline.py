from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


@dataclass
class ImageMetrics:
    edge_density: float
    entropy: float
    contrast: float
    saturation_mean: float
    brightness_mean: float


def _compute_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / (hist.sum() + 1e-12)
    return float(-np.sum(prob * np.log2(prob + 1e-12)))


def extract_metrics(image_rgb: np.ndarray) -> ImageMetrics:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, threshold1=70, threshold2=160)

    return ImageMetrics(
        edge_density=float(np.mean(edges > 0)),
        entropy=_compute_entropy(gray),
        contrast=float(np.std(gray) / 255.0),
        saturation_mean=float(np.mean(hsv[:, :, 1]) / 255.0),
        brightness_mean=float(np.mean(hsv[:, :, 2]) / 255.0),
    )


def metrics_to_vector(metrics: ImageMetrics) -> List[float]:
    return [
        metrics.edge_density,
        metrics.entropy,
        metrics.contrast,
        metrics.saturation_mean,
        metrics.brightness_mean,
    ]


def _load_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def load_dataset(dataset_root: str) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(dataset_root)
    classes = ["virus", "non_virus"]

    features: List[List[float]] = []
    labels: List[int] = []

    for idx, class_name in enumerate(classes):
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for image_path in class_dir.glob("*.*"):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            img = _load_image(image_path)
            metrics = extract_metrics(img)
            features.append(metrics_to_vector(metrics))
            labels.append(idx)

    if len(features) < 10:
        raise ValueError("Dataset trop petit. Il faut au moins 10 images au total.")

    if len(set(labels)) < 2:
        raise ValueError("Le dataset doit contenir les 2 classes: virus et non_virus.")

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def train_and_save_model(dataset_root: str, model_path: str, random_state: int = 42) -> Dict[str, object]:
    x, y = load_dataset(dataset_root)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)

    payload = {
        "model": model,
        "classes": ["virus", "non_virus"],
        "feature_names": [
            "edge_density",
            "entropy",
            "contrast",
            "saturation_mean",
            "brightness_mean",
        ],
    }

    model_target = Path(model_path)
    model_target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_target)

    metrics = {
        "samples": int(len(y)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "accuracy": float(acc),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
        "classification_report": classification_report(y_test, pred, zero_division=0),
    }
    return metrics


def load_model(model_path: str) -> Dict[str, object]:
    return joblib.load(model_path)


def predict_label(payload: Dict[str, object], image_rgb: np.ndarray) -> Tuple[str, float]:
    model = payload["model"]
    classes = payload["classes"]
    metrics = extract_metrics(image_rgb)
    vector = np.array([metrics_to_vector(metrics)], dtype=np.float32)

    label_idx = int(model.predict(vector)[0])
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(vector)[0]))

    return str(classes[label_idx]), confidence
