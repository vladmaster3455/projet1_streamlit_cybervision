import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from ml_pipeline import extract_metrics, load_model, predict_label


@dataclass
class ImageMetrics:
    edge_density: float
    entropy: float
    contrast: float
    saturation_mean: float
    brightness_mean: float
    artifact_score: float


def init_model_and_data() -> None:
    model_path = Path("models/cybervision_model.joblib")
    data_path = Path("data/dataset_alt")

    if model_path.exists() and data_path.exists():
        return

    with st.spinner("Initialization in progress: dataset and virus model setup..."):
        try:
            if not data_path.exists():
                from make_alt_dataset import create_dataset

                create_dataset(str(data_path), per_class=50)

            if not model_path.exists():
                from train_model import main as train_main

                old_argv = sys.argv
                sys.argv = [
                    "train_model.py",
                    "--dataset",
                    str(data_path),
                    "--model-out",
                    str(model_path),
                    "--metrics-out",
                    "models/train_metrics.json",
                ]
                train_main()
                sys.argv = old_argv
        except Exception as exc:
            st.error(f"Initialization failed: {exc}")


def analyze_image_metrics(image: np.ndarray) -> ImageMetrics:
    m = extract_metrics(image)
    artifact_score = 0.45 * m.edge_density + 0.35 * m.contrast + 0.20 * min(m.entropy / 8.0, 1.0)
    return ImageMetrics(
        edge_density=m.edge_density,
        entropy=m.entropy,
        contrast=m.contrast,
        saturation_mean=m.saturation_mean,
        brightness_mean=m.brightness_mean,
        artifact_score=float(artifact_score),
    )


def build_edge_overlay(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    heat = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.75, heat[:, :, ::-1], 0.25, 0)


def classify_virus_risk(score: float) -> Tuple[str, str]:
    if score >= 0.58:
        return "HIGH", "#C1121F"
    if score >= 0.42:
        return "MEDIUM", "#D97706"
    return "LOW", "#1B5E20"


def _resolve_local_weights(candidates: List[str]) -> Optional[str]:
    roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
    ]
    for root in roots:
        for name in candidates:
            path = root / name
            if path.exists():
                return str(path)
    return None


@st.cache_resource
def load_detector(detector_name: str):
    if detector_name == "YOLO":
        from ultralytics import YOLO

        weights = _resolve_local_weights(["yolo11n.pt", "yolov8n.pt"])
        return YOLO(weights or "yolov8n.pt")

    if detector_name == "RT-DETR":
        from ultralytics import RTDETR

        weights = _resolve_local_weights(["rtdetr-l.pt", "rtdetr-l.yaml"])
        return RTDETR(weights or "rtdetr-l.pt")

    if detector_name == "DINO":
        from transformers import pipeline

        return pipeline(
            task="zero-shot-object-detection",
            model="IDEA-Research/grounding-dino-tiny",
        )

    raise ValueError(f"Unsupported detector: {detector_name}")


def _detect_with_ultralytics(detector_name: str, image: np.ndarray) -> Dict[str, object]:
    try:
        model = load_detector(detector_name)
        results = model(image, verbose=False)
    except Exception as exc:
        return {"error": f"{detector_name} loading or inference failed: {exc}"}

    detections: List[Dict[str, object]] = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            names = results[0].names
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append(
                    {
                        "label": str(names.get(cls_id, cls_id)).replace("_", " "),
                        "confidence": conf,
                    }
                )

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "detections": detections,
        "count": len(detections),
        "note": f"{detector_name} pre-trained knowledge on generic object categories.",
    }


def _detect_with_dino(image: np.ndarray) -> Dict[str, object]:
    candidate_labels = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "dog",
        "cat",
        "bird",
        "horse",
        "chair",
        "bottle",
        "laptop",
        "cell phone",
        "book",
        "backpack",
        "traffic light",
        "stop sign",
        "tree",
    ]
    try:
        dino = load_detector("DINO")
        predictions = dino(Image.fromarray(image), candidate_labels=candidate_labels)
    except Exception as exc:
        return {"error": f"DINO loading or inference failed: {exc}"}

    detections: List[Dict[str, object]] = []
    for pred in predictions:
        score = float(pred.get("score", 0.0))
        if score >= 0.25:
            detections.append(
                {
                    "label": str(pred.get("label", "unknown")).replace("_", " "),
                    "confidence": score,
                }
            )

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "detections": detections,
        "count": len(detections),
        "note": "DINO pre-trained knowledge on broad open-vocabulary object categories.",
    }


def detect_objects_pretrained(image: np.ndarray, detector_name: str) -> Dict[str, object]:
    if detector_name in {"YOLO", "RT-DETR"}:
        return _detect_with_ultralytics(detector_name, image)
    if detector_name == "DINO":
        return _detect_with_dino(image)
    return {"error": f"Unknown detector: {detector_name}"}


@st.cache_resource
def load_virus_model(model_path: str):
    try:
        return load_model(model_path)
    except Exception:
        return None


def render_virus_metrics(metrics: ImageMetrics) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Edge Density", f"{metrics.edge_density:.2%}")
    with c2:
        st.metric("Entropy", f"{metrics.entropy:.2f}")
    with c3:
        st.metric("Contrast", f"{metrics.contrast:.3f}")


def render_non_virus_object_step(image: np.ndarray, detector_name: str) -> None:
    st.subheader("Step 2 - Pre-trained Object Recognition")
    st.caption(
        "Image classified as non-virus. The selected pre-trained model is used to recognize known object types "
        "(person, vehicle, animal, daily objects, etc.)."
    )

    with st.spinner(f"Running {detector_name} inference..."):
        result = detect_objects_pretrained(image, detector_name)

    if "error" in result:
        st.error(result["error"])
        return

    st.info(result.get("note", ""))

    detections = result.get("detections", [])
    if not detections:
        st.warning(
            "No confident object detected. This can happen on empty scenes, normal backgrounds, or images "
            "without recognizable pre-trained categories."
        )
        return

    st.success(f"Detected objects: {result.get('count', 0)}")
    table = pd.DataFrame(
        [
            {
                "Object": d["label"].title(),
                "Confidence": f"{d['confidence']:.1%}",
            }
            for d in detections
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Benchmark DINO YOLO RT-DETR", layout="wide")

    st.markdown(
        """
        <style>
            .main { padding-top: 0.6rem; }
            h1 {
                background: linear-gradient(90deg, #0f4c81 0%, #1f7a8c 50%, #6c9a8b 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                letter-spacing: 0.3px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h1>Benchmarking Project: DINO vs YOLO vs RT-DETR</h1>", unsafe_allow_html=True)
    st.caption(
        "Pipeline: virus triage first. If image is non-virus, object recognition is run using the selected pre-trained detector."
    )

    init_model_and_data()

    selected_detector = st.selectbox(
        "Choose pre-trained detector for non-virus images",
        ["YOLO", "RT-DETR", "DINO"],
        index=0,
    )

    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    tab_analysis, tab_details = st.tabs(["Analysis", "Details"])

    with tab_analysis:
        if uploaded is None:
            st.info("Upload an image to start benchmarking.")
            return

        image = np.array(Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB"))
        model_path = "models/cybervision_model.joblib"
        payload = load_virus_model(model_path)
        metrics = analyze_image_metrics(image)
        risk_label, risk_color = classify_virus_risk(metrics.artifact_score)

        left, right = st.columns(2)
        with left:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)
        with right:
            st.subheader("Edge Overlay")
            st.image(build_edge_overlay(image), use_container_width=True)

        st.subheader("Step 1 - Virus / Non-Virus Triage")
        render_virus_metrics(metrics)

        predicted_label = "unknown"
        confidence = 0.0
        if payload is not None:
            predicted_label, confidence = predict_label(payload, image)
        else:
            st.warning("Virus model unavailable. Heuristic indicators only.")

        if predicted_label == "virus":
            st.markdown(
                f"""
                <div style='background:#fef2f2;border-left:6px solid #b91c1c;padding:14px;border-radius:6px;'>
                    <strong>Virus-like pattern detected</strong><br>
                    Confidence: {confidence:.1%}<br>
                    Heuristic Risk: <span style='color:{risk_color};font-weight:700;'>{risk_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "For virus-like images, the application prioritizes artifact-oriented analysis metrics rather than "
                "generic object labels."
            )
        else:
            st.markdown(
                f"""
                <div style='background:#ecfdf5;border-left:6px solid #166534;padding:14px;border-radius:6px;'>
                    <strong>Non-virus image detected</strong><br>
                    Confidence: {confidence:.1%}<br>
                    Heuristic Risk: <span style='color:{risk_color};font-weight:700;'>{risk_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_non_virus_object_step(image, selected_detector)

    with tab_details:
        if uploaded is None:
            st.info("Upload an image to display details.")
            return

        image = np.array(Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB"))
        metrics = analyze_image_metrics(image)

        details = pd.DataFrame(
            [
                {"Metric": "Edge Density", "Value": f"{metrics.edge_density:.6f}"},
                {"Metric": "Entropy", "Value": f"{metrics.entropy:.6f}"},
                {"Metric": "Contrast", "Value": f"{metrics.contrast:.6f}"},
                {"Metric": "Saturation Mean", "Value": f"{metrics.saturation_mean:.6f}"},
                {"Metric": "Brightness Mean", "Value": f"{metrics.brightness_mean:.6f}"},
                {"Metric": "Artifact Score", "Value": f"{metrics.artifact_score:.6f}"},
            ]
        )
        st.dataframe(details, use_container_width=True, hide_index=True)

        st.markdown("### Model Notes")
        st.write("- YOLO and RT-DETR are pre-trained for generic object detection categories.")
        st.write("- DINO uses zero-shot pre-trained knowledge with text-guided labels.")
        st.write(
            "- If no object is detected, the image can be empty, normal, or outside recognizable pre-trained categories."
        )


if __name__ == "__main__":
    main()
