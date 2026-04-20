import io
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from ml_pipeline import extract_metrics, load_model, predict_label


@dataclass
class FrameMetrics:
    edge_density: float
    entropy: float
    contrast: float
    artifact_score: float


def compute_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / (hist.sum() + 1e-12)
    return float(-np.sum(prob * np.log2(prob + 1e-12)))


def analyze_frame(image: np.ndarray) -> FrameMetrics:
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


    edge_density = float(np.mean(edges > 0))
    entropy = compute_entropy(gray)
    contrast = float(np.std(gray) / 255.0)

    artifact_score = 0.45 * edge_density + 0.35 * contrast + 0.20 * min(entropy / 8.0, 1.0)

    return FrameMetrics(
        edge_density=edge_density,
        entropy=entropy,
class ImageMetrics:
        artifact_score=artifact_score,
    )


def build_overlay(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def init_model_and_data():
    """Initialize model and dataset if they don't exist."""
    model_path = Path("models/cybervision_model.joblib")
    data_path = Path("data/dataset_alt")
    
    if not model_path.exists() or not data_path.exists():
        with st.spinner("Initializing model and dataset..."):
            try:
                if not data_path.exists():
                    from make_alt_dataset import create_dataset
                    create_dataset("data/dataset_alt", per_class=50)
                
                if not model_path.exists():
                    from train_model import main as train_main
                    import sys
                    old_argv = sys.argv
                    sys.argv = [
                        "train_model.py",
                        "--dataset", "data/dataset_alt",
                        "--model-out", str(model_path),
                        "--metrics-out", "models/train_metrics.json"
                    ]
                    train_main()
                    sys.argv = old_argv
            except Exception as e:
                st.error(f"Error during initialization: {e}")


    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    heat = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.75, heat[:, :, ::-1], 0.25, 0)


def analyze_image_metrics(image: np.ndarray) -> ImageMetrics:
    if score >= 0.58:
    edges = cv2.Canny(gray, threshold1=70, threshold2=160)
    
        return "🟠 MOYEN", "#FFA500"
    return "🟢 BAS", "#4CAF50"

    artifact_score = 0.45 * edge_density + 0.35 * contrast + 0.20 * min(entropy / 8.0, 1.0)
    
    return ImageMetrics(
        return load_model(model_path)
    except Exception:
        return None


def render_detection_panel(image: np.ndarray, model_path: str) -> dict:
    """Render detection results in a modern card-style layout."""
def build_edge_overlay(image: np.ndarray) -> np.ndarray:
    
    metrics = analyze_frame(image)
    risk_label, risk_color = classify_risk(metrics.artifact_score)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"""
def load_virus_model(model_path: str):
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; font-size: 14px; opacity: 0.8;'>Densité d'artefacts</h3>
            <h2 style='margin: 10px 0 0 0; font-size: 32px;'>{metrics.edge_density:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
@st.cache_resource
def load_yolo_model():
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception:
        return None


def classify_virus_risk(score: float) -> Tuple[str, str]:
    if score >= 0.58:
        return "HIGH", "#CC0000"
    if score >= 0.42:
        return "MEDIUM", "#FF6600"
    return "LOW", "#009900"


def detect_objects_yolo(image: np.ndarray) -> dict:
    """Detect objects in image using YOLO."""
    yolo = load_yolo_model()
    if yolo is None:
        return {"error": "YOLO model not available"}
    
    try:
        results = yolo(image, verbose=False)
        detections = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = results[0].names[cls_id]
                    detections.append({
                        "class": class_name,
                        "confidence": conf
                    })
        
        return {
            "detections": detections,
            "count": len(detections)
        }
    except Exception as e:
        return {"error": str(e)}


def render_virus_analysis(image: np.ndarray, model_path: str):
    """Render virus detection analysis."""
    st.subheader("Virus Detection Analysis")
    
    payload = load_virus_model(model_path)
    metrics = analyze_image_metrics(image)
    risk_level, risk_color = classify_virus_risk(metrics.artifact_score)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Edge Density", f"{metrics.edge_density:.2%}")
    with col2:
        st.metric("Entropy", f"{metrics.entropy:.2f}")
    with col3:
        st.metric("Contrast", f"{metrics.contrast:.3f}")
            <div style='background: {pred_color}20; border-left: 4px solid {pred_color}; 
                        padding: 15px; border-radius: 5px;'>
    col_pred, col_risk = st.columns(2)
    
            </div>
        with col_pred:
            """, unsafe_allow_html=True)
            result_text = "VIRUS DETECTED" if predicted_label == "virus" else "CLEAN IMAGE"
            result_color = "#CC0000" if predicted_label == "virus" else "#009900"
            
            st.markdown(f"""
            <div style='background-color: {result_color}20; border-left: 5px solid {result_color}; 
                        padding: 15px; border-radius: 5px; margin: 10px 0;'>
                <h4 style='margin: 0; color: {result_color};'>{result_text}</h4>
                <p style='margin: 5px 0 0 0; font-size: 13px;'>Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
    # Custom CSS
        with col_pred:
            st.warning("Model not loaded - heuristic analysis only")
    
    with col_risk:
        st.markdown(f"""
        <div style='background-color: {risk_color}20; border-left: 5px solid {risk_color}; 
                    padding: 15px; border-radius: 5px; margin: 10px 0;'>
            <h4 style='margin: 0; color: {risk_color};'>Risk Level: {risk_level}</h4>
            <p style='margin: 5px 0 0 0; font-size: 13px;'>Artifact Score: {metrics.artifact_score:.2f}</p>
        </div>
        """, unsafe_allow_html=True)


def render_object_detection(image: np.ndarray):
    """Render object detection analysis."""
    st.subheader("Object Detection Analysis")
    
    with st.spinner("Analyzing objects..."):
        result = detect_objects_yolo(image)
    
    if "error" in result:
        st.error(f"Detection error: {result['error']}")
        return
    
    detections = result.get("detections", [])
    count = result.get("count", 0)
    
    st.metric("Objects Detected", count)
    
    if detections:
        st.divider()
        st.subheader("Detected Objects")
        
        df_data = []
        for det in detections:
            df_data.append({
                "Object": det["class"].title(),
                "Confidence": f"{det['confidence']:.1%}"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No objects detected in this image")
    <style>
        .main { padding-top: 0; }
        h1 { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
    """, unsafe_allow_html=True)
    
        initial_sidebar_state="collapsed"
    st.markdown("<h1 style='text-align: center; padding: 20px 0;'>🔬 ImageAnalyzer Pro</h1>", 
    
    init_model_and_data()
    
                "Détection avancée d'anomalies visuelles par IA</p>", unsafe_allow_html=True)
    
        .title { text-align: center; padding: 20px 0; }
        h1 { color: #1a1a1a; font-weight: 600; }
        .subtitle { text-align: center; color: #666; margin-top: -15px; font-size: 14px; }
    
    with tab1:
    
    st.markdown("<div class='title'><h1>ImageAnalyzer Pro</h1></div>", 
            label_visibility="collapsed"
    st.markdown("<p class='subtitle'>Advanced Image Analysis Platform</p>", unsafe_allow_html=True)
    
    col_spacer1, col_mode, col_spacer2 = st.columns([1, 2, 1])
    with col_mode:
        analysis_mode = st.radio(
            "Select Analysis Mode",
            ["Virus Detection", "Object Recognition"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
            pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    
    tab1, tab2 = st.tabs(["Analysis", "Details"])
    
    with tab1:
        uploaded = st.file_uploader(
            "Upload Image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        
                           unsafe_allow_html=True)
                overlay = build_overlay(image)
                st.image(overlay, use_column_width=True, clamp=True)
            
            
            # Detection panel
                st.subheader("Original Image")
        else:
            
    
                st.subheader("Edge Detection")
                overlay = build_edge_overlay(image)
            image = np.array(pil_img)
            
            st.markdown("<h3>Statistiques détaillées</h3>", unsafe_allow_html=True)
            
            if analysis_mode == "Virus Detection":
                model_path = "models/cybervision_model.joblib"
                render_virus_analysis(image, model_path)
            else:
                render_object_detection(image)
            detail_cols = st.columns(2)
            st.info("Upload an image to begin analysis")
    
                **Métriques structurelles:**
                - Edge Density: """ + f"{metrics.edge_density:.4f}" + """
                - Entropy: """ + f"{metrics.entropy:.4f}" + """
                - Contrast: """ + f"{metrics.contrast:.4f}" + """
            
            st.subheader("Image Information")
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Resolution", f"{image.shape[1]} x {image.shape[0]}")
                st.metric("Format", "RGB")
            with col_info2:
                st.metric("File Size", f"{len(uploaded.getvalue()) / 1024:.1f} KB")
                st.metric("Channels", "3")
            
            st.divider()
            st.subheader("Detailed Metrics")
            
            metrics = analyze_image_metrics(image)
                **Métriques couleur:**
            
            metric_data = {
                "Metric": [
                    "Edge Density",
                    "Entropy",
                    "Contrast",
                    "Saturation",
                    "Brightness",
                    "Artifact Score"
                ],
                "Value": [
                    f"{metrics.edge_density:.4f}",
                    f"{metrics.entropy:.4f}",
                    f"{metrics.contrast:.4f}",
                    f"{extracted.saturation_mean:.4f}",
                    f"{extracted.brightness_mean:.4f}",
                    f"{metrics.artifact_score:.4f}"
                ]
            }
            
            df = pd.DataFrame(metric_data)
        
        with col_config2:
