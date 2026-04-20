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
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=70, threshold2=160)

    edge_density = float(np.mean(edges > 0))
    entropy = compute_entropy(gray)
    contrast = float(np.std(gray) / 255.0)

    artifact_score = 0.45 * edge_density + 0.35 * contrast + 0.20 * min(entropy / 8.0, 1.0)

    return FrameMetrics(
        edge_density=edge_density,
        entropy=entropy,
        contrast=contrast,
        artifact_score=artifact_score,
    )


def build_overlay(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    heat = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.75, heat[:, :, ::-1], 0.25, 0)


def classify_risk(score: float) -> tuple:
    if score >= 0.58:
        return "🔴 CRITIQUE", "#FF1744"
    if score >= 0.42:
        return "🟠 MOYEN", "#FFA500"
    return "🟢 BAS", "#4CAF50"


@st.cache_resource
def load_trained_payload(model_path: str):
    try:
        return load_model(model_path)
    except Exception:
        return None


def render_detection_panel(image: np.ndarray, model_path: str) -> dict:
    """Render detection results in a modern card-style layout."""
    payload = load_trained_payload(model_path)
    
    metrics = analyze_frame(image)
    risk_label, risk_color = classify_risk(metrics.artifact_score)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; font-size: 14px; opacity: 0.8;'>Densité d'artefacts</h3>
            <h2 style='margin: 10px 0 0 0; font-size: 32px;'>{metrics.edge_density:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; font-size: 14px; opacity: 0.8;'>Contraste</h3>
            <h2 style='margin: 10px 0 0 0; font-size: 32px;'>{metrics.contrast:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; font-size: 14px; opacity: 0.8;'>Entropie</h3>
            <h2 style='margin: 10px 0 0 0; font-size: 32px;'>{metrics.entropy:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ML Prediction
    if payload is not None:
        predicted_label, confidence = predict_label(payload, image)
        is_virus = predicted_label == "virus"
        pred_icon = "⚠️ VIRUS" if is_virus else "✅ SAIN"
        pred_color = "#FF1744" if is_virus else "#4CAF50"
        
        col_ml, col_risk = st.columns(2)
        with col_ml:
            st.markdown(f"""
            <div style='background: {pred_color}20; border-left: 4px solid {pred_color}; 
                        padding: 15px; border-radius: 5px;'>
                <h4 style='margin: 0; color: {pred_color};'>{pred_icon}</h4>
                <p style='margin: 5px 0 0 0; font-size: 12px; color: #666;'>Confiance: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_risk:
            st.markdown(f"""
            <div style='background: {risk_color}20; border-left: 4px solid {risk_color}; 
                        padding: 15px; border-radius: 5px;'>
                <h4 style='margin: 0; color: {risk_color};'>{risk_label}</h4>
                <p style='margin: 5px 0 0 0; font-size: 12px; color: #666;'>Score: {metrics.artifact_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚙️ Modèle non disponible - analyse heuristique seulement")
    
    return {"metrics": metrics, "risk_label": risk_label}


def main() -> None:
    st.set_page_config(
        page_title="ImageAnalyzer Pro",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        [data-testid="stSidebarNav"] { display: none; }
        .main { padding-top: 0; }
        h1 { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 style='text-align: center; padding: 20px 0;'>🔬 ImageAnalyzer Pro</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; margin-top: -10px;'>"
                "Détection avancée d'anomalies visuelles par IA</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Main layout with tabs
    tab1, tab2, tab3 = st.tabs(["📤 Analyse", "📊 Détails", "⚙️ Configuration"])
    
    with tab1:
        uploaded = st.file_uploader(
            "Déposez votre image ici",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        
        if uploaded:
            pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            image = np.array(pil_img)
            
            # Display images
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown("<h3 style='text-align: center;'>📸 Image source</h3>", 
                           unsafe_allow_html=True)
                st.image(image, use_column_width=True, clamp=True)
            
            with col_img2:
                st.markdown("<h3 style='text-align: center;'>🌡️ Heatmap</h3>", 
                           unsafe_allow_html=True)
                overlay = build_overlay(image)
                st.image(overlay, use_column_width=True, clamp=True)
            
            st.divider()
            
            # Detection panel
            model_path = "models/cybervision_model.joblib"
            render_detection_panel(image, model_path)
        else:
            st.info("📂 Chargez une image PNG ou JPG pour commencer l'analyse")
    
    with tab2:
        if uploaded:
            pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            image = np.array(pil_img)
            
            st.markdown("<h3>Statistiques détaillées</h3>", unsafe_allow_html=True)
            
            metrics = analyze_frame(image)
            extracted = extract_metrics(image)
            
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.markdown("""
                **Métriques structurelles:**
                - Edge Density: """ + f"{metrics.edge_density:.4f}" + """
                - Entropy: """ + f"{metrics.entropy:.4f}" + """
                - Contrast: """ + f"{metrics.contrast:.4f}" + """
                """)
            
            with detail_cols[1]:
                st.markdown("""
                **Métriques couleur:**
                - Saturation: """ + f"{extracted.saturation_mean:.4f}" + """
                - Luminosité: """ + f"{extracted.brightness_mean:.4f}" + """
                - Densité d'artefacts: """ + f"{metrics.artifact_score:.4f}" + """
                """)
            
            # Top zones
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 70, 160)
            h, w = edges.shape
            patch = max(16, min(h, w) // 12)
            
            rows = []
            for y in range(0, h - patch + 1, patch):
                for x in range(0, w - patch + 1, patch):
                    window = edges[y : y + patch, x : x + patch]
                    density = float(np.mean(window > 0))
                    rows.append({"x": x, "y": y, "densité": density})
            
            df = pd.DataFrame(rows).sort_values("densité", ascending=False).head(15)
            st.markdown("**🎯 Top 15 zones suspectes:**")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Chargez une image dans l'onglet Analyse pour voir les détails")
    
    with tab3:
        st.markdown("<h3>Paramètres d'analyse</h3>", unsafe_allow_html=True)
        
        col_config1, col_config2 = st.columns(2)
        with col_config1:
            model_path = st.text_input(
                "Chemin du modèle ML",
                value="models/cybervision_model.joblib"
            )
        
        with col_config2:
            threshold = st.slider(
                "Seuil de risque (Critique)",
                min_value=0.4,
                max_value=0.8,
                value=0.58,
                step=0.01
            )
        
        st.info(
            "💡 **Conseil**: Entraînez d'abord le modèle avec `python train_model.py` "
            "avant d'utiliser cette application."
        )


if __name__ == "__main__":
    main()
