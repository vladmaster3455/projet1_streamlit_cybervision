# Deploiement Streamlit Cloud - Projet 1

## 0) Entrainer le modele en local
- python3 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
- python make_alt_dataset.py
- python train_model.py --dataset data/dataset_alt --model-out models/cybervision_model.joblib --metrics-out models/train_metrics.json

Verifier que ces fichiers existent avant push:
- models/cybervision_model.joblib
- models/train_metrics.json

## 1) Preparer GitHub
- Creer un nouveau repository
- Pousser le contenu de projet1_streamlit_cybervision

## 2) Parametres Streamlit Cloud
- Repository: ton repo
- Branch: main
- Main file path: app.py
- Python version: 3.12

## 3) Fichiers requis
- requirements.txt
- packages.txt
- .streamlit/config.toml
- models/cybervision_model.joblib

## 4) Verification rapide
- Upload d'image fonctionne
- Prediction ML affiche Virus detecte ou Image non virus
- Tableau top zones s'affiche
- Aucun crash OpenCV
