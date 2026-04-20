# Projet 1 - CyberVision Lite (Streamlit)

Application Streamlit de triage visuel pour analyser des images suspectes.
Le projet combine:
- une prediction ML (virus / non_virus),
- un mode heuristique explicable (edge, entropy, contrast),
- une visualisation des zones a inspecter.

## Objectif

Fournir une base deployable pour:
- entrainer un modele sur un dataset image,
- predire si une image ressemble plutot a la classe virus ou non_virus,
- conserver des indicateurs interpretable pour presentation pedagogique.

## Signification des classes

- `virus`: images avec artefacts/bruit/patterns visuels plus aggressifs.
- `non_virus`: images visuellement plus stables.

Important: c'est un prototype de triage. La sortie ne remplace pas un antivirus ni une analyse forensique complete.

## Signification des indicateurs affiches

- `Edge density`: proportion de contours detectes. Plus c'est haut, plus l'image est structuree ou bruitée.
- `Entropy`: diversite d'information pixel. Plus c'est haut, plus la texture est complexe.
- `Contrast`: dispersion d'intensite luminance.
- `Artifact score`: score heuristique combine pour aider l'analyse humaine.

## Pipeline ML

1. Extraction de features image:
	- edge_density
	- entropy
	- contrast
	- saturation_mean
	- brightness_mean
2. Train/test split stratifie
3. Entrainement RandomForest
4. Export du modele (`joblib`) + metriques (`json`)
5. Inference dans l'app Streamlit

## Dataset de travail alternatif

Ce projet utilise un dataset :
- `data/dataset_alt/virus`
- `data/dataset_alt/non_virus`

Generation demo:
- `python make_alt_dataset.py`

## Lancement local (train + app)

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python make_alt_dataset.py`
5. `python train_model.py --dataset data/dataset_alt --model-out models/cybervision_model.joblib --metrics-out models/train_metrics.json`
6. `streamlit run app.py`

## Fichiers importants

- `app.py`: interface Streamlit + inference
- `ml_pipeline.py`: extraction features + train + prediction
- `train_model.py`: script d'entrainement
- `make_alt_dataset.py`: generation de dataset alternatif
- `models/cybervision_model.joblib`: modele entraine
- `models/train_metrics.json`: metriques d'entrainement

## Interpretation des resultats

- `Prediction ML`: classe predite (`virus` ou `non_virus`) et confiance.
- `Verdict heuristique`: niveau de risque visuel (faible/modere/eleve).
- `Top zones inspectables`: patches avec forte densite de contours.

## Deploiement Streamlit Cloud

Voir le guide:
- `DEPLOY_STREAMLIT.md`

## Limites

- Dataset demo synthetique: utile pour workflow, pas pour validation industrielle.
- Le modele doit etre re-entraine avec donnees reelles pour usage production.

