# 🌿 EcoVision - Détection de déchets avec YOLOv8

## 📁 Structure du Projet

```
EcoVision/
├── config.py                  # Configuration centralisée
├── requirements.txt           # Dépendances Python
├── Dockerfile                 # Configuration Docker
│
├── models/                    # 🤖 Modèles YOLO
│   ├── pretrained/           
│   │   └── yolov8n.pt        # Modèle YOLOv8n pré-entraîné
│   └── trained/              
│       └── ecovision_waste_v1/  # Modèle entraîné sur les déchets
│           └── weights/
│               ├── best.pt      # Meilleur modèle
│               └── last.pt      # Dernier checkpoint
│
├── datasets/                  # 📊 Données d'entraînement
│   ├── data.yaml             # Configuration du dataset
│   ├── train/                # Images d'entraînement
│   │   ├── images/
│   │   └── labels/
│   ├── valid/                # Images de validation
│   │   ├── images/
│   │   └── labels/
│   └── test/                 # Images de test
│       ├── images/
│       └── labels/
│
├── notebooks/                 # 📓 Jupyter Notebooks
│   ├── step1_train.ipynb     # Entraînement du modèle
│   └── emissions.csv         # Tracking CO2
│
├── src/                       # 💻 Code source
│   ├── api.py                # API FastAPI
│   └── models/
│       ├── train_waste.py    # Script d'entraînement
│       └── detect_webcam.py  # Détection webcam
│
└── data/                      # 📦 Données brutes/traitées
    ├── raw/
    └── processed/
```

## 🚀 Utilisation

### 1. Installation

```bash
# Créer l'environnement conda
conda create -n ecovision python=3.12
conda activate ecovision

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Entraînement du modèle

Ouvrir et exécuter `notebooks/step1_train.ipynb`

Le modèle entraîné sera sauvegardé dans :
- `models/trained/ecovision_waste_v1/weights/best.pt`

### 3. Lancer l'API

```bash
uvicorn src.api:app --reload
```

L'API sera accessible sur : http://127.0.0.1:8000

### 4. Docker

```bash
docker build -t ecovision-api:v1 .
docker run -p 8000:8000 ecovision-api:v1
```

## 📊 Classes de déchets détectées

1. BIODEGRADABLE
2. CARDBOARD
3. GLASS
4. METAL
5. PAPER
6. PLASTIC

## 🌱 Tracking CO2

Le projet utilise CodeCarbon pour mesurer l'empreinte carbone de l'entraînement.
Les résultats sont sauvegardés dans `notebooks/emissions.csv`
