"""
Configuration centralisée pour le projet EcoVision
"""
import os

# Racine du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modèles
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PRETRAINED_DIR = os.path.join(MODELS_DIR, 'pretrained')
TRAINED_DIR = os.path.join(MODELS_DIR, 'trained')

YOLO_PRETRAINED = os.path.join(PRETRAINED_DIR, 'yolov8n.pt')
YOLO_TRAINED_BEST = os.path.join(TRAINED_DIR, 'ecovision_waste_v1', 'weights', 'best.pt')

# Dataset
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
DATA_YAML = os.path.join(DATASETS_DIR, 'data.yaml')

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Training
TRAINING_EPOCHS = 50
TRAINING_BATCH_SIZE = 16
TRAINING_IMG_SIZE = 640
TRAINING_PATIENCE = 10

# Classes de déchets
WASTE_CLASSES = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
