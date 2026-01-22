#!/usr/bin/env python3
"""
Script de vérification de la structure du projet EcoVision
"""
import os
import sys

def check_file(path, description):
    """Vérifie l'existence d'un fichier"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_dir(path, description):
    """Vérifie l'existence d'un dossier"""
    exists = os.path.isdir(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def main():
    print("🔍 Vérification de la structure EcoVision\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_ok = True
    
    # Modèles
    print("📦 MODÈLES")
    all_ok &= check_dir(os.path.join(base_dir, 'models'), "Dossier models")
    all_ok &= check_dir(os.path.join(base_dir, 'models', 'pretrained'), "Dossier pretrained")
    all_ok &= check_dir(os.path.join(base_dir, 'models', 'trained'), "Dossier trained")
    all_ok &= check_file(os.path.join(base_dir, 'models', 'pretrained', 'yolov8n.pt'), "Modèle pré-entraîné")
    
    print("\n📊 DATASET")
    all_ok &= check_dir(os.path.join(base_dir, 'datasets'), "Dossier datasets")
    all_ok &= check_file(os.path.join(base_dir, 'datasets', 'data.yaml'), "Config dataset")
    all_ok &= check_dir(os.path.join(base_dir, 'datasets', 'train'), "Images train")
    all_ok &= check_dir(os.path.join(base_dir, 'datasets', 'valid'), "Images valid")
    all_ok &= check_dir(os.path.join(base_dir, 'datasets', 'test'), "Images test")
    
    print("\n💻 CODE SOURCE")
    all_ok &= check_file(os.path.join(base_dir, 'src', 'api.py'), "API FastAPI")
    all_ok &= check_file(os.path.join(base_dir, 'config.py'), "Configuration")
    all_ok &= check_file(os.path.join(base_dir, 'requirements.txt'), "Requirements")
    
    print("\n📓 NOTEBOOKS")
    all_ok &= check_file(os.path.join(base_dir, 'notebooks', 'step1_train.ipynb'), "Notebook d'entraînement")
    
    print("\n" + "="*50)
    if all_ok:
        print("✅ Tout est en ordre !")
        return 0
    else:
        print("❌ Certains fichiers/dossiers sont manquants")
        return 1

if __name__ == "__main__":
    sys.exit(main())
