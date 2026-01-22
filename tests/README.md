# EcoVision Tests

## Installation des dépendances de test

```bash
pip install -r requirements.txt
```

## Lancer tous les tests

```bash
pytest tests/ -v
```

## Lancer les tests avec couverture de code

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Lancer un fichier de test spécifique

```bash
pytest tests/test_api.py -v
pytest tests/test_model.py -v
```

## Tests inclus

### test_api.py
- ✅ Test route GET / (health check)
- ✅ Test POST /detect avec image valide
- ✅ Test POST /detect sans fichier (error handling)
- ✅ Test format de réponse
- ✅ Test du modèle (existence, chargement, classes)
- ✅ Test des chemins et configuration
- ✅ Test end-to-end avec images réelles

### test_model.py
- ✅ Test chargement du modèle YOLO
- ✅ Test méthode predict
- ✅ Test format de sortie des prédictions
- ✅ Test nombre de classes (6 classes waste)
- ✅ Test vitesse d'inférence
- ✅ Test du script d'entraînement
- ✅ Test des chemins de données

## Exemple de sortie

```
tests/test_api.py::TestAPI::test_home_route PASSED
tests/test_api.py::TestAPI::test_detect_with_valid_image PASSED
tests/test_api.py::TestAPI::test_detect_missing_file PASSED
tests/test_api.py::TestModel::test_model_path_exists PASSED
tests/test_model.py::TestYOLOModel::test_model_loads PASSED
tests/test_model.py::TestYOLOModel::test_model_predict_output_format PASSED
tests/test_model.py::TestTrainingScript::test_training_script_exists PASSED
tests/test_model.py::TestDataPaths::test_dataset_has_images PASSED

============= 20 passed in 5.23s =============
```
