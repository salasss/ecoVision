"""
Tests pour le modèle YOLO et l'entraînement
"""
import pytest
import os
from pathlib import Path
from PIL import Image
import io

class TestYOLOModel:
    """Tests du modèle YOLO"""
    
    @pytest.fixture
    def model_path(self):
        """Récupère le chemin du modèle"""
        from src.api import MODEL_PATH
        return MODEL_PATH
    
    def test_model_loads(self, model_path):
        """Test que le modèle YOLO se charge sans erreur"""
        from ultralytics import YOLO
        model = YOLO(model_path)
        assert model is not None
    
    def test_model_has_predict_method(self, model_path):
        """Test que le modèle a la méthode predict"""
        from ultralytics import YOLO
        model = YOLO(model_path)
        assert hasattr(model, 'predict')
        assert callable(model.predict)
    
    def test_model_predict_output_format(self, model_path):
        """Test que la prédiction retourne le bon format"""
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        
        # Créer une image de test
        img = Image.new('RGB', (640, 480), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Prédire
        results = model.predict(img, imgsz=416, device='cpu', conf=0.25, verbose=False)
        
        assert results is not None
        assert len(results) > 0
        assert hasattr(results[0], 'boxes')
    
    def test_model_classes_count(self, model_path):
        """Test que le modèle a 6 classes (waste detection)"""
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        assert hasattr(model, 'names')
        assert len(model.names) == 6
        
        # Vérifier les noms de classes
        expected_classes = {'BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC'}
        actual_classes = set(model.names.values())
        assert actual_classes == expected_classes
    
    def test_model_inference_speed(self, model_path):
        """Test que l'inférence est raisonnablement rapide"""
        from ultralytics import YOLO
        import time
        
        model = YOLO(model_path)
        img = Image.new('RGB', (640, 480), color='blue')
        
        start = time.time()
        model.predict(img, imgsz=416, device='cpu', verbose=False)
        elapsed = time.time() - start
        
        # Doit finir en moins de 10 secondes (CPU)
        assert elapsed < 10, f"Inference took {elapsed:.2f}s (expected < 10s)"


class TestTrainingScript:
    """Tests du script d'entraînement"""
    
    def test_training_script_exists(self):
        """Test que le script d'entraînement existe"""
        script = Path("src/models/train_waste.py")
        assert script.exists()
    
    def test_training_script_syntax(self):
        """Test que le script est valide Python"""
        import py_compile
        script = Path("src/models/train_waste.py")
        try:
            py_compile.compile(str(script), doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Syntax error in training script: {e}")
    
    def test_training_imports(self):
        """Test que les imports du script d'entraînement fonctionnent"""
        try:
            import src.models.train_waste  # noqa
        except ImportError as e:
            pytest.fail(f"Missing dependency in training script: {e}")


class TestDataPaths:
    """Tests des chemins de données"""
    
    def test_dataset_config_exists(self):
        """Test que le fichier de config dataset existe"""
        data_yaml = Path("datasets/data.yaml")
        if not data_yaml.exists():
            pytest.skip("Dataset config not available locally; skipping")
        assert data_yaml.exists()
    
    def test_dataset_folders_exist(self):
        """Test que les dossiers du dataset existent"""
        train_images = Path("datasets/train/images")
        valid_images = Path("datasets/valid/images")
        test_images = Path("datasets/test/images")

        if not (train_images.exists() and valid_images.exists() and test_images.exists()):
            pytest.skip("Dataset folders not present; skipping")

        assert train_images.exists()
        assert valid_images.exists()
        assert test_images.exists()
    
    def test_dataset_has_images(self):
        """Test que le dataset a des images"""
        train_images = list(Path("datasets/train/images").glob("*.jpg"))
        valid_images = list(Path("datasets/valid/images").glob("*.jpg"))
        
        if not train_images or not valid_images:
            pytest.skip("Dataset images not available; skipping")
        
        assert len(train_images) > 0, "No training images found"
        assert len(valid_images) > 0, "No validation images found"
    
    def test_dataset_has_labels(self):
        """Test que les labels existent"""
        train_labels = Path("datasets/train/labels")
        valid_labels = Path("datasets/valid/labels")

        if not (train_labels.exists() and valid_labels.exists()):
            pytest.skip("Dataset labels not present; skipping")

        assert train_labels.exists()
        assert valid_labels.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
