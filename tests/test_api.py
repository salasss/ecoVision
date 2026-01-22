"""
Tests pour EcoVision API
"""
import pytest
import os
from pathlib import Path
from PIL import Image
import io
from fastapi.testclient import TestClient

# Import de l'API
from src.api import app, MODEL_PATH

client = TestClient(app)

class TestAPI:
    """Tests des routes FastAPI"""
    
    def test_home_route(self):
        """Test GET /"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "online"
    
    def test_detect_with_valid_image(self):
        """Test POST /detect avec une image valide"""
        # Créer une image de test simple (100x100 RGB)
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert "detections_count" in data
        assert "detections" in data
        assert isinstance(data["detections"], list)
    
    def test_detect_missing_file(self):
        """Test POST /detect sans fichier"""
        response = client.post("/detect")
        assert response.status_code == 422  # Validation error
    
    def test_detect_response_format(self):
        """Test que la réponse a le bon format"""
        img = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        
        data = response.json()
        assert isinstance(data["detections_count"], int)
        assert data["detections_count"] >= 0
        
        # Vérifier structure des détections
        for detection in data["detections"]:
            assert "object" in detection
            assert "confidence" in detection
            assert "box" in detection
            assert isinstance(detection["box"], list)
            assert len(detection["box"]) == 4


class TestModel:
    """Tests du modèle YOLO"""
    
    def test_model_path_exists(self):
        """Test que le modèle existe"""
        assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
    
    def test_model_can_load(self):
        """Test que le modèle peut être chargé"""
        from ultralytics import YOLO
        try:
            model = YOLO(MODEL_PATH)
            assert model is not None
            assert hasattr(model, 'predict')
        except Exception as e:
            pytest.fail(f"Could not load model: {e}")
    
    def test_model_has_classes(self):
        """Test que le modèle a les bonnes classes"""
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        assert hasattr(model, 'names')
        assert isinstance(model.names, dict)
        # Doit avoir 6 classes pour waste detection
        assert len(model.names) == 6


class TestConfig:
    """Tests de configuration et chemins"""
    
    def test_base_dir_exists(self):
        """Test que le dossier base existe"""
        assert os.path.exists(os.path.dirname(os.path.dirname(MODEL_PATH)))
    
    def test_datasets_exist(self):
        """Test que les datasets existent"""
        base_dir = Path(MODEL_PATH).parent.parent.parent.parent
        data_yaml = base_dir / "datasets" / "data.yaml"
        train_images = base_dir / "datasets" / "train" / "images"
        valid_images = base_dir / "datasets" / "valid" / "images"

        if not (data_yaml.exists() and train_images.exists() and valid_images.exists()):
            pytest.skip("Dataset not available locally; skipping dataset path checks")

        assert data_yaml.exists()
        assert train_images.exists()
        assert valid_images.exists()
    
    def test_pretrained_model_exists(self):
        """Test que le modèle pré-entraîné existe"""
        base_dir = Path(MODEL_PATH).parent.parent.parent.parent
        pretrained = base_dir / "models" / "pretrained" / "yolov8n.pt"
        if not pretrained.exists():
            pytest.skip(f"Pretrained weight missing at {pretrained}; skipping")
        assert pretrained.exists(), f"Pretrained model not found at {pretrained}"


class TestEndToEnd:
    """Tests end-to-end"""
    
    def test_api_detects_objects_in_real_image(self):
        """Test détection sur une image réelle du dataset"""
        from pathlib import Path
        
        # Chercher une image de test
        base_dir = Path(MODEL_PATH).parent.parent.parent.parent
        test_images = list((base_dir / "datasets" / "test" / "images").glob("*.jpg"))
        
        if not test_images:
            pytest.skip("No test images found")
        
        test_image = test_images[0]
        
        with open(test_image, 'rb') as f:
            response = client.post(
                "/detect",
                files={"file": (test_image.name, f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == test_image.name
    
    def test_full_pipeline(self):
        """Test complet: API -> détection -> réponse"""
        img = Image.new('RGB', (640, 480), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = client.post(
            "/detect",
            files={"file": ("pipeline_test.jpg", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "pipeline_test.jpg"
        assert "detections" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
