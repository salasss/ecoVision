from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI(title="EcoVision API 🌿", description="API de détection de déchets")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemin fixe du modèle (simplifié)
MODEL_PATH = os.path.join(
    BASE_DIR, 'models', 'trained', 'ecovision_waste_v13', 'weights', 'best.pt'
)

if not os.path.exists(MODEL_PATH):
    FALLBACK_MODEL = os.path.join(BASE_DIR, 'models', 'trained', 'ecovision_waste_best.pt')
    if os.path.exists(FALLBACK_MODEL):
        MODEL_PATH = FALLBACK_MODEL
    else:
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH} or fallback {FALLBACK_MODEL}"
        )

print(f"Loading modèle: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

@app.get("/")
def home():
    return {"status": "online", "message": "EcoVision API is running! 🚀"}


@app.post("/detect")
async def detect_waste(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Prediction
    results = model.predict(image, conf=0.35, iou=0.5, imgsz=448, device='cpu')

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            detections.append({
                "object": class_name,
                "confidence": round(confidence, 2),
                "box": box.xyxy[0].tolist() 
            })

    return {
        "filename": file.filename,
        "detections_count": len(detections),
        "detections": detections
    }