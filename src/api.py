from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="EcoVision API 🌿", description="API de détection de déchets")
model = YOLO('weights/ecovision_waste_v1/weights/best.pt')

@app.get("/")
def home():
    return {"status": "online", "message": "EcoVision API is running! 🚀"}

@app.post("/detect")
async def detect_waste(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Prediction
    results = model(image)

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