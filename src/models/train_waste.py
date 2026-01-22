from ultralytics import YOLO
from codecarbon import EmissionsTracker
import os

# Chemins absolus
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataset_path = os.path.join(BASE_DIR, "datasets", "data.yaml")
model_path = os.path.join(BASE_DIR, "models", "pretrained", "yolov8n.pt")

print(f"📂 Base directory: {BASE_DIR}")
print(f"📂 Dataset: {dataset_path}")
print(f"🤖 Model: {model_path}")

# Vérifier que les fichiers existent
assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"
assert os.path.exists(model_path), f"Model not found: {model_path}"

tracker = EmissionsTracker(project_name="EcoVision_FineTuning")
tracker.start()

print(f"\n🚀 Fine-Tuning started...")

model = YOLO(model_path) 

results = model.train(
    data=dataset_path,
    epochs=4,              
    imgsz=416,             
    batch=4,               
    project=os.path.join(BASE_DIR, 'models', 'trained'),
    name='ecovision_waste_v1',
    patience=2,
    save=True,
    plots=False,           
    device='cpu',
    workers=2,             
    cache=False,           
    augment=True,          
    scale=0.5,
    translate=0.1,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    seed=0,
    val=True,
    verbose=True
)

emissions = tracker.stop()
print(f"\n🌿 Fine-Tuning finished!")
print(f"💨 CO2 emissions : {emissions:.6f} kg")

# Sauvegarder le modèle
output_dir = os.path.join(BASE_DIR, 'models', 'trained', 'ecovision_waste_v1', 'weights', 'best.pt')
print(f"✅ Modèle saved : {output_dir}")