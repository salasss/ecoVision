from ultralytics import YOLO
from codecarbon import EmissionsTracker
import os

tracker = EmissionsTracker(project_name="EcoVision_FineTuning")
tracker.start()

dataset_path = os.path.abspath("datasets/data.yaml")

print(f"Fine-Tuning of : {dataset_path}")

model = YOLO('weights/ecovision_waste_v1/weights/best.pt') 

results = model.train(
    data=dataset_path,  
    epochs=5,           
    imgsz=320,          
    batch=8,            
    project='runs/detect', 
    name='ecovision_turbo',
    device='cpu'       
)

emissions = tracker.stop()
print(f"🌿 Fine-Tuning finished !  CO2 emissions : {emissions} kg")

success = model.export(format='pt')
print(f"Modele saved for prod : ../weights/ecovision_waste_v1/weights/best.pt")