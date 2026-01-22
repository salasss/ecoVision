from ultralytics import YOLO
from codecarbon import EmissionsTracker
import mlflow
import mlflow.pytorch
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataset_path = os.path.join(BASE_DIR, "datasets", "data.yaml")
model_path = os.path.join(BASE_DIR, "models", "pretrained", "yolov8n.pt")

print(f"📂 Base directory: {BASE_DIR}")
print(f"📂 Dataset: {dataset_path}")
print(f"🤖 Model: {model_path}")

assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"
assert os.path.exists(model_path), f"Model not found: {model_path}"

mlruns_path = Path(BASE_DIR) / "mlruns"
mlflow.set_tracking_uri(mlruns_path.as_uri())
mlflow.set_experiment("EcoVision_Waste_Detection")

with mlflow.start_run(run_name="yolov8n_waste_training"):
    
    # Log  hyperparamètres
    params = {
        "model": "yolov8n",
        "epochs": 4,
        "imgsz": 416,
        "batch": 4,
        "patience": 2,
        "device": "cpu",
        "workers": 2,
        "augment": True,
        "scale": 0.5,
        "translate": 0.1,
        "fliplr": 0.5,
    }
    mlflow.log_params(params)
    
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
    
    # Log  metriques from YOLO results
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        mlflow.log_metrics({
            "final_mAP50": metrics.get("metrics/mAP50(B)", 0),
            "final_mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
            "final_precision": metrics.get("metrics/precision(B)", 0),
            "final_recall": metrics.get("metrics/recall(B)", 0),
        })
    
    # Log CO2 emissions
    mlflow.log_metric("co2_emissions_kg", emissions)
    
    # Log du modele trained
    model_output = os.path.join(BASE_DIR, 'models', 'trained', 'ecovision_waste_v1', 'weights', 'best.pt')
    if os.path.exists(model_output):
        mlflow.log_artifact(model_output, "model")
    
    print(f"\n🌿 Fine-Tuning finished!")
    print(f"💨 CO2 emissions : {emissions:.6f} kg")
    print(f"📊 MLflow run: {mlflow.active_run().info.run_id}")
    print(f"✅ Modele saved : {model_output}")
    print(f"\n🔍 The results: mlflow ui --backend-store-uri {os.path.join(BASE_DIR, 'mlruns')}")