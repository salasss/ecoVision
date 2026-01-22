from ultralytics import YOLO
from codecarbon import EmissionsTracker
import mlflow
import mlflow.pytorch
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "datasets" / "data.yaml"
MODEL_PATH = BASE_DIR / "models" / "pretrained" / "yolov8n.pt"


def run_training(params=None):
    """Launch YOLO fine-tuning with MLflow and CodeCarbon."""
    params = params or {
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

    # Verify required assets are present before training
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Pretrained model not found at {MODEL_PATH}")

    print(f"Base directory: {BASE_DIR}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {MODEL_PATH}")

    mlruns_path = BASE_DIR / "mlruns"
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment("EcoVision_Waste_Detection")

    with mlflow.start_run(run_name="yolov8n_waste_training"):
        mlflow.log_params(params)

        tracker = EmissionsTracker(project_name="EcoVision_FineTuning")
        tracker.start()

        print("\nStarting fine-tuning...")

        model = YOLO(str(MODEL_PATH))

        results = model.train(
            data=str(DATASET_PATH),
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            project=str(BASE_DIR / "models" / "trained"),
            name="ecovision_waste_v1",
            patience=params["patience"],
            save=True,
            plots=False,
            device=params["device"],
            workers=params["workers"],
            cache=False,
            augment=params["augment"],
            scale=params["scale"],
            translate=params["translate"],
            shear=0.0,
            flipud=0.0,
            fliplr=params["fliplr"],
            seed=0,
            val=True,
            verbose=True,
        )

        emissions = tracker.stop()

        if hasattr(results, "results_dict"):
            metrics = results.results_dict
            mlflow.log_metrics({
                "final_mAP50": metrics.get("metrics/mAP50(B)", 0),
                "final_mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
                "final_precision": metrics.get("metrics/precision(B)", 0),
                "final_recall": metrics.get("metrics/recall(B)", 0),
            })

        mlflow.log_metric("co2_emissions_kg", emissions)

        model_output = BASE_DIR / "models" / "trained" / "ecovision_waste_v1" / "weights" / "best.pt"
        if model_output.exists():
            mlflow.log_artifact(str(model_output), "model")

        print("\nFine-tuning finished.")
        print(f"CO2 emissions: {emissions:.6f} kg")
        print(f"MLflow run: {mlflow.active_run().info.run_id}")
        print(f"Model saved: {model_output}")
        print(f"See results with: mlflow ui --backend-store-uri {BASE_DIR / 'mlruns'}")


if __name__ == "__main__":
    run_training()