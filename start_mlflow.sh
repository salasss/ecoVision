#!/bin/bash
# Lance l'interface MLflow pour visualiser les expériences

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLRUNS_DIR="$BASE_DIR/mlruns"

echo "🚀 Lancement de MLflow UI..."
echo "📂 Tracking URI: $MLRUNS_DIR"
echo ""
echo "🌐 Interface disponible sur: http://127.0.0.1:5000"
echo ""

mlflow ui --backend-store-uri "$MLRUNS_DIR" --host 127.0.0.1 --port 5000
