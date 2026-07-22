from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = BASE_DIR /"models" / "yolo" / "best.pt"

print("Model path:", YOLO_MODEL_PATH)
print("Exists:", YOLO_MODEL_PATH.exists())