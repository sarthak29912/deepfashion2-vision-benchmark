!pip install -q ultralytics

import os
import gc
import yaml
import torch
from ultralytics import YOLO

gc.collect()
torch.cuda.empty_cache()

device = 0 if torch.cuda.is_available() else "cpu"
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = "/kaggle/input/yolo-redwing/yolo_dataset"
DATA_YAML_PATH = "/kaggle/working/data.yaml"

data_yaml = {
    "path": DATASET_ROOT,
    "train": "train/images",
    "val": "val/images",
    "nc": 5,
    "names": [
        "short sleeve top",
        "long sleeve top",
        "shorts",
        "trousers",
        "skirt"
    ]
}

with open(DATA_YAML_PATH, "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

EPOCHS = 10
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 2
WORK_DIR = "/kaggle/working"

# -------- Transfer --------
gc.collect()
torch.cuda.empty_cache()

model_transfer = YOLO("yolov8n.pt")
model_transfer.train(
    data=DATA_YAML_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=device,
    workers=WORKERS,
    project=WORK_DIR,
    name="yolo_transfer_10",
    pretrained=True,
    verbose=True
)

# -------- Scratch --------
gc.collect()
torch.cuda.empty_cache()

model_scratch = YOLO("yolov8n.yaml")
model_scratch.train(
    data=DATA_YAML_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=device,
    workers=WORKERS,
    project=WORK_DIR,
    name="yolo_scratch_10",
    pretrained=False,
    verbose=True
)

# -------- Optional validation --------
best_transfer = YOLO("/kaggle/working/yolo_transfer_10/weights/best.pt")
print(best_transfer.val(data=DATA_YAML_PATH, imgsz=IMG_SIZE, batch=BATCH_SIZE, device=device))

best_scratch = YOLO("/kaggle/working/yolo_scratch_10/weights/best.pt")
print(best_scratch.val(data=DATA_YAML_PATH, imgsz=IMG_SIZE, batch=BATCH_SIZE, device=device))