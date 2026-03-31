import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0, mobilenet_v3_large
from sklearn.metrics import precision_score, recall_score, f1_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# DEVICE + PATHS
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "/kaggle/input/deepfashion-redwing/deepfashion2_pruned"
VAL_IMG_DIR  = os.path.join(DATA_ROOT, "validation", "image")
VAL_ANNO_DIR = os.path.join(DATA_ROOT, "validation", "annos")

RESNET_PATH = "/kaggle/working/resnet50_transfer_10.pth"
EFFICIENTNET_PATH = "/kaggle/working/efficientnet_b0_transfer_10.pth"
MOBILENET_PATH = "/kaggle/working/mobilenetv3_transfer_10.pth"

# =========================
# CLASS MAPPING
# =========================
catid_to_index = {
    1: 0,   # short sleeve top
    8: 1,   # long sleeve top
    7: 2,   # shorts
    2: 3,   # trousers
    9: 4    # skirt
}

CLASS_NAMES = [
    "short sleeve top",
    "long sleeve top",
    "shorts",
    "trousers",
    "skirt"
]

NUM_CLASSES = 5

# =========================
# DATASET
# =========================
class DeepFashion2MultiLabelDataset(Dataset):
    def __init__(self, img_dir, anno_dir, transform=None, category_map=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transform = transform
        self.category_map = category_map
        self.samples = []

        anno_files = sorted([f for f in os.listdir(anno_dir) if f.endswith(".json")])

        for anno_file in anno_files:
            anno_path = os.path.join(anno_dir, anno_file)

            try:
                with open(anno_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            label = np.zeros(NUM_CLASSES, dtype=np.float32)

            for _, value in data.items():
                if isinstance(value, dict) and "category_id" in value:
                    cid = value["category_id"]
                    if cid in self.category_map:
                        label[self.category_map[cid]] = 1.0

            if label.sum() > 0:
                img_name = anno_file.replace(".json", ".jpg")
                img_path = os.path.join(img_dir, img_name)

                if os.path.exists(img_path):
                    self.samples.append((img_path, torch.tensor(label, dtype=torch.float32)))

        print(f"Loaded {len(self.samples)} usable samples from {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_dataset = DeepFashion2MultiLabelDataset(
    img_dir=VAL_IMG_DIR,
    anno_dir=VAL_ANNO_DIR,
    transform=val_transform,
    category_map=catid_to_index
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# =========================
# MODEL BUILDERS
# =========================
def build_resnet50():
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model

def build_efficientnet_b0():
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model

def build_mobilenetv3():
    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return model

# =========================
# EVALUATION
# =========================
def evaluate_model(model, loader, model_name, threshold=0.5):
    model.to(device)
    model.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if i % 250 == 0:
                print(f"{model_name} - Batch {i}")

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).float()

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    overall_result = {
        "Model": model_name,
        "Precision Macro": precision_macro,
        "Recall Macro": recall_macro,
        "F1 Macro": f1_macro,
        "Precision Micro": precision_micro,
        "Recall Micro": recall_micro,
        "F1 Micro": f1_micro
    }

    per_class_result = []
    for idx, class_name in enumerate(CLASS_NAMES):
        per_class_result.append({
            "Model": model_name,
            "Class": class_name,
            "Precision": precision_per_class[idx],
            "Recall": recall_per_class[idx],
            "F1": f1_per_class[idx]
        })

    return overall_result, per_class_result

# =========================
# LOAD MODELS
# =========================
resnet_model = build_resnet50()
resnet_model.load_state_dict(torch.load(RESNET_PATH, map_location=device))

efficientnet_model = build_efficientnet_b0()
efficientnet_model.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))

mobilenet_model = build_mobilenetv3()
mobilenet_model.load_state_dict(torch.load(MOBILENET_PATH, map_location=device))

# =========================
# RUN EVALUATION
# =========================
resnet_overall, resnet_per_class = evaluate_model(resnet_model, val_loader, "ResNet50 Transfer")
eff_overall, eff_per_class = evaluate_model(efficientnet_model, val_loader, "EfficientNet-B0 Transfer")
mob_overall, mob_per_class = evaluate_model(mobilenet_model, val_loader, "MobileNetV3 Transfer")

overall_df = pd.DataFrame([resnet_overall, eff_overall, mob_overall]).round(4)
per_class_df = pd.DataFrame(resnet_per_class + eff_per_class + mob_per_class).round(4)

print("\nFINAL OVERALL COMPARISON")
print(overall_df)

print("\nFINAL PER-CLASS COMPARISON")
print(per_class_df)

overall_df.to_csv("/kaggle/working/classification_overall_metrics.csv", index=False)
per_class_df.to_csv("/kaggle/working/classification_per_class_metrics.csv", index=False)

print("\nSaved:")
print("/kaggle/working/classification_overall_metrics.csv")
print("/kaggle/working/classification_per_class_metrics.csv")