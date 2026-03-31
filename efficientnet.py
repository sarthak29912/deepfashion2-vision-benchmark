import os
import gc
import json
import time
import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "/kaggle/input/deepfashion-redwing/deepfashion2_pruned"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "image")
TRAIN_ANNO_DIR = os.path.join(DATA_ROOT, "train", "annos")
VAL_IMG_DIR   = os.path.join(DATA_ROOT, "validation", "image")
VAL_ANNO_DIR  = os.path.join(DATA_ROOT, "validation", "annos")

catid_to_index = {
    1: 0,
    8: 1,
    7: 2,
    2: 3,
    9: 4
}
NUM_CLASSES = 5

class DeepFashion2MultiLabelDataset(Dataset):
    def __init__(self, img_dir, anno_dir, transform=None, category_map=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transform = transform
        self.category_map = category_map
        self.samples = []

        anno_files = sorted([f for f in os.listdir(anno_dir) if f.endswith(".json")])
        for anno_file in tqdm(anno_files, desc=f"Building dataset: {os.path.basename(img_dir)}"):
            anno_path = os.path.join(anno_dir, anno_file)
            with open(anno_path, "r") as f:
                data = json.load(f)

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = DeepFashion2MultiLabelDataset(TRAIN_IMG_DIR, TRAIN_ANNO_DIR, train_transform, catid_to_index)
val_dataset = DeepFashion2MultiLabelDataset(VAL_IMG_DIR, VAL_ANNO_DIR, val_transform, catid_to_index)

train_loader = DataLoader(train_dataset, batch_size=250, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=250, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

def compute_pos_weight(dataset):
    all_labels = torch.stack([label for _, label in dataset.samples], dim=0).cpu().numpy()
    pos_counts = all_labels.sum(axis=0)
    neg_counts = len(all_labels) - pos_counts
    return torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_dataset).to(device))

def evaluate_multilabel(model, loader, threshold=0.5):
    model.eval()
    all_targets, all_preds = [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).float()

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)

    return {
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, save_path):
    model.to(device)

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, (images, labels) in pbar:
            if i % 250 == 0:
                print(f"Batch {i}")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{running_loss/(i+1):.4f}")

        avg_train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_multilabel(model, val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] | Time: {time.time()-epoch_start:.2f}s | Train Loss: {avg_train_loss:.4f}")
        print(val_metrics)

    torch.save(model.state_dict(), save_path)
    print(f"Saved: {save_path}")

# -------- Transfer --------
model_transfer = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
in_features = model_transfer.classifier[3].in_features
model_transfer.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
optimizer_transfer = torch.optim.Adam(model_transfer.parameters(), lr=1e-4)
train_model(model_transfer, train_loader, val_loader, criterion, optimizer_transfer, 10, "/kaggle/working/mobilenetv3_transfer_10.pth")

# -------- Scratch --------
model_scratch = mobilenet_v3_large(weights=None)
in_features = model_scratch.classifier[3].in_features
model_scratch.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
optimizer_scratch = torch.optim.Adam(model_scratch.parameters(), lr=1e-3)
train_model(model_scratch, train_loader, val_loader, criterion, optimizer_scratch, 10, "/kaggle/working/mobilenetv3_scratch_10.pth")