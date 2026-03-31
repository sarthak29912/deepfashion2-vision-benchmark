!pip install -q segmentation-models-pytorch

import os
import gc
import json
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp

ImageFile.LOAD_TRUNCATED_IMAGES = True

gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "/kaggle/input/deepfashion-redwing/deepfashion2_pruned"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "image")
TRAIN_ANNO_DIR = os.path.join(DATA_ROOT, "train", "annos")
VAL_IMG_DIR   = os.path.join(DATA_ROOT, "validation", "image")
VAL_ANNO_DIR  = os.path.join(DATA_ROOT, "validation", "annos")

CATEGORY_TO_MASK_CLASS = {
    1: 1,   # short sleeve top
    8: 2,   # long sleeve top
    7: 3,   # shorts
    2: 4,   # trousers
    9: 5    # skirt
}

NUM_CLASSES = 6  # background + 5 classes
EPOCHS = 10
BATCH_SIZE = 8
IMG_SIZE = 256

class DeepFashion2UNetDataset(Dataset):
    def __init__(self, img_dir, anno_dir, img_size=256):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.img_size = img_size
        self.samples = []

        anno_files = sorted([f for f in os.listdir(anno_dir) if f.endswith(".json")])
        for anno_file in tqdm(anno_files, desc=f"Building dataset: {os.path.basename(img_dir)}"):
            anno_path = os.path.join(anno_dir, anno_file)
            img_name = anno_file.replace(".json", ".jpg")
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            try:
                with open(anno_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            has_valid_item = False
            for _, value in data.items():
                if isinstance(value, dict) and "category_id" in value:
                    cid = value["category_id"]
                    segs = value.get("segmentation", None)
                    if cid in CATEGORY_TO_MASK_CLASS and segs is not None and len(segs) > 0:
                        has_valid_item = True
                        break

            if has_valid_item:
                self.samples.append((img_path, anno_path))

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def polygons_to_mask(self, polygons, width, height, class_id):
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)

        for poly in polygons:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            try:
                draw.polygon(xy, outline=class_id, fill=class_id)
            except Exception:
                continue

        return np.array(mask_img, dtype=np.uint8)

    def __getitem__(self, idx):
        img_path, anno_path = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.fromarray(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))

        orig_w, orig_h = image.size

        try:
            with open(anno_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

        semantic_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for _, value in data.items():
            if not (isinstance(value, dict) and "category_id" in value):
                continue

            cid = value["category_id"]
            if cid not in CATEGORY_TO_MASK_CLASS:
                continue

            segs = value.get("segmentation", None)
            if segs is None or len(segs) == 0:
                continue

            class_id = CATEGORY_TO_MASK_CLASS[cid]
            item_mask = self.polygons_to_mask(segs, orig_w, orig_h, class_id)
            semantic_mask[item_mask > 0] = class_id

        image = self.img_transform(image)

        mask_pil = Image.fromarray(semantic_mask)
        mask_pil = mask_pil.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        semantic_mask = np.array(mask_pil, dtype=np.uint8)
        mask_tensor = torch.tensor(semantic_mask, dtype=torch.long)

        return image, mask_tensor

train_dataset = DeepFashion2UNetDataset(TRAIN_IMG_DIR, TRAIN_ANNO_DIR, img_size=IMG_SIZE)
val_dataset = DeepFashion2UNetDataset(VAL_IMG_DIR, VAL_ANNO_DIR, img_size=IMG_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

criterion = nn.CrossEntropyLoss()

def compute_segmentation_metrics(preds, targets, num_classes=6):
    ious, dices = [], []

    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(1, num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        pred_sum = pred_cls.sum().item()
        target_sum = target_cls.sum().item()

        iou = 1.0 if union == 0 else intersection / (union + 1e-6)
        dice = 1.0 if (pred_sum + target_sum) == 0 else (2 * intersection) / (pred_sum + target_sum + 1e-6)

        ious.append(iou)
        dices.append(dice)

    return float(np.mean(ious)), float(np.mean(dices))

def get_unet_transfer():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    )

def get_unet_scratch():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES
    )

def train_one_epoch(model, loader, optimizer, epoch, run_name):
    model.train()
    running_loss = 0.0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"{run_name} Epoch {epoch+1}/{EPOCHS}")
    for i, (images, masks) in pbar:
        if i % 250 == 0:
            print(f"Batch {i}")

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{running_loss/(i+1):.4f}")

    return running_loss / len(loader)

def validate_one_epoch(model, loader):
    model.eval()
    running_loss = 0.0
    all_ious, all_dices = [], []

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Validation")
        for _, (images, masks) in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            miou, dice = compute_segmentation_metrics(preds.cpu(), masks.cpu(), num_classes=NUM_CLASSES)
            all_ious.append(miou)
            all_dices.append(dice)

    return running_loss / len(loader), float(np.mean(all_ious)), float(np.mean(all_dices))

def run_training(model, optimizer, save_path, run_name):
    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        gc.collect()
        torch.cuda.empty_cache()

        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, run_name)
        val_loss, miou, dice = validate_one_epoch(model, val_loader)

        print(f"{run_name} Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {miou:.4f} | Dice: {dice:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved: {save_path}")

        torch.save(model.state_dict(), save_path.replace(".pth", f"_epoch_{epoch+1}.pth"))

# -------- Transfer --------
model_transfer = get_unet_transfer()
optimizer_transfer = torch.optim.Adam(model_transfer.parameters(), lr=1e-4)
run_training(model_transfer, optimizer_transfer, "/kaggle/working/unet_transfer_10.pth", "UNET_TRANSFER")

# -------- Scratch --------
model_scratch = get_unet_scratch()
optimizer_scratch = torch.optim.Adam(model_scratch.parameters(), lr=1e-3)
run_training(model_scratch, optimizer_scratch, "/kaggle/working/unet_scratch_10.pth", "UNET_SCRATCH")