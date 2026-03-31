!pip install -q ultralytics segmentation-models-pytorch pycocotools

import os
import gc
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as F
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from ultralytics import YOLO
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# DEVICE + PATHS
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "/kaggle/input/deepfashion-redwing/deepfashion2_pruned"
VAL_IMG_DIR = os.path.join(DATA_ROOT, "validation", "image")
VAL_ANNO_DIR = os.path.join(DATA_ROOT, "validation", "annos")

YOLO_MODEL_PATH = "/kaggle/input/evaluation_2/best.pt"
MASKRCNN_MODEL_PATH = "/kaggle/input/evaluation_2/maskrcnn_scratch_fast_5.pth"
UNET_MODEL_PATH = "/kaggle/input/evaluation_2/unet_transfer_10.pth"

YOLO_EVAL_ROOT = "/kaggle/working/yolo_eval_dataset"
YOLO_VAL_IMAGES = os.path.join(YOLO_EVAL_ROOT, "val", "images")
YOLO_VAL_LABELS = os.path.join(YOLO_EVAL_ROOT, "val", "labels")
YOLO_DATA_YAML = os.path.join(YOLO_EVAL_ROOT, "data.yaml")

CATEGORY_TO_YOLO = {
    1: 0,
    8: 1,
    7: 2,
    2: 3,
    9: 4
}

CATEGORY_TO_INSTANCE_CLASS = {
    1: 1,
    8: 2,
    7: 3,
    2: 4,
    9: 5
}

NUM_CLASSES_SEG = 6

# =========================
# BUILD YOLO VAL DATA FROM DEEPFASHION
# =========================
def prepare_yolo_val_dataset_from_deepfashion(img_dir, anno_dir, out_root):
    if os.path.exists(out_root):
        shutil.rmtree(out_root)

    os.makedirs(YOLO_VAL_IMAGES, exist_ok=True)
    os.makedirs(YOLO_VAL_LABELS, exist_ok=True)

    anno_files = sorted([f for f in os.listdir(anno_dir) if f.endswith(".json")])

    for anno_file in tqdm(anno_files, desc="Preparing YOLO val dataset"):
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

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue

        yolo_lines = []

        for _, value in data.items():
            if not (isinstance(value, dict) and "category_id" in value):
                continue

            cid = value["category_id"]
            if cid not in CATEGORY_TO_YOLO:
                continue

            bbox = value.get("bounding_box", None)
            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(1, min(x2, w))
            y2 = max(1, min(y2, h))

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            xc = (x1 + x2) / 2.0 / w
            yc = (y1 + y2) / 2.0 / h
            bw_n = bw / w
            bh_n = bh / h

            class_id = CATEGORY_TO_YOLO[cid]
            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}")

        if len(yolo_lines) == 0:
            continue

        shutil.copy2(img_path, os.path.join(YOLO_VAL_IMAGES, img_name))
        label_name = anno_file.replace(".json", ".txt")
        with open(os.path.join(YOLO_VAL_LABELS, label_name), "w") as f:
            f.write("\n".join(yolo_lines))

    yaml_text = f"""path: {out_root}
train: val/images
val: val/images
nc: 5
names:
  0: short sleeve top
  1: long sleeve top
  2: shorts
  3: trousers
  4: skirt
"""
    with open(YOLO_DATA_YAML, "w") as f:
        f.write(yaml_text)

prepare_yolo_val_dataset_from_deepfashion(VAL_IMG_DIR, VAL_ANNO_DIR, YOLO_EVAL_ROOT)

# =========================
# YOLO EVALUATION
# =========================
def evaluate_yolo(model_path, data_yaml):
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=True
    )
    return {
        "Model": "YOLO",
        "Precision": float(metrics.box.mp),
        "Recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map)
    }

# =========================
# MASK R-CNN DATASET
# =========================
class DeepFashion2MaskRCNNDataset(Dataset):
    def __init__(self, img_dir, anno_dir):
        self.samples = []
        anno_files = sorted([f for f in os.listdir(anno_dir) if f.endswith(".json")])

        for anno_file in tqdm(anno_files, desc="Building Mask R-CNN eval dataset"):
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

            valid_items = []

            for _, value in data.items():
                if not (isinstance(value, dict) and "category_id" in value):
                    continue

                cid = value["category_id"]
                if cid not in CATEGORY_TO_INSTANCE_CLASS:
                    continue

                bbox = value.get("bounding_box", None)
                segs = value.get("segmentation", None)

                if bbox is None or len(bbox) != 4 or segs is None or len(segs) == 0:
                    continue

                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue

                valid_items.append({
                    "class_id": CATEGORY_TO_INSTANCE_CLASS[cid],
                    "bbox": [x1, y1, x2, y2],
                    "segmentation": segs,
                    "img_path": img_path
                })

            if len(valid_items) > 0:
                self.samples.append((img_path, valid_items))

    def __len__(self):
        return len(self.samples)

    def polygons_to_mask(self, polygons, width, height):
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        for poly in polygons:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            try:
                draw.polygon(xy, outline=1, fill=1)
            except Exception:
                continue
        return np.array(mask_img, dtype=np.uint8)

    def __getitem__(self, idx):
        img_path, items = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        boxes, labels, masks = [], [], []

        for item in items:
            x1, y1, x2, y2 = item["bbox"]
            mask = self.polygons_to_mask(item["segmentation"], width, height)
            if mask.sum() == 0:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(item["class_id"])
            masks.append(mask)

        image = F.to_tensor(image)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if len(boxes) else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if len(labels) else torch.zeros((0,), dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks), dtype=torch.uint8) if len(masks) else torch.zeros((0, height, width), dtype=torch.uint8),
            "image_id": torch.tensor([idx])
        }
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def build_maskrcnn(num_classes):
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=256, max_size=512)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def mask_iou(pred_mask, true_mask):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    inter = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return 1.0 if union == 0 else inter / (union + 1e-6)

def dice_score(pred_mask, true_mask):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    inter = np.logical_and(pred_mask, true_mask).sum()
    denom = pred_mask.sum() + true_mask.sum()
    return 1.0 if denom == 0 else (2 * inter) / (denom + 1e-6)

def evaluate_maskrcnn(model_path, img_dir, anno_dir, limit=100):
    dataset = DeepFashion2MaskRCNNDataset(img_dir, anno_dir)
    if limit < len(dataset):
        dataset = Subset(dataset, list(range(limit)))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = build_maskrcnn(NUM_CLASSES_SEG)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    ious, dices = [], []
    total_pred, total_true, matched = 0, 0, 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Mask R-CNN Eval"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            pred = outputs[0]
            target = targets[0]

            total_pred += len(pred["labels"])
            total_true += len(target["labels"])

            pred_masks = pred["masks"].cpu().numpy() if len(pred["masks"]) else np.zeros((0, 1, 1, 1))
            true_masks = target["masks"].cpu().numpy() if len(target["masks"]) else np.zeros((0, 1, 1))

            n = min(len(pred_masks), len(true_masks))
            matched += n

            for i in range(n):
                pm = (pred_masks[i, 0] > 0.5).astype(np.uint8)
                tm = true_masks[i].astype(np.uint8)

                if pm.shape != tm.shape:
                    continue

                ious.append(mask_iou(pm, tm))
                dices.append(dice_score(pm, tm))

    precision = matched / (total_pred + 1e-6)
    recall = matched / (total_true + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    return {
        "Model": "Mask R-CNN",
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mIoU": float(np.mean(ious)) if len(ious) else 0.0,
        "Dice": float(np.mean(dices)) if len(dices) else 0.0
    }

# =========================
# U-NET DATASET + MODEL
# =========================
class DeepFashion2UNetDataset(Dataset):
    def __init__(self, img_dir, anno_dir, img_size=256):
        self.img_size = img_size
        self.samples = []

        anno_files = sorted([f for f in os.listdir(anno_dir) if f.endswith(".json")])
        for anno_file in tqdm(anno_files, desc="Building U-Net eval dataset"):
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
                    if cid in CATEGORY_TO_INSTANCE_CLASS and segs is not None and len(segs) > 0:
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
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        with open(anno_path, "r") as f:
            data = json.load(f)

        semantic_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for _, value in data.items():
            if not (isinstance(value, dict) and "category_id" in value):
                continue

            cid = value["category_id"]
            if cid not in CATEGORY_TO_INSTANCE_CLASS:
                continue

            segs = value.get("segmentation", None)
            if segs is None or len(segs) == 0:
                continue

            class_id = CATEGORY_TO_INSTANCE_CLASS[cid]
            item_mask = self.polygons_to_mask(segs, orig_w, orig_h, class_id)
            semantic_mask[item_mask > 0] = class_id

        image = self.img_transform(image)

        mask_pil = Image.fromarray(semantic_mask)
        mask_pil = mask_pil.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        semantic_mask = np.array(mask_pil, dtype=np.uint8)

        return image, torch.tensor(semantic_mask, dtype=torch.long)

def build_unet():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES_SEG
    )

def compute_seg_metrics(preds, targets, num_classes=6):
    ious, dices = [], []

    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(1, num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        inter = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        pred_sum = pred_cls.sum().item()
        target_sum = target_cls.sum().item()

        iou = 1.0 if union == 0 else inter / (union + 1e-6)
        dice = 1.0 if (pred_sum + target_sum) == 0 else (2 * inter) / (pred_sum + target_sum + 1e-6)

        ious.append(iou)
        dices.append(dice)

    return float(np.mean(ious)), float(np.mean(dices))

def evaluate_unet(model_path, img_dir, anno_dir, limit=200):
    dataset = DeepFashion2UNetDataset(img_dir, anno_dir, img_size=256)
    if limit < len(dataset):
        dataset = Subset(dataset, list(range(limit)))

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    model = build_unet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_ious, all_dices = [], []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="U-Net Eval"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            miou, dice = compute_seg_metrics(preds.cpu(), masks.cpu(), num_classes=NUM_CLASSES_SEG)
            all_ious.append(miou)
            all_dices.append(dice)

    return {
        "Model": "U-Net",
        "mIoU": float(np.mean(all_ious)),
        "Dice": float(np.mean(all_dices))
    }

# =========================
# RUN EVALUATION
# =========================
gc.collect()
torch.cuda.empty_cache()
yolo_results = evaluate_yolo(YOLO_MODEL_PATH, YOLO_DATA_YAML)

gc.collect()
torch.cuda.empty_cache()
maskrcnn_results = evaluate_maskrcnn(MASKRCNN_MODEL_PATH, VAL_IMG_DIR, VAL_ANNO_DIR, limit=100)

gc.collect()
torch.cuda.empty_cache()
unet_results = evaluate_unet(UNET_MODEL_PATH, VAL_IMG_DIR, VAL_ANNO_DIR, limit=200)

summary_rows = [
    {
        "Model": "YOLO",
        "Precision": yolo_results.get("Precision", None),
        "Recall": yolo_results.get("Recall", None),
        "F1": None,
        "mAP50": yolo_results.get("mAP50", None),
        "mAP50-95": yolo_results.get("mAP50-95", None),
        "mIoU": None,
        "Dice": None
    },
    {
        "Model": "Mask R-CNN",
        "Precision": maskrcnn_results.get("Precision", None),
        "Recall": maskrcnn_results.get("Recall", None),
        "F1": maskrcnn_results.get("F1", None),
        "mAP50": None,
        "mAP50-95": None,
        "mIoU": maskrcnn_results.get("mIoU", None),
        "Dice": maskrcnn_results.get("Dice", None)
    },
    {
        "Model": "U-Net",
        "Precision": None,
        "Recall": None,
        "F1": None,
        "mAP50": None,
        "mAP50-95": None,
        "mIoU": unet_results.get("mIoU", None),
        "Dice": unet_results.get("Dice", None)
    }
]

summary_df = pd.DataFrame(summary_rows).round(4)

print("\nFINAL COMPARISON TABLE")
print(summary_df)

summary_df.to_csv("/kaggle/working/detection_segmentation_evaluation.csv", index=False)
print("\nSaved: /kaggle/working/detection_segmentation_evaluation.csv")