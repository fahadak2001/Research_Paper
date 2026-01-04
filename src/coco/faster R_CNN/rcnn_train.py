"""
train_rcnn.py

Beginner-friendly Faster R-CNN training script (CPU only)
WITH RESUME TRAINING SUPPORT
"""

import os
import json
import random
import time
from typing import List

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()

# -------------------------
# Input helper
# -------------------------
def ask_input(prompt, default=None, cast_func=str):
    inp = input(f"{prompt} [{'Default: '+str(default) if default else 'Required'}]: ").strip()
    if inp == "":
        if default is not None:
            return default
        return ask_input(prompt, default, cast_func)
    return cast_func(inp)

# -------------------------
# User inputs
# -------------------------
train_img_path = ask_input("Path to training images")
train_label_file = ask_input("Path to training COCO JSON")
test_img_path = ask_input("Path to test images")
test_label_file = ask_input("Path to test COCO JSON")
epochs = ask_input("Total number of epochs", default=50, cast_func=int)
batch_size = ask_input("Batch size", default=4, cast_func=int)
lr = ask_input("Learning rate", default=1e-4, cast_func=float)
threshold = ask_input("Prediction score threshold", default=0.5, cast_func=float)
output_dir = ask_input("Output folder", default="output")
use_pretrained = ask_input("Use pretrained weights? (yes/no)", "yes").lower() in ["yes", "y"]
resume_epoch = ask_input("Resume from epoch? (0 = start fresh)", default=0, cast_func=int)

device = torch.device("cpu")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "sample_predictions"), exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def calculate_iou(b1, b2):
    x1, y1, x2, y2 = b1
    x1b, y1b, x2b, y2b = b2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (x2 - x1)*(y2 - y1) + (x2b - x1b)*(y2b - y1b) - inter
    return inter / union if union > 0 else 0

# -------------------------
# Transforms
# -------------------------
class ToTensor:
    def __call__(self, img, boxes):
        return F.to_tensor(img), boxes

train_tf = ToTensor()
val_tf = ToTensor()

# -------------------------
# Dataset
# -------------------------
class VehicleDataset(Dataset):
    def __init__(self, img_dir, data, transform=None):
        self.img_dir = img_dir
        self.data = data
        self.transform = transform
        self.imgs = data["images"]
        self.anns = {}
        for a in data["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        info = self.imgs[idx]
        img_id = info["id"]
        path = os.path.join(self.img_dir, info["file_name"])

        img = Image.fromarray(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

        boxes, labels = [], []
        for a in self.anns.get(img_id, []):
            boxes.append(xywh_to_xyxy(a["bbox"]))
            labels.append(a["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        img, boxes = self.transform(img, boxes)
        return img, boxes, labels

    def collate_fn(self, batch):
        imgs = torch.stack([b[0] for b in batch])
        boxes = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        return imgs, boxes, labels

# -------------------------
# Load data
# -------------------------
train_data = json.load(open(train_label_file))
test_data = json.load(open(test_label_file))

num_classes = max(c["id"] for c in train_data["categories"]) + 1

train_ds = VehicleDataset(train_img_path, train_data, train_tf)
test_ds = VehicleDataset(test_img_path, test_data, val_tf)

train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=train_ds.collate_fn)
test_loader = DataLoader(test_ds, 1, shuffle=False, collate_fn=test_ds.collate_fn)

# -------------------------
# Model
# -------------------------
model = fasterrcnn_resnet50_fpn(pretrained=use_pretrained)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//3), gamma=0.1)

# -------------------------
# Resume training
# -------------------------
if resume_epoch > 0:
    ckpt = os.path.join(output_dir, f"model_epoch_{resume_epoch}.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"✅ Resumed from epoch {resume_epoch}")

# -------------------------
# Train loop
# -------------------------
def train_epoch(epoch):
    model.train()
    total = 0
    for i, (imgs, boxes, labels) in enumerate(train_loader):
        if any(len(b) == 0 for b in boxes):
            continue

        imgs = imgs.to(device)
        targets = [{"boxes": b.to(device), "labels": l.to(device)} for b, l in zip(boxes, labels)]

        optimizer.zero_grad()
        loss = sum(model(imgs, targets).values())
        loss.backward()
        optimizer.step()

        total += loss.item()
        print(f"[Epoch {epoch+1}/{epochs}] Batch {i+1}/{len(train_loader)} Loss: {loss.item():.4f}", end="\r")

    scheduler.step()
    print(f"\nEpoch {epoch+1} done | Avg Loss: {total/len(train_loader):.4f}")

# -------------------------
# Main
# -------------------------
for ep in range(resume_epoch, epochs):
    train_epoch(ep)
    torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{ep+1}.pth"))

torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
print("🎉 Training complete")
