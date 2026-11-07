"""Classify one satellite image with the model trained by train.py.

Usage:  python predict.py path/to/image.jpg [--model outputs/model.pt]
"""
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--model", default="outputs/model.pt")
parser.add_argument("--topk", type=int, default=3)
args = parser.parse_args()

ckpt = torch.load(args.model, map_location="cpu")
classes, img_size = ckpt["classes"], ckpt["img_size"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["state_dict"])
model.eval()

tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]

top = probs.topk(args.topk)
for p, i in zip(top.values, top.indices):
    print(f"{classes[int(i)]:<22} {float(p) * 100:5.1f}%")
