"""Satellite land-use classifier (EuroSAT, ResNet18 transfer learning).

Run:  python train.py            (real run, needs internet for the dataset)
      SMOKE_TEST=1 python train.py   (tiny fake-data run to test the code)
Generated from landuse_classifier.ipynb - same code.
"""

import os, sys, json, random, time, copy
from pathlib import Path

import numpy as np
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")            # no display needed when run as a plain script
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"   # tiny fake-data run, only to test the code

SEED = 42
IMG_SIZE = 64 if SMOKE_TEST else 128      # EuroSAT images are 64x64; we upscale to 128 for the pretrained net
BATCH_SIZE = 32 if SMOKE_TEST else 64
EPOCHS_HEAD = 1 if SMOKE_TEST else 3      # stage 1: train only the new last layer
EPOCHS_FINETUNE = 1 if SMOKE_TEST else 4  # stage 2: fine-tune the whole network
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4                        # smaller learning rate so we do not destroy pretrained knowledge
NUM_WORKERS = 0
DATA_ROOT = "data"
OUT = Path("outputs_smoke" if SMOKE_TEST else "outputs")
OUT.mkdir(exist_ok=True)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| smoke test:", SMOKE_TEST)

def save_show(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet statistics

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def denorm(t):
    """Undo normalisation so an image tensor can be plotted."""
    t = t.clone().cpu() * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    return t.clamp(0, 1).permute(1, 2, 0).numpy()

CLASS_NAMES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
               "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

class SyntheticLandUse(Dataset):
    """Tiny fake dataset (coloured noise) used ONLY to smoke-test this code when EuroSAT is unavailable."""
    classes = CLASS_NAMES
    def __init__(self, transform=None, per_class=40):
        rng = np.random.RandomState(0)
        self.transform, self.images, self.targets = transform, [], []
        for c in range(len(self.classes)):
            base = rng.randint(30, 225, size=3)
            for _ in range(per_class):
                self.images.append(np.clip(base + rng.randn(64, 64, 3) * 25, 0, 255).astype(np.uint8))
                self.targets.append(c)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        return (self.transform(img) if self.transform else img), self.targets[i]

def make_dataset(tf):
    if SMOKE_TEST:
        return SyntheticLandUse(transform=tf)
    return datasets.EuroSAT(root=DATA_ROOT, download=True, transform=tf)

full_train = make_dataset(train_tf)    # same images, two different transform pipelines
full_eval = make_dataset(eval_tf)
classes = full_eval.classes
targets = np.array(full_eval.targets)
print(f"{len(full_eval)} images, {len(classes)} classes:", classes)

counts = np.bincount(targets, minlength=len(classes))
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(x=classes, y=counts, ax=ax, color="#4C72B0")
ax.set_title("Images per class"); ax.set_ylabel("count"); ax.tick_params(axis="x", rotation=45)
for lbl in ax.get_xticklabels():
    lbl.set_ha("right")
save_show(fig, "class_distribution.png")

fig, axes = plt.subplots(len(classes), 5, figsize=(8, 16))
for c, name in enumerate(classes):
    idxs = np.where(targets == c)[0][:5]
    for j, i in enumerate(idxs):
        axes[c, j].imshow(denorm(full_eval[i][0])); axes[c, j].axis("off")
    axes[c, 0].set_title(name, loc="left", fontsize=9)
save_show(fig, "sample_images.png")

idx = np.arange(len(targets))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=targets, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=targets[temp_idx], random_state=SEED)
print(f"train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")

train_ds = Subset(full_train, train_idx)   # with augmentation
val_ds = Subset(full_eval, val_idx)        # without augmentation
test_ds = Subset(full_eval, test_idx)

pin = device.type == "cuda"
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)   # new classification head
    return m

def set_backbone_trainable(m, flag):
    for name, p in m.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = flag

model = build_model(len(classes), pretrained=not SMOKE_TEST).to(device)
criterion = nn.CrossEntropyLoss()
print("Parameters:", sum(p.numel() for p in model.parameters()))

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n

history, best = [], {"val_acc": -1.0, "state": None, "epoch": None}

def fit(stage, epochs, lr):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader)
        history.append(dict(stage=stage, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc))
        if va_acc > best["val_acc"]:
            best.update(val_acc=va_acc, state=copy.deepcopy(model.state_dict()), epoch=len(history))
        print(f"[{stage}] epoch {len(history):2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time()-t0:.0f}s")

set_backbone_trainable(model, False)      # stage 1: only the head learns
fit("head", EPOCHS_HEAD, LR_HEAD)
set_backbone_trainable(model, True)       # stage 2: fine-tune everything
fit("finetune", EPOCHS_FINETUNE, LR_FINETUNE)
print(f"Best validation accuracy {best['val_acc']:.4f} at epoch {best['epoch']}")

epochs_axis = range(1, len(history) + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(epochs_axis, [h["train_loss"] for h in history], label="train")
axes[0].plot(epochs_axis, [h["val_loss"] for h in history], label="validation")
axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].plot(epochs_axis, [h["train_acc"] for h in history], label="train")
axes[1].plot(epochs_axis, [h["val_acc"] for h in history], label="validation")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
n_head = sum(h["stage"] == "head" for h in history)
for ax in axes:
    ax.axvline(n_head + 0.5, color="grey", ls="--", lw=1)   # where fine-tuning starts
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
save_show(fig, "training_curves.png")

model.load_state_dict(best["state"])
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        all_preds.append(model(x.to(device)).argmax(1).cpu().numpy())
        all_labels.append(y.numpy())
preds, labels = np.concatenate(all_preds), np.concatenate(all_labels)

test_acc = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average="macro")
rng = np.random.RandomState(SEED)
hits = (preds == labels).astype(float)
boot = [hits[rng.randint(0, len(hits), len(hits))].mean() for _ in range(1000)]
ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

print(f"Test accuracy: {test_acc:.4f}  (95% CI {ci_low:.4f} - {ci_high:.4f})")
print(f"Macro F1:      {macro_f1:.4f}\n")
print(classification_report(labels, preds, target_names=classes, digits=3, zero_division=0))

cm = confusion_matrix(labels, preds)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (each row sums to 1)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
save_show(fig, "confusion_matrix.png")

# the most confused pairs of classes
off = cm.copy(); np.fill_diagonal(off, 0)
pairs = sorted(((off[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if off[i, j] > 0),
               reverse=True)[:5]
print("Most common mistakes (true -> predicted):")
for count, t, p in pairs:
    print(f"  {t} -> {p}: {count} images")

report = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
fig, ax = plt.subplots(figsize=(9, 4))
f1s = [report[c]["f1-score"] for c in classes]
sns.barplot(x=classes, y=f1s, ax=ax, color="#55A868")
ax.set_ylim(min(0.5, min(f1s) - 0.05), 1.0); ax.set_title("F1-score per class"); ax.tick_params(axis="x", rotation=45)
for lbl in ax.get_xticklabels():
    lbl.set_ha("right")
save_show(fig, "per_class_f1.png")

wrong = np.where(preds != labels)[0][:12]
if len(wrong):
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k in zip(axes.ravel(), wrong):
        ax.imshow(denorm(test_ds[k][0]))
        ax.set_title(f"true: {classes[labels[k]]}\npred: {classes[preds[k]]}", fontsize=8)
    save_show(fig, "misclassified_examples.png")

metrics = {
    "test_accuracy": float(test_acc),
    "test_accuracy_95ci": [float(ci_low), float(ci_high)],
    "macro_f1": float(macro_f1),
    "best_val_accuracy": float(best["val_acc"]),
    "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score")} for c in classes},
    "top_confusions": [{"true": t, "predicted": p, "count": int(c)} for c, t, p in pairs],
    "splits": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
    "config": {"img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "epochs_head": EPOCHS_HEAD,
               "epochs_finetune": EPOCHS_FINETUNE, "lr_head": LR_HEAD, "lr_finetune": LR_FINETUNE,
               "seed": SEED, "device": str(device), "smoke_test": SMOKE_TEST},
    "history": history,
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
torch.save({"state_dict": best["state"], "classes": classes, "img_size": IMG_SIZE}, OUT / "model.pt")
print("Saved to", OUT.resolve())
if SMOKE_TEST:
    print("\n>>> SMOKE TEST on fake data: these numbers are meaningless. Re-run without SMOKE_TEST for real results.")
else:
    print(f"\n>>> For your README / resume: {test_acc*100:.1f}% test accuracy, macro-F1 {macro_f1:.3f} "
          f"on {len(test_idx)} held-out images across {len(classes)} classes")

def predict(image, topk=3):
    """image: a PIL image. Returns the top-k (class, probability) pairs."""
    model.eval()
    x = eval_tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    top = probs.topk(topk)
    return [(classes[i], float(p)) for p, i in zip(top.values, top.indices)]

if hasattr(full_eval, "samples"):            # real EuroSAT: grab a test image from disk
    path, true_idx = full_eval.samples[test_idx[0]]
    img = Image.open(path)
    print("True class:", classes[true_idx])
    print("Prediction:", predict(img))
    plt.imshow(img); plt.axis("off"); plt.show()
