import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from dataset_class import VertebraDataset
from monai.networks.nets import DenseNet121
from utils import DEFAULT_DATA_ROOT

CSV_PATH = "data/vertebra_dataset.csv"
ROOT_DIR = DEFAULT_DATA_ROOT

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
USE_SAMPLE_STRATIFIED_SPLIT = True
USE_WEIGHTED_SAMPLER = False
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0
SEED = 42
NUM_WORKERS = 0
PIN_MEMORY = False
USE_PATCH_CACHE = True
PREBUILD_PATCH_CACHE = True
PATCH_CACHE_DIR = "data/vertebra_patch_cache"
PATIENT_CACHE_SIZE = 1
PATCH_SIZE = (96, 96, 64)
NORM_MODE = "zscore_sigmoid"
ZSCORE_SCALE = 1.5
FOREGROUND_FLOOR = 0.15
RUN_DATE = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = Path("output/stage1/cancer_type") / RUN_DATE
BEST_MODEL_PATH = OUTPUT_DIR / "best.pth"
LAST_MODEL_PATH = OUTPUT_DIR / "last.pth"
HISTORY_PATH = OUTPUT_DIR / "history.csv"
PARAMS_PATH = OUTPUT_DIR / "param.json"
CURVES_PATH = OUTPUT_DIR / "curves.png"
TRAIN_CONFUSION_PATH = OUTPUT_DIR / "train_confusion_matrix.csv"
TRAIN_REPORT_PATH = OUTPUT_DIR / "train_classification_report.txt"
TRAIN_REPORT_JSON_PATH = OUTPUT_DIR / "train_classification_report.json"
VAL_CONFUSION_PATH = OUTPUT_DIR / "best_val_confusion_matrix.csv"
VAL_REPORT_PATH = OUTPUT_DIR / "best_val_classification_report.txt"
VAL_REPORT_JSON_PATH = OUTPUT_DIR / "best_val_classification_report.json"
TEST_CONFUSION_PATH = OUTPUT_DIR / "test_confusion_matrix.csv"
TEST_REPORT_PATH = OUTPUT_DIR / "test_classification_report.txt"
TEST_REPORT_JSON_PATH = OUTPUT_DIR / "test_classification_report.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["blastic", "lytic", "mixed"]
NUM_CLASSES = len(CLASS_NAMES)
SPLIT_MODE = "sample_stratified" if USE_SAMPLE_STRATIFIED_SPLIT else "patient_random"
TYPE_MAP = {1: 0, 2: 1, 3: 2}


class CancerTypeDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.indices = [i for i, label in enumerate(base_dataset.df["label"].astype(int).tolist()) if label > 0]
        self.df = base_dataset.df.iloc[self.indices].copy().reset_index(drop=True)
        self.df["label"] = self.df["label"].astype(int).map(TYPE_MAP)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        patch, label = self.base[base_idx]
        return patch, torch.tensor(TYPE_MAP[int(label)], dtype=torch.long)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[targets] * loss
        return loss.mean()


def build_patient_splits(dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED):
    patients = dataset.df["patient_id"].astype(str).unique().tolist()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(patients), generator=g).tolist()
    patients = [patients[i] for i in perm]

    n_total = len(patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for idx, pid in enumerate(dataset.df["patient_id"].astype(str).tolist()):
        if pid in train_patients:
            train_idx.append(idx)
        elif pid in val_patients:
            val_idx.append(idx)
        elif pid in test_patients:
            test_idx.append(idx)
    return train_idx, val_idx, test_idx


def build_sample_stratified_splits(dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED):
    indices = list(range(len(dataset.df)))
    labels = dataset.df["label"].astype(int).tolist()
    test_ratio = 1.0 - train_ratio - val_ratio
    train_idx, holdout_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    holdout_labels = [labels[i] for i in holdout_idx]
    val_fraction = val_ratio / max(val_ratio + test_ratio, 1e-8)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=val_fraction,
        random_state=seed,
        stratify=holdout_labels,
    )
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


def build_splits(dataset):
    if USE_SAMPLE_STRATIFIED_SPLIT:
        return build_sample_stratified_splits(dataset)
    return build_patient_splits(dataset)


def make_loader(subset, shuffle=False):
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def make_weighted_train_loader(dataset, indices):
    labels = dataset.df.iloc[indices]["label"].astype(int).tolist()
    counts = Counter(labels)
    sample_weights = [1.0 / max(counts[label], 1) for label in labels]
    sampler = WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(
        Subset(dataset, indices),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def label_counts(df, indices):
    return dict(Counter(df.iloc[indices]["label"].astype(int).tolist()))


def compute_class_weights(df, indices):
    labels = df.iloc[indices]["label"].astype(int).tolist()
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for label in labels:
        counts[label] += 1.0
    total = counts.sum().item()
    weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for c in range(NUM_CLASSES):
        if counts[c] > 0:
            weights[c] = total / (NUM_CLASSES * counts[c])
    return weights, counts


def build_report_from_confusion(conf):
    eps = 1e-8
    f1_scores = []
    for c in range(conf.size(0)):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_scores.append((2 * precision * recall) / (precision + recall + eps))
    return sum(f1_scores) / len(f1_scores)


def run_epoch(model, loader, criterion, optimizer=None, collect_predictions=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
        preds = out.argmax(dim=1)
        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_examples += x.size(0)
        y_cpu = y.detach().cpu()
        preds_cpu = preds.detach().cpu()
        for t, p in zip(y_cpu, preds_cpu):
            conf[int(t), int(p)] += 1
        if collect_predictions:
            y_true.extend(int(v) for v in y_cpu.tolist())
            y_pred.extend(int(v) for v in preds_cpu.tolist())

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
        "f1": build_report_from_confusion(conf),
        "confusion_matrix": conf,
    }
    if collect_predictions:
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
    return metrics


def save_confusion_matrix_csv(conf, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for i, row in enumerate(conf.tolist()):
            writer.writerow([CLASS_NAMES[i], *row])


def save_classification_report(y_true, y_pred, txt_path, json_path):
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )
    report_json = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
        output_dict=True,
    )
    with open(txt_path, "w") as f:
        f.write(report_text)
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)


def save_split_reports(metrics, confusion_path, report_path, report_json_path, split_name):
    save_confusion_matrix_csv(metrics["confusion_matrix"], confusion_path)
    save_classification_report(metrics["y_true"], metrics["y_pred"], report_path, report_json_path)
    print(f"Saved {split_name} confusion matrix: {confusion_path}")
    print(f"Saved {split_name} classification report: {report_path}")


dataset = CancerTypeDataset(
    VertebraDataset(
        csv_path=CSV_PATH,
        root_dir=ROOT_DIR,
        use_patch_cache=USE_PATCH_CACHE,
        cache_dir=PATCH_CACHE_DIR,
        patient_cache_size=PATIENT_CACHE_SIZE,
        patch_size=PATCH_SIZE,
        norm_mode=NORM_MODE,
        zscore_scale=ZSCORE_SCALE,
        foreground_floor=FOREGROUND_FLOOR,
    )
)

if PREBUILD_PATCH_CACHE and USE_PATCH_CACHE:
    start_time = datetime.now()
    print(f"Precomputing patch cache -> {PATCH_CACHE_DIR}")
    dataset.base.precompute_cache(force=False, verbose=True)
    print(f"Precomputation time: {datetime.now() - start_time}")

train_idx, val_idx, test_idx = build_splits(dataset)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
test_set = Subset(dataset, test_idx)

train_loader = make_weighted_train_loader(dataset, train_idx) if USE_WEIGHTED_SAMPLER else make_loader(train_set, shuffle=True)
val_loader = make_loader(val_set, shuffle=False)
test_loader = make_loader(test_set, shuffle=False)

print(f"Device: {DEVICE}")
print(f"Split mode: {SPLIT_MODE}")
print(f"Weighted sampler: {USE_WEIGHTED_SAMPLER}")
print(f"Focal loss: {USE_FOCAL_LOSS}")
print(f"Count: train - {len(train_set)} val - {len(val_set)} test - {len(test_set)}")
print(f"Train label counts: {label_counts(dataset.df, train_idx)}")
print(f"Val label counts:   {label_counts(dataset.df, val_idx)}")
print(f"Test label counts:  {label_counts(dataset.df, test_idx)}")

class_weights, train_counts = compute_class_weights(dataset.df, train_idx)
params = {
    "csv_path": CSV_PATH,
    "root_dir": ROOT_DIR,
    "split_mode": SPLIT_MODE,
    "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
    "use_focal_loss": USE_FOCAL_LOSS,
    "focal_gamma": FOCAL_GAMMA,
    "train_label_counts": label_counts(dataset.df, train_idx),
    "val_label_counts": label_counts(dataset.df, val_idx),
    "test_label_counts": label_counts(dataset.df, test_idx),
    "class_weights": [float(x) for x in class_weights.tolist()],
    "class_count_vector": [int(x) for x in train_counts.tolist()],
}
with open(PARAMS_PATH, "w") as f:
    json.dump(params, f, indent=2)

model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
criterion = FocalLoss(alpha=class_weights.to(DEVICE), gamma=FOCAL_GAMMA) if USE_FOCAL_LOSS else nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

history = []
best_val_loss = float("inf")
best_val_f1 = float("-inf")
best_epoch = 0

for epoch in range(1, EPOCHS + 1):
    train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer)
    val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)
    history.append(
        {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
        }
    )
    print(
        f"Epoch {epoch:02d} | train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
        f"train_f1={train_metrics['f1']:.4f} | val_loss={val_metrics['loss']:.4f} "
        f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f}"
    )
    if val_metrics["f1"] > best_val_f1 or (val_metrics["f1"] == best_val_f1 and val_metrics["loss"] < best_val_loss):
        best_val_f1 = val_metrics["f1"]
        best_val_loss = val_metrics["loss"]
        best_epoch = epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_f1,
            },
            BEST_MODEL_PATH,
        )

with open(HISTORY_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"])
    writer.writeheader()
    writer.writerows(history)

torch.save(
    {
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
    },
    LAST_MODEL_PATH,
)

epochs = [r["epoch"] for r in history]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
axes[0].plot(epochs, [r["val_loss"] for r in history], label="val")
axes[0].set_title("Loss")
axes[0].legend()
axes[1].plot(epochs, [r["train_acc"] for r in history], label="train")
axes[1].plot(epochs, [r["val_acc"] for r in history], label="val")
axes[1].set_title("Accuracy")
axes[1].legend()
axes[2].plot(epochs, [r["train_f1"] for r in history], label="train")
axes[2].plot(epochs, [r["val_f1"] for r in history], label="val")
axes[2].set_title("Macro F1")
axes[2].legend()
fig.tight_layout()
fig.savefig(CURVES_PATH, dpi=140)
plt.close(fig)

if BEST_MODEL_PATH.exists():
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

train_metrics = run_epoch(model, train_loader, criterion, optimizer=None, collect_predictions=True)
val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, collect_predictions=True)
test_metrics = run_epoch(model, test_loader, criterion, optimizer=None, collect_predictions=True)

save_split_reports(train_metrics, TRAIN_CONFUSION_PATH, TRAIN_REPORT_PATH, TRAIN_REPORT_JSON_PATH, "train")
save_split_reports(val_metrics, VAL_CONFUSION_PATH, VAL_REPORT_PATH, VAL_REPORT_JSON_PATH, "best-val")
save_split_reports(test_metrics, TEST_CONFUSION_PATH, TEST_REPORT_PATH, TEST_REPORT_JSON_PATH, "test")

params["best_val_loss"] = best_val_loss
params["best_val_f1"] = best_val_f1
params["best_epoch"] = best_epoch
params["train_loss"] = train_metrics["loss"]
params["train_acc"] = train_metrics["acc"]
params["train_f1"] = train_metrics["f1"]
params["val_loss_at_best"] = val_metrics["loss"]
params["val_acc_at_best"] = val_metrics["acc"]
params["val_f1_at_best"] = val_metrics["f1"]
params["test_loss"] = test_metrics["loss"]
params["test_acc"] = test_metrics["acc"]
params["test_f1"] = test_metrics["f1"]
with open(PARAMS_PATH, "w") as f:
    json.dump(params, f, indent=2)
