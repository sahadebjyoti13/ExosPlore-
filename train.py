# train.py
"""
Trains the ExosPlore transit classifier on a real, human-built catalog.

This file used to call generate_mock_csv() unconditionally on every run,
silently overwriting data/curated/train.csv with 20 rows that were all
the same light curve file (wasp18_test.fits) under 5 fake labels. That
is why the shipped checkpoint scored 20% accuracy and predicted class 0
for everything — it never had more than one real signal to learn from.

There is no mock data path anymore. If data/curated/train.csv (and
val.csv) don't exist, this script fails loudly and tells you what to
run instead: src/data_acquisition.py to build a real catalog, then
scripts/split_dataset.py to produce train/val/test splits.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report

from src.preprocess import preprocess_lightcurve
from src.bls_detector import run_bls
from src.classifier import TransitCNN, save_checkpoint, CLASS_NAMES


def _augment_and_normalize(segment, augment):
    segment = segment.copy()
    if augment:
        if np.random.rand() < 0.5:
            segment = segment + np.random.normal(0, 0.001, segment.shape)
        if np.random.rand() < 0.5:
            segment = np.roll(segment, np.random.randint(-10, 10))

    seg_min, seg_max = np.min(segment), np.max(segment)
    if (seg_max - seg_min) > 0:
        segment = (segment - seg_min) / (seg_max - seg_min)
    else:
        segment = np.zeros_like(segment)
    return segment


class TESSDataset(Dataset):
    """On-the-fly dataset: re-runs Module 1+2 for every __getitem__ call.
    Correct, but slow at real dataset sizes — prefer CachedDataset
    (built via `python -m src.cache_dataset`) once you have more than a
    few dozen light curves."""

    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fits_path = row["fits_path"]
        label = int(row["label"])

        t, f, ferr, _, _ = preprocess_lightcurve(fits_path)
        candidates = run_bls(t, f, ferr)
        if not candidates:
            raise RuntimeError(
                f"No BLS candidates found for {fits_path} (TIC {row.get('tic_id')}). "
                f"This light curve can't be used for training as-is."
            )
        segment = _augment_and_normalize(candidates[0]["phase_folded_flux_200"], self.augment)
        return torch.tensor(segment, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

    def labels(self):
        return self.df["label"].values


class CachedDataset(Dataset):
    """Fast dataset backed by .npy arrays produced by src/cache_dataset.py."""

    def __init__(self, x_path, y_path, augment=False):
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        segment = _augment_and_normalize(self.X[idx], self.augment)
        return torch.tensor(segment, dtype=torch.float32).unsqueeze(0), torch.tensor(int(self.y[idx]), dtype=torch.long)

    def labels(self):
        return self.y


def build_dataset(csv_path, cache_dir, split_name, augment):
    x_path = os.path.join(cache_dir, f"{split_name}_X.npy")
    y_path = os.path.join(cache_dir, f"{split_name}_y.npy")
    if os.path.exists(x_path) and os.path.exists(y_path):
        return CachedDataset(x_path, y_path, augment=augment)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No {split_name} data found at {csv_path} or {x_path}.\n"
            f"Build a real catalog first:\n"
            f"  1. python -m src.data_acquisition\n"
            f"  2. python scripts/split_dataset.py\n"
            f"  3. (optional, faster) python -m src.cache_dataset --csv {csv_path} --out {cache_dir}"
        )
    print(f"No cache found at {x_path} — computing {split_name} features on the fly "
          f"(slower; run `python -m src.cache_dataset --csv {csv_path}` first to speed this up).")
    return TESSDataset(pd.read_csv(csv_path), augment=augment)


def train_pipeline_model(
    train_csv="data/curated/train.csv",
    val_csv="data/curated/val.csv",
    cache_dir="data/processed",
    checkpoint_path="models/transit_cnn.pth",
    epochs=40,
    batch_size=32,
    lr=1e-3,
):
    train_dataset = build_dataset(train_csv, cache_dir, "train", augment=True)
    val_dataset = build_dataset(val_csv, cache_dir, "val", augment=False)

    if len(train_dataset) == 0:
        raise RuntimeError(f"Training set at {train_csv} is empty.")

    train_labels = train_dataset.labels()
    class_counts = np.bincount(train_labels, minlength=5)
    missing = [CLASS_NAMES[i] for i, c in enumerate(class_counts) if c == 0]
    if missing:
        print(f"WARNING: zero training examples for: {missing}. "
              f"The model has no way to learn these classes — predictions for them will be meaningless "
              f"until you add real examples.")

    class_counts_safe = np.where(class_counts == 0, 1, class_counts)
    sample_weights = 1.0 / class_counts_safe[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitCNN().to(device)

    criterion_weights = torch.tensor(1.0 / class_counts_safe, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=criterion_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {len(train_dataset)} real examples, validating on {len(val_dataset)}, device={device}")
    print(f"Train class counts: {dict(zip(CLASS_NAMES, class_counts.tolist()))}")

    best_val_loss = float("inf")
    last_val_preds, last_val_targets = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        epoch_val_loss = val_loss / max(len(val_loader.dataset), 1)

        print(f"Epoch {epoch + 1}/{epochs} | train_loss={epoch_loss:.4f} | val_loss={epoch_val_loss:.4f}")

        if epoch_val_loss <= best_val_loss:
            best_val_loss = epoch_val_loss
            save_checkpoint(model, path=checkpoint_path)
            last_val_preds, last_val_targets = val_preds, val_targets

    print("\nValidation report at the best checkpoint (this is for model selection during "
          "training — for the honest, held-out number run evaluate.py against test.csv):")
    if last_val_targets:
        print(classification_report(last_val_targets, last_val_preds, target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ExosPlore transit classifier on real data.")
    parser.add_argument("--train_csv", default="data/curated/train.csv")
    parser.add_argument("--val_csv", default="data/curated/val.csv")
    parser.add_argument("--cache_dir", default="data/processed")
    parser.add_argument("--checkpoint", default="models/transit_cnn.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_pipeline_model(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        cache_dir=args.cache_dir,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
