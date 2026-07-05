# train.py

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# Internal Module Imports
from src.preprocess import preprocess_lightcurve
from src.bls_detector import run_bls
from src.classifier import TransitCNN, save_checkpoint


class TESSDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fits_path = row["fits_path"]
        label = int(row["label"])

        try:
            t, f, ferr, _, _ = preprocess_lightcurve(fits_path)
            candidates = run_bls(t, f, ferr)

            if len(candidates) > 0:
                segment = candidates[0]["phase_folded_flux_200"]
            else:
                segment = np.ones(200)
        except Exception:
            segment = np.ones(200)

        if self.augment:
            if np.random.rand() < 0.5:
                segment += np.random.normal(0, 0.001, segment.shape)
            if np.random.rand() < 0.5:
                shift = np.random.randint(-10, 10)
                segment = np.roll(segment, shift)

        seg_min, seg_max = np.min(segment), np.max(segment)
        if (seg_max - seg_min) > 0:
            segment = (segment - seg_min) / (seg_max - seg_min)
        else:
            segment = np.zeros_like(segment)

        return torch.tensor(segment, dtype=torch.float32).unsqueeze(0), torch.tensor(
            label, dtype=torch.long
        )


def generate_mock_csv():
    csv_dir = "data/curated"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "train.csv")

    print("Creating layout verification data in 'data/curated/train.csv'...")
    mock_data = {
        "tic_id": [100100827, 99999, 88888, 77777, 66666] * 4,
        "label": [0, 1, 2, 3, 4] * 4,
        "fits_path": ["wasp18_test.fits"] * 20,
    }
    pd.DataFrame(mock_data).to_csv(csv_path, index=False)
    return csv_path


def train_pipeline_model(epochs=3, batch_size=2):
    print("Initializing Training Workflow Script...")
    csv_path = generate_mock_csv()
    df = pd.read_csv(csv_path)

    # RUGGED SPLIT HANDLING: Direct conditional choice to guarantee zero scikit-learn split errors
    if len(df) <= 25:
        print("Small dataset detected: using simple index separation.")
        # Grab first 15 for train, last 5 for validation
        train_df = df.iloc[:15].copy()
        val_df = df.iloc[15:].copy()
    else:
        print("Large dataset detected: running stratified train/test split.")
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )

    train_dataset = TESSDataset(train_df, augment=True)
    val_dataset = TESSDataset(val_df, augment=False)

    # Handle Sampler configuration cleanly
    class_counts = np.bincount(train_df["label"], minlength=5)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[label] for label in train_df["label"]])

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitCNN().to(device)

    all_labels = df["label"].values
    global_counts = np.bincount(all_labels, minlength=5)
    global_counts = np.where(global_counts == 0, 1, global_counts)
    criterion_weights = torch.tensor(1.0 / global_counts, dtype=torch.float32).to(
        device
    )
    criterion = nn.CrossEntropyLoss(weight=criterion_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Beginning training over device target: {device}")
    best_loss = float("inf")

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
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / max(len(val_loader.dataset), 1)
        print(
            f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss <= best_loss:
            best_loss = epoch_val_loss
            save_checkpoint(model, path="models/transit_cnn.pth")

    print("\n--- Model Training Execution Complete ---")


if __name__ == "__main__":
    train_pipeline_model(epochs=3, batch_size=2)
