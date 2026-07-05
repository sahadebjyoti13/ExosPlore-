# evaluate.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Internal Module Imports
from train import TESSDataset
from src.classifier import TransitCNN


def run_evaluation(test_csv_path="data/curated/train.csv", batch_size=16):
    print("🔬 Initializing Pipeline Evaluation Suite...")

    if not os.path.exists(test_csv_path):
        print(f"❌ Error: Evaluation catalog tracker not found at {test_csv_path}")
        return

    # Load evaluation dataset
    df = pd.read_csv(test_csv_path)

    # If using mock testing setup, isolate a small validation chunk
    if len(df) <= 25:
        print("💡 Small testing database detected. Evaluating on validation slice.")
        eval_df = df.iloc[15:].copy()
    else:
        # Split a distinct evaluation pool (or point this to a separate val.csv if provided)
        from sklearn.model_selection import train_test_split

        _, eval_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )

    eval_dataset = TESSDataset(eval_df, augment=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Load model configuration weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitCNN().to(device)

    model_weights_path = "models/transit_cnn.pth"
    if not os.path.exists(model_weights_path):
        print(
            f"❌ Error: Trained weight checkpoint not found at {model_weights_path}. Run train.py first!"
        )
        return

    model.load_state_dict(
        torch.load(
            model_weights_path, map_location=device if device.type == "cpu" else None
        )
    )
    model.eval()

    all_preds = []
    all_targets = []

    print(f"🔄 Processing evaluation telemetry over: {device}")
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    # --- METRICS COMPILATION SYSTEM ---
    class_names = [
        "Planetary Transit (0)",
        "Eclipsing Binary (1)",
        "Stellar Blend (2)",
        "Stellar Variability (3)",
        "Noise/Unknown (4)",
    ]

    print("\n" + "=" * 60)
    print("📊 ISRO PROBLEM STATEMENT 7 — CLASSIFICATION REPORT")
    print("=" * 60)

    # Print clean precision, recall, f1-score sheets
    print(
        classification_report(
            all_targets, all_preds, target_names=class_names, zero_division=0
        )
    )

    print("\n🧩 CONFUSION MATRIX (Rows: True, Columns: Predicted):")
    print("-" * 55)
    matrix = confusion_matrix(all_targets, all_preds, labels=range(5))
    print(matrix)
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
