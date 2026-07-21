# evaluate.py
"""
Evaluates the trained checkpoint on data/curated/test.csv — the split
scripts/split_dataset.py carves out and that train.py never sees, so
these numbers are an honest estimate of real-world performance rather
than a number computed on data the model was tuned against.

The previous version of this file re-split train.csv itself (same seed,
same test_size as train.py's internal split) and called the result an
"evaluation." At real dataset sizes that's still leakage-adjacent —
val and "test" ended up being the same slice, used both for checkpoint
selection and for the reported number. Point this at a real held-out
test.csv instead.
"""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from train import build_dataset
from src.classifier import TransitCNN

CLASS_NAMES_INDEXED = [
    "Planetary Transit (0)",
    "Eclipsing Binary (1)",
    "Stellar Blend (2)",
    "Stellar Variability (3)",
    "Noise/Unknown (4)",
]


def run_evaluation(
    test_csv="data/curated/test.csv",
    cache_dir="data/processed",
    checkpoint_path="models/transit_cnn.pth",
    batch_size=32,
    out_path="models/metrics.json",
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No trained checkpoint at {checkpoint_path}. Run train.py first — "
            f"there is no pretrained model shipped in this repo."
        )

    test_dataset = build_dataset(test_csv, cache_dir, "test", augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_targets.extend(labels.numpy())

    print(f"Evaluated on {len(all_targets)} held-out test examples from {test_csv} "
          f"(not used in training or checkpoint selection).")

    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES_INDEXED, zero_division=0))
    matrix = confusion_matrix(all_targets, all_preds, labels=range(5))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(matrix)

    report_dict = classification_report(
        all_targets, all_preds, target_names=CLASS_NAMES_INDEXED, zero_division=0, output_dict=True
    )
    metrics_export = {
        "n_test_examples": len(all_targets),
        "test_csv": test_csv,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "note": "Computed on a held-out test split never used during training or checkpoint selection.",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the held-out test set.")
    parser.add_argument("--test_csv", default="data/curated/test.csv")
    parser.add_argument("--cache_dir", default="data/processed")
    parser.add_argument("--checkpoint", default="models/transit_cnn.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", default="models/metrics.json")
    args = parser.parse_args()
    run_evaluation(args.test_csv, args.cache_dir, args.checkpoint, args.batch_size, args.out)
