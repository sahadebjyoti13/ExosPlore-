# src/cache_dataset.py

import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Import our working physics modules
from src.preprocess import preprocess_lightcurve
from src.bls_detector import run_bls


def pre_cache_pipeline(csv_path="data/curated/train.csv", output_dir="data/processed"):
    print("⚡ Initializing Space Science Pre-Caching Engine...")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"❌ Error: Target catalog file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"📋 Found {len(df)} telemetry targets registered in catalog registry.")

    cached_features = []
    cached_labels = []

    # Process files sequentially with a progress tracking bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing FITS"):
        fits_path = row["fits_path"]
        label = row["label"]

        if not os.path.exists(fits_path):
            print(f"⚠️ Warning: File skipped, not found at path: {fits_path}")
            continue

        try:
            # Step 1: Run physics detrending
            time, flux = preprocess_lightcurve(fits_path)

            # Step 2: Run Box Least Squares grid search and extract 200-point phase-fold
            folded_time, folded_flux, bls_stats = run_bls(time, flux)

            # Save the processed array shape (1, 200) matching our CNN input tensor layout
            cached_features.append(folded_flux)
            cached_labels.append(label)

        except Exception as e:
            print(
                f"⚠️ Error processing index {idx} (TIC {row.get('tic_id', 'Unknown')}): {str(e)}"
            )
            continue

    if len(cached_features) == 0:
        print("❌ Error: No features were successfully cached.")
        return

    # Convert lists into highly optimized numpy memory matrices
    X = np.array(cached_features, dtype=np.float32)
    y = np.array(cached_labels, dtype=np.int64)

    # Save arrays to disk as ultra-fast binary dumps
    np.save(os.path.join(output_dir, "X_train.npy"), X)
    np.save(os.path.join(output_dir, "y_train.npy"), y)

    print("\n" + "=" * 50)
    print("✅ PRE-CACHING STORAGE ROUTINE COMPLETE!")
    print(f"📦 Total Cached Tensors: {len(X)}")
    print(
        f"💾 Features File: {os.path.join(output_dir, 'X_train.npy')} ({X.nbytes / 1024 / 1024:.2f} MB)"
    )
    print(f"💾 Labels File:   {os.path.join(output_dir, 'y_train.npy')}")
    print(
        "💡 Update train.py to read directly from these numpy arrays for instant training!"
    )
    print("=" * 50)


if __name__ == "__main__":
    pre_cache_pipeline()
