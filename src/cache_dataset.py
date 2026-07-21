# src/cache_dataset.py
"""
Pre-computes Module 1 + Module 2 (preprocessing + BLS) once per light
curve and caches the resulting 200-point phase-folded segment to disk.

Why this exists: the BLS grid search in src/bls_detector.py scans 10,000
periods per light curve. Recomputing that on every training epoch (which
is what happens if you feed raw .fits paths straight into a DataLoader)
is needlessly slow once you have more than a handful of real light
curves. Run this once per data split, then train.py will pick up the
cached arrays automatically.

NOTE: the previous version of this file did not work — it unpacked
preprocess_lightcurve()'s 5 return values into 2 variables, and called
run_bls() without the required flux_err argument while assuming a
return shape it doesn't have. It had never been run successfully.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocess import preprocess_lightcurve
from src.bls_detector import run_bls


def pre_cache_pipeline(csv_path="data/curated/train.csv", output_dir="data/processed", prefix=None):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Catalog not found at {csv_path}. Build one with "
            f"`python -m src.data_acquisition` and `python scripts/split_dataset.py` first."
        )

    df = pd.read_csv(csv_path)
    prefix = prefix or os.path.splitext(os.path.basename(csv_path))[0]

    features, labels, tic_ids, failed = [], [], [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Caching {prefix}"):
        fits_path = row["fits_path"]
        label = row["label"]
        tic_id = row.get("tic_id", idx)

        if not os.path.exists(fits_path):
            failed.append((tic_id, f"file not found: {fits_path}"))
            continue

        try:
            time, flux, flux_err, _, _ = preprocess_lightcurve(fits_path)
            candidates = run_bls(time, flux, flux_err)
            if not candidates:
                failed.append((tic_id, "no BLS candidates found"))
                continue
            features.append(candidates[0]["phase_folded_flux_200"])
            labels.append(label)
            tic_ids.append(tic_id)
        except Exception as e:
            failed.append((tic_id, str(e)))
            continue

    if not features:
        raise RuntimeError(
            "No light curves were successfully cached — every row failed. "
            "See the errors above before continuing."
        )

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    ids = np.array(tic_ids)

    np.save(os.path.join(output_dir, f"{prefix}_X.npy"), X)
    np.save(os.path.join(output_dir, f"{prefix}_y.npy"), y)
    np.save(os.path.join(output_dir, f"{prefix}_tic_ids.npy"), ids)

    print(f"Cached {len(X)}/{len(df)} light curves -> {output_dir}/{prefix}_X.npy, {prefix}_y.npy")
    if failed:
        print(f"{len(failed)} rows failed and were skipped:")
        for tic_id, reason in failed[:20]:
            print(f"  TIC {tic_id}: {reason}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    return X, y, ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-cache phase-folded segments for a data split.")
    parser.add_argument("--csv", default="data/curated/train.csv", help="Path to a train/val/test CSV")
    parser.add_argument("--out", default="data/processed", help="Output directory for .npy arrays")
    args = parser.parse_args()
    pre_cache_pipeline(args.csv, args.out)
