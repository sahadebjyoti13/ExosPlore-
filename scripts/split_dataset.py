# scripts/split_dataset.py
"""
Splits data/curated/catalog.csv (built by src/data_acquisition.py) into
train/val/test CSVs, stratified by label, with a fixed seed.

Run this once after building or updating the catalog. train.py and
evaluate.py read the resulting train.csv/val.csv/test.csv directly —
they never touch catalog.csv or each other's split, so the test set
stays genuinely held-out.

Rows flagged needs_review=True (see src/data_acquisition.py — this
covers most of classes 2/3/4, which don't have a single clean public
catalog) are excluded from the split by default and written to
needs_review.csv instead. Move them into catalog.csv with
needs_review=False once a human has actually looked at the light curve
and confirmed the label.
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_catalog(catalog_path, out_dir, val_size=0.15, test_size=0.15, seed=42, exclude_needs_review=True):
    df = pd.read_csv(catalog_path)
    os.makedirs(out_dir, exist_ok=True)

    if "needs_review" in df.columns and exclude_needs_review:
        flagged = df["needs_review"].astype(bool)
        if flagged.any():
            print(f"Excluding {flagged.sum()} rows flagged needs_review=True — "
                  f"see {out_dir}/needs_review.csv. These are mostly automatically "
                  f"bucketed class 2/3/4 examples that haven't been manually verified.")
            df[flagged].to_csv(os.path.join(out_dir, "needs_review.csv"), index=False)
        df = df[~flagged].copy()

    if df.empty:
        raise RuntimeError(
            "No rows left to split after excluding needs_review rows. "
            "You need at least some manually-verified examples per class."
        )

    counts = df["label"].value_counts()
    print(f"Usable rows by class:\n{counts}")
    if (counts < 3).any():
        print("WARNING: at least one class has fewer than 3 examples — a train/val/test "
              "split can't give that class meaningful coverage in all three splits yet.")

    try:
        train_df, temp_df = train_test_split(
            df, test_size=val_size + test_size, random_state=seed, stratify=df["label"]
        )
        rel_test = test_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df, test_size=rel_test, random_state=seed, stratify=temp_df["label"]
        )
    except ValueError as e:
        print(f"Stratified split failed ({e}) — falling back to a non-stratified split. "
              f"Treat metrics from this run as provisional until you have more data per class.")
        train_df, temp_df = train_test_split(df, test_size=val_size + test_size, random_state=seed)
        rel_test = test_size / (val_size + test_size)
        val_df, test_df = train_test_split(temp_df, test_size=rel_test, random_state=seed)

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"\ntrain={len(train_df)}  val={len(val_df)}  test={len(test_df)}  -> {out_dir}/")
    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/curated/catalog.csv")
    parser.add_argument("--out_dir", default="data/curated")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_needs_review", action="store_true",
                         help="Include unverified rows in the split anyway. Not recommended.")
    args = parser.parse_args()
    split_catalog(
        args.catalog, args.out_dir, args.val_size, args.test_size, args.seed,
        exclude_needs_review=not args.include_needs_review,
    )
