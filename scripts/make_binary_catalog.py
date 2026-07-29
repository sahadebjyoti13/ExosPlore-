# scripts/make_binary_catalog.py
"""
Collapses the multi-class catalog.csv (from src/data_acquisition.py) into
a 2-class catalog: 0 = Planetary Transit, 1 = Not a Planet.

Why: after a real acquisition run, we ended up with real, verified data
for classes 0 (planets, 100 examples) and 1 (eclipsing binaries, 100
examples), but zero for stellar variability and noise, and 57
unverified "blend" guesses. Rather than ship a 5-class model held up by
two empty categories, v1 ships as Planet vs Not-a-Planet, built only
from the verified data.

Usage:
    python scripts/make_binary_catalog.py
    python scripts/make_binary_catalog.py --include_unverified
        (folds the needs_review rows into class 1 too — only do this
        after you've actually looked at them; see CONTRIBUTING.md)
"""

import argparse
import os
import pandas as pd


def make_binary_catalog(catalog_path="data/curated/catalog.csv",
                          out_path="data/curated/catalog_binary.csv",
                          include_unverified=False):
    df = pd.read_csv(catalog_path)

    if not include_unverified and "needs_review" in df.columns:
        before = len(df)
        df = df[~df["needs_review"].astype(bool)].copy()
        print(f"Dropped {before - len(df)} unverified rows (pass --include_unverified to keep them).")

    df["label"] = df["label"].apply(lambda x: 0 if int(x) == 0 else 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nBinary catalog written to {out_path}: {len(df)} rows")
    print(df["label"].value_counts().rename({0: "Planetary Transit", 1: "Not a Planet"}))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/curated/catalog.csv")
    parser.add_argument("--out", default="data/curated/catalog_binary.csv")
    parser.add_argument("--include_unverified", action="store_true")
    args = parser.parse_args()
    make_binary_catalog(args.catalog, args.out, args.include_unverified)
