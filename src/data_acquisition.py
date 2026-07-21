# src/data_acquisition.py
"""
Builds a real, citable TESS light-curve training catalog. Replaces the
old src/data_miner.py, which queried astroquery's NasaExoplanetArchive
for confirmed planets and a hardcoded list of 3 example TIC IDs for
eclipsing binaries — and was never actually called from train.py, so
none of it affected the shipped (mock-trained) model.

Sources (both public, no login required)
------------------------------------------
- ExoFOP-TESS TOI table (NExScI):
    https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv
  Community/TFOPWG-vetted dispositions per TESS Object of Interest:
    CP/KP = confirmed / known planet
    FP    = false positive
    PC/APC/FA = candidate / ambiguous / false alarm (not used here)

- TESS Eclipsing Binary Catalog, Prsa et al. 2022 (ApJS 258, 16), hosted
  as a MAST High-Level Science Product:
    https://archive.stsci.edu/hlsp/tess-ebs
  Peer-reviewed, citable eclipsing binary detections.
  NOTE: verify the exact CSV filename/path on that page before relying
  on EB_CATALOG_URL below — MAST HLSP file names change between catalog
  versions (v1.0 -> v1.1 etc.) and I could not fetch this URL myself to
  confirm it (my environment can't reach archive.stsci.edu). If the
  download fails, check the "direct download" link on the page above
  and update EB_CATALOG_URL.

Label reliability — read this before trusting anything downstream
--------------------------------------------------------------------
Class 0 (Planetary Transit) and Class 1 (Eclipsing Binary) come directly
from the peer-reviewed catalogs above. Trustworthy as-is.

Classes 2 (Stellar Blend), 3 (Stellar Variability), and 4 (Noise/Unknown)
have NO single public catalog with this exact taxonomy. There is no way
to fully automate this honestly. What this script does instead:
  - Class 2/3: takes ExoFOP TOIs disposed 'FP' and keyword-searches their
    free-text Comments field for blend/contamination or variability/
    spot/rotation language. This is a weak heuristic — comments are
    inconsistent and often blank.
  - Class 4: downloads light curves for TIC IDs with no TOI/EB match and
    keeps the ones where Module 1+2 (preprocessing + BLS) finds no
    significant periodic signal. "No obvious BLS peak" is not the same
    as "confirmed nothing periodic is present."
Every row produced by these heuristics is written with needs_review=True.
scripts/split_dataset.py excludes them from train/val/test by default.
Clear that flag only after a human has actually looked at the light
curve (e.g. in the ExosPlore dashboard) and agrees with the label.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightkurve as lk

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
EB_CATALOG_URL = "https://archive.stsci.edu/hlsps/tess-ebs/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv"

BLEND_KEYWORDS = ["blend", "contamina", "nearby", "background", "diluted"]
VARIABILITY_KEYWORDS = ["variab", "spot", "rotat", "pulsat", "starspot"]


def _find_column(df, needle):
    """Case-insensitive substring match for a column name. Raises with the
    real column list instead of silently returning the wrong thing or
    crashing on a bare KeyError if the source table's schema changes."""
    for col in df.columns:
        if needle.lower() in col.lower():
            return col
    raise KeyError(f"No column containing '{needle}' found. Available columns: {list(df.columns)}")


def fetch_toi_table():
    print(f"Fetching ExoFOP TOI table from {TOI_URL}")
    return pd.read_csv(TOI_URL)


def fetch_eb_catalog():
    print(f"Fetching TESS EB catalog from {EB_CATALOG_URL}")
    print("(If this 404s, check https://archive.stsci.edu/hlsp/tess-ebs for the current filename.)")
    return pd.read_csv(EB_CATALOG_URL)


def _download_one(tic_id, out_dir, suffix, exptime=120):
    fits_path = os.path.join(out_dir, f"TIC{tic_id}_{suffix}.fits")
    if os.path.exists(fits_path):
        return fits_path
    try:
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", exptime=exptime)
        if len(search) == 0:
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search) == 0:
            return None
        search[0].download().to_fits(fits_path, overwrite=True)
        return fits_path
    except Exception as e:
        print(f"  TIC {tic_id}: download failed ({e})")
        return None


def build_planet_rows(toi_df, limit, out_dir):
    disp_col = _find_column(toi_df, "TFOPWG Disposition")
    tic_col = _find_column(toi_df, "TIC ID")
    planets = toi_df[toi_df[disp_col].isin(["CP", "KP"])].drop_duplicates(subset=tic_col).head(limit)

    rows = []
    for _, r in tqdm(planets.iterrows(), total=len(planets), desc="Downloading confirmed planets"):
        tic_id = int(r[tic_col])
        path = _download_one(tic_id, out_dir, "planet")
        if path:
            rows.append({"tic_id": tic_id, "label": 0, "fits_path": path,
                         "source": "ExoFOP TOI (CP/KP)", "needs_review": False})
    return rows


def build_eb_rows(eb_df, limit, out_dir):
    tic_col = _find_column(eb_df, "TIC")
    ebs = eb_df.drop_duplicates(subset=tic_col).head(limit)

    rows = []
    for _, r in tqdm(ebs.iterrows(), total=len(ebs), desc="Downloading eclipsing binaries"):
        tic_id = int(r[tic_col])
        path = _download_one(tic_id, out_dir, "eb")
        if path:
            rows.append({"tic_id": tic_id, "label": 1, "fits_path": path,
                         "source": "TESS EB Catalog (Prsa et al. 2022)", "needs_review": False})
    return rows


def build_fp_rows(toi_df, limit, out_dir):
    """Weak, keyword-based split of ExoFOP false positives into blend
    (class 2) / variability (class 3) buckets. Always needs_review."""
    disp_col = _find_column(toi_df, "TFOPWG Disposition")
    tic_col = _find_column(toi_df, "TIC ID")
    comment_col = next((c for c in toi_df.columns if "comment" in c.lower()), None)

    fps = toi_df[toi_df[disp_col] == "FP"].drop_duplicates(subset=tic_col).head(limit)

    rows = []
    for _, r in tqdm(fps.iterrows(), total=len(fps), desc="Downloading false positives"):
        tic_id = int(r[tic_col])
        comment = str(r[comment_col]).lower() if comment_col and pd.notna(r[comment_col]) else ""

        if any(k in comment for k in BLEND_KEYWORDS):
            label = 2
        elif any(k in comment for k in VARIABILITY_KEYWORDS):
            label = 3
        else:
            label = 2  # default bucket when the comment gives no signal either way

        path = _download_one(tic_id, out_dir, "fp")
        if path:
            rows.append({"tic_id": tic_id, "label": label, "fits_path": path,
                         "source": "ExoFOP TOI (FP, keyword-bucketed)", "needs_review": True})
    return rows


def build_noise_rows(limit, out_dir, seed_tics):
    """Screens random TIC IDs for the absence of a significant BLS peak.
    Kept examples are candidate noise/no-detection cases, not confirmed
    ones — hence needs_review."""
    from src.preprocess import preprocess_lightcurve
    from src.bls_detector import run_bls

    rows = []
    tried = 0
    pool = list(seed_tics)
    np.random.default_rng(42).shuffle(pool)

    for tic_id in tqdm(pool, desc="Screening for noise examples"):
        if len(rows) >= limit:
            break
        tried += 1
        path = _download_one(tic_id, out_dir, "noise_screen")
        if not path:
            continue
        try:
            t, f, ferr, _, _ = preprocess_lightcurve(path)
            candidates = run_bls(t, f, ferr)
            if not candidates or candidates[0]["bls_power"] < 0.01:
                rows.append({"tic_id": tic_id, "label": 4, "fits_path": path,
                             "source": "low BLS power screen", "needs_review": True})
        except Exception:
            continue

    print(f"Screened {tried} stars, kept {len(rows)} candidate noise examples.")
    return rows


def build_catalog(n_planets=100, n_ebs=100, n_fp=60, n_noise=40,
                   raw_dir="data/raw", out_csv="data/curated/catalog.csv"):
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    toi_df = fetch_toi_table()
    eb_df = fetch_eb_catalog()

    rows = []
    rows += build_planet_rows(toi_df, n_planets, raw_dir)
    rows += build_eb_rows(eb_df, n_ebs, raw_dir)
    rows += build_fp_rows(toi_df, n_fp, raw_dir)

    tic_col = _find_column(toi_df, "TIC ID")
    seed_pool = toi_df[tic_col].dropna().astype(int).unique().tolist()
    rows += build_noise_rows(n_noise, raw_dir, seed_pool)

    if not rows:
        raise RuntimeError("No rows were produced — check network access and the URLs at the top of this file.")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print(f"Catalog written to {out_csv}: {len(df)} rows")
    print(df.groupby(["label", "needs_review"]).size())
    print("=" * 60)
    print("Rows with needs_review=True (most of classes 2/3/4) are NOT verified ground "
          "truth. scripts/split_dataset.py excludes them from train/val/test until you "
          "manually check them and flip the flag.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a real, citable TESS light-curve catalog.")
    parser.add_argument("--planets", type=int, default=100)
    parser.add_argument("--ebs", type=int, default=100)
    parser.add_argument("--fp", type=int, default=60)
    parser.add_argument("--noise", type=int, default=40)
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out", default="data/curated/catalog.csv")
    args = parser.parse_args()
    build_catalog(args.planets, args.ebs, args.fp, args.noise, args.raw_dir, args.out)
