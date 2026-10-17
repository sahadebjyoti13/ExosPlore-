"""
ExosPlore — Test Data Downloader
Downloads verified TESS light curves for pipeline calibration.
Run this BEFORE the hackathon on reliable internet.
"""

import lightkurve as lk
import os

os.makedirs("data/raw", exist_ok=True)

targets = [
    ("TIC 100100827", "wasp18"),    # WASP-18b  — very deep transit, easiest to detect
    ("TIC 420814525", "hd209458"),  # HD 209458b — classic benchmark exoplanet
    ("TIC 59873330",  "toi132"),    # TOI-132b   — TESS discovery target
]

for tic, name in targets:
    print(f"Searching {tic} ({name})...")
    try:
        sr = lk.search_lightcurve(tic, mission="TESS", exptime=120)
        if len(sr) > 0:
            lc = sr[0].download()
            path = f"data/raw/{name}_test.fits"
            lc.to_fits(path, overwrite=True)
            print(f"  ✓ Saved: {path}")
        else:
            print(f"  ✗ No 2-min cadence data found for {tic}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\nDone. Upload any of these .fits files into the dashboard to verify the pipeline.")
