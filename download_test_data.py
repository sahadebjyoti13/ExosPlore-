# save as download_test_data.py and run: python download_test_data.py
import lightkurve as lk
import os

os.makedirs("data/raw", exist_ok=True)

targets = [
    ("TIC 100100827", "wasp18"),
    ("TIC 420814525", "hd209458"),
    ("TIC 59873330", "toi132"),
]

for tic, name in targets:
    print(f"Downloading {name}...")
    sr = lk.search_lightcurve(tic, mission="TESS", exptime=120)
    if len(sr) > 0:
        lc = sr[0].download()
        lc.to_fits(f"data/raw/{name}_test.fits", overwrite=True)
        print(f"  Saved: data/raw/{name}_test.fits")
    else:
        print(f"  Not found.")
