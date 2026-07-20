# src/data_miner.py

import os
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import lightkurve as lk
from tqdm import tqdm


def build_custom_dataset(download_limit=50):
    print("🔭 Initiating NASA Exoplanet Archive API Connection...")
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/curated", exist_ok=True)

    dataset_rows = []

    # ---------------------------------------------------------
    # 1. FETCH CONFIRMED PLANETS (Class 0)
    # ---------------------------------------------------------
    print("📊 Querying Confirmed TESS Planets...")
    # Using astroquery to search the Planetary Systems table for TESS discoveries
    tess_planets = NasaExoplanetArchive.query_criteria(
        table="pscomppars", select="hostname", where="disc_facility like '%TESS%'"
    )

    # Convert astropy table to a list of unique host star names
    planet_hosts = list(set(tess_planets["hostname"]))[:download_limit]

    print(f"📥 Downloading {len(planet_hosts)} Planetary Light Curves...")
    for star in tqdm(planet_hosts, desc="Downloading Planets"):
        try:
            # Search MAST archive and download short cadence data
            search = lk.search_lightcurve(star, mission="TESS", exptime=120)
            if len(search) > 0:
                fits_path = f"data/raw/{star.replace(' ', '_')}_planet.fits"
                search[0].download().to_fits(fits_path, overwrite=True)

                # Append to our CSV structure: Label 0 = Planet
                dataset_rows.append(
                    {"tic_id": star, "label": 0, "fits_path": fits_path}
                )
        except Exception as e:
            continue

    # ---------------------------------------------------------
    # 2. FETCH KNOWN ECLIPSING BINARIES (Class 1)
    # ---------------------------------------------------------
    # To train the network on false positives, we download known binary stars
    # (In a full scale run, you would query the TESS Eclipsing Binary Catalog)
    known_binaries = ["TIC 272828941", "TIC 11624633", "TIC 55525572"]  # Example IDs

    print(f"📥 Downloading Eclipsing Binaries for False Positive Training...")
    for binary in tqdm(known_binaries, desc="Downloading Binaries"):
        try:
            search = lk.search_lightcurve(binary, mission="TESS")
            if len(search) > 0:
                fits_path = f"data/raw/{binary.replace(' ', '')}_binary.fits"
                search[0].download().to_fits(fits_path, overwrite=True)

                # Append to our CSV structure: Label 1 = Eclipsing Binary
                dataset_rows.append(
                    {"tic_id": binary, "label": 1, "fits_path": fits_path}
                )
        except Exception as e:
            continue

    # ---------------------------------------------------------
    # 3. COMPILE AND SAVE THE REGISTRY
    # ---------------------------------------------------------
    df = pd.DataFrame(dataset_rows)
    csv_path = "data/curated/train.csv"
    df.to_csv(csv_path, index=False)

    print("==================================================")
    print(f"✅ DATASET COMPILED SUCCESSFULLY!")
    print(f"📦 Total Files Downloaded: {len(df)}")
    print(f"💾 Registry Saved To: {csv_path}")
    print("💡 You are now ready to run python -m src.cache_dataset")
    print("==================================================")


if __name__ == "__main__":
    build_custom_dataset()
