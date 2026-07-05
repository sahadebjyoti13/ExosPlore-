# src/preprocess.py

import os
import numpy as np
import lightkurve as lk


def preprocess_lightcurve(fits_path: str, window_length: int = 401):
    """
    Module 1: Physics-Aware Preprocessing
    Loads a local TESS .fits file, applies quality bitmasks, removes outliers,
    normalizes the flux, and flattens/detrends stellar variability.
    """
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS file not found at: {fits_path}")

    # 1. Load via lightkurve with standard science quality bitmask
    lc_raw_data = lk.read(fits_path, quality_bitmask=175)

    # Safely pull properties. Lightkurve objects map PDCSAP_FLUX directly to .flux
    # when loading standard TESS lightcurves.
    time_raw = lc_raw_data.time.value

    # Fail-safe check for explicit PDCSAP column attributes if mapping variations occur
    if hasattr(lc_raw_data, "pdcsap_flux") and lc_raw_data.pdcsap_flux is not None:
        flux_raw = lc_raw_data.pdcsap_flux.value
        flux_err_raw = lc_raw_data.pdcsap_flux_err.value
    else:
        # Fall back to default primary flux columns (which are PDCSAP in standardized TESS files)
        flux_raw = lc_raw_data.flux.value
        flux_err_raw = lc_raw_data.flux_err.value

    # Remove any NaN values inherent in raw telemetry before processing
    nan_mask = ~np.isnan(time_raw) & ~np.isnan(flux_raw)
    time_clean = time_raw[nan_mask]
    flux_clean = flux_raw[nan_mask]
    flux_err_clean = flux_err_raw[nan_mask]

    # Gracefully handle missing or entirely NaN flux errors
    if len(flux_err_clean) == 0 or np.all(np.isnan(flux_err_clean)):
        median_err = np.median(flux_clean) * 0.001
        flux_err_clean = np.full_like(flux_clean, fill_value=median_err)
    else:
        flux_err_clean[np.isnan(flux_err_clean)] = np.nanmedian(flux_err_clean)

    # Reconstruct a clean LightCurve object for downstream manipulations
    lc_clean = lk.LightCurve(time=time_clean, flux=flux_clean, flux_err=flux_err_clean)

    # 2. Sigma-clip outliers (sigma=5 to eliminate extreme spikes safely)
    lc_clipped = lc_clean.remove_outliers(sigma=5)

    # 3. Normalize flux so median = 1.0
    lc_norm = lc_clipped.normalize()

    # 4. Flatten/detrend using Savitzky-Golay filter to remove long-term stellar variations
    if window_length % 2 == 0:
        window_length += 1

    lc_flat = lc_norm.flatten(window_length=window_length)

    # Extract output numpy arrays
    time = lc_flat.time.value
    flux_flat = lc_flat.flux.value
    flux_err_flat = lc_flat.flux_err.value

    return time, flux_flat, flux_err_flat, lc_norm, lc_flat


if __name__ == "__main__":
    print("Testing Preprocessing Module standalone...")
    test_file = "wasp18_test.fits"

    if os.path.exists(test_file):
        try:
            t, f, ferr, raw_obj, flat_obj = preprocess_lightcurve(test_file)
            print("\n--- Preprocessing Successful! ---")
            print(f"Processed Array Shapes - Time: {t.shape}, Flux: {f.shape}")
            print(f"Flux Median (Normalized): {np.median(f):.4f}")
            print(f"Flux Standard Deviation (Noise Floor): {np.std(f):.6f}")
        except Exception as e:
            print(f"Error during preprocessing runtime validation:\n{e}")
    else:
        print(f"Test file {test_file} not found. Run script again to test.")
