# src/bls_detector.py

import numpy as np
from astropy.timeseries import BoxLeastSquares
import astropy.units as u


def phase_fold_and_resample(time, flux, period, t0, n_points=200):
    """
    Folds the light curve at a given period and t0, centers the transit at 0.0,
    and resamples it cleanly to exactly `n_points` using linear interpolation.
    """
    # 1. Phase fold: phases range from -0.5 to +0.5
    phase = ((time - t0) / period) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    # Sort by phase so np.interp works correctly
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]

    # 2. Resample to exactly n_points evenly spaced across [-0.5, 0.5]
    resampled_phases = np.linspace(-0.5, 0.5, n_points)
    resampled_flux = np.interp(resampled_phases, phase_sorted, flux_sorted)

    return resampled_flux


def run_bls(time, flux, flux_err):
    """
    Module 2: Candidate Event Detection using Box Least Squares
    Runs a rigorous grid search to extract periodic transit signals.

    Returns:
    --------
    candidates : list of dict
        Top 3 candidate detections sorted descending by BLS power.
    """
    # Initialize the astropy BLS engine
    bls = BoxLeastSquares(time * u.day, flux, dy=flux_err)

    # Define exact grid as per ISRO requirements
    periods = np.linspace(0.5, 15.0, 10000)
    durations = np.array([0.05, 0.1, 0.15, 0.2])  # fraction of period

    # Run power spectrum search
    bls_power = bls.power(periods, durations)

    candidates = []

    # Find top 3 unique peaks by locating local maxima in the spectrum
    sorted_indices = np.argsort(bls_power.power)[::-1]

    # Ensure items inside bls_power arrays are treated as raw values
    power_arr = np.array(bls_power.power)
    period_arr = np.array(bls_power.period)
    duration_arr = np.array(bls_power.duration)
    t0_arr = np.array(bls_power.transit_time)
    depth_arr = np.array(bls_power.depth)

    seen_periods = []
    for idx in sorted_indices:
        p_curr = period_arr[idx]

        # Avoid duplicate counting of identical periods/harmonics within a 5% window
        if any(np.isclose(p_curr, p_seen, rtol=0.05) for p_seen in seen_periods):
            continue

        seen_periods.append(p_curr)

        p = period_arr[idx]
        d = duration_arr[idx]
        t0 = t0_arr[idx]
        pwr = power_arr[idx]
        depth = depth_arr[idx]

        # Estimate astronomical Signal-to-Noise Ratio (SNR) via standard BLS stats
        phase = ((time - t0) / p) % 1.0
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        out_transit = np.abs(phase) > (d / (2.0 * p))

        noise_floor = np.std(flux[out_transit]) if np.any(out_transit) else np.std(flux)
        snr_bls = depth / noise_floor if noise_floor > 0 else 0.0

        # Extract the mandatory 200-point phase-folded window
        folded_flux_200 = phase_fold_and_resample(time, flux, p, t0, n_points=200)

        candidates.append(
            {
                "period": p,
                "t0": t0,
                "duration": d,
                "depth_estimate": depth,
                "bls_power": pwr,
                "snr_bls": snr_bls,
                "phase_folded_flux_200": folded_flux_200,
                "bls_power_array": power_arr,
                "period_grid": periods,
            }
        )

        if len(candidates) >= 3:
            break

    return candidates


if __name__ == "__main__":
    print("Testing BLS Detection Module using output from Module 1...")
    from preprocess import preprocess_lightcurve

    test_file = "wasp18_test.fits"
    t, f, ferr, _, _ = preprocess_lightcurve(test_file)

    print("Running BLS Engine (This grid scan takes a few seconds)...")
    candidates = run_bls(t, f, ferr)

    print("\n--- BLS Search Complete! Top Candidate Details: ---")
    top = candidates[0]
    print(f"Best Period: {top['period']:.5f} days (Expected for WASP-18b: ~0.94 days)")
    print(f"Transit Center (t0): {top['t0']:.4f}")
    print(f"Transit Duration: {top['duration'] * 24.0:.2f} hours")
    print(f"Transit Depth: {top['depth_estimate'] * 1e6:.1f} ppm")
    print(f"Calculated BLS SNR: {top['snr_bls']:.2f}")
    print(f"Resampled Array Shape: {top['phase_folded_flux_200'].shape}")
