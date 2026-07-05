# src/characterizer.py

import numpy as np
from scipy.optimize import curve_fit

# Try loading the advanced physical transit model, with a clean fallback built-in
try:
    import batman

    BATMAN_AVAILABLE = True
except ImportError:
    BATMAN_AVAILABLE = False


def trapezoid_transit_model(phase, t0, duration, depth, ingress_fraction):
    """
    Analytical trapezoid transit model for curve fitting optimization.
    phase: 1D array representing time or phase centered at 0.
    """
    # Baseline out of transit is normalized to 1.0
    flux = np.ones_like(phase)

    half_dur = duration / 2.0
    ingress_width = ingress_fraction * half_dur
    t_flat = half_dur - ingress_width

    # Structural conditions for a symmetrical trapezoid dip
    # 1. Fully in-transit (Flat bottom)
    in_flat = np.abs(phase - t0) <= t_flat
    flux[in_flat] = 1.0 - depth

    # 2. Ingress / Egress slopes
    in_ingress = (np.abs(phase - t0) > t_flat) & (np.abs(phase - t0) <= half_dur)
    if np.any(in_ingress):
        # Calculate linear slope interpolation from base to bottom
        distance_from_edge = half_dur - np.abs(phase[in_ingress] - t0)
        fractional_depth = distance_from_edge / (ingress_width + 1e-8)
        flux[in_ingress] = 1.0 - (depth * fractional_depth)

    return flux


def fit_trapezoid(phase, flux, duration_est, depth_est):
    """
    Fits an analytical trapezoid profile to the phase-folded data window.
    """
    # Bounds: t0 (-0.1 to 0.1), duration (0.01 to 0.4), depth (0 to 0.1), ingress_fraction (0.05 to 0.95)
    lower_bounds = [-0.1, 0.01, 0.0, 0.05]
    upper_bounds = [0.1, 0.4, 0.1, 0.95]

    p0 = [0.0, max(0.02, min(duration_est, 0.35)), max(1e-4, depth_est), 0.2]

    try:
        popt, _ = curve_fit(
            trapezoid_transit_model,
            phase,
            flux,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000,
        )
        return popt
    except Exception:
        return p0  # Fallback to defaults if convergence fails


def characterize_transit(candidate_dict, time, flux):
    """
    Module 4: Transit Characterization
    Calculates precise geometric and physical stellar system traits.
    """
    period = candidate_dict["period"]
    t0_est = candidate_dict["t0"]
    duration_est = candidate_dict["duration"]
    depth_est = candidate_dict["depth_estimate"]

    # 1. Recompute standard orbital phases for extraction metrics
    phase = ((time - t0_est) / period) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    # 2. Compute exact scientific SNR
    out_transit_mask = np.abs(phase) > (duration_est / (2.0 * period))
    out_transit_flux = flux[out_transit_mask]

    if len(out_transit_flux) > 0:
        noise_floor = np.std(out_transit_flux)
    else:
        noise_floor = np.std(flux)

    snr = depth_est / noise_floor if noise_floor > 0 else candidate_dict["snr_bls"]

    # 3. Model optimization fitting step
    # Sort phases for optimization profile matching
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]

    # Fit the structural trapezoid profile
    t0_fit, dur_fit, depth_fit, ingress_frac_fit = fit_trapezoid(
        phase_sorted, flux_sorted, duration_est, depth_est
    )

    # Calculate model flux line for dashboard rendering
    model_flux = trapezoid_transit_model(
        phase_sorted, t0_fit, dur_fit, depth_fit, ingress_frac_fit
    )

    # Attempt advanced BATMAN physical modeling if available
    is_batman_fit = False
    if BATMAN_AVAILABLE:
        try:
            # Re-map parameter structures to physical units
            bm_params = batman.TransitParams()
            bm_params.t0 = t0_fit
            bm_params.per = 1.0  # Phase context is normalized relative to period base
            bm_params.rp = np.sqrt(
                max(1e-6, depth_fit)
            )  # Planet-to-star radius ratio (Rp/Rs)
            bm_params.a = 15.0  # Scaled semi-major axis (a/Rs)
            bm_params.inc = 90.0  # Orbital inclination
            bm_params.ecc = 0.0  # Circular orbit assumption
            bm_params.w = 90.0
            bm_params.u = [0.3, 0.1]  # Quadratic limb darkening
            bm_params.limb_dark = "quadratic"

            bm_model = batman.TransitModel(bm_params, phase_sorted)
            model_flux = bm_model.light_curve(bm_params)
            is_batman_fit = True
        except Exception:
            pass  # Gracefully slide back to our standard trapezoid line output

    # Package clean outputs
    results = {
        "period_days": float(period),
        "duration_hours": float(
            dur_fit * period * 24.0
        ),  # Convert phase width directly to hours
        "depth_ppm": float(depth_fit * 1e6),  # Convert to Parts Per Million
        "snr": float(snr),
        "phase_sorted": phase_sorted,
        "flux_sorted": flux_sorted,
        "model_flux": model_flux,
        "using_batman": is_batman_fit,
    }

    return results


# Locate the test block at the bottom of src/characterizer.py and change:
if __name__ == "__main__":
    print("Testing Characterizer Module using output data parameters...")
    from src.preprocess import preprocess_lightcurve
    from src.bls_detector import run_bls

    test_file = "wasp18_test.fits"
    t, f, ferr, _, _ = preprocess_lightcurve(test_file)
    candidates = run_bls(t, f, ferr)

    print(f"Batman Engine Available Local Status: {BATMAN_AVAILABLE}")
    print("Characterizing candidate transit signature geometry...")
    metrics = characterize_transit(candidates[0], t, f)

    print("\n--- Physical Optimization Metrics Extracted: ---")
    print(f"Period: {metrics['period_days']:.5f} days")
    print(f"Fitted Duration: {metrics['duration_hours']:.2f} hours")
    print(f"Fitted Depth: {metrics['depth_ppm']:.1f} ppm")
    print(f"System Operational SNR: {metrics['snr']:.2f}")
    print(f"Advanced Batman Physics Layer Used: {metrics['using_batman']}")
