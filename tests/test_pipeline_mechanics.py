# tests/test_pipeline_mechanics.py
"""
These tests validate that the pipeline's *mechanics* are correct —
shapes, phase-folding math, curve fitting convergence — using synthetic
injection-recovery data (a known fake transit injected into synthetic
noise). This is standard practice in transit-search software (see e.g.
the injection-recovery tests in Astropy's BoxLeastSquares test suite)
and is fundamentally different from the mock training data this repo
used to ship: these tests check that the code does what it claims to
do; they say nothing about how well the trained model classifies real
astrophysical signals, and are not used anywhere to report accuracy.

No network access or real TESS data required, so these can run in CI.
"""

import numpy as np
import pytest

from src.bls_detector import phase_fold_and_resample, run_bls
from src.classifier import TransitCNN
from src.characterizer import trapezoid_transit_model, fit_trapezoid


def make_synthetic_transit(period=3.0, depth=0.01, duration_days=0.1, n_points=2000, seed=0):
    """A box-shaped dip on top of flat, noisy flux — enough to exercise
    the BLS search and characterization fit, not a realistic light curve."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 27, n_points)  # ~one TESS sector
    flux = np.ones_like(time) + rng.normal(0, 0.001, n_points)

    phase = (time % period) / period
    in_transit = np.abs(phase - 0.5) < (duration_days / period / 2)
    flux[in_transit] -= depth

    flux_err = np.full_like(flux, 0.001)
    return time, flux, flux_err, period


class TestPhaseFold:
    def test_output_length_is_exact(self):
        time = np.linspace(0, 10, 500)
        flux = np.sin(time)
        result = phase_fold_and_resample(time, flux, period=2.0, t0=0.0, n_points=200)
        assert result.shape == (200,)

    def test_phase_zero_centers_the_transit(self):
        time, flux, flux_err, period = make_synthetic_transit()
        folded = phase_fold_and_resample(time, flux, period=period, t0=period / 2, n_points=200)
        # the injected dip should show up as a minimum near the center of the array
        center = folded[90:110]
        edges = np.concatenate([folded[:20], folded[-20:]])
        assert center.mean() < edges.mean()


class TestBLSDetection:
    def test_recovers_injected_period(self):
        time, flux, flux_err, true_period = make_synthetic_transit(period=3.0)
        candidates = run_bls(time, flux, flux_err)
        assert len(candidates) > 0
        best = candidates[0]
        assert abs(best["period"] - true_period) < 0.05, (
            f"Recovered period {best['period']:.3f} too far from injected {true_period}"
        )

    def test_returns_at_most_three_candidates(self):
        time, flux, flux_err, _ = make_synthetic_transit()
        candidates = run_bls(time, flux, flux_err)
        assert 0 < len(candidates) <= 3

    def test_each_candidate_has_200_point_segment(self):
        time, flux, flux_err, _ = make_synthetic_transit()
        candidates = run_bls(time, flux, flux_err)
        for c in candidates:
            assert c["phase_folded_flux_200"].shape == (200,)


class TestClassifierShapes:
    def test_forward_pass_shape(self):
        model = TransitCNN()
        x = np.random.randn(4, 1, 200).astype(np.float32)
        import torch
        out = model(torch.tensor(x))
        assert out.shape == (4, 5)

    def test_silently_accepts_wrong_input_length(self):
        """AdaptiveAvgPool1d(32) makes the network shape-compatible with
        ANY input length, not just 200 — it will NOT raise or crash on a
        150-point segment. That's the opposite of what the README's
        'Do Not Touch' table implies ("Changing this breaks the model").
        It's arguably worse: silently running on out-of-distribution
        input instead of failing loudly. Documenting the real behavior
        here so it isn't relied on as a safety net elsewhere."""
        model = TransitCNN()
        import torch
        out = model(torch.randn(1, 1, 150))
        assert out.shape == (1, 5)  # runs fine; the *meaning* of the output is undefined


class TestCharacterizer:
    def test_trapezoid_fit_recovers_depth(self):
        phase = np.linspace(-0.5, 0.5, 200)
        true_depth = 0.02
        clean_flux = trapezoid_transit_model(phase, t0=0.0, duration=0.1, depth=true_depth, ingress_fraction=0.2)
        noisy_flux = clean_flux + np.random.default_rng(1).normal(0, 0.0005, 200)

        popt = fit_trapezoid(phase, noisy_flux, duration_est=0.1, depth_est=0.015)
        fitted_depth = popt[2]
        assert abs(fitted_depth - true_depth) < 0.01
