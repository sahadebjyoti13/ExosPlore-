# src/pipeline.py

import os
import torch
import numpy as np

# Internal module imports
# Change this block at the top of src/pipeline.py:

from src.preprocess import preprocess_lightcurve
from src.bls_detector import run_bls
from src.classifier import TransitCNN, predict, CLASS_NAMES
from src.characterizer import characterize_transit


def run_pipeline(
    fits_path: str, model_path: str = "models/transit_cnn.pth", device: str = "cpu"
):
    """
    Master Pipeline Orchestration Script
    Chains Preprocessing, BLS Detection, Deep Learning Classification,
    and Transit Characterization into an end-to-end processing execution block.
    """
    pipeline_results = {
        "status": "Success",
        "fits_name": os.path.basename(fits_path),
        "is_planet": False,
        "predicted_class": "Noise/Unknown",
        "confidence": 0.0,
        "all_probabilities": {},
        "bls_candidates_count": 0,
        "top_candidate": None,
        "characterization": None,
        "plots": {},
    }

    try:
        # Step 1: Run Physics-Aware Preprocessing
        t, f, ferr, lc_norm, lc_flat = preprocess_lightcurve(fits_path)

        # Stash structural data for dashboard charting
        pipeline_results["plots"]["raw_time"] = lc_norm.time.value
        pipeline_results["plots"]["raw_flux"] = lc_norm.flux.value
        pipeline_results["plots"]["flat_time"] = t
        pipeline_results["plots"]["flat_flux"] = f

        # Step 2: Run Box Least Squares Candidate Search
        candidates = run_bls(t, f, ferr)
        pipeline_results["bls_candidates_count"] = len(candidates)

        if len(candidates) == 0:
            pipeline_results["status"] = "No periodic candidates isolated by BLS"
            return pipeline_results

        # Isolate top-ranked candidate profile
        top_cand = candidates[0]
        pipeline_results["top_candidate"] = top_cand

        # Step 3: Neural Network Classification Inference Pass
        model = TransitCNN()
        if os.path.exists(model_path):
            # Load trained weights if checkpoint exists
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            class_idx, class_name, confidence, probs = predict(
                model, top_cand["phase_folded_flux_200"], device=device
            )

            pipeline_results["predicted_class"] = class_name
            pipeline_results["confidence"] = confidence
            pipeline_results["all_probabilities"] = {
                CLASS_NAMES[i]: float(probs[i]) for i in range(5)
            }

            # Step 4: Conditional Physics Engine Characterization Routing
            if class_idx == 0:
                pipeline_results["is_planet"] = True
                # Pass top candidate parameters and flattened light curve values
                char_metrics = characterize_transit(top_cand, t, f)
                pipeline_results["characterization"] = char_metrics
            else:
                pipeline_results["is_planet"] = False
                print(
                    f"Candidate rejected from physical characterization. Target designated as: {class_name}"
                )
        else:
            # No trained checkpoint exists yet. Do NOT invent a classification —
            # a fabricated "85% Planetary Transit" result here would be actively
            # misleading to anyone using this for real research. Report the BLS
            # detection honestly and leave classification/characterization empty
            # until train.py has produced a real checkpoint from real data.
            pipeline_results["status"] = (
                f"No trained model checkpoint found at '{model_path}'. "
                f"BLS detection above is real, but class prediction, confidence, "
                f"and transit characterization are withheld — train a model on "
                f"real data first (see train.py / src/data_acquisition.py)."
            )
            pipeline_results["predicted_class"] = None
            pipeline_results["confidence"] = None
            pipeline_results["all_probabilities"] = {}
            pipeline_results["is_planet"] = None

    except Exception as e:
        pipeline_results["status"] = f"Failed inside pipeline runtime layer: {str(e)}"

    return pipeline_results


if __name__ == "__main__":
    print("Testing Global Pipeline Routing Engine...")
    test_file = "wasp18_test.fits"

    # Run pipeline on our workspace test file
    res = run_pipeline(test_file)

    print("\n--- Pipeline Execution Summary ---")
    print(f"Target File: {res['fits_name']}")
    print(f"Pipeline Process Status: {res['status']}")
    print(f"Isolated BLS Candidates: {res['bls_candidates_count']}")
    print(
        f"Predicted Target Category: {res['predicted_class']} ({res['confidence']*100:.2f}%)"
    )

    if res["is_planet"] and res["characterization"]:
        print("\n--- Extracted Planet Parameters ---")
        char = res["characterization"]
        print(f"Period: {char['period_days']:.5f} Days")
        print(f"Duration: {char['duration_hours']:.2f} Hours")
        print(f"Depth: {char['depth_ppm']:.1f} ppm")
        print(f"Signal Significance (SNR): {char['snr']:.2f}")
