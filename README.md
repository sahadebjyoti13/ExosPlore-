# 🌌 ExosPlore

### An open-source pipeline for TESS exoplanet transit detection

ExosPlore ingests raw TESS space telescope light curves and runs them
through physics-aware preprocessing, Box Least Squares transit
detection, a 1D-CNN + attention classifier, and transit
characterization — with an interactive Streamlit dashboard on top.

It started as a submission for the **Bharatiya Antariksh Hackathon
2026 (ISRO Problem Statement 7)** — we weren't selected for the
finale, so we're open-sourcing it instead. See
[Project History](#project-history) at the bottom.

## Status (read this before trusting any output)

| Component | Status |
| --- | --- |
| **Preprocessing** (`src/preprocess.py`) | ✅ Real. Quality-flag filtering, sigma-clipping, normalization, Savitzky-Golay detrending — verified against a known target (see [Calibration](#calibration-check) below). |
| **BLS transit detection** (`src/bls_detector.py`) | ✅ Real. Verified to recover known orbital parameters for WASP-18b. |
| **Transit characterization** (`src/characterizer.py`) | ✅ Real. Trapezoid fit (or `batman` physical model if installed) on the phase-folded signal. |
| **1D-CNN + Attention classifier** (`src/classifier.py`) | ⚠️ **Architecture only, currently untrained.** No checkpoint ships in this repo — see [Training on Real Data](#training-on-real-data). Until you train one, the dashboard reports BLS detections honestly and skips classification rather than guessing. |
| **Dashboard** (`app.py`) | ✅ Real, and now gates the classification UI correctly when no trained model is present. |

An earlier version of this repo shipped a `models/transit_cnn.pth`
checkpoint trained on 20 rows that were all the same light curve file
under 5 fabricated labels — it scored 20% accuracy (chance level) and
predicted "Planetary Transit" for every input, and the dashboard
silently faked an 85%-confidence result whenever no checkpoint was
present. Both have been removed. See `models/README.md` and
`CONTRIBUTING.md` for how to fix that for real.

---

## Pipeline Overview

```
Raw TESS .fits
      │
      ▼
┌─────────────────────────────────┐
│  MODULE 1 · Preprocessing       │  Quality flags → sigma-clip → normalize → Savitzky-Golay flatten
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  MODULE 2 · BLS Detection       │  Box Least Squares period search → phase-fold → 200-pt segment
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  MODULE 3 · CNN Classifier      │  1D-CNN + Multihead Attention → 5-class prediction + confidence
│                                  │  (needs a real trained checkpoint — see Status above)
└───────────────┬─────────────────┘
                │
        ┌───────┴────────┐
        │ Planetary       │ Other class
        │ Transit?        │ → logged as rejected
        ▼                 │
┌───────────────┐         │
│  MODULE 4     │         │
│  Characterize │         │
│  Period/Depth │         │
│  Duration/SNR │         │
└───────┬───────┘         │
        └────────┬────────┘
                 ▼
┌─────────────────────────────────┐
│  MODULE 5 · Streamlit Dashboard │  Interactive plots · Parameters · Confidence · Rejected examples
└─────────────────────────────────┘
```

**5 Output Classes:** `0` Planetary Transit · `1` Eclipsing Binary · `2` Stellar Blend · `3` Stellar Variability / Starspots · `4` Noise / Unknown

---

## Repository Structure

```
ExosPlore/
├── data/
│   ├── raw/                     # .fits files (git-ignored — large, and personal to your download)
│   └── curated/
│       ├── catalog.csv          # master labeled index built by src/data_acquisition.py
│       ├── needs_review.csv     # auto-labeled rows awaiting human verification
│       └── train.csv / val.csv / test.csv   # produced by scripts/split_dataset.py
├── models/
│   └── (no checkpoint shipped — see models/README.md)
├── src/
│   ├── preprocess.py            # Module 1 — physics-aware preprocessing
│   ├── bls_detector.py          # Module 2 — BLS period search & phase-folding
│   ├── classifier.py            # Module 3 — 1D-CNN + Attention model
│   ├── characterizer.py         # Module 4 — trapezoid/batman transit fitting
│   ├── pipeline.py              # Orchestrator — wires modules 1–4 end-to-end
│   ├── data_acquisition.py      # Builds a real labeled catalog from public archives
│   └── cache_dataset.py         # Pre-computes features so training doesn't redo BLS every epoch
├── scripts/
│   └── split_dataset.py         # Stratified train/val/test split, held out consistently
├── tests/
│   └── test_pipeline_mechanics.py  # Synthetic injection-recovery tests (no network needed)
├── train.py                     # Model training
├── evaluate.py                  # Held-out test-set evaluation: F1, confusion matrix
├── app.py                       # Module 5 — Streamlit dashboard
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── CONTRIBUTING.md
```

---

## Setup

### Prerequisites

- Python 3.10 – 3.14
- Windows: ensure **"Add Python to PATH"** was checked during installation

### Install Dependencies

```
pip install -r requirements.txt
```
> **Note on `batman-package`:** requires Microsoft C++ Build Tools on
> Windows. It's optional — the pipeline falls back to a trapezoid model
> if `batman` isn't installed. To install it: get the
> [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> (select "Desktop development with C++"), then `pip install batman-package`.

### Verify the Installation

```
python -c "
import lightkurve, astropy, numpy, scipy, pandas, torch, sklearn, matplotlib, plotly, streamlit, shap
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
try:
    import batman; print('Batman: OK')
except ImportError:
    print('Batman: not installed — trapezoid fallback active')
"
```

---

## Training on Real Data

There is no shortcut here, and no mock-data fallback in this repo
anymore. Three steps:

### 1. Build a real, citable catalog

```
python -m src.data_acquisition
```

This pulls confirmed planets and false positives from the
[ExoFOP-TESS TOI table](https://exofop.ipac.caltech.edu/tess/) and
eclipsing binaries from the
[TESS EB Catalog](https://archive.stsci.edu/hlsp/tess-ebs) (Prša et al.
2022) — both public, peer-reviewed sources. **Classes 0 and 1 are
trustworthy as downloaded.** Classes 2–4 (blend / variability / noise)
don't have one clean public catalog with this exact taxonomy, so the
script makes a best-effort automatic guess for them and flags every one
of those rows `needs_review=True`. Read the docstring at the top of
`src/data_acquisition.py` before relying on those.

### 2. Verify the flagged rows, then split

Check `data/curated/needs_review.csv` in the dashboard, move confirmed
rows into `catalog.csv`, then:

```
python scripts/split_dataset.py
```

This produces `train.csv` / `val.csv` / `test.csv`, stratified by
class, with `needs_review` rows excluded by default. The test split is
written once and never touched again by training — see `evaluate.py`.

### 3. Cache features and train

```
python -m src.cache_dataset --csv data/curated/train.csv --out data/processed
python -m src.cache_dataset --csv data/curated/val.csv   --out data/processed
python train.py --epochs 40 --batch_size 32
```

No GPU? CPU training is fine at realistic dataset sizes for a
hackathon-scale project — a few minutes to tens of minutes depending on
how much data you've gathered. For a much larger run, Google Colab's
free tier GPU works too.

### 4. Evaluate honestly

```
python evaluate.py
```

Reports precision/recall/F1/confusion matrix on `test.csv` — a split
the model has never seen, computed once you're actually done tuning.
Don't re-run `train.py` after looking at these numbers and expect the
next `evaluate.py` run to still be a fair test.

---

## Running the Dashboard

```
python -m streamlit run app.py
```

Open **<http://localhost:8501>**. Upload any TESS `.fits` file:

- **Tab 1 — Light Curve:** Raw flux, preprocessed flux, BLS periodogram
- **Tab 2 — Detection:** Phase-folded curve, predicted class + confidence (only if a trained checkpoint exists — see Status)
- **Tab 3 — Parameters:** Period, Duration, Depth, SNR, Confidence
- **Tab 4 — Rejected Detections:** Correctly flagged eclipsing binaries and blends, for transparency
- **Tab 5 — Model Evaluation Metrics:** Whatever `evaluate.py` last produced, with the test-set size shown

### Calibration check

The BLS + preprocessing modules (not the classifier) have been verified
end-to-end against a known target:

```
python -c "
import lightkurve as lk
sr = lk.search_lightcurve('TIC 100100827', mission='TESS', exptime=120)
sr[0].download().to_fits('wasp18_test.fits', overwrite=True)
"
```

Upload `wasp18_test.fits` to the dashboard. Expected:

| Parameter | Expected Value |
| --- | --- |
| Period | ~0.941 days |
| Duration | ~2.4 hours |
| Depth | ~9,960 ppm |
| SNR | ~13–14 |

These match the published WASP-18b ephemeris — good evidence the
physics pipeline (Modules 1, 2, 4) is correct. It says nothing about
the classifier, which is architecture without training data until you
run the steps above.

---

## Architecture: What to Change vs. What Not to Touch

### ✅ Safe to Modify

| File / Setting | What you can change |
| --- | --- |
| `train.py` → `--epochs`, `--batch_size`, `--lr` | CLI args now, not hardcoded |
| `data/curated/catalog.csv` | Add, remove, or relabel rows freely (see CONTRIBUTING.md) |
| `app.py` — UI layout, tab order, colors, text | Cosmetic changes only — math backend is untouched |
| `evaluate.py` | Add or remove metrics as needed |

### ❌ Do Not Touch Without Understanding the Consequences

| File / Setting | Why it's locked |
| --- | --- |
| `src/preprocess.py` → `window_length=401` | Tuned for 2-minute TESS cadence. Shortening it will erase transit dips by treating them as stellar variability. |
| `src/preprocess.py` → `quality_bitmask=175` | Removes corrupted cadences. Changing this passes bad telemetry downstream. |
| `src/bls_detector.py` → period grid (0.5–15 days, 10,000 steps) | Hard-coded to the original hackathon's problem scope. Reducing resolution causes the search to miss short or shallow planetary periods. |
| `src/classifier.py` → layer channels and dimensions | `Conv1d(1→16→32→64)`, `MultiheadAttention(embed_dim=64)` are coupled. Changing one channel without updating downstream dimensions causes tensor shape crashes. |
| `src/bls_detector.py` → output length (200 points) | **Correction from the original README:** this does *not* crash the model at other lengths — `AdaptiveAvgPool1d(32)` makes the classifier shape-compatible with any input length. That's actually worse: the model will silently run on a 150- or 500-point segment and produce a meaningless prediction with no error. Keep this at 200 because that's what any trained checkpoint expects, not because anything will crash if you don't. |

---

## Model Architecture Reference

```
Input: (batch, 1, 200)  ← phase-folded, normalized 200-point segment

Conv1d(1→16, k=7, pad=3) → BatchNorm1d → ReLU → MaxPool1d(2)
Conv1d(16→32, k=5, pad=2) → BatchNorm1d → ReLU → MaxPool1d(2)
Conv1d(32→64, k=3, pad=1) → BatchNorm1d → ReLU → AdaptiveAvgPool1d(32)

Permute → (batch, 32, 64)
MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
Permute back → (batch, 64, 32)

Flatten → Linear(2048→128) → ReLU → Dropout(0.3) → Linear(128→5)

Output: logits over 5 classes (softmax → confidence score)
```

---

## Known Constraints

- **`batman-package`** requires Microsoft C++ Build Tools on Windows; the pipeline falls back to a trapezoid model automatically.
- **No CUDA GPU required.** CPU training works at realistic dataset sizes.
- **Classes 2–4 have no clean public label source** — see `src/data_acquisition.py`'s docstring. This is a real, unsolved data problem, not a bug; contributions here are the most valuable kind (see `CONTRIBUTING.md`).

---

## Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python 3.10–3.14 |
| Astronomy I/O | Lightkurve, Astropy |
| Signal Detection | Astropy BLS (`BoxLeastSquares`) |
| Deep Learning | PyTorch (1D-CNN + MultiheadAttention) |
| Transit Fitting | SciPy `curve_fit` (trapezoid) · batman (optional) |
| Scientific Computing | NumPy, SciPy, Pandas |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Explainability | SHAP |
| Data sources | ExoFOP-TESS, TESS EB Catalog (Prsa et al. 2022), MAST |

---

## Citation

If you use this in research, see `CITATION.cff` (GitHub's "Cite this
repository" button will pick it up automatically).

## License

MIT — see `LICENSE`.

## Project History

Originally built for the **Bharatiya Antariksh Hackathon 2026**
(ISRO Problem Statement 7 — AI-enabled Detection of Exoplanets from
Noisy Astronomical Light Curves, organized by ISRO, powered by
Hack2skill; grand finale August 6–7, 2026). Not selected for the
finale. Open-sourced afterward so the physics pipeline — which is
real and verified — doesn't go to waste, and so the classifier can
actually get trained on real data by whoever wants to pick it up.
