# 🌌 ExosPlore
### AI-Enabled Exoplanet Transit Detection from TESS Light Curves

**Bharatiya Antariksh Hackathon 2026 — Problem Statement 7 (ISRO)**

ExosPlore is an end-to-end AI pipeline that ingests raw TESS space telescope light curves and automatically detects, classifies, and physically characterizes exoplanet transit signals. It distinguishes true planetary transits from look-alike phenomena — eclipsing binaries, stellar blends, and starspots — and reports orbital parameters with confidence scores, all through an interactive dashboard.

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
│   ├── raw/                     # .fits files (git-ignored — large files)
│   └── curated/
│       └── train.csv            # Training index: [tic_id, label, fits_path]
├── models/
│   └── transit_cnn.pth          # Trained PyTorch model checkpoint
├── src/
│   ├── preprocess.py            # Module 1 — physics-aware preprocessing
│   ├── bls_detector.py          # Module 2 — BLS period search & phase-folding
│   ├── classifier.py            # Module 3 — 1D-CNN + Attention model
│   ├── characterizer.py         # Module 4 — trapezoid/batman transit fitting
│   └── pipeline.py              # Orchestrator — wires modules 1–4 end-to-end
├── train.py                     # Standalone model training script
├── evaluate.py                  # Test-set evaluation: F1, confusion matrix
├── app.py                       # Module 5 — Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10 – 3.14
- Windows: ensure **"Add Python to PATH"** was checked during installation

### Install Dependencies

```bash
python -m pip install lightkurve astropy numpy scipy pandas torch torchvision scikit-learn matplotlib plotly streamlit shap requests tqdm
```

> **Note on `batman-package`:** This package requires Microsoft C++ Build Tools on Windows. It is optional — the pipeline automatically falls back to a trapezoid model if batman is not installed.
>
> To install it: download [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), select **"Desktop development with C++"**, then run `python -m pip install batman-package`.

### Add Scripts to PATH (Windows)

If you see warnings like `streamlit.exe is not on PATH`, add this to your user PATH variable:

```
C:\Users\<YOUR_USERNAME>\AppData\Roaming\Python\Python3XX\Scripts
```

Then reopen PowerShell and verify:

```bash
streamlit --version
```

### Verify the Full Installation

```python
# python verify_setup.py
import lightkurve, astropy, numpy, scipy, pandas, torch, sklearn, matplotlib, plotly, streamlit, shap
print("Python:", __import__('sys').version)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
try:
    import batman; print("Batman: OK")
except ImportError:
    print("Batman: not installed — trapezoid fallback active")
print("All checks passed.")
```

---

## Download Test Data

Before the hackathon (do this on reliable internet — venue WiFi is unpredictable):

```python
# python download_test_data.py
import lightkurve as lk, os
os.makedirs("data/raw", exist_ok=True)

targets = [
    ("TIC 100100827", "wasp18"),      # WASP-18b — deep transit, easy to detect
    ("TIC 420814525", "hd209458"),    # HD 209458b — classic benchmark
    ("TIC 59873330",  "toi132"),      # TOI-132b — TESS discovery target
]
for tic, name in targets:
    sr = lk.search_lightcurve(tic, mission="TESS", exptime=120)
    if len(sr) > 0:
        sr[0].download().to_fits(f"data/raw/{name}_test.fits", overwrite=True)
        print(f"Saved: data/raw/{name}_test.fits")
```

---

## Training the Model

### Step 1 — Prepare the Training Index

Open `data/curated/train.csv` and populate it with one row per labelled light curve:

```csv
tic_id,label,fits_path
100100827,0,data/raw/wasp18_test.fits
...
```

**Label mapping:**

| Label | Class |
|---|---|
| `0` | Planetary Transit |
| `1` | Eclipsing Binary |
| `2` | Stellar Blend |
| `3` | Stellar Variability / Starspots |
| `4` | Noise / Unknown |

When ISRO provides the official curated dataset, place the `.fits` files into `data/raw/` and update this CSV to point to them.

### Step 2 — Run Training

```bash
python train.py
```

The script:
1. Reads `train.csv`, routes each file through Modules 1 and 2 to extract a 200-point phase-folded segment
2. Handles class imbalance via `WeightedRandomSampler` (prevents the model from predicting "Noise" for everything)
3. Trains the 1D-CNN + Attention model and saves the best checkpoint to `models/transit_cnn.pth`
4. Prints per-class accuracy and a confusion matrix at the end

### Training Parameters to Adjust

When running on the full ISRO dataset (thousands of rows), open `train.py` and change:

```python
epochs = 3      →  epochs = 40      # or 50
batch_size = 2  →  batch_size = 32  # or 64 depending on RAM
```

> No GPU? Training the small model on CPU takes 10–30 minutes on a typical laptop, which is fine within the 30-hour hackathon window. Alternatively, train on [Google Colab](https://colab.research.google.com) (free T4 GPU, ~5 minutes), download `models/transit_cnn.pth`, and place it locally.

---

## Running the Dashboard

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

Upload any TESS `.fits` file. The pipeline runs automatically and displays:

- **Tab 1 — Light Curve:** Raw flux, preprocessed flux, BLS periodogram
- **Tab 2 — Detection:** Phase-folded curve with transit model overlay, predicted class, confidence bar chart
- **Tab 3 — Parameters:** Period (days), Duration (hours), Depth (ppm), SNR, Confidence %
- **Tab 4 — Rejected Detections:** Correctly flagged eclipsing binaries and blends shown for transparency

### Quick one-line calibration test

```bash
python -c "
import lightkurve as lk
sr = lk.search_lightcurve('TIC 100100827', mission='TESS', exptime=120)
sr[0].download().to_fits('wasp18_test.fits', overwrite=True)
print('Download complete.')
"
```

Then upload `wasp18_test.fits` into the dashboard. Expected results:

| Parameter | Expected Value |
|---|---|
| Period | ~0.941 days |
| Duration | ~2.4 hours |
| Depth | ~9,960 ppm |
| SNR | ~13–14 |
| Classification | Planetary Transit |

---

## Evaluation

Once you have a labelled test set, run:

```bash
python evaluate.py
```

This reports overall accuracy, per-class F1-score, precision, recall, and a confusion matrix. It also displays three example detections: one confirmed planet, one correctly rejected eclipsing binary, and one correctly rejected stellar blend.

---

## Architecture: What to Change vs. What Not to Touch

### ✅ Safe to Modify

| File / Setting | What you can change |
|---|---|
| `train.py` → `epochs`, `batch_size` | Scale up when ISRO data arrives |
| `data/curated/train.csv` | Add, remove, or update rows freely |
| `app.py` — UI layout, tab order, colors, text | Cosmetic changes only — math backend is untouched |
| `evaluate.py` | Add or remove metrics as needed |

### ❌ Do Not Touch Without Understanding the Consequences

| File / Setting | Why it's locked |
|---|---|
| `src/preprocess.py` → `window_length=401` | Tuned for 2-minute TESS cadence. Shortening it will erase transit dips by treating them as stellar variability. |
| `src/preprocess.py` → `quality_bitmask=175` | Removing corrupted cadences. Changing this passes bad telemetry downstream. |
| `src/bls_detector.py` → period grid (0.5–15 days, 10,000 steps) | Hard-coded to ISRO's problem scope. Reducing resolution causes the search to miss short or shallow planetary periods. |
| `src/classifier.py` → layer channels and dimensions | `Conv1d(1→16→32→64)`, `AdaptiveAvgPool1d(32)`, `MultiheadAttention(embed_dim=64)` are coupled. Changing one channel without updating downstream dimensions causes tensor shape crashes. |
| `src/bls_detector.py` → output length (200 points) | The CNN input is hardcoded to accept exactly 200 points. Changing this breaks the model. |

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

- **`batman-package`** requires Microsoft C++ Build Tools on Windows. The pipeline automatically uses a trapezoid fallback if batman is not available. This is fully acceptable for evaluation.
- **No CUDA GPU required.** Training runs on CPU. For faster training, use Google Colab and transfer the `.pth` checkpoint.
- **Python 3.14** is fully supported. All packages verified with the latest wheels.
- **ISRO curated dataset** has not yet been received. The current `transit_cnn.pth` was trained on a minimal mock setup for verification only. Retrain with `epochs=40`, `batch_size=32` once the official data arrives.

---

## Verified Results (WASP-18b Calibration Run)

| Metric | Value |
|---|---|
| Period | 0.94084 days |
| Transit Duration | 2.41 hours |
| Transit Depth | 9964.1 ppm |
| SNR | 13.78 |
| Pipeline Status | ✅ End-to-end success |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10–3.14 |
| Astronomy I/O | Lightkurve, Astropy |
| Signal Detection | Astropy BLS (`BoxLeastSquares`) |
| Deep Learning | PyTorch (1D-CNN + MultiheadAttention) |
| Transit Fitting | SciPy `curve_fit` (trapezoid) · batman (optional) |
| Scientific Computing | NumPy, SciPy, Pandas |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Explainability | SHAP |
| Version Control | Git + GitHub |

---

## Hackathon Context

- **Event:** Bharatiya Antariksh Hackathon 2026
- **Organizer:** ISRO, powered by Hack2skill
- **Problem Statement:** 7 — AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves
- **Grand Finale:** August 6–7, 2026 (30-hour window)
- **Data Source:** TESS light curves from [MAST Archive](https://archive.stsci.edu/tess/tic_ctl.html)

---

*Built for ISRO BAH 2026. Pipeline verified on real TESS telemetry (WASP-18b, TIC 100100827).*