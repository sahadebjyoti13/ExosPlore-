"""
ExosPlore — Environment Verification Script
Run this after installing dependencies to confirm everything is working.
Expected: all packages print version numbers, no ImportError.
"""

import sys
print(f"Python:       {sys.version}")
print()

checks = [
    ("numpy",       "NumPy"),
    ("scipy",       "SciPy"),
    ("pandas",      "Pandas"),
    ("astropy",     "Astropy"),
    ("lightkurve",  "Lightkurve"),
    ("torch",       "PyTorch"),
    ("sklearn",     "Scikit-learn"),
    ("matplotlib",  "Matplotlib"),
    ("plotly",      "Plotly"),
    ("streamlit",   "Streamlit"),
    ("shap",        "SHAP"),
]

all_ok = True
for module, label in checks:
    try:
        m = __import__(module)
        version = getattr(m, "__version__", "installed")
        print(f"  ✓ {label:<18} {version}")
    except ImportError:
        print(f"  ✗ {label:<18} NOT FOUND — run: python -m pip install {module}")
        all_ok = False

import torch
print(f"\n  CUDA available:  {torch.cuda.is_available()}")
print(f"  Device:          {'GPU — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (training will use CPU)'}")

try:
    import batman
    print(f"\n  ✓ batman-package  installed — physical transit models enabled")
except ImportError:
    print(f"\n  ⚠ batman-package  not installed — trapezoid fallback will be used (acceptable)")

print()
if all_ok:
    print("All core checks passed. Environment is ready.")
else:
    print("Some packages are missing. Run: python -m pip install lightkurve astropy numpy scipy pandas torch torchvision scikit-learn matplotlib plotly streamlit shap requests tqdm")
