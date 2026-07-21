# Contributing to ExosPlore

Thanks for considering contributing. The two most valuable things this
project needs right now are **more verified training data** and
**scientific review of the pipeline itself** — not just code.

## Ways to help

### 1. Verify a `needs_review` light curve
`scripts/split_dataset.py` sets aside rows that were auto-labeled by
keyword-matching or a "no BLS peak" heuristic (see the docstring in
`src/data_acquisition.py` for exactly which classes/why). To help:

1. Run the dashboard (`python -m streamlit run app.py`) and open a
   `.fits` file listed in `data/curated/needs_review.csv`.
2. Look at the phase-folded curve and BLS periodogram.
3. If you agree with the assigned label, move that row into
   `data/curated/catalog.csv` with `needs_review` set to `False`.
   If you disagree, correct the `label` column instead, or drop the row
   if it's unusable. Open a PR with the updated CSVs.

You don't need to be a professional astronomer to help here, but please
say in the PR description what you based the call on (periodogram
shape, comparison to a known catalog, etc.) so it's reviewable.

### 2. Expand the label sources
Right now Class 0 (planets) and Class 1 (eclipsing binaries) are
sourced from real, peer-reviewed catalogs (ExoFOP TOI table, Prsa et
al. 2022 TESS EB catalog). Classes 2–4 don't have an equivalent clean
source. If you know of one — a published stellar-blend catalog, a
starspot/rotation catalog with TIC cross-matches, etc. — please open an
issue or PR wiring it into `src/data_acquisition.py`.

### 3. Code contributions
- Run `python -m pytest` (see `tests/`) before opening a PR.
- Don't touch the locked architecture parameters documented in the
  README's "What to Change vs. What Not to Touch" section without also
  updating every downstream shape assumption — that's how the model
  breaks silently.
- If you change `src/preprocess.py` or `src/bls_detector.py`'s output
  format, existing cached `.npy` files (from `src/cache_dataset.py`)
  become stale — regenerate them.

### 4. Reporting issues
Please include: the `.fits` file (or a MAST/TIC identifier if you can't
share the file), the exact command you ran, and the full traceback.

## Code of conduct
Be respectful. This is a scientific tool other researchers may rely on
— correctness and honest reporting of limitations matter more than
looking impressive.
