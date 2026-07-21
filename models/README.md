# models/

No checkpoint is committed here.

The version that used to ship in this repo (`transit_cnn.pth`) was
trained on 20 rows of a single duplicated light curve file with
arbitrary labels — it scored 20% accuracy (chance level for 5 classes)
and predicted "Planetary Transit" for every input. Shipping it would
have implied a working classifier existed when it didn't.

To get a real checkpoint:

```
python -m src.data_acquisition      # build a real catalog from ExoFOP + the TESS EB catalog
python scripts/split_dataset.py     # stratified train/val/test split
python -m src.cache_dataset --csv data/curated/train.csv --out data/processed
python -m src.cache_dataset --csv data/curated/val.csv   --out data/processed
python train.py                     # writes models/transit_cnn.pth
python evaluate.py                  # honest metrics on the held-out test set -> models/metrics.json
```

If you train a checkpoint you're willing to stand behind, it's fine to
commit it here (see `.gitignore` — `models/*.pth` is allowed through)
so the Streamlit dashboard has weights to load. Please also commit the
`metrics.json` `evaluate.py` produces alongside it, so anyone using the
dashboard can see what the reported accuracy is actually based on.
