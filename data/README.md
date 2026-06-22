# Data Notes

Raw datasets are not included.

Use `code/real_prep.py` to document or rebuild the real-data caches from the
public sources after verifying access and license terms. The duplicate-safe run
expects the project-local `02_data/` cache files:

- `creditcard_pi10.npz`
- `creditcard_pi5.npz`
- `iotid20_compact.npz`

The package includes dataset and split manifests in `results/`, not raw data.
