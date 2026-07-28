# Data Notes

Raw datasets are not included and must not be committed to this repository.

Use `code/real_prep.py` to document or rebuild the real-data caches from the
public sources after verifying access and license terms. The duplicate-safe run
expects the project-local `02_data/` cache files:

- `creditcard_pi10.npz`
- `creditcard_pi5.npz`
- `iotid20_compact.npz`

The package includes dataset and split manifests in `results/`, not raw data.

## Sources and terms

- Credit Card Fraud Detection: obtain version 3 from
  <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>. The official
  Kaggle record identifies the owner as Machine Learning Group - ULB and labels
  the license `Database: Open Database, Contents: Database Contents`.
- IoTID20: obtain the data through the author-maintained page at
  <https://sites.google.com/view/iot-network-intrusion-dataset/home>. The page
  grants perpetual free use for academic research and requires citation of
  Ullah and Mahmoud, *A Scheme for Generating a Dataset for Anomalous Activity
  Detection in IoT Networks*, DOI
  <https://doi.org/10.1007/978-3-030-47358-7_52>.
- Breast Cancer Wisconsin is loaded through
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html>.
- The twelve expansion datasets come from the Zenodo collection at
  <https://doi.org/10.5281/zenodo.61452> and are loaded through
  <https://imbalanced-learn.org/stable/references/generated/imblearn.datasets.fetch_datasets.html>.
- Synthetic datasets are generated locally with
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html>.

Download the raw files from their canonical sources and review the current
source terms before use. This repository distributes only code, manifests, and
aggregate derived evidence.
